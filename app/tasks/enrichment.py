"""Celery tasks for data enrichment via Apollo.io."""

import asyncio
import logging
from uuid import UUID

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database_celery import celery_session
from app.models.company import Company
from app.services.enrichment import ApolloEnrichmentService
from app.tasks.celery_app import celery_app
from app.tasks.org_runner import for_each_org, org_scope

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async code in Celery's sync context with a fresh event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="app.tasks.enrichment.enrich_company", bind=True, max_retries=3)
def enrich_company(self, company_id: str):
    """Enrich a single company via Apollo.io API.

    Args:
        company_id: UUID string of the company to enrich
    """
    logger.info("Enriching company %s", company_id)

    async def _run():
        service = ApolloEnrichmentService()
        async with celery_session() as db:
            try:
                # Per-record task: context is lost over the Celery wire, so load
                # the company unfiltered and bind ITS org before enriching, so
                # contacts/writes land in the owning tenant.
                result = await db.execute(
                    select(Company)
                    .where(Company.id == UUID(company_id))
                    .execution_options(skip_org_filter=True)
                )
                company = result.scalar_one_or_none()
                if not company:
                    return {"status": "not_found", "company_id": company_id}

                async with org_scope(company.org_id):
                    enrich_result = await service.enrich_company(db, company)
                    contacts = await service.find_contacts(db, company)
                    await db.commit()

                return {
                    "company_id": company_id,
                    "enrichment": enrich_result,
                    "contacts_found": len(contacts),
                }
            except Exception as e:
                await db.rollback()
                raise

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("Enrichment failed for %s: %s", company_id, str(exc))
        raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))


@celery_app.task(name="app.tasks.enrichment.enrich_batch", bind=True)
def enrich_batch(self, company_ids: list[str]):
    """Enrich multiple companies. Dispatches individual tasks.

    Args:
        company_ids: List of UUID strings
    """
    logger.info("Dispatching enrichment for %d companies", len(company_ids))
    results = []
    for cid in company_ids:
        task = enrich_company.delay(cid)
        results.append({"company_id": cid, "task_id": task.id})
    return {"dispatched": len(results), "tasks": results}


@celery_app.task(name="app.tasks.enrichment.auto_enrich_companies")
def auto_enrich_companies():
    """Periodic task: find companies missing industry or employee data and enrich via Apollo."""
    logger.info("Auto-enriching companies missing data")

    async def _run():
        service = ApolloEnrichmentService()

        # Per-org: companies are org-scoped.
        async def _body(db):
            # Find companies missing key data that have a domain (needed for Apollo)
            result = await db.execute(
                select(Company)
                .where(
                    Company.domain.isnot(None),
                    Company.domain != "",
                    or_(
                        Company.industry_segment.is_(None),
                        Company.industry_segment == "",
                        Company.employee_count.is_(None),
                    ),
                )
                .limit(10)  # Process in small batches to avoid API rate limits
            )
            companies = list(result.scalars().all())

            if not companies:
                logger.info("No companies need enrichment")
                return {"enriched": 0}

            enriched = 0
            for company in companies:
                try:
                    await service.enrich_company(db, company)
                    enriched += 1
                except Exception as e:
                    logger.warning("Auto-enrich failed for %s: %s", company.name, str(e))
                    continue

            await db.commit()
            logger.info("Auto-enriched %d / %d companies", enriched, len(companies))
            return {"enriched": enriched, "total_candidates": len(companies)}

        return await for_each_org(_body, label="auto_enrich_companies")

    return _run_async(_run())


async def _score_with_org_keys(db: AsyncSession, company: Company, ai) -> dict | None:
    """Score one company on the org's own LLM keys, or ``None`` if that fails too.

    ``AIService.score_lead`` runs on the deployment's keys and degrades to a
    neutral 50 when they are exhausted — which is what happened when the
    platform Anthropic key ran out of credit and Gemini was rate-limited: every
    lead in every org scored "C, review manually". Discovery already falls back
    to the org's own keys (``org_gemini``: Gemini → Anthropic → Groq →
    OpenRouter); scoring now does the same before settling for the default.
    """
    from pydantic import BaseModel, Field

    from app.org_context import get_current_org_id
    from app.services import org_gemini

    class _LeadScoreAnswer(BaseModel):
        score: int = Field(ge=0, le=100)
        grade: str = "C"
        reasoning: str = ""
        recommended_action: str = ""
        priority: str = "medium"

    org_id = get_current_org_id()
    if org_id is None:
        return None
    prompt = ai.lead_score_prompt(
        company.name,
        company.industry_segment or "Unknown",
        company.employee_count,
        f"{company.city or ''}, {company.state_province or ''}, {company.country or ''}".strip(", "),
    )
    try:
        answer = await org_gemini.generate_json(
            db,
            org_id,
            system="You score B2B leads. Answer with a single JSON object only.",
            prompt=prompt,
            schema=_LeadScoreAnswer,
            max_tokens=1024,
        )
    except (org_gemini.GeminiKeyMissing, org_gemini.GeminiError) as exc:
        logger.info("Org-key lead scoring unavailable for %s: %s", company.name, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — never fail the batch over a fallback
        logger.warning("Org-key lead scoring failed for %s: %s", company.name, exc)
        return None
    return answer.model_dump()


@celery_app.task(name="app.tasks.enrichment.auto_score_leads")
def auto_score_leads():
    """Periodic task: find companies with no AI score and score them."""
    logger.info("Auto-scoring unscored leads")

    async def _run():
        from app.services.ai_service import get_ai_service
        from app.models.lead_score import LeadScore

        ai = get_ai_service()

        # Per-org: companies and their lead scores are org-scoped.
        async def _body(db):
            # Get IDs of companies that already have AI scores
            scored_result = await db.execute(
                select(LeadScore.company_id).where(
                    LeadScore.score_type == "ai"
                ).distinct()
            )
            scored_ids = set()
            for row in scored_result.all():
                scored_ids.add(row[0])

            # Get unscored companies
            if scored_ids:
                result = await db.execute(
                    select(Company).where(
                        Company.id.notin_(scored_ids)
                    ).limit(10)
                )
            else:
                result = await db.execute(
                    select(Company).limit(10)
                )
            companies = list(result.scalars().all())

            if not companies:
                logger.info("No companies need scoring")
                return {"scored": 0}

            scored = 0
            for company in companies:
                try:
                    score_result = await ai.score_lead(
                        company_name=company.name,
                        industry=company.industry_segment or "Unknown",
                        employee_count=company.employee_count,
                        location=f"{company.city or ''}, {company.state_province or ''}, {company.country or ''}".strip(", "),
                    )

                    if score_result.get("degraded"):
                        # The deployment's keys couldn't answer; try the org's own.
                        org_result = await _score_with_org_keys(db, company, ai)
                        if org_result is not None:
                            score_result = org_result

                    if "error" not in score_result:
                        # Map AI response to LeadScore model fields
                        raw_score = score_result.get("score", 50)
                        lead_score = LeadScore(
                            company_id=company.id,
                            score_type="ai",
                            fit_score=raw_score,
                            factors={
                                "grade": score_result.get("grade", "C"),
                                "reasoning": score_result.get("reasoning", ""),
                                "recommended_action": score_result.get("recommended_action", ""),
                                "priority": score_result.get("priority", "medium"),
                            },
                        )
                        db.add(lead_score)

                        # Update company stage if still 'new'
                        if company.stage == "new":
                            company.stage = "scored"

                        scored += 1
                except Exception as e:
                    logger.warning("Auto-score failed for %s: %s", company.name, str(e))
                    continue

            await db.commit()
            logger.info("Auto-scored %d / %d companies", scored, len(companies))
            return {"scored": scored, "total_candidates": len(companies)}

        return await for_each_org(_body, label="auto_score_leads")

    return _run_async(_run())
