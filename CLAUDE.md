# CLAUDE.md

This file tells any Claude Code session everything it needs to know to work on the prospect-backend codebase effectively.

## Project Overview
Landline is a B2B lead generation and client management platform. FastAPI backend with SQLAlchemy async, PostgreSQL, Redis, Celery Beat.

## Tech Stack
- Python 3.11, FastAPI, SQLAlchemy async, Alembic migrations
- PostgreSQL (docker, port 5433→5432, db: landline_db)
- Redis (port 6380→6379)
- Celery Beat for scheduled automation
- Anthropic Claude API for AI features (tactics, news analysis, scoring)
- Gmail OAuth + SMTP for email sending

## Code Structure
- `app/routers/` — API endpoint files (companies, contacts, email_sends, sequences, outreach, news, scoring, tactics, gmail, smtp, deals, tasks, interactions, pipeline, suggestions, reports)
- `app/models/` — SQLAlchemy models
- `app/services/` — Business logic (health scoring, email dispatch, etc.)
- `app/config.py` — Settings from environment
- `alembic/` — Database migrations (currently at 018)
- `tests/` — Pytest test suite (175+ tests)

## Common Tasks

### Run tests
```bash
pytest tests/ -v
```

### Create a new migration
```bash
alembic revision --autogenerate -m "description"
```

### Run migrations locally
```bash
alembic upgrade head
```

### Add a new router
1. Create `app/routers/new_router.py`
2. Register in `app/main.py`: `app.include_router(new_router.router, prefix="/api/v1/new", tags=["new"])`
3. Add tests in `tests/test_new.py`

## Deploy
Push to main. Deploy is handled externally via SSM to EC2 at `/opt/landline`.
- `git pull origin main`
- `docker compose -f docker-compose.prod.yml build`
- `docker compose -f docker-compose.prod.yml up -d`
- `docker compose -f docker-compose.prod.yml exec -T api alembic upgrade head`

## Conventions
- Push directly to main, no PRs
- Every new feature needs tests
- Use async SQLAlchemy patterns (see existing routers for examples)
- Pipeline stages: `new_lead → researching → outreach → engaged → negotiating → client → churned`
- API responses follow existing patterns — check similar routers
- Alembic migrations must chain correctly (check `down_revision`)

## Environment Variables
Key vars in `.env.production`: `DATABASE_URL`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `APOLLO_API_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, `FRONTEND_URL`, `FERNET_KEY`

## AI provider chain
Every AI feature routes through `AIService` → `app/services/llm_providers.py`, which
tries providers in order (`LLM_PROVIDER_CHAIN`, default
`openrouter,gemini,groq,cerebras`), then Anthropic, then the caller's static
template. Keys: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`,
`CEREBRAS_API_KEY` — an unkeyed provider is skipped, not attempted.

A provider failing 3× in a row is skipped for 5 minutes
(`LLM_BREAKER_THRESHOLD` / `LLM_BREAKER_COOLDOWN_SECONDS`).

### Rate limits are a routing decision, not a wait

`app/services/llm_rate_limit.py` paces each provider to its own cap
(`LLM_PROVIDER_RPM`, `"name:rpm"` pairs) and honours `Retry-After` on a 429 by
parking that provider. The breaker is the wrong instrument for this in both
directions: it needs three consecutive failures, so a batch send hits the cap and
then burns three more round trips before anything stops — and then it blackholes
a *healthy* provider for five minutes over a limit that clears in one.

Three rules hold this together:

- **Throttled is not failed.** A paced or parked provider records no failure and
  never moves the breaker. Being early is not being broken.
- **Don't wait when another provider is idle.** `LLM_RATE_LIMIT_MAX_WAIT_SECONDS`
  (3s) is deliberately small — the chain's whole premise is an independent quota
  one hop away. The exception is when *every* provider is paced out before a byte
  is sent, where returning would degrade the caller to static mail-merge copy
  because we were two seconds early; then the chain stalls for the soonest one, up
  to `LLM_RATE_LIMIT_MAX_STALL_SECONDS`. It does **not** stall after real 429s —
  those round trips are already spent.
- **A park picks providers; it doesn't interrupt one mid-call.** The slot is
  claimed once per provider, before its model list. OpenRouter's free model 429ing
  is exactly when the paid sibling — separate allowance — should be tried, and a
  success clears the park, because a provider that just served us is not limited.

`GET /api/v1/ai/providers` reports `rate_limits` next to `circuit_breakers`:
"outreach went out as templates" and "outreach went out personalized but
throttled" are otherwise indistinguishable from outside the process.

Services that predate `AIService` and call `client.messages.create(...)` directly
(sales coach, orchestrator, meeting/relationship/competitive intelligence,
sentiment, script generator, investor outreach) use `chain_client()` from the
same module — it keeps the Anthropic call shape and response
(`message.content[0].text`) but serves it from the chain. Gate those on
`any_llm_available()`, **not** on `settings.anthropic_api_key`: a box carrying
only a Groq key has working AI, and the narrow gate would report it dark.

**Never add a bare `httpx.post` to a model endpoint, and never construct
`anthropic.AsyncAnthropic` in a service.** Twice now a single-provider
call has run out of quota and silently degraded outreach to static mail-merge
copy while every log line said "fine". `GET /api/v1/ai/providers` reports which
provider is actually serving.

## Tenant isolation is opt-in, so the opt-in is the thing to check

A model is isolated because it mixes in `OrgScopedMixin`. That gives it a NOT
NULL `org_id` and enrols it in the `do_orm_execute` filter that scopes every
SELECT to the caller's org. A model that doesn't mix it in is not "unscoped
pending review" — it is a table every customer on the platform shares, and it
looks identical from every angle except the mixin.

`company_groups` was one for two years. The chain that hid it is worth knowing,
because it is the chain, not the missing mixin, that will do it again:

- `app/models/__init__.py` did not import `company_group`, so the class was
  never registered on `Base.metadata`;
- `alembic/env.py` built `target_metadata` from a hand-written list of 19 module
  names, last extended around migration 030, so it didn't have it either;
- migration 078 — the autogenerated one that put `org_id` on 71 tables —
  therefore had nothing to notice, and skipped it.

`GET /groups` listed every org's groups, and the id it returned was enough to
rename, delete or re-populate another tenant's folder. Auto-grouping was the
quiet one: it asks "is there already a group called 'Aerospace'?", found another
tenant's, and filed this tenant's companies into it. Migration 111 fixed the
table; `alembic/env.py` now imports the package.

Three rules:

- **Every new model mixes in `OrgScopedMixin`** unless it is genuinely
  platform-wide, and every new model is imported in `app/models/__init__.py`.
  `tests/test_org_scoping_coverage.py` enforces both — it walks the model
  *directory*, not the package, so a missing import cannot hide a model from it.

  The registration half of that check runs in a **subprocess**, and has to.
  `Base.registry` is process-global and only grows, so once conftest and the
  tests above it have imported every module, "what does `import app.models`
  register?" can no longer be asked in-process — the first version of this check
  passed with the `company_group` import deleted again, i.e. against the exact
  bug it was written for. Any future check of the form "is X reachable from a
  bare import?" has the same trap.
- **Three models are exempt, and the reasons don't generalise.**
  `Organization` (scoping the tenant to a tenant is circular), `TenantUsage`
  (keyed by `tenant_id` already), and `User` (membership *is* the `org_id`
  column; login and the platform-admin cross-org views read users before any org
  context is bound).

  **Every `select(User)` scopes itself.** There is no filter behind it, and an
  unscoped one is invisible on review precisely because every *other* query in
  the codebase looks identical and is scoped. Use
  `dependencies.scope_users_to_org(stmt, caller)` when a caller is in hand, or
  `scope_users_to_current_org(stmt)` in services and Celery tasks, which have an
  org binding instead. Both exempt `platform_admin`, which exists to cross orgs.

  Four hand-rolled copies of the predicate existed and three call sites had
  none: `?view_as_user_id=`, `GET /coaching/team` (which enumerated every user
  on the platform *and* wrote a `RepPerformance` row for each, stamped with the
  caller's org), and `POST /coaching/generate?target_user_id=`. The worst was
  `generate_weekly_coaching`: it runs inside `for_each_org`, which binds each org
  precisely so scoped reads come out right, and then read users — so every org's
  pass generated snapshots for every user on the platform, N-1 of them attributed
  to the wrong tenant, every week.

  Reading users is not automatically a leak — `compute_team_benchmarks` was safe
  without scoping because every *metric* it reads is org-scoped, so a foreign rep
  came back with no activity and was dropped. That is a property of the tables it
  happens to read, not a rule; it scopes its user query anyway.
- **The filter skips relationship *lazy* loads** (`is_relationship_load`), but
  loader criteria injected into a statement do propagate into its `selectinload`
  legs. That is why a foreign group came back reading `member_count: 0`: the
  group row leaked, the companies did not. Don't rely on it for a new surface —
  an association table like `company_group_members` has no `org_id` of its own,
  so what scopes it is the owning row you looked up to reach it.

## The read filter is a *read* filter

`do_orm_execute` checks `execute_state.is_select`. A Core `update()` or
`delete()` is not a select, so it gets no `org_id` predicate and runs across
every tenant on the platform.

What makes this the hardest of the tenancy bugs to see is that the unsafe
statement is written exactly like the safe ones around it. Everywhere else in
this codebase, **not writing an org predicate is the correct shape** — the
filter supplies it. So an unscoped bulk write has nothing to notice, in a file
full of queries that also have no visible scoping and are fine.

`app.org_context.scope_write_to_current_org(stmt, Model)` is the write-side
counterpart. Its semantics match the read filter deliberately: no org bound
(platform admin, or a background job outside `for_each_org`) means no filter.

Most bulk writes don't need it — they pin a row by a primary key read back
through a scoped SELECT, which already proved ownership. Two shapes do:

- **a broad predicate.** `where(is_default == True)` in the template and
  signature routers cleared *every tenant's* default when one org promoted its
  own — and the default is what unattended send paths resolve when no template
  is named, so it decided which copy a stranger's next mail went out as.
- **client-supplied ids.** `POST /companies/bulk-update-stage` took
  `where(Company.id.in_(payload.company_ids))` straight off the request body: a
  write IDOR against any company on the platform. Ids are not secrets; they come
  back in exports, webhooks and error text. Note that `bulk_tag` directly below
  it takes the same ids and was never exposed, because it routes through
  `select(Company)` and mutates the objects it gets back. The SELECT is scoped,
  so foreign ids never come back. That is the entire difference between the two
  endpoints, and it is invisible.

The kill switch was the worst instance and is worth keeping in mind, because
being *half* scoped was what made it dangerous. `system_settings` is per-tenant
(migration 080 made `(org_id, key)` unique precisely so each org could hold its
own switch), and only the enrollment cascade was global. So the control reads as
scoped in the model, in the database and in the UI — and halted the platform.
The resume side did not merely restart strangers' outreach, it **defeated their
kill switch**: org B pauses because its domain is on fire, org A releases its own
switch, B starts sending again with B's UI still showing it engaged.

`clear_expired_snoozes` shows the quiet version. It runs inside `for_each_org`,
so the first org's pass cleared every tenant's expired snoozes and was credited
with the platform's count. The end state was correct — clearing an elapsed
snooze is idempotent and every org's rows were due — which is exactly why it
would have sat there: the loop has been touching every tenant's rows on pass one
all along, and the first per-org side effect anyone adds inherits that. Same
shape as `generate_weekly_coaching`, which *did* have one.

`tests/test_bulk_write_scoping.py` enforces this mechanically, for the same
reason the `scalar_one_or_none` scan does — there is nothing to see at the call
site. It carries self-tests that plant violations, including an `id.in_(...)`,
so a checker that merely asked "is the primary key mentioned?" fails its own
suite. It does **not** catch an `id ==` whose id came from an unscoped source,
association-table writes, or raw `text()` SQL.

## The filter needs an org to filter by, and one line supplies it

Exactly one thing in a request binds the caller's tenant:
`dependencies.get_current_user` calls `set_current_org_id(user.org_id)`. A route
that never runs it takes the **documented no-op branch** — the same one a
platform admin takes — so every query in it runs unfiltered.

What decides that, for a whole router, is one keyword argument in `app/main.py`:
`dependencies=_auth`. It sits four screens away from the endpoints it governs,
and a router without it looks correct from inside its own file.

`health.router` sat that way. It reads as a liveness probe — the module is
called `health`, the router is mounted alongside one — and it serves **client
health scoring** on `/companies/{id}/health`. With no org bound, any logged-in
user of any tenant could pull another org's account health and snapshot series
by company id, and `POST /health/snapshot` **wrote** a row: no org context also
means `org_id_for_insert()` falls back to the first-party org, so the row landed
under Apprend's org pointing at a stranger's company, in that tenant's own
history. Ids are not secrets — they come back in exports, webhooks and error
bodies.

Two things made it survive review:

- **`ApiKeyMiddleware` fails closed, so the route *was* authenticated.** Every
  reviewer's instinct that you need a valid token to reach it is correct and
  beside the point: the request was authenticated and simply not attributed to
  anyone. Authentication and tenancy are separate mechanisms here and only one
  of them is in the middleware.
- **The endpoint has nothing to see.** It reads `Company` by id like every
  scoped read in the codebase, because in this codebase not writing an org
  predicate *is* the correct shape.

`tests/test_route_org_binding.py` walks the live route table and requires every
route to either reach an org-binding dependency or be named in `PUBLIC_ROUTES`
with a reason — login, the OAuth callbacks, the tracking pixel, the one-click
unsubscribe, the HMAC'd events webhook. Adding a router and forgetting
`dependencies=_auth` fails there instead of shipping. It self-tests by mounting
a router the way `health` was mounted and asserting the scan names it.

## `scalar_one_or_none` asserts a uniqueness the schema rarely promises

This has now cost four times, in unrelated code, the same way every time: the
query reads as the careful option, and is in fact an assertion that raises
`MultipleResultsFound` — a 500 — the moment a second row exists.

- `contacts.email` has no unique constraint and the scraper re-creates rows, so
  one duplicate contact broke **every unsubscribe link** (see the opt-out
  section). The same duplicate also aborted the **inbox sync**, where the
  failure is worse than loud: a mailbox that stops ingesting replies looks
  exactly like a mailbox with no replies.
- `company_groups.name` has no unique constraint and `POST /groups` accepts any
  name, so a tenant that created "Aerospace" twice got a 500 from **all three**
  `/groups/auto-group-by-*` endpoints, with no way out that the UI explains.
- `gmail_accounts` is unique on `email` and `google_sub` and nothing else, so
  `get_active_account` broke on the **second connected mailbox** — while the
  function directly below it returns a *list* of connected accounts, because
  multi-mailbox sending is the whole design. `welcome_email` and `notifications`
  call it with no rotation in front, so for them it was unconditional.
- `companies` is unique on `domain`, not `name`. Two firms sharing a name is
  ordinary B2B data, and it 500'd **quick-add** — the business-card scanner's
  entire job.

Before writing one, check for the constraint. If there isn't one, take
`.order_by(...).limit(1)` and `.scalars().first()` — and order by something
stable, or repeat runs pick different rows and the work scatters between
duplicates instead of accumulating in one. For a pure existence check,
`select(Model.id)…limit(1)` is both correct and cheaper.

Two things this experience is worth:

**Order by more than a timestamp.** `created_at` is a `server_default`, so rows
written in one transaction — which is how duplicates usually arrive — share a
reading and tie. Without a second key the database may return either, which is
the scattering above with extra steps. `(<timestamp>, id)`.

**Reviewing for this does not work, so it is checked mechanically.** The unsafe
call is character-for-character identical to the safe one; what separates them
lives in another file, in the model's constraints.
`tests/test_single_row_assertions.py` walks every `scalar_one_or_none` in `app/`
and requires each to be covered by a `.limit(1)`, by an equality filter on
genuinely unique columns, or by a reasoned entry in its `REVIEWED` map. It
carries self-tests that plant a violation and assert the scan catches it — a
checker matching nothing passes forever and reads as coverage.

An "existence check" is the most common false sense of safety here: code that
asks "is there already one?" and enforces uniqueness itself. It is not a
constraint. It cannot stop a race, it does not bind other write paths, and the
row it fails on is the one it was written to detect.

## The rate-limit key is the caller, and behind a proxy that takes saying so

`request.client.host` is the peer that opened the socket, which is the caller
only when nothing sits in front of the app. In production nginx proxies to
`127.0.0.1:8001` — a Docker published port — so the peer every request presents
is the bridge gateway. Uvicorn ships a `ProxyHeadersMiddleware` that would fix
this and does not: its `forwarded_allow_ips` defaults to `127.0.0.1`, the
gateway is not that, so it declines to rewrite anything and says nothing.

That is not a loose limit, it is an inverted one. `AUTH_RATE_LIMIT` is
10/minute, and on a single shared key that is ten logins a minute **for the
whole platform**, spendable by any unauthenticated stranger. A control installed
to stop credential stuffing became the cheapest denial of service the API
offers. Note the shape: the limiter was working exactly as designed and the
tests were green, because what was wrong was the identity it was counting.

`TRUSTED_PROXY_HOPS` (set to `1` in `docker-compose.prod.yml`) is how the app is
told what is in front of it. `X-Forwarded-For` cannot simply be trusted — it is
attacker-controlled, and reading its leftmost entry lets anyone mint a fresh
identity per request, removing the limit entirely rather than sharing it. What
*is* trustworthy is position: each proxy appends the peer it saw, so with one
hop the **last** entry is the one nginx wrote and no client can reach. Hence a
count rather than a boolean, a default of 0, and a clamp so an over-configured
count cannot walk left into client-supplied entries.

Two related traps in slowapi, both of which 500 the endpoint:

- **`headers_enabled` must stay False.** It injects `X-RateLimit-*` by reaching
  for a `response: Response` kwarg the endpoints don't declare.
- **`swallow_errors` is necessary and not sufficient.** When storage raises,
  slowapi catches it and returns without setting `request.state.view_rate_limit`
  — which its endpoint wrapper then reads unconditionally, before any
  `headers_enabled` check. `RateLimitStateMiddleware` pre-seeds the attribute.
  Without it a Redis blip returns a clean 401 for every *rejected* login and a
  500 for the correct password: failures fine, success broken, and invisible to
  any test that only checks that bad credentials are refused.

`conftest` sets `limiter.enabled = False` for the whole session, which is the
right default — it stops unrelated tests tripping a 429 — and it means a
rate-limit test that forgets to re-enable it asserts nothing while looking
exactly like one that works. Anything testing limiter behaviour needs the
`live_limiter` fixture in `tests/test_rate_limit_storage_outage.py`, and needs a
control proving the limiter was on.

## The Apprend org id is copied into every migration, so it is checked

Migrations inline `APPREND_ORG_ID = "a9910000-0000-4000-8000-000000000001"`
rather than importing `app.org_context.APPREND_ORG_ID`, and should keep doing
so: a migration describes the schema at its own revision and must not change
meaning when application code is refactored. The cost is a literal copied a
dozen times with nothing comparing the copies, and two of them read
`00000000-0000-0000-0000-000000000001` — an id that looks exactly like what a
first-party tenant would have and matches no row in `organizations`.

`tests/test_migration_org_id_constant.py` pins every copy. It is worth having
because the two failure modes look nothing alike:

- **111** backfilled `company_groups.org_id` with the bad id and *then* added the
  foreign key, so it aborts mid-deploy on any database that has ever held a
  group — which is every real one.
- **081** used it as a `server_default` on two new, empty tables. Nothing failed,
  and nothing would: ORM inserts are stamped by the `before_flush` hook before
  the default is consulted. The default is what a raw insert, a backfill script
  or a `COPY` falls back on, and there it is an immediate FK violation. Migration
  112 corrects the databases already carrying it; 081's own literal is corrected
  for databases built from scratch. **Both are needed** — editing the old
  migration only fixes future builds, and adding the new one only fixes existing
  ones.

## Sent mail lives in two tables

There is **no single "sent emails" table**. A send lands in one of two places
depending on which path put it on the wire:

- `email_sends` (`EmailSend`) — manual sends, template sends, inbox replies,
  smart-reply auto-sends.
- `outreach_events` (`OutreachEvent`, `event_type='sent'`) — **the sequence
  engine** (`app.tasks.outreach.send_sequence_step`), which is where the bulk of
  outreach volume goes. It writes **no** `EmailSend` row; the message body,
  subject, recipient and dispatcher result live in the event's `metadata`.

Anything that answers "what did we send?" must read **both**. Reading only
`EmailSend` is how the unified inbox went blind to every sequence send — mail
sitting in Gmail's Sent folder, nothing in Landline, and prospect replies shown
as threads with no opening message. `EmailHistoryService._fetch_sequence_sends`
is the read-side union (inbox, thread view, lead view); mirror it rather than
adding a second write path, because the send accounting (warm-up counters,
deliverability, A/B attribution) is keyed on today's split.

**The meeting-booking agent wrote to neither**, which is the third state and the
easiest to miss: not a half-migrated *read*, an unrecorded *write*. Its four
sends (propose, re-propose, confirm, remind) went out over a `GmailSender` and
left nothing behind anywhere. The inbox showed a prospect's "Tuesday works" as a
thread with no opening message, and the daily cap's row-count seed forgot the
send. The one that bit hardest is circular: `_is_outreach_reply` decides whether
an inbound message is outreach or inbox noise, and all three of its tests read
`EmailSend` or an enrollment — so a proposal on a *fresh* thread to a contact
with no live enrollment matched none of them, the reply was dropped as noise,
and `check_booking_replies` polled `EmailReply` by thread for a row nothing had
written. **The agent's own loop depended on the record it wasn't keeping.**
`_record_sent_mail` writes the `EmailSend` row after a landed send.

**`status='skipped'` is not a send.** `dashboard.SENT_STATUSES` and
`platform._SENT_STATUSES` were `("sent", "skipped")`, on the theory that skipped
meant "SES creds aren't configured — still a successful compose". It never meant
only that, and it means it less every release: the template router writes
`skipped` for a contact with no address, one who unsubscribed and one on the
suppression list, and `send_outreach_email` returns it for a spam-gate block, a
warm-up hold, a kill-switch halt and a spent daily cap. None of them put a
message on the wire — the SES branch logs "email not sent" — so every one was a
non-send counted as a send, on a metric whose *denominator* role makes it
flattering, and it grew exactly when things went wrong: the day a cap holds a
blast is the day the dashboard reported the biggest send. Both are `("sent",)`.
Note how it hid: `test_email_trends_after_compose` composed with no transport
configured and asserted the trend showed a send, so the suite pinned the wrong
definition rather than the behaviour the test was named for.

The last two surfaces reading `EmailSend` alone — `get_inbox_analytics` and the
dashboard's `emails_sent` tiles — now union both. Note *why* that mattered on the
dashboard, because the same shape will recur: `emails_opened` beside it already
came from `OutreachEvent`, so the tiles divided sequence opens by non-sequence
sends and could show an open rate above 100%. A half-migrated read surface is
worse than an unmigrated one — it is wrong in the direction that flatters.

**"Have we emailed them?" is the same question, and three surfaces answered it
from one table.** All three are "chase this lead, nobody has" features, so the
error compounds in one direction — the contacts we mail most are the ones they
nominate hardest:

- `tasks/followups.py` filed a prospect three steps into a live sequence as
  *"Active-pipeline lead has never been emailed — send the first touch"* and put
  a reminder in a human's queue. The task exists to stop leads going cold and was
  manufacturing cold first touches on top of running cadences. It is
  self-sustaining: the operator's follow-up writes no `EmailSend` row either
  unless they reply from the inbox, so the same contact resurfaces every run and
  takes a slot of `auto_followup_batch_limit` a genuinely stale lead never gets.
- `GET /follow-ups/stale-contacts` listed those same prospects with
  `last_emailed_at: null`.
- `suggestions._stale_leads` proposed *"In 'outreach' but never emailed —
  re-engage"* — about a pipeline stage the enrollment that had been mailing them
  for weeks had itself set.

**The per-contact communication timeline was the same read, one surface over.**
`communication_history.get_contact_communications` — the thing behind
`GET /contacts/{id}/communications`, whose own docstring calls itself a unified,
user-attributed history of *every* touchpoint — unioned `EmailSend`, replies,
calls and interactions, and not the sequence engine. So the contact it was most
wrong about is the one being worked hardest: a prospect five steps into a
cadence showed their replies with nothing they were replying to, and the
timeline opened with the prospect speaking first. It reuses
`EmailHistoryService._fetch_sequence_sends` rather than writing a third query,
so there stays one place that knows how a sequence send is stored. A cadence
step has no `user_id` of its own and is attributed to the owner of the mailbox
it left from — the same fallback the `EmailSend` branch already used, which is
what keeps `?mine_only=true` from hiding it.

`app/services/last_contact.py` is the one resolver (`last_send_by_contact`,
`last_send_by_company`). Note that `OutreachEvent` carries no `contact_id` — the
enrollment holds it — which is exactly why it is the easy table to forget:
including it means a join, so the shortest correct query is longer than the
wrong one.

All three also counted every `EmailSend` row whatever its status, which is the
*opposite* error from the same omission — a `skipped` row stamps `sent_at` and
put nothing on the wire, so a spent cap or an engaged kill switch read as
"contacted recently" and held the lead out. `WIRE_STATUSES` now, which keeps
`bounced` (mail that went out) and every engagement-advanced status. Both
directions are pinned in `tests/test_stale_followup_send_sources.py` and
`tests/test_stale_surfaces_read_both_send_tables.py`, with controls proving a
long-quiet lead still surfaces.

## The daily send cap is a Redis reservation, not a row count

`SendLimiter.can_send_today` counts `OutreachEvent(event_type='sent')` rows and
compares them to the limit. It is **advisory only** — safe for dashboards and
for the queue sweep's cheap pre-check, and wrong for anything that then sends.
Between that read and the event insert sit recipient validation, Hunter
verification, AI composition and an SMTP round-trip, and every concurrent worker
reads the same pre-send count. N workers each conclude they are the last one
under the cap, and the cap is exceeded by N.

The send path must claim its slot with `await limiter.reserve_daily_send(db)`
immediately before the wire, and hand it back with `release_daily_send()` on any
outcome that isn't a real send (soft skip, dispatcher exception, bounce). One
`INCR` per intended send, so exactly `limit` workers get a slot however many ask
at once. The counter seeds from the DB (`SET … NX`) so a restart or an eviction
resumes at the real number. With Redis unreachable it degrades to the row-count
check — per-worker, never looser — and says `[degraded: …]` in the reason.

## "Today" is the sender's day, and it was UTC's

Three counters answer "has this mailbox sent enough today?" — the cap's Redis
key, the row count that seeds it, and the per-account warm-up counter — and all
three keyed on the UTC date. That is only the right boundary if the sending day
sits inside a UTC day, and this system deliberately does not send inside one:
`within_send_window` is evaluated in **the contact's** timezone
(`tasks/outreach.py` passes `contact.timezone` at both dispatch sites).

Take the shipped defaults — 08:00–18:00, `America/Toronto` — and a contact in
Los Angeles. Their window runs 15:00 UTC to 01:00 UTC the next day, so 00:00 UTC
is 17:00 for them: an hour *before* it closes. Every counter reset while the
mailbox was still sending, and that last hour got a fresh full allowance.

The warm-up ramp is the half that matters. A day-1 mailbox on a 5/day plan sent
five, rolled over at 17:00 local, sent five more — and every warm-up check after
that read `5/5`, because the counter it reads is the one that rolled. The single
number governing a cold domain's volume, doubled by the clock, with nothing to
notice: no error, a textbook ramp graph, and a receiving side that finds out
weeks later.

`app.services.send_day` owns the boundary and everything that counts a day uses
it: `_daily_key`, `_db_sent_today`, `get_send_stats`, `warmup.used_today`,
`record_account_send`'s rollover `CASE`. They have to agree — a seed taken over
a different window than the key it is written under hands back allowance the
mailbox already spent, which is the one direction a cap must never be wrong in.

**It is anchored to `send_window_timezone`, and that is the only setting that
can carry it.** The window is `start_hour`–`end_hour` on a 24h clock in that
zone, so it cannot span local midnight — which makes local midnight the one
instant the sender is guaranteed not to be sending at, whatever hours an
operator configures. `daily_summary_timezone` is the operator's *reporting* zone
(shipped `Asia/Kolkata`) and would put the rollover at 18:30 UTC, squarely
inside the default Toronto window: worse than the UTC boundary it replaced.

Contacts far enough east still cross it — a Tokyo business day spans Toronto's
local midnight no matter what — but that residue is bounded by the sender's own
ten-hour window, where the UTC anchor had no bound and broke on the ordinary
case of a North American sender mailing the West Coast.

**Two more surfaces were still on the UTC boundary, and they are the two shapes
a leftover one takes.** Both now call `send_day`:

- **A display.** The dashboard's two `emails_sent_today` tiles
  (`/dashboard/overview` and `/dashboard/kpis`) sit beside a cap readout that
  counts against the new boundary, so between 20:00 and midnight in Toronto the
  tile read zero while the cap reported the day's allowance spent — and a send
  in the last hour of a US-Pacific window appeared here as today's while the cap
  booked it against yesterday. Two "today"s on one screen, neither labelled.
- **A decision.** `OrchestrationPolicy.max_daily_touches` compared UTC *dates*,
  so the per-contact touch cap refilled at 17:00 in Los Angeles — inside that
  prospect's own window — and a cadence could put twice the policy's touches
  into one of their business days. Not a reporting inconsistency: the same
  doubling the warm-up ramp had, one level down. `orchestrator.count_touches_today`
  is split out of `_gather_context` so the guardrail is testable without a
  session; a guardrail whose only test has to mock a database is untested.

Two boundaries were left alone on purpose. `api_credits._day_start` is a
*vendor's* quota day (Apollo, Google Places) and belongs to their clock, not
ours. `check_lead_gen_health` runs at 21:30 UTC — 17:30 Toronto, inside the
window and on the same calendar date either way — so its UTC window already
covers the whole sending day it reports on.

## A cap that counts one of six sources is not a cap

The reservation above was claimed in exactly one place — `app.tasks.outreach` —
and the per-account warm-up ramp in exactly one other: inside
`send_outreach_email`, which is the only thing that incremented
`GmailAccount.daily_send_count`. Both therefore governed the sequence queue and
nothing else, while five other paths sent from the same mailbox for free.

**This is the same shape as the compliance gate above and fails differently.**
Skipping suppression is a breach the recipient notices. Skipping the volume
caps is invisible in both directions: a day-1 mailbox on a 5/day ramp sends its
5 sequence steps and stops exactly as designed, and then the meeting agent
proposes, the smart-reply answers, a human replies from the inbox and an
operator fires two "tests" — and every warm-up check still reads `5/5`. Nothing
errors, the ramp reports itself enforced, `daily_send_count` graphs a textbook
curve, and the mailbox is putting four times that volume on the wire from a cold
IP. What notices is the receiving side, weeks later, by which time the domain is
the broken thing. Note that the *template blast* skipped the daily cap while
going straight through `send_outreach_email` — the function whose docstring
claims every outreach path funnels through it.

Three pieces, in `app.services.email_dispatch`:

- **`claim_send_volume(db, account, *, refusable=, daily_slot_held=)`** takes
  the warm-up gate and the daily reservation together and returns a
  `VolumeSlot`. `send_outreach_email` calls it for every caller by default;
  `daily_slot_held=True` is the queue, which claims earlier so its slot releases
  alongside the send claim and the pacing window.
- **`settle_send_volume(db, slot, sent=)`** closes the claim against the *day*
  — release if nothing landed, consume if the slot was non-refusable. Call it on
  every outcome including an exception, or a flapping SMTP host eats the day's
  allowance without sending.
- **`record_account_send(db, account, sent=)`** books it against the *mailbox*.
  Kept separate because the two disagree about an unsent attempt: the day gets
  its slot back, and the mailbox is still stamped as the one we last tried,
  which is what rotation reads.

**`refusable` is a product decision, so it is stated per call site.** The
compliance floors are absolute because they are about consent. The volume caps
are *operational* — they keep a cold domain from burning itself — so they bind
cold volume and not the second half of a conversation the recipient started:

- **Refused when the day is spent:** the sequence queue, meeting proposals and
  re-proposals, smart-reply auto-sends.
- **Never refused, always counted** (`refusable=False`): a human hitting Reply
  in the inbox, a meeting confirmation (they picked a slot and the invite is on
  their calendar), a meeting reminder (`reminder_sent_at` is stamped *before*
  the send, so a refusal is permanent silence), the per-account test send, and
  the welcome email carrying an operator their own password.

Not counting what we couldn't refuse is what makes a cap drift, so
`consume_daily_send` increments past the limit rather than declining to record.

Two consequences worth keeping in mind:

- **`_db_sent_today` reads both send tables**, per "Sent mail lives in two
  tables" above. It is what the Redis counter is *seeded* from, so an
  undercounted seed hands back allowance the mailbox already spent — the one
  direction a cap must never be wrong in.
- **A caller that ignored a failed send now ignores a routine one.**
  `_guarded_send` returned `False` only on a dead Gmail token before, so
  `propose_meeting` discarding it was a rare bug; with the caps declining
  routinely it became a common one. It now adds the `BookedMeeting` row *after*
  the send, and `repropose_meeting` replaces `proposed_slots` after the send —
  otherwise the meeting offers times the prospect was never shown, and their
  reply picking an old one matches nothing.

  Three callers were reporting a send that didn't happen as one that did, and
  they are worth reading together because the shape recurs wherever a send
  result is turned back into a decision:

  - **`GmailSender.send_email` returns `{"status": "error"}`; it does not
    raise.** An expired refresh token and a 4xx from the API both come back as
    a dict, so `_guarded_send`'s `except` never saw the *most common* way a
    send fails. It settled the volume correctly and returned `True` anyway —
    one outcome with two answers. It returns `landed` now.
  - **`send_meeting_reminders_async` counted attempts.** `reminder_sent_at` is
    stamped before the send (deliberately — one missed reminder beats mailing
    someone every 15 minutes), so nothing downstream ever corrects the report.
    It counts the return value.
  - **`deal_progression_agent.execute_action` marked follow-ups `executed`
    whatever the dispatcher said**, and the hourly sweep only looks at
    `approved` ones. So a follow-up held by a spent cap was retired as though
    it had gone out. A decline that will clear leaves the action `approved`;
    `suppressed` — the one that never clears — becomes `expired`, because
    retrying it hourly is a queue that cannot drain. The `DealActivity` row is
    written only for a real send, or one "dropped" entry, so a held blast
    doesn't put an hourly "sent a follow-up" under a deal nobody emailed.

`tests/test_send_volume_coverage.py` enforces it mechanically, mirroring
`test_send_gate_coverage` (same AST walk, its own `REVIEWED`). Its self-test
plants a send that is fully *compliant* and simply costs the mailbox nothing —
the exact shape all five paths had after the compliance fix landed, which is why
it is a second file rather than a second assertion in the first.

## A step goes on the wire at most once

Every send gate in this codebase asks the *database* — and the window between
"Gmail has the message" and "the transaction saying so committed" has not
written any of it. `send_sequence_step` sends and then records the
`OutreachEvent`, the advanced `current_step` and the thread ids; a dropped
`COMMIT` (or a raise, or a killed worker) in that window rolls the record back
and leaves the mail. The Celery retry then finds `step N, active, due` and sends
the same cold email to the same prospect again.

`app.services.send_claim.acquire(db, enrollment_id, step)` is the interlock: a
conditional UPDATE on `outreach_enrollments.send_claim_step`, **committed on its
own session** immediately before the dispatcher call, so it survives that
rollback. The row lock also decides the concurrent case — two workers, one
enrollment, neither having committed.

- **Never release a claim on an ambiguous outcome.** `send_outreach_email` does
  work *after* the transport returns, so an exception out of it does not prove
  the message stayed home. Release only on a returned soft-skip status
  (`skipped` / `suppressed` / `halted`) — the dispatcher declined, it didn't fail.
- **A stale claim is reconciled, never re-sent.** `send_claim.reconcile` writes
  the missing `sent` event (flagged `reconciled` / `wire_outcome: unconfirmed`,
  and counted, because deliverability rates divide by *attempts*) and advances
  the cadence, so the enrollment stops being re-dispatched and refused forever.
- **Any new send path must claim.** A path that sends without one reintroduces
  the duplicate for itself.

`OutreachEnrollment.dispatched_at` is the other half — a lease taken by the
queue sweep before the task exists. The sweep marked nothing before, so an
enrollment whose task was still in the broker was dispatched again five minutes
later; a worker half an hour behind held six copies of every task in its
backlog. Time-based, never explicitly cleared, floored above the sweep interval
(a lease shorter than 5 minutes is inert).

## An opt-out stops the mail, and keeps stopping it

There are three ways out of Dialtone mail — a reply saying "unsubscribe", the
bulk compliance route, and **the link in the footer of every email plus the RFC
8058 one-click header Gmail and Yahoo require**. The first two set
`contact.suppressed`, wrote a consent record and stopped every live enrollment.
The third set `contact.unsubscribed` and nothing else — and no send gate has ever
read that column. So the only opt-out a recipient can actually reach was the only
one that let steps 2 through 5 keep arriving on schedule.

**`app.services.opt_out.record(db, email, org_id=…)` is the one way to record an
opt-out.** It sets both flags, writes the revoked `ConsentRecord`, stops every
enrollment in `ACTIONABLE_STATUSES`, *and* adds the address to the org
suppression list. Never write the flags directly.

- **The list is the durable half.** Flags live on a contact row and the scraper
  creates rows: someone who opts out today and is rediscovered next month arrives
  with `suppressed=False`. `SuppressionEntry` is keyed by `(org, email)` and
  outlives every row. `unsubscribe` was a legal `SUPPRESSION_REASONS` value that
  nothing had ever written.
- **Match every row, case-insensitively.** `contacts.email` has no unique
  constraint. A `scalar_one_or_none` here raised `MultipleResultsFound` on a
  re-scraped duplicate, which became a 400 — "That link didn't work" to the human,
  a broken one-click affordance to the mailbox provider, and nothing recorded.
- **One serializer, `opt_out.unsubscribe_serializer()`.** The sender used to fall
  back to a hardcoded secret and the endpoint to a random per-process key, so on
  a box with no `SECRET_KEY` every unsubscribe link we sent was unverifiable.
- **Tokens carry the sending org**, so an opt-out lands on that sender's list
  rather than every org holding the address. Bare-string tokens are the legacy
  shape, stay valid a year, and are in real inboxes — they decode to
  `org_id=None`, which deliberately applies the opt-out everywhere.
- **Only promise what happened.** The confirmation page and the one-click 200 are
  claims that the address is suppressed; return the error page when it isn't.

The send path checks `suppressed` **or** `unsubscribed` **or** the list, in both
`send_sequence_step` and the queue sweep. The `unsubscribed` read is deliberate
redundancy: rows written by the old link handler are in the database now.

**And that redundancy was in the sequence path only.** `check_send_allowed` —
"the two compliance floors, in one place", the gate *every other* path calls —
read the kill switch and the list. Not the flags. So the state the sequence path
keeps a redundant read for, and says so in its own comment, was the state that
walked straight past the gate for the smart-reply auto-send, all four
meeting-agent sends, the inbox reply composer, the per-account test send, and
`send_outreach_email` itself. The opt-out stopped the cadence and stopped
nothing else. Same shape as the gate's origin story one level down: the floor
was consolidated out of its callers and one of its clauses stayed behind.

`suppression.address_blocking_reason(db, email)` is the three-clause test for a
caller holding an **address** rather than a `Contact` — which is what a send path
has, since the gate takes what goes in the `To:` header.
`blocking_reason(db, contact)` is the row version, for the enrollment gate. Both
match case-insensitively across every row carrying the address: `contacts.email`
has no unique constraint, and which duplicate a path happened to load is not a
fact about the person.

## `lower(email)` is what is queried, so `lower(email)` is what is indexed

Sixteen call sites filter `func.lower(Contact.email)` and none filter the raw
column, for the reason `can_send_to_contact` documents: the column is not
normalised on write, so a lookup on the raw value is about a *row* and not about
a *recipient*. `ix_contacts_email` from migration 001 indexes the raw column and
can serve none of them — Postgres plans a sequential scan.

Migration 119 adds `ix_contacts_email_lower` on the expression. Worth knowing
where the scans were, because it is not the reporting surfaces: the compliance
gate on every outbound message, the limiter's monthly cap and minimum-gap checks
twice more per send, reply/bounce/complaint attribution on their beats, and
`gdpr.erase_subject`. The fastest-growing table in the schema, scanned by the
paths that run per send and per inbound message — it degrades in step with the
product working, which is why it doesn't look like a problem until it is one.

An expression index rather than a normalised column: a column needs every writer
to maintain it, and a writer that forgets is a compliance lookup that silently
misses.

## Every transport sets the one-click unsubscribe headers

`List-Unsubscribe` + `List-Unsubscribe-Post` are a Gmail/Yahoo bulk-sender
requirement, and there are **three** transports. Gmail and SMTP each carried
their own copy of the header builder. Amazon SES — the fallback
`send_outreach_email` takes when no account resolves — carried neither, and
could not: SES's `send_email` action takes a structured message with nowhere to
put a custom header, so setting one means `send_raw_email` with a MIME document.

The gap was invisible from every angle except the transport that lacked it, and
that transport is the one that fires when the operator's mailboxes are down —
carrying the day's volume, on the sending domain's shared reputation, rated by
every bulk filter as a sender who offers no one-click affordance.

`opt_out.unsubscribe_headers(url, sender_email)` is the one builder; all three
transports splat it. A fourth transport that hand-rolls the pair shows up as a
difference in `tests/test_ses_list_unsubscribe.py` rather than as mail that
quietly lacks the affordance.

**`ses:SendRawEmail` is a different IAM action from `ses:SendEmail`.** Changing
which API call a transport makes changes which permission it needs, and an
identity provisioned back when this path only ever sent structured mail holds
only the old one — so the fix would have turned every fallback send into an
`AccessDenied`, on precisely the day the fallback is what's carrying the mail.
Missing headers are a deliverability problem; sending nothing is an outage, so
an authorization failure specifically retries the structured call and logs the
permission to grant. **Only** that one: a throttle, a rejected address or an
unverified sending domain fails identically on both actions, so retrying them is
a second round trip, a duplicate log line, and — for a throttle — more pressure
on the thing already refusing us.

## The test suite must not depend on the machine it runs on

Two autouse fixtures in `tests/conftest.py` exist because the suite was reading
ambient state and calling it a result. Both were found the same way — a change
elsewhere flipped tests that had nothing to do with it — and both failure modes
read as a regression in whatever they landed on:

- **`_isolated_send_counter`.** `reserve_daily_send` claims its slot with an
  `INCR` in **real Redis**, which on a developer box is reachable, so
  `dialtone:daily_sends:<org>:<day>` was shared by every test in the suite *and*
  by every earlier run that calendar day. About a hundred sends in, the cap is
  spent and the next test that sends fails on `status == 'sent'` — an assertion
  about sending, not about limits. Green on CI and on a fresh container, red on
  a laptop by mid-afternoon, and *which* test fails is decided by ordering, so
  it moves between runs and reads as several unrelated flakes. The fixture gives
  each test a private in-memory counter rather than disabling the limiter: the
  reservation is the mechanism that makes the cap correct under concurrency, and
  a suite that stubs it out stops testing it.
- **`_no_ambient_google_oauth`.** `Settings` reads `.env`, and several tests
  assert the *unconfigured* path (`/gmail/oauth/authorize` → 503,
  `/gmail/status` → `configured: false`, three more in `test_postmaster.py`).
  They passed because the developer's `.env` happened to hold no Google
  credentials, and went red the moment rotating the Google project required
  putting a working client id there. `test_gmail.py` already had a
  `_google_configured` fixture turning them *on* for the tests that need them;
  nothing turned them off for the tests that need them off.

The general rule: a test whose outcome depends on Redis being reachable, on
`.env` contents, or on the time of day is asserting a property of the machine.
Fixtures for both directions — on *and* off — or the absence is the bug.

## A test that reads the clock twice is a test that fails by the calendar

`tests/test_daily_summary.py` stamped its fixture events from a module-level
`NOW = datetime.now(timezone.utc)` — evaluated at **import**, which in a full
suite run is a quarter of an hour before the test that uses it — and resolved
the day it then asked the endpoint about at **call** time. Cross a day boundary
in between and the events are in one day while the question is about the next.

The timezone is what makes the window findable and is not the bug. The
endpoints reckon a day in `daily_summary_timezone` (`Asia/Kolkata`, UTC+5:30),
so the boundary lands at 18:30 UTC — inside the working day, in the middle of a
suite run, roughly every day. Note the file already carried a docstring about
exactly this hazard and had fixed **half** of it: the target date was moved to
call time and the seeded events were left at import time, which narrowed the
failure from 5.5 hours to 15 minutes and made it read as flake.

Rules: take one reading per test and derive everything from it, and test the
boundary on a **fixed** instant rather than on `now` — a test that reads the
real clock can be right but cannot be proven. `test_a_minute_before_local_midnight_still_belongs_to_that_day`
is the deterministic half; the `Pacific/Kiritimati` / `Pacific/Midway`
parametrization is the generalisation, because a scenario exercised only in the
shipped timezone can only fail during that timezone's own boundary window.

## Every Google/SMTP client in this codebase is blocking

`googleapiclient`, `smtplib` and `imaplib` are synchronous. A call to one of
them made directly from a coroutine holds the event loop for the whole network
round trip — not a slow function, a *stopped process*: every other request, beat
task and coroutine on that worker waits.

Nearly everything here already knows that. `email_dispatch` wraps the senders in
`asyncio.to_thread`, `smart_reply` has `_gexec`, the calendar and DNS helpers are
offloaded at their call sites, and `smtp_sender` / `smtp_reader` say in their
module docstrings that callers must do it. `EmailHistoryService.sync_replies_for_account`
was the one that did not, and it is the worst place for it to have been missed:

- it makes **one `messages().get` per message**, up to 50 per mailbox, on top of
  the `messages().list` — so the stall is tens of round trips deep, not one;
- `sync_all_accounts` loops that over every connected mailbox the caller owns;
- and it is reached from **`POST /api/v1/emails/sync`**, a request-path endpoint.
  One operator pressing "Sync now" froze the API for every tenant sharing the
  worker, for as long as Gmail took to answer.

`EmailHistoryService._gexec` is the offload, mirroring `smart_reply`'s.
`_build_gmail_service` goes through `to_thread` too — `build()` refreshes
credentials and, depending on the client version, fetches a discovery document.

**`bounce_ingestion.ingest_account` was the other one**, and it is the same
function as `complaint_ingestion.ingest_account` — whose docstring says it
*mirrors* this one — with the offload present in the mirror and absent in the
original. It is the deeper stall of the two: a `messages().get` per DSN, up to
**100** per mailbox, across every connected mailbox on the beat. What it holds
is the beat worker, so what waits is every other scheduled task on it,
including the send queue. `ingest_smtp_account` directly below it wraps its
IMAP reads in `asyncio.to_thread`, which is what makes this a slip rather than
a decision. `tests/test_bounce_scan_offloaded.py` pins both halves of the pair —
the fixed one and the control — because "these two are the same function twice"
is the fact that let them drift.

`tests/test_gmail_sync_offloaded.py` pins the behaviour rather than the call
shape: the Gmail calls must land on a thread that is not the loop's, and the
loop must keep servicing timers while they run. **The responsiveness threshold
cannot be "the loop advanced at all"** — a blocked sweep still yields at each of
its `await db.execute(...)` points, so the broken version ticks a handful of
times. What separates them is the rate.

## A credential can be rotated from the dashboard, and the environment is still the base

`app_credentials` (migration 120) holds **overrides**: a row wins for its key,
and no row means the `.env` value stands. An empty table is therefore a no-op,
and clearing an override restores the deployed value rather than blanking it.
`Platform → API Credentials` (platform_admin only) is the UI; the registry
`credential_store.MANAGED_CREDENTIALS` is an allowlist, without which the
endpoint would be an arbitrary write primitive over `Settings`.

Four things about it that are not obvious from the code:

- **It hydrates `Settings` rather than being a lookup.** Fifteen call sites read
  `get_settings().google_client_id`, several inside synchronous OAuth helpers
  with no session in hand (`GmailSender`, `google_calendar`, `postmaster_oauth`,
  both ingestion sweeps). Resolving once and writing onto the `lru_cache`d
  singleton is what avoids threading a session through all of them.
- **`_ENV_BASELINE` is snapshotted before the first hydrate**, because by the
  second one the attribute holds the *previous override*, not the environment.
  A store that read it back at clear-time would "restore" to the value being
  backed out — discovered while backing out a bad credential, i.e. at the worst
  moment. `test_clearing_an_override_restores_the_deployed_value` pins it.
- **Propagation is per process and therefore explicit.** The worker that served
  the write has the value at once; the rest get it from
  `app.tasks.credentials.refresh_credentials` (5-minute beat) or a restart.
  Without that task the UI would report success while the beat worker went on
  using the retired key — worse than not having the feature. `applied_at` in
  the API response is per-worker, so the propagation is visible rather than
  assumed.
- **Not org-scoped, and it must not become so.** These identify the deployment
  to a third party — one Google Cloud project, one redirect URI — and are
  resolved where no org is bound (the OAuth callback runs before any user
  context exists). A tenant admin who could write `google_client_id` could point
  the consent flow at a client they control. The exemptions in
  `test_org_scoping_coverage.py` and `test_task_org_binding.py` both say so.

**Rotating the Google client invalidates every stored Gmail refresh token.** A
refresh token is issued *to* an OAuth client; the new client cannot refresh the
old one's grants. Every connected mailbox has to be reconnected, and until it
is, `sync_replies_for_account` raises `RefreshError` per account and
`sync_all_accounts` self-heals by flipping each to `status='revoked'` — which is
correct behaviour that looks exactly like a mass revocation incident.

## Gmail strips the base64 padding, and three of four decoders didn't restore it

`base64.urlsafe_b64decode` raises `binascii.Error: Incorrect padding` unless the
encoded length is a multiple of 4, and an **unpadded** encoded length is a
multiple of 4 for only one raw byte-length in three. Gmail returns
`body.data` unpadded. So decoding it inline fails on roughly **two message
bodies in three** — selected by nothing but the message's length, which is why
it never looked like a pattern.

There were four readers of Gmail bodies and one of them padded. Where each put
the failure is the whole story:

- **`email_history._decode_base64url` padded**, so the *inbox* — the surface
  anyone actually looks at — worked. Solved ground, visibly.
- **`bounce_ingestion` and `complaint_ingestion` wrapped the decode in
  `except Exception: pass`** and fell back to Gmail's ~100-character `snippet`.
  A DSN's `Final-Recipient` and an ARF report's `Original-Rcpt-To` are in the
  body, not the snippet, so the sweep logged *"matched the query but parsed no
  recipient"* at DEBUG and moved on: no `BounceRecord`, no suppression, and the
  cadence kept mailing a dead mailbox. That is precisely the loop
  `bounce_ingestion`'s own module docstring exists to close, failing two times
  in three while reporting a clean sweep.
- **`smart_reply._extract_body` didn't catch**, so it raised — an ordinary
  prospect reply took the auto-responder down instead of being answered. The
  loud failure, and by far the cheaper one.

`app.services.gmail_body.decode_body` is the one decoder; all four go through
it. `tests/test_gmail_body_padding.py` parametrizes every case over the three
length classes and asserts no module outside `gmail_body` calls
`urlsafe_b64decode`.

**Every test body must be length-parametrized.** A body that happens to encode
to a multiple of 4 passes against the broken decoder — `test_gmail_sync_offloaded`'s
`"Sounds good."` is one, which is exactly how a suite of 4700 tests contained no
evidence of this. Note also the second-order lesson about defensive `except`
blocks around *parsers*: a swallowed decode error reads as "this message had
nothing in it", which is indistinguishable from the quiet correct case. Nothing
here was ever going to show up as an error rate.

## One bad inbound message must not take the mailbox — or the sweep — with it

The reply sync ran a whole mailbox in one transaction with no way back from a
statement that failed inside it. Two faults compounded, and both are about
input arriving from the open internet:

- **The unbounded header.** `_is_outreach_reply` matches an inbound message's
  `In-Reply-To` / `References` against ids we recorded when sending, by putting
  every token into two `IN` clauses. Those headers are written by the *sender of
  the inbound message* — attacker-controlled, bounded by nothing between the
  mail server and here — and every driver refuses a statement past its own
  parameter ceiling (psycopg: 65535). `reply_matching._message_id_tokens` was
  capped for exactly this; `email_history._parse_message_ids` is a second parser
  of the same two headers, written earlier, and never was. Both now go through
  `reply_matching.bound_header_tokens`, which owns the policy: `In-Reply-To` is
  the *direct parent* and is kept whole, and only the `References` chain is cut,
  from its **oldest** end.
- **The cascade.** A failed statement leaves the session unusable until someone
  rolls back. Nothing did, so the message after the bad one failed, and the one
  after that, and the `commit` at the end — and then the **next mailbox** in the
  sweep, on the same session, and the caller's session after that. One crafted
  reply to one mailbox stopped reply ingestion for the whole org, and "no
  replies at all" is indistinguishable from "no replies yet".

Recovering has two consequences that are easy to get wrong:

- **Commit per message, not per mailbox.** A rollback discards everything still
  pending, so recovering from message 2 with a batched commit means throwing
  away message 1.
- **A rollback expires every instance on the session**, and reading an expired
  attribute back is a database round trip the sync ORM accessor cannot make from
  a coroutine (`MissingGreenlet`). So the recovery breaks the iteration it was
  added to save unless the loop holds plain values: `sync_replies_for_account`
  captures `account.id`/`.email` up front, and `sync_all_accounts` iterates ids
  and re-`get`s each row.

**`tests/test_gmail_sync_resilience.py` poisons the session with a failed
*flush*, not a failed SELECT, and that is the whole test.** Postgres aborts the
transaction on any error, so in production a bad SELECT cascades exactly as
described — but SQLite does not, and the test engine is SQLite. Written the
obvious way, both cascade tests passed with the rollback deleted. A failed flush
deactivates the transaction in SQLAlchemy itself, driver-independently, so the
test now fails without the fix. Any resilience test asserting "the thing after
the broken one still works" has this trap: verify it fails without the fix, or
it is pinning nothing.

## Every do-not-contact writer writes the list, erasure most of all

Five paths suppress an address (unsubscribe link, reply, bounce, complaint, bulk
compliance) and a sixth erases a subject outright. **All of them write a
`SuppressionEntry`**, because the flags live on a contact row and the scraper
makes rows — suppress someone today, rediscover them next month through another
source, and the new row arrives `suppressed=False`.

**`gdpr.erase_subject` is the one to understand.** It sets the flags and then
overwrites `contact.email` with `anonymized+<id>@redacted.invalid`. The flags
survive; the address they were about does not. Before the list write, erasure
did not merely fail to be durable — it *deleted the only link* between the
address and the decision, for the one class of subject where re-contact is a
regulatory violation rather than an annoyance.

- **Record the address the request named**, never one read back off a contact
  row. The write sits above the scrub loop to keep that obviously true rather
  than merely true today.
- **Unconditional, not gated on a contact existing.** An erasure request for an
  address we hold no row for — or whose rows a previous erasure already
  anonymised — still has to stick if that address is ever rediscovered.
- **`reason="erasure"` is its own value** (migration 110 widened
  `ck_suppression_reason`). An operator can re-verify a `bounce` and argue about
  a `manual`; an erasure entry is the one whose removal re-opens legal exposure.
  `DELETE /dialtone/suppression/{id}` refuses it with a 409.

**Un-suppressing must lift something, but only its own.**
`suppression.remove_entry` deletes entries in `REVERSIBLE_REASONS` — `manual`
alone. Without it, `bulk-suppress {suppressed: false}` takes the flag off, leaves
the entry, and the send gate goes on refusing with no explanation. With it
widened, a checkbox round-trip would erase a complaint or an erasure record.
Removing any other reason stays an explicit, per-entry action on the list itself.

**Erasure ends the cadence too.** The other four writers close the subject's
enrollments in `ACTIONABLE_STATUSES`; erasure set the flags, scrubbed every
column, wrote the list entry and left them running. The queue sweep stops an
`active` enrollment whose contact is suppressed, so no mail went out — but it
selects `status == "active"` and therefore never sees a `pending_review` one,
and that is not a resting state: `expire_stale_review_drafts` un-parks it back
to `active`, stamps it due now and dispatches. An erased subject's parked
enrollment was revived *after* the erasure completed. There is also no second
chance at it — erasure overwrites the address with the `anonymized+<id>@
redacted.invalid` placeholder, so re-running it finds no contacts at all.

## Enrolling is a do-not-contact decision, and six paths disagreed about it

The send gate is the compliance floor and it holds. What it cannot stop is the
**enrollment**, and six writers created those with six different ideas of who is
mailable. Only `autonomous_prospector` asked all three questions:

- `signal_enrollment` and `POST /sequences/{id}/enroll` read the two contact
  flags and not the list;
- `POST /outreach/enrollments` and the public API's `POST /sequences/enrollments`
  asked nothing beyond "is there an address" — and the first of those dispatches
  `send_sequence_step` immediately;
- `social_listening_agent.auto_enroll_lead` asked nothing at all, while its own
  docstring claimed it mirrored the prospector.

**The list is the clause that gets dropped, and it is the one the contact row
cannot answer.** `suppression.add_entry` deliberately does not touch
`Contact.suppressed` — an address outlives every row that carries it — so a
compliance CSV upload, a bounce recorded against a re-scraped duplicate, and an
erasure whose subject the scraper later rediscovered all present as a contact
with clean flags whose address must never be mailed. That last one is what
`erase_subject` writes its entry *for*.

Nothing leaked, which is why it accumulated. What it cost is everything up to
the gate: a live enrollment for someone who opted out, dispatched and paid for,
counted as outreach on a campaign that can never mail it — and in the
social-listening path a company advanced to `pipeline_stage='outreach'` on the
strength of it, which is a claim about the *company* that no downstream gate
corrects.

`suppression.blocking_reason(db, contact)` is the shared test and returns the
operator-facing reason, so a bulk endpoint can report *which* clause refused
rather than a bare False. `blocking_flag` is its row-only half and
`suppressed_addresses` the batch list query, for callers holding a page of
contacts — use both rather than a per-row existence check in a loop.

**A reminder is a do-not-contact decision too.** The stale-lead sweep's entire
output is "reach out to this person", handed to a human who will — and it read
the two contact flags and not the list, so it chased addresses nothing is
allowed to mail. There is no send gate behind a task queue; this is the only
gate in front of it. It filters *before* the batch limit is spent, because a
listed candidate dropped from an already-capped batch takes a slot on every run
and never yields it — nothing about it changes.

`tests/test_enrollment_gate_coverage.py` enforces it mechanically and
**function-locally**, mirroring `test_send_gate_coverage`: "some caller checks
first" is exactly the assumption that was wrong. It carries self-tests that
plant each shape, including the flags-only one that five of the six had — that
plant must *fail*, because reading `contact.suppressed` by hand is precisely the
check that cannot see the list.

## "The single choke point" was a claim about the callers, not the system

`send_outreach_email` holds the two compliance floors — the kill switch and the
do-not-contact list — and its docstring called itself "the single choke point
every outreach path funnels through". That was true of the paths that call it,
and the set of things that put mail on this platform's wire is larger than that
set. **Five paths held a `GmailSender`/`SmtpSender`, or the raw Gmail client,
and sent without it:**

- **The meeting-booking agent** (four sends: propose, re-propose, confirm,
  remind) answered to *neither* floor. It is autonomous — a 15-minute beat, no
  human between the decision and the wire — so an operator engaging the kill
  switch because their domain was on fire stopped the queue, the sequences and
  every manual send, watched the UI report the platform paused, and this agent
  kept proposing meetings.
- **The smart-reply auto-send** had the kill switch and three other floors, and
  not suppression. A reply arriving on an old thread from an address that has
  since unsubscribed or filed a complaint was auto-answered.
- **The inbox reply composer** and **the per-account test send** are
  human-driven, which is not an exemption — a person clicking Reply, or typing a
  third-party address into a "test", is how a do-not-contact address actually
  gets mailed.

The gate is now `email_dispatch.check_send_allowed`, callable from outside the
dispatcher, and `send_outreach_email` calls the same function rather than
carrying its own copy.

**What made this invisible is worth naming.** A send that should not have
happened produces no error: the transport returns success, the row records a
sent message, nothing is logged as wrong, and the only party who finds out is
the recipient who asked to never hear from us again. Compare the send *window*
and the *daily cap*, which are enforced in the queue and so cover only the queue
— but a path that skips them merely sends at an odd hour. Skipping this one is a
compliance breach. **The two floors are not the same kind of thing:** the kill
switch is operational (`bypass_kill_switch` exists for internal callers, and a
connectivity test to your own mailbox passes `check_kill_switch=False`), and
suppression is a compliance gate with no flag that turns it off.

Reminders and confirmations clear the same gate as proposals rather than
claiming a transactional exemption: the list does not record which mail a
recipient meant to escape, a complaint is the entry most likely sitting there,
and the override is an operator removing the entry — a decision with a record —
not a branch in a module. Two sends genuinely aren't outreach and say so in
`REVIEWED`: the kill-switch admin notification (applying the switch to the mail
announcing the switch is circular) and the welcome email carrying a new
operator their own temporary password.

`tests/test_send_gate_coverage.py` enforces it mechanically, **function-locally**
— "some caller checks" is the assumption that was wrong to begin with. It found a
send site a grep for the transport class names had missed, and it carries
self-tests that plant each of the three send shapes, including the
`asyncio.to_thread(sender.send_email, ...)` form where the send is an *argument*
rather than a call. `tests/test_send_compliance_gate.py` is the behavioural half.

## Pre-send spam gate

Every outbound path runs the deterministic content scan
(`app.services.spam_guard`) on the final rendered copy. `flag` records the score
on the send; `block` is a **soft skip** that holds the enrollment for a copy
edit. Blocking is off by default (`SPAM_GUARD_BLOCK_ENABLED`) — a content
heuristic that stops outreach on its own judgement is a worse failure than the
spam it prevents.

The sequence path scans **before** `apply_tracking` and passes
`skip_spam_scan=True` to the dispatcher: scoring the post-tracking body would
count our own rewrapped links as "too many links". Any new send path either
scans its own pre-tracking copy or lets `send_outreach_email` do it.

## An alarm on a platform-wide sum asks "is *everyone* dead?"

`check_lead_gen_health` is the one beat task that deliberately does not use
`for_each_org` — it reads platform-wide so its totals match what a
platform_admin sees on `/leads/stats`. That part is fine. What was wrong is that
both of its alert conditions were evaluated against those totals:
`sends_today == 0` and `ai_composed == 0`. Each therefore asked whether *every*
tenant was dead, and answered "no" the moment one healthy tenant existed.

On a single-tenant install that is the same question. From migration 078 onward
it is not, and the alarm silently stopped covering anyone but the last org
standing — with no failure anywhere to notice, because **the reading was correct
and the aggregation was the bug**. A green health check does not tell you which
question it answered.

Both failure modes are per-tenant to begin with: a stalled queue, a mailbox with
`sending_enabled=False`, a spent daily cap, an engaged kill switch. AI
composition most of all — per-org LLM keys (084) mean one tenant's key running
out kills composition *for that tenant only*, which a platform-wide rate cannot
represent at any threshold.

The totals stay, and the per-org checks run underneath them:

- **A platform-level alert subsumes the per-org ones of the same kind.** An
  outage that took out every tenant is one line, not one per customer.
- **The per-org zero-send alert is gated on an `active` enrollment.** An org
  with nothing enrolled sent nothing because there was nothing to send; a daily
  ERROR per idle tenant is how the whole channel gets muted. `paused` doesn't
  count — that is the system doing what it was told. Same reasoning as
  `DangerMonitor`'s one-alert-per-account rule.
- **The per-org AI check needs no gate**, being already conditioned on that org
  having sent sequence mail.

Worth noting where this *didn't* happen, because it bounds the class: `/leads/
stats` and `/dialtone/status` call the same `tally_by_org` and both keep the
per-org breakdown next to the total. They are display surfaces, and a reader can
see which org is which. The monitor was the only place a *decision* was taken on
the sum, and the only one that discarded the breakdown to take it — the
per-tenant numbers were in hand and summed away.

The tests are the other half. Every case in `tests/test_lead_monitor.py` ran one
org, so the suite was green throughout and was measuring a sum of one. **A
single-tenant fixture cannot distinguish an aggregate from a per-tenant value**,
so anything asserting on an aggregate needs two orgs or it is pinning nothing.

## Auto-pause: match the response to the scope

- **Global-scope** danger breach → the system-wide kill switch. An aggregate
  bounce rate that high is a shared domain/IP reputation problem.
- **Account-scope** breach → `DangerMonitor.pause_accounts` disables just that
  mailbox (`sending_enabled = False`) and the rest of the fleet keeps sending.
  One pause and one alert per account; a breach that persists across sweeps
  stays silent, because an alert that repeats every 5 minutes gets muted.
- **Out-of-office reply** → `out_of_office.pause_for_ooo` sets the enrollment to
  `paused` with `pause_reason='out_of_office'` and `next_send_at` carrying the
  return date; the hourly `resume_out_of_office` task brings it back.

`OutreachEnrollment.pause_reason` is what makes an automatic resume safe. A
sweep that resumed every paused enrollment with a past due date would also
revive the ones paused for a missing address, a dead domain, a consent block, or
an operator's decision. **NULL means "not ours — leave it alone."**

## One resolver decides when a step is due

`app.services.step_delay.due_at(step, anchor, factor=, enrollment_id=)` computes
every due date in the system. **Do not add a sixth `anchor + timedelta(days=step.delay_days)`.**

There were five, and they had drifted: the post-send advance, the optimizer
advance and `adaptive_delay.reschedule_pending_step` applied
`OutreachEnrollment.adaptive_delay_factor`; the queue sweep's legacy path and the
sequences router's preview did not. An enrollment whose factor had moved off 1.0
was therefore "due" to the sweep and "not yet due" to the send task the sweep
dispatched — a step that oscillates instead of sending.

A step's cadence is `delay_days` + `delay_hours`, optionally counted in business
days (`delay_business_days`) and spread by `delay_jitter_minutes`. Three rules
worth knowing before touching it:

- **Day-granular steps must keep their exact schedule.** A step with
  `delay_hours == 0` delegates to `adaptive_delay.effective_delay_days` — whole-day
  rounding and the `[1, 14]`-day clamp included. Every sequence in the database is
  one of these, and changing where they land silently reschedules live outreach.
- **Jitter is derived from the enrollment id, never drawn fresh.** The sweep
  recomputes due dates for enrollments with no `next_send_at`, so a random offset
  would move the deadline on every poll.
- **This decides the schedule, not the send.** Business hours, weekends,
  per-contact timezone, warm-up, the daily cap and per-mailbox pacing are all
  enforced at dispatch by `send_limiter` / `send_pacing`. What `step_delay` owns is
  the *spacing between touches*, which nothing downstream can reconstruct — the
  send window holds a Sunday send to Monday, which is exactly why a Thursday +3
  and a Friday +2 used to land on the same Monday.

## A/B tests are judged on their own metric

`evaluate_experiment(experiment, metric)` and `auto_declare_winner(..., metric=)`
take the metric the test declared; the resolver passes `test.metric` through, the
hard-cap default winner sorts by that metric's rate, and the conclusion alert
counts it. Subject-line tests default to `opened` — the subject decides whether a
message is opened and barely touches what happens after, and cold reply rates
(a few percent) need an order of magnitude more volume than open rates to reach
significance. Omitting `metric` keeps the legacy reply-then-open behaviour.

## The MCP server is a second client of the same handlers

`app/mcp_server/` (`python -m app.mcp_server`, stdio) exposes Prospect to Claude
Desktop/Code. See `FEATURE_MCP.md` for setup. Three things to keep true:

- **Writes call the route functions** (`routers.campaigns.create_campaign`,
  `add_contacts_to_campaign`) and never re-implement them. A tool that writes
  directly is a second path past the gates the UI's path goes through.
- **Auth is the public-API key, re-checked on every call** through
  `api_key_auth.get_api_key_context`, inside `org_scope(None)`. That function
  leaves the org bound, which is fine for a request task. In a long-lived
  process the binding would outlive the call, so the reset is not optional.
  Write tools act as the key's `created_by` user and refuse without one.
- **fd 1 is the protocol.** `__main__.isolate_stdout` points fd 1 at stderr
  before anything else is imported. SQLAlchemy `echo` writes to stdout.

`mcp` needs pydantic>=2.11 and the API pins 2.10.3, so the server has its own
`requirements-mcp.txt` and runs from its own venv. `tests/test_mcp_server.py`
skips (and says so) under the prod pins. Run it from an MCP venv.
