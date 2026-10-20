# Feature: Prospect MCP server

Lets Claude Desktop / Claude Code users work with DoAide Prospect from their AI
tool: search leads, inspect a lead, create and fill Autopilot campaigns, read
campaign stats, check mailboxes and the unsubscribe list. It is a
[Model Context Protocol](https://modelcontextprotocol.io) server built on the
official Python SDK (`mcp`), shipped as a package inside this repo.

## Shape

- **Package:** `app/mcp_server/` — inside `app/` deliberately, so the
  mechanical checks that walk `app/` (`test_single_row_assertions`,
  `test_send_gate_coverage`, `test_bulk_write_scoping`, …) cover it too.
  - `tools.py` — the tool logic as plain async functions taking a session and
    an authenticated context. No MCP types, so it is testable directly.
  - `auth.py` — API key → org-bound context, per call.
  - `server.py` — the `FastMCP` app: registers the tools, opens a session per
    call, turns errors into tool errors.
  - `__main__.py` — `python -m app.mcp_server`, stdio transport.
- **Database:** the same PostgreSQL as the API, through `app.database` and the
  existing SQLAlchemy models. No new tables, no migration.
- **Process:** standalone, stdio. Claude Desktop launches it; it is not part of
  the API container and does not affect the API deploy.

## Authentication and tenancy

It reuses the **public-API keys** (`/org/api-keys`, `llk_…`). The key goes in the
`PROSPECT_API_KEY` env var of the Claude config. **Every tool call** runs it
through the same `api_key_auth.get_api_key_context` the public REST API uses, so
revocation, expiry, org suspension and the per-key rate limit apply at once —
there is no session token cached in the MCP process that outlives a revoke.

That call binds the key's org as the current context, so the global
`OrgScopedMixin` read filter scopes every query to the key's tenant. The
binding is reset after each call.

New scopes, added to `ALL_SCOPES` (existing keys don't gain them, so you have
to grant them on purpose):

| Scope | Tools |
|---|---|
| `contacts:read` (existing) | `search_leads`, `get_lead_details` |
| `campaigns:read` | `list_campaigns`, `get_campaign_stats`, `list_discovery_documents` |
| `campaigns:write` | `create_campaign`, `add_leads_to_campaign` |
| `mailboxes:read` | `search_mailboxes` |
| `suppression:read` | `get_unsubscribes` |

**The write tools act as the user who created the key** (`api_keys.created_by`),
because campaign creation needs a user: an owner, and a default sending
mailbox. If that user is gone, inactive or in another org, the write tools
refuse instead of guessing.

## Tools

"Leads" are `contacts`. Each one comes back with its company and pipeline stage.

| Tool | What it does |
|---|---|
| `search_leads` | Free-text search over name/email/title/company, with filters for company, pipeline stage and has-email. Paginated in a stable order (`created_at, id`). |
| `get_lead_details` | The full contact, its company, the campaigns it is in, when it was last emailed (across **both** send tables, via `last_contact`), and whether it is mailable (`suppression.blocking_reason`: the flags *and* the org suppression list). |
| `create_campaign` | Creates an Autopilot campaign by calling the existing `/campaigns` handler, so you get the same document check, AI-key check, sender check, ICP build and defaults (review-before-send on). Needs ready discovery documents. |
| `list_discovery_documents` | Lists the documents `create_campaign` can use. |
| `list_campaigns` | Campaigns with their status and settings, with an optional status filter. |
| `get_campaign_stats` | The campaign's existing metrics (by status, sent, replies, bounces, rates, today's cap) plus **opens and clicks** from the `EmailSend` rows campaign sends write. |
| `add_leads_to_campaign` | Calls the existing add-contacts handler: emails are verified before drafting, contacts that are suppressed or already mailed are skipped, and the result tells you why. |
| `search_mailboxes` | Connected Gmail and SMTP mailboxes: status, whether sending is on, today's count, owner. **Tokens and passwords are never returned.** |
| `get_unsubscribes` | The org suppression list (by default `reason=unsubscribe`), plus contacts that carry only the legacy `unsubscribed` flag, so opt-outs recorded by the old link handler are not missed. |

The write tools go through the same handlers as the UI, so no MCP call can skip
a compliance gate. No tool sends mail directly. Sending stays with the campaign
pipeline and `send_outreach_email`.

## Dependencies

`mcp` requires `pydantic>=2.11`, and the API image pins `pydantic==2.10.3`.
Rather than move the API's pins for a process that doesn't run in that image,
the server has its own `requirements-mcp.txt`. It is the runtime requirements
with the pydantic pins relaxed, plus `mcp`. Install it into a separate venv.

## stdio hygiene

On stdio, stdout *is* the protocol stream, so one stray byte corrupts it.
SQLAlchemy's `echo` (on when `APP_ENV=development`) writes SQL to stdout. The
entry point turns echo off, sends logging to stderr, and hands the transport the
real stdout while `sys.stdout` points at stderr. A stray `print` in any imported
module therefore lands in the log, not the stream.

## Claude Desktop config

```json
{
  "mcpServers": {
    "doaide-prospect": {
      "command": "/path/to/mcp-venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/prospect-backend",
      "env": {
        "PROSPECT_API_KEY": "llk_…",
        "DATABASE_URL": "postgresql+asyncpg://user:pass@host:5432/landline_db",
        "REDIS_URL": "redis://localhost:6380/0"
      }
    }
  }
}
```

Claude Code: `claude mcp add doaide-prospect --env PROSPECT_API_KEY=llk_… --env DATABASE_URL=… -- /path/to/mcp-venv/bin/python -m app.mcp_server`
(run from the repo root).

## Tests

`tests/test_mcp_server.py` covers each tool against the SQLite test DB:
- the happy path;
- org isolation (another org's key sees nothing of this one);
- a missing scope is refused;
- a revoked key is refused;
- write tools refuse when there is no acting user;
- the server registers exactly these tools;
- the entry point keeps stdout clean;
- a real `python -m app.mcp_server` subprocess completes an MCP handshake.

Run them from the MCP venv (the prod-pinned venv skips the file):

```bash
python3.11 -m venv .venv-mcp && .venv-mcp/bin/pip install -r requirements-mcp.txt pytest pytest-asyncio aiosqlite
.venv-mcp/bin/python -m pytest tests/test_mcp_server.py -v
```

## Out of scope

- No HTTP/SSE transport: stdio only, for desktop use.
- No tools that send mail or change suppression.
