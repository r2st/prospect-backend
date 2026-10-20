"""``python -m app.mcp_server`` — serve the Prospect tools over stdio.

On stdio, file descriptor 1 *is* the protocol stream: one stray byte on it and
the client drops the connection with a JSON parse error that names nothing.
Things in this codebase write there without meaning to: SQLAlchemy's ``echo``
(on whenever ``APP_ENV=development``) installs a stdout handler for every SQL
statement, and any ``print`` in any module the tools import would land there
too. So before anything else is imported, :func:`isolate_stdout` keeps a
private duplicate of fd 1 for the protocol and points fd 1 itself at stderr.
Stray output, Python or C level, then lands in the log, where it belongs.
"""

from __future__ import annotations

import io
import logging
import os
import sys


def isolate_stdout() -> io.TextIOWrapper:
    """Return a private text stream on the real stdout; redirect fd 1 to stderr."""
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", closefd=False), encoding="utf-8",
                                  line_buffering=True)
    return io.TextIOWrapper(os.fdopen(protocol_fd, "wb"), encoding="utf-8")


def silence_sql_echo() -> None:
    """Turn off ``echo`` on the shared engine (it logs SQL to stdout)."""
    from app.database import engine

    engine.sync_engine.echo = False


async def serve(protocol_out: io.TextIOWrapper) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    from app.mcp_server.server import build_server

    server = build_server()
    # FastMCP.run_stdio_async() always wraps sys.stdout; this is that method
    # with the protocol stream passed in explicitly.
    async with stdio_server(stdout=anyio.wrap_file(protocol_out)) as (read, write):
        await server._mcp_server.run(read, write, server._mcp_server.create_initialization_options())


def main() -> None:
    protocol_out = isolate_stdout()
    logging.basicConfig(
        stream=sys.stderr,
        level=os.environ.get("MCP_LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    silence_sql_echo()
    if not os.environ.get("PROSPECT_API_KEY"):
        logging.getLogger("app.mcp_server").warning(
            "PROSPECT_API_KEY is not set; every tool call will be refused."
        )

    import anyio

    anyio.run(serve, protocol_out)


if __name__ == "__main__":
    main()
