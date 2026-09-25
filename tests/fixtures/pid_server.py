"""Tiny stdio MCP server for tests: reports its own process id."""
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pid")


@mcp.tool()
def jira_whoami() -> str:
    """Return this server process's PID."""
    return str(os.getpid())


if __name__ == "__main__":
    mcp.run()
