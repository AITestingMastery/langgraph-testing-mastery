"""tools/native_tools.py — native @tool functions used by the agents."""
from __future__ import annotations
import re
from langchain_core.tools import tool
from rag.retriever import get_retriever

@tool
def search_docs(query: str) -> str:
    """Search the QA knowledge base (test plans, known bugs, API cases)."""
    docs = get_retriever().invoke(query)
    if not docs:
        return "No relevant passages found."
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source','?')}]\n{d.page_content}" for d in docs)

@tool
def format_bug_report(title: str, steps: str, severity: str = "Medium",
                      environment: str = "Not specified") -> str:
    """Turn rough notes into a standardized bug report."""
    severity = severity.capitalize()
    if severity not in {"Low", "Medium", "High", "Critical"}:
        severity = "Medium"
    lines = [re.sub(r"^\s*\d+[.)]\s*", "", s.strip()) for s in steps.splitlines() if s.strip()]
    numbered = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1)) or "1. (none)"
    return (f"**Bug Report**\n\n**Title:** {title}\n**Severity:** {severity}\n"
            f"**Environment:** {environment}\n\n**Steps to Reproduce:**\n{numbered}\n\n**Status:** Open")

@tool
def check_duplicate_bug(summary: str) -> str:
    """Check whether a bug summary looks like a likely duplicate of a known bug.
    Use before creating a Jira ticket to avoid filing duplicates."""
    known = {
        "login button": "BUG-101 (Chrome login button unresponsive)",
        "session": "BUG-087 (session not expiring on reports page)",
        "password reset": "BUG-112 (password reset email delayed)",
    }
    low = summary.lower()
    for key, ref in known.items():
        if key in low:
            return f"Possible duplicate of {ref}. Consider updating that ticket instead."
    return "No obvious duplicate found in known bugs."


@tool
def generate_test_cases(feature: str) -> str:
    """Generate a starter set of test cases for a feature: positive, negative,
    boundary, and one security case."""
    return (f"Test cases for: {feature}\n"
            f"1. Positive — valid input produces the expected result.\n"
            f"2. Negative — invalid input is rejected with a clear message.\n"
            f"3. Boundary — behaviour at limits (empty, max length, edge values).\n"
            f"4. Security — unauthorized/malformed input cannot bypass checks.")


NATIVE_TOOLS = [search_docs, format_bug_report, check_duplicate_bug, generate_test_cases]