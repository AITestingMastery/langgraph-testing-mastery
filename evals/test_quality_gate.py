"""
evals/test_quality_gate.py — the evaluation layer as a DeepEval / pytest quality gate.

    deepeval test run evals/test_quality_gate.py          # DeepEval's runner + report
    pytest evals/test_quality_gate.py -q                  # plain pytest

Scores the RECORDED runs in results/eval_runs.jsonl (create them first with
`python -m evals.run_evals --live`). Deterministic metrics always run; set
EVAL_GATE_JUDGE=true to add the LLM-judge suites (needs OPENAI_API_KEY). Metrics listed in
evals.metrics.INFORMATIONAL (e.g. Contextual Relevancy) are reported by run_evals but never gate.
Not part of `pytest tests` — it needs recorded runs and, for the judge, an API key.
"""
import os
from pathlib import Path

import pytest

try:                                     # a real install, not a leftover empty folder
    from deepeval.metrics import BaseMetric  # noqa: F401
except ImportError:
    pytest.skip("DeepEval not installed — pip install -r requirements-eval.txt", allow_module_level=True)
from deepeval import assert_test  # noqa: E402

from evals.golden import GOLDEN  # noqa: E402
from evals.harness import load_runs  # noqa: E402
from evals.metrics import (DEFAULT_JUDGE_SUITES, deterministic_metrics, is_gating,  # noqa: E402
                           judge_metrics, to_test_case)

RUNS = load_runs(Path("results/eval_runs.jsonl"))
CASES = {c["id"]: c for c in GOLDEN}
PAIRS = [(CASES[r["id"]], r) for r in RUNS if r["id"] in CASES]

if not PAIRS:
    pytest.skip("no recorded runs — run: python -m evals.run_evals --live", allow_module_level=True)


@pytest.mark.parametrize("case,run", PAIRS, ids=[c["id"] for c, _ in PAIRS])
def test_deterministic_quality(case, run):
    assert_test(to_test_case(case, run), deterministic_metrics(case), run_async=False)


@pytest.mark.skipif(os.getenv("EVAL_GATE_JUDGE", "").lower() != "true",
                    reason="set EVAL_GATE_JUDGE=true to run the LLM-judge gate")
@pytest.mark.parametrize("case,run", PAIRS, ids=[c["id"] for c, _ in PAIRS])
def test_judged_quality(case, run):
    # check EVERY metric, then report all failures together (the first failure must not
    # hide the others); informational metrics are reported by run_evals but never gate
    failures = []
    for _suite, metric, test_case in judge_metrics(case, run, DEFAULT_JUDGE_SUITES):
        if not is_gating(metric.__name__):
            continue
        try:
            assert_test(test_case, [metric], run_async=False)
        except AssertionError as exc:
            failures.append(str(exc).split(" failed.")[0])
    assert not failures, " | ".join(failures)