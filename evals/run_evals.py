"""
evals/run_evals.py — the evaluation layer's runner.

    python -m evals.run_evals --live                 # run the agent on the golden set (real LLM,
                                                     # FAKE Jira/Gmail), record, then score:
                                                     # deterministic + LLM-judge (rag, agent, qa)
    python -m evals.run_evals --live --no-judge      # record + deterministic only (cheapest)
    python -m evals.run_evals                        # re-score the RECORDED runs: deterministic, free
    python -m evals.run_evals --judge                # re-score recorded runs with the LLM judge
    python -m evals.run_evals --judge --suites all   # + the 10-metric matrix
    python -m evals.run_evals --live --only E1,E8

Judge metrics run in parallel (EVAL_WORKERS, default 8). Judge model: EVAL_JUDGE_MODEL
(default gpt-4o-mini). Both the agent's and the judge's cost are printed.

Recorded runs:  results/eval_runs.jsonl   (re-score them as often as you like)
Reports:        results/eval_report.md · results/eval_scores.csv
Exit code 1 if any DETERMINISTIC check fails (LLM-judge scores are reported, not gated,
unless --strict), so it can be used as a CI quality gate.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

RUNS_FILE = Path("results/eval_runs.jsonl")


def score_runs(runs: list[dict], suites=("deterministic",), workers: int | None = None) -> list[dict]:
    """Score recorded runs. Returns one row per (case, metric).

    Deterministic metrics run in order (instant). LLM-judge metrics are network-bound,
    so they run in parallel threads (EVAL_WORKERS, default 8); rows keep a stable order."""
    from concurrent.futures import ThreadPoolExecutor

    from evals.golden import GOLDEN
    from evals.metrics import deterministic_metrics, judge_metrics, run_metric, to_test_case

    by_id = {c["id"]: c for c in GOLDEN}
    rows, jobs = [], []
    for run in runs:
        case = by_id.get(run["id"])
        if case is None:
            continue
        tc = to_test_case(case, run)
        if "deterministic" in suites:
            for metric in deterministic_metrics(case):
                rows.append({"id": case["id"], "category": case["category"], "suite": "deterministic",
                             **run_metric(metric, tc)})
        judge = [s for s in suites if s != "deterministic"]
        if judge:
            jobs += [(case, suite, metric, test_case)
                     for suite, metric, test_case in judge_metrics(case, run, judge)]
    if jobs:
        n = workers or int(os.getenv("EVAL_WORKERS", "8"))
        with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
            results = list(pool.map(lambda j: run_metric(j[2], j[3]), jobs))   # map keeps order
        rows += [{"id": c["id"], "category": c["category"], "suite": suite, **res}
                 for (c, suite, _m, _tc), res in zip(jobs, results)]
    return rows


def _gating(metric: str) -> bool:
    from evals.metrics import is_gating
    return is_gating(metric)


def summarize(rows: list[dict], runs: list[dict]) -> dict:
    per_metric: dict = defaultdict(lambda: {"scores": [], "passed": 0, "total": 0, "suite": ""})
    for r in rows:
        m = per_metric[r["metric"]]
        m["suite"] = r["suite"]
        m["total"] += 1
        m["passed"] += bool(r["success"])
        if r["score"] is not None:
            m["scores"].append(r["score"])
    det = [r for r in rows if r["suite"] == "deterministic"]
    cases_ok = {}
    for r in det:
        cases_ok[r["id"]] = cases_ok.get(r["id"], True) and bool(r["success"])
    return {
        "per_metric": {k: {"suite": v["suite"], "avg": round(sum(v["scores"]) / len(v["scores"]), 3)
                           if v["scores"] else None, "passed": v["passed"], "total": v["total"]}
                       for k, v in per_metric.items()},
        "cases_passed": sum(cases_ok.values()), "cases": len(cases_ok),
        "det_failures": [f"{r['id']}:{r['metric']}" for r in det if not r["success"]],
        "judge_failures": [f"{r['id']}:{r['metric']}" for r in rows
                           if r["suite"] != "deterministic" and not r["success"]
                           and _gating(r["metric"])],
        "agent_tokens": sum(r.get("tokens", 0) for r in runs),
        "agent_cost": round(sum(r.get("cost_usd", 0) for r in runs), 5),
        "judge_cost": round(sum(r.get("cost", 0) for r in rows), 5),
    }


def write_reports(rows: list[dict], runs: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "eval_scores.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    s = summarize(rows, runs)
    lines = ["# Evaluation report", "",
             f"- **Cases passing every deterministic check:** {s['cases_passed']}/{s['cases']}",
             f"- **Agent:** {s['agent_tokens']:,} tokens · ${s['agent_cost']:.4f}"
             f" · **Judge:** ${s['judge_cost']:.4f}", "",
             "## By metric", "", "| Suite | Metric | Avg score | Passed |", "|---|---|---|---|"]
    order = {"deterministic": 0, "rag": 1, "agent": 2, "qa": 3, "matrix": 4}
    for name, v in sorted(s["per_metric"].items(), key=lambda kv: (order.get(kv[1]["suite"], 9), kv[0])):
        avg = "—" if v["avg"] is None else f"{v['avg']:.2f}"
        label = name if _gating(name) else f"{name} *(informational)*"
        lines.append(f"| {v['suite']} | {label} | {avg} | {v['passed']}/{v['total']} |")
    lines += ["", "*Informational* metrics are tracked but never block a release — see "
              "`INFORMATIONAL` in evals/metrics.py.", "",
              "## By case", "", "| Case | Route | Tokens | Failed checks |", "|---|---|---|---|"]
    fails = defaultdict(list)
    for r in rows:
        if not r["success"]:
            fails[r["id"]].append(r["metric"])
    for run in runs:
        lines.append(f"| {run['id']} | {' → '.join(run.get('route') or ['(none)'])} | "
                     f"{run.get('tokens', 0):,} | {', '.join(fails.get(run['id'], [])) or '—'} |")
    bad = [r for r in rows if not r["success"]]
    if bad:
        lines += ["", "## Why checks failed", "", "| Case | Metric | Score | Reason |", "|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['metric']} | {r['score']} | {r['reason'].replace('|', '/')[:160]} |"
                  for r in bad]
    path = out_dir / "eval_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="run the agent on the golden set first (real LLM)")
    ap.add_argument("--judge", action="store_true", help="add LLM-judge suites when re-scoring")
    ap.add_argument("--no-judge", action="store_true", help="with --live: deterministic metrics only")
    ap.add_argument("--suites", default="", help="judge suites: rag,agent,qa,matrix or 'all'")
    ap.add_argument("--only", help="comma-separated case ids, e.g. E1,E8")
    ap.add_argument("--out", default="results", help="report folder (default: results/)")
    ap.add_argument("--strict", action="store_true", help="also fail on LLM-judge scores below threshold")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()
    from evals.golden import GOLDEN
    from evals.harness import load_runs, run_case, save_runs
    from evals.metrics import DEFAULT_JUDGE_SUITES

    out = Path(args.out)
    runs_file = out / RUNS_FILE.name
    wanted = {x.strip() for x in args.only.split(",")} if args.only else None

    if args.live:
        cases = [c for c in GOLDEN if not wanted or c["id"] in wanted]
        print(f"\nRUN      {len(cases)} golden case(s) through the real agent (fake Jira/Gmail)…")
        runs = []
        for c in cases:
            r = run_case(c)
            runs.append(r)
            print(f"  {c['id']:4s} {c['category']:15s} {' → '.join(r['route']) or '(none)':34s} "
                  f"{r['tokens']:>7,} tok  ${r['cost_usd']:.4f}  {r['seconds']:>5.1f}s"
                  + (f"  ERROR {r['error'][:60]}" if r["error"] else ""))
        if wanted:                                    # keep the other recorded runs
            keep = [r for r in load_runs(runs_file) if r["id"] not in wanted]
            runs_to_save = keep + runs
        else:
            runs_to_save = runs
        save_runs(runs_to_save, runs_file)
        print(f"  recorded → {runs_file}")
    else:
        runs = [r for r in load_runs(runs_file) if not wanted or r["id"] in wanted]
        if not runs:
            print(f"No recorded runs in {runs_file}. Run first:  python -m evals.run_evals --live")
            return 1

    suites = ["deterministic"]
    if (args.live and not args.no_judge) or args.judge:
        chosen = (["rag", "agent", "qa", "matrix"] if args.suites == "all"
                  else [s.strip() for s in args.suites.split(",") if s.strip()] or list(DEFAULT_JUDGE_SUITES))
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set — LLM-judge suites skipped.")
        else:
            suites += chosen
    print(f"\nSCORE    suites: {', '.join(suites)}")
    rows = score_runs(runs, suites)
    s = summarize(rows, runs)

    for name, v in sorted(s["per_metric"].items(), key=lambda kv: kv[1]["suite"]):
        avg = "   —" if v["avg"] is None else f"{v['avg']:.2f}"
        note = "" if _gating(name) else "   (informational — not gated)"
        print(f"  {v['suite']:13s} {name:26s} avg {avg}   passed {v['passed']}/{v['total']}{note}")
    print(f"\n  cases passing all deterministic checks: {s['cases_passed']}/{s['cases']}")
    print(f"  agent cost ${s['agent_cost']:.4f} ({s['agent_tokens']:,} tokens) · judge cost ${s['judge_cost']:.4f}")
    for f in s["det_failures"]:
        print(f"  ❌ {f}")
    path = write_reports(rows, runs, out)
    print(f"\nReport: {path}")
    failed = bool(s["det_failures"]) or (args.strict and bool(s["judge_failures"]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())