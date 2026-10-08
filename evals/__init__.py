"""Evaluation layer — measures answer and agent QUALITY (the red team measures safety).

    python -m evals.run_evals --live      # run the agent once, record it, score it
    python -m evals.run_evals             # re-score the recorded runs (free, deterministic)
    python -m evals.run_evals --judge     # re-score with DeepEval LLM-judge metrics

Install the extra dependencies first:  pip install -r requirements-eval.txt
"""