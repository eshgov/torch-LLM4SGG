#!/usr/bin/env python3
"""
Run triplet sampling on captions, then planning experiments (greedy vs UA) on the same output dir.

Example:
  export OPENAI_API_KEY=sk-...
  python -m thesis_mvp.run_full_pipeline --out_dir thesis_mvp/outputs/full_pipeline_latest

Requires: pip install -r thesis_mvp/requirements.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Full thesis MVP: sampling + planning eval")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(root / "thesis_mvp" / "outputs" / "full_pipeline_latest"),
        help="Fixed output directory for sampling (and planning_eval/ inside it)",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=str(root / "thesis_mvp" / "captions.txt"),
    )
    parser.add_argument("--runs", type=int, default=20, help="K runs per caption per temperature")
    parser.add_argument(
        "--compare_temps",
        type=str,
        default="0.0,0.5,0.7,1.0,1.2",
        help="Comma-separated temperatures (more temps + higher T -> more disagreement / planning rows)",
    )
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--max_captions",
        type=int,
        default=None,
        help="Limit captions (for smoke tests); omit for full captions.txt",
    )
    parser.add_argument(
        "--no_hardcoded",
        action="store_true",
        help="Pass to sampling: use only captions from captions.txt (no 3 default examples)",
    )
    parser.add_argument("--skip_sampling", action="store_true", help="Only run planning on existing out_dir")
    parser.add_argument(
        "--spatial_only",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Passed to run_planning_experiments (spatial predicate filter)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    py = sys.executable

    if not args.skip_sampling:
        cmd = [
            py,
            "-m",
            "thesis_mvp.run_triplet_sampling",
            "--captions_file",
            args.captions_file,
            "--out_dir",
            str(out_dir),
            "--runs",
            str(args.runs),
            "--compare_temps",
            args.compare_temps,
            "--model",
            args.model,
        ]
        if args.max_captions is not None:
            cmd.extend(["--max_captions", str(args.max_captions)])
        if args.no_hardcoded:
            cmd.append("--no_hardcoded")
        print("Running:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            sys.exit(r.returncode)

    cmd2 = [
        py,
        "-m",
        "thesis_mvp.run_planning_experiments",
        "--outputs_dir",
        str(out_dir),
        "--spatial_only",
        args.spatial_only,
    ]
    print("Running:", " ".join(cmd2), flush=True)
    r2 = subprocess.run(cmd2, cwd=str(root))
    if r2.returncode != 0:
        sys.exit(r2.returncode)

    eval_dir = out_dir / "planning_eval"
    detail = eval_dir / "planning_experiment_detail.csv"
    agg = eval_dir / "planning_experiment_aggregate.csv"
    print("\n--- Pipeline complete ---", flush=True)
    print(f"Sampling + plots: {out_dir}", flush=True)
    print(f"Planning tables/figures: {eval_dir}", flush=True)
    if detail.exists():
        import pandas as pd

        df = pd.read_csv(detail)
        print(f"\nPredicate-disagreement search cases (planning rows): {len(df)}", flush=True)
        if len(df) < 20:
            print(
                "Tip: raise --runs (e.g. 30), add a higher temp in --compare_temps, "
                "or add more vague captions if you need 20+ cases.",
                flush=True,
            )
        else:
            print("Enough rows for aggregate analysis (>= 20).", flush=True)
        if agg.exists():
            print("\nAggregate by temperature:", flush=True)
            print(pd.read_csv(agg).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
