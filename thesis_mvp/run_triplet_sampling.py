"""
Thesis MVP: repeated triplet extraction with stochastic decoding, belief graph,
and uncertainty metrics (edge frequency, entropy). No GPU/CUDA/GLIP required.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Load .env from thesis_mvp/ if present (so OPENAI_API_KEY can live in thesis_mvp/.env)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

import openai
import pandas as pd
from tqdm import tqdm

from thesis_mvp.belief_graph import (
    build_belief_graph,
    global_entropy,
    predicate_disagreement,
    summary_stats_per_caption,
    to_dataframe,
)
from thesis_mvp.canonicalize import canonicalize_triplets, load_synonyms_file
from thesis_mvp.triplet_parser import parse_triplets_from_response
from thesis_mvp.search_planner import (
    plan_locations,
    simulate_search,
)
from thesis_mvp.caption_hist_plot import save_caption_edge_histogram

# --- Few-shot prompt (LLM4SGG-style, single caption) ---
SINGLE_CAPTION_PROMPT_PREFIX = """\
From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being).
Let's take a few examples to understand how to extract meaningful triplets.
Question: Given the sentence 'a slice of bread is covered with a sour cream and quacamole,' extract meaningful triplets. Answer:
Meaningful triplets are <bread, covered with, sour cream>, and <bread, covered with, guacamole>.
Question: Given the sentence 'A beautiful woman walking a dog on top of a beach,' extract meaningful triplets. Answer:
Meaningful triplets are <woman, walking with, dog>, <woman, on, beach>, and <dog, on, beach>.
Question: Given the sentence 'Four clock sitting on a floor next to a woman's feet,' extract meaningful triplets. Answer:
Meaningful triplets are <clock, sitting on, floor> and <clock, next to, feet>.
Question: Given the sentence 'One person sits in a chair looking at her phone while another rests on the couch,' extract meaningful triplets. Answer:
Meaningful triplets are <person, sits in, chair>, <person, looking at, phone>, and <person, rests on, couch>.
Question: Given the sentence 'A lady and a child near a park bench with kites and ducks flying in the sky and on the ground,' extract meaningful triplets. Answer:
Meaningful triplets are <lady, near, park bench>, <child, near, park bench>, <kites, flying in sky>, and <ducks, on, ground>.
Question: Given the sentence 'Two men sit on a bench near the sidewalk and one of them talks on a cell phone,' extract meaningful triplets. Answer:
Meaningful triplets are <men, sit on, bench>, <bench, near, sidewalk>, and <man, talks on, phone>.
Please answer the following one question.
Question: Given the sentence '"""

SINGLE_CAPTION_PROMPT_SUFFIX = "', extract meaningful triplets. Answer: "


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("Error: OPENAI_API_KEY is not set. Set it in your environment or thesis_mvp/.env", file=sys.stderr)
        sys.exit(1)
    return key


def get_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return ""


def extract_triplets_once(caption: str, model: str, temperature: float) -> str:
    """One API call for one caption; returns raw response text."""
    caption_clean = caption.strip().strip("\n").strip("'").strip(".")
    content = SINGLE_CAPTION_PROMPT_PREFIX + caption_clean + SINGLE_CAPTION_PROMPT_SUFFIX
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    return completion.choices[0].message.content


def load_captions(captions_file: str | None, hardcoded: list[str], max_captions: int | None) -> list[str]:
    """Merge hardcoded captions with optional file (one caption per line). Apply max_captions if set."""
    captions = list(hardcoded)
    if captions_file and Path(captions_file).exists():
        with open(captions_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    captions.append(line)
    if max_captions is not None and max_captions > 0:
        captions = captions[:max_captions]
    return captions


def main():
    parser = argparse.ArgumentParser(description="Thesis MVP: repeated triplet extraction and belief graph")
    parser.add_argument("--captions_file", type=str, default=None, help="Text file with one caption per line")
    parser.add_argument(
        "--no_hardcoded",
        action="store_true",
        help="Do not prepend the 3 default example captions; use only --captions_file (required unless max_captions tests)",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of extraction runs per caption (K)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (ignored if --compare_temps set)")
    parser.add_argument("--compare_temps", type=str, default=None, help="Comma-separated temps, e.g. 0.0,0.7; run each caption at each temp")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI chat model")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: thesis_mvp/outputs/<timestamp>)")
    parser.add_argument("--plot_type", type=str, default="entropy", choices=["entropy", "prob"], help="Histogram: entropy or probability")
    parser.add_argument("--max_captions", type=int, default=None, help="Limit number of captions to process (for quick tests)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (stored in metadata for reproducibility)")
    parser.add_argument("--canonicalize", type=str, default="true", choices=["true", "false"], help="Apply entity/predicate canonicalization (default true)")
    parser.add_argument("--synonyms_file", type=str, default=None, help="Optional CSV 'from,to' for extra synonym mappings")
    parser.add_argument(
        "--plan_target",
        type=str,
        default=None,
        help="If set, run object-search planning: marginal P(location) for this subject (canonicalized)",
    )
    parser.add_argument(
        "--plan_spatial_only",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Only sum edges whose predicate looks spatial (default true)",
    )
    parser.add_argument(
        "--plan_truth",
        type=str,
        default=None,
        help="Ground-truth location for simulated search (canonicalized); use with --plan_target",
    )
    parser.add_argument(
        "--plan_simulate",
        action="store_true",
        help="If set with --plan_target and --plan_truth, simulate greedy vs uncertainty_aware search order",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    openai.api_key = api_key

    # Resolve temperatures
    if args.compare_temps:
        temps = [float(t.strip()) for t in args.compare_temps.split(",") if t.strip()]
        if not temps:
            temps = [0.0, 0.7]
    else:
        temps = [args.temperature]

    do_canonicalize = args.canonicalize.lower() == "true"
    synonyms = load_synonyms_file(args.synonyms_file) if args.synonyms_file else {}

    hardcoded = [] if args.no_hardcoded else [
        "A person sits in a chair looking at her phone while another rests on the couch.",
        "A living room with a sofa, a coffee table, and a lamp in the corner.",
        "Two men sit on a bench near the sidewalk and one of them talks on a cell phone.",
    ]
    captions = load_captions(args.captions_file, hardcoded, args.max_captions)
    if not captions:
        print("No captions to process. Add --captions_file or use hardcoded list.", file=sys.stderr)
        sys.exit(1)

    if args.plan_simulate and not args.plan_truth:
        print("Warning: --plan_simulate ignored without --plan_truth (ground-truth location).", file=sys.stderr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir or f"thesis_mvp/outputs/{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    summary_rows = []
    planning_rows: list[dict] = []
    plan_spatial = args.plan_spatial_only.lower() == "true"
    # Per-temp aggregates for final console summary: temp -> {global_entropies[], n_unique_edges[], total_disagreements}
    per_temp = {t: {"global_entropies": [], "n_unique_edges": [], "disagreements": 0} for t in temps}

    for temp in temps:
        temp_dir = out_root / f"temp_{temp}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for cap_idx, caption in enumerate(captions):
            print(f"\n--- Caption {cap_idx + 1}/{len(captions)} (temp={temp}) ---")
            print(caption[:80] + ("..." if len(caption) > 80 else ""))

            run_triplets = []
            for _ in tqdm(range(args.runs), desc="Runs"):
                try:
                    raw = extract_triplets_once(caption, args.model, temp)
                    triplets = parse_triplets_from_response(raw)
                    run_triplets.append(triplets)
                except Exception as e:
                    print(f"  API/parse error: {e}", file=sys.stderr)
                    run_triplets.append([])

            k = len(run_triplets)
            n_entity_variants = 0
            if do_canonicalize:
                run_triplets, n_entity_variants = canonicalize_triplets(run_triplets, synonyms)

            edge_counts, edge_probs, edge_entropies = build_belief_graph(run_triplets, k)
            df = to_dataframe(edge_counts, edge_probs, edge_entropies)

            safe_label = f"caption_{cap_idx + 1}"
            csv_path = temp_dir / f"{safe_label}_edges.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Wrote {csv_path}")

            stats = summary_stats_per_caption(edge_counts, edge_entropies, k)
            disagreements_list = predicate_disagreement(edge_counts)
            n_predicate_disagreements = len(disagreements_list)

            per_temp[temp]["global_entropies"].append(stats["global_entropy"])
            per_temp[temp]["n_unique_edges"].append(stats["n_unique_edges"])
            per_temp[temp]["disagreements"] += n_predicate_disagreements

            summary_rows.append({
                "caption_id": cap_idx + 1,
                "caption_text": caption[:200],
                "temperature": temp,
                "runs": k,
                "n_unique_edges": stats["n_unique_edges"],
                "global_entropy": round(stats["global_entropy"], 4),
                "mean_entropy_per_edge": round(stats["mean_entropy_per_edge"], 4),
                "n_predicate_disagreements": n_predicate_disagreements,
                "n_entity_variants": n_entity_variants,
                "timestamp": timestamp,
            })

            # Top 10 most frequent
            top_freq = df.head(10)
            print("\n  Top 10 most frequent edges:")
            for _, row in top_freq.iterrows():
                print(f"    ({row['count']}/{k}) ({row['subject']}, {row['predicate']}, {row['object']})")

            # Top 10 highest entropy
            df_ent = df.sort_values("entropy", ascending=False).head(10)
            print("\n  Top 10 highest-entropy edges:")
            for _, row in df_ent.iterrows():
                print(f"    H={row['entropy']:.3f} ({row['subject']}, {row['predicate']}, {row['object']})")

            # Predicate disagreement (top 5)
            if disagreements_list:
                print("\n  Predicate disagreement (top 5):")
                for (s, o), pred_list in sorted(disagreements_list, key=lambda x: -sum(c for _, c in x[1]))[:5]:
                    pred_str = ", ".join(f"{p}({c})" for p, c in pred_list)
                    print(f"    ({s}, {o}): {pred_str}")
            else:
                print("\n  No predicate disagreement in this caption.")

            if args.plan_target:
                raw_m, P_tilde, g_loc, u_loc = plan_locations(
                    edge_probs,
                    args.plan_target,
                    synonyms=synonyms,
                    spatial_only=plan_spatial,
                )
                print(f"\n  --- Planning (target={args.plan_target!r}) ---")
                if not P_tilde:
                    print("  No location mass for this target in the belief graph (check spelling / canonical form).")
                else:
                    print("  P̃(location) | greedy baseline (argmax P̃) →", g_loc, "| uncertainty-aware →", u_loc)
                    for loc in sorted(P_tilde.keys(), key=lambda x: -P_tilde[x]):
                        r = raw_m.get(loc, 0.0)
                        print(f"    {loc!r}: P̃={P_tilde[loc]:.4f} (raw mass={r:.4f})")
                planning_rows.append({
                    "caption_id": cap_idx + 1,
                    "temperature": temp,
                    "plan_target": args.plan_target,
                    "greedy_location": g_loc or "",
                    "uncertainty_aware_location": u_loc or "",
                    "n_location_candidates": len(P_tilde),
                })
                if args.plan_simulate and args.plan_truth:
                    for pol in ("greedy", "uncertainty_aware"):
                        sim = simulate_search(
                            edge_probs,
                            args.plan_target,
                            args.plan_truth,
                            policy=pol,  # type: ignore[arg-type]
                            synonyms=synonyms,
                            spatial_only=plan_spatial,
                        )
                        print(
                            f"  Simulate [{pol}]: steps={sim.steps}, success={sim.success}, order={sim.visited}"
                        )
                        planning_rows[-1][f"sim_{pol}_steps"] = sim.steps
                        planning_rows[-1][f"sim_{pol}_success"] = sim.success
                        planning_rows[-1][f"sim_{pol}_order"] = ";".join(sim.visited)

            # Plot (full scene caption in title, word-wrapped; figure height scales with text)
            if plt is not None and len(df) > 0:
                plot_path = temp_dir / f"{safe_label}_hist_{args.plot_type}.png"
                save_caption_edge_histogram(
                    df,
                    cap_idx + 1,
                    float(temp),
                    caption,
                    plot_path,
                    "entropy" if args.plot_type == "entropy" else "prob",
                )
                print(f"  Wrote {plot_path}")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")

    if planning_rows:
        plan_df = pd.DataFrame(planning_rows)
        plan_path = out_root / "planning_summary.csv"
        plan_df.to_csv(plan_path, index=False)
        print(f"Wrote {plan_path}")

    # metadata.json
    metadata = {
        "model": args.model,
        "temperatures": temps,
        "runs": args.runs,
        "canonicalize": do_canonicalize,
        "plot_type": args.plot_type,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "seed": args.seed,
        "n_captions": len(captions),
        "plan_target": args.plan_target,
        "plan_spatial_only": plan_spatial,
        "plan_simulate": bool(args.plan_simulate and args.plan_truth),
    }
    meta_path = out_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {meta_path}")

    # Global entropy by temp (bar chart)
    if plt is not None and temps:
        fig, ax = plt.subplots(figsize=(6, 4))
        x_temps = [str(t) for t in temps]
        mean_entropies = []
        for t in temps:
            vals = per_temp[t]["global_entropies"]
            mean_entropies.append(sum(vals) / len(vals) if vals else 0)
        ax.bar(x_temps, mean_entropies, edgecolor="black")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Mean global entropy (across captions)")
        ax.set_title("Mean global entropy by temperature")
        plt.tight_layout()
        entropy_plot_path = out_root / "global_entropy_by_temp.png"
        plt.savefig(entropy_plot_path, dpi=100)
        plt.close()
        print(f"Wrote {entropy_plot_path}")

    # Compact console summary
    print("\n" + "=" * 60)
    print("COMPACT SUMMARY BY TEMPERATURE")
    print("=" * 60)
    for t in temps:
        vals = per_temp[t]
        n_c = len(vals["global_entropies"])
        mean_ent = sum(vals["global_entropies"]) / n_c if n_c else 0
        mean_edges = sum(vals["n_unique_edges"]) / n_c if n_c else 0
        print(f"  Temp {t}: mean global entropy = {mean_ent:.4f}, mean unique edges = {mean_edges:.2f}, total predicate disagreements = {vals['disagreements']}")
    print(f"\nOutputs saved under: {out_root}")


if __name__ == "__main__":
    main()
