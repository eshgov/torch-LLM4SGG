# Thesis MVP: Repeated Triplet Extraction & Belief Graph

Lightweight pipeline for **repeated LLM-based triplet extraction** (subject–predicate–object) from captions, then building a **belief graph** with edge frequencies and entropy. No GPU, CUDA, or GLIP required.

---

## Quick start (step-by-step, from scratch)

Do this from the **repository root** (the folder that contains `thesis_mvp/`).

**Step 1 — Open Terminal and go to the repo:**
```bash
cd /Users/eshaangovil/Desktop/Princeton/Thesis/torch-LLM4SGG
```

**Step 2 — Create a virtual environment:**
```bash
python3 -m venv .venv
```

**Step 3 — Activate the virtual environment:**  
*(You need to do this every time you open a new terminal to run the pipeline.)*
```bash
source .venv/bin/activate
```
You should see `(.venv)` at the start of your prompt.

**Step 4 — Install dependencies:**
```bash
pip install -r thesis_mvp/requirements.txt
```

**Step 5 — Set your OpenAI API key:**  
Either put it in `thesis_mvp/.env` as `OPENAI_API_KEY=sk-...` (if you use `python-dotenv`), or:
```bash
export OPENAI_API_KEY='your-key-here'
```

**Step 6 — Run the pipeline:**  
See **Example commands** below.

---

## Example commands

**Quick test** (5 captions, 3 runs per caption, compare temp 0 vs 0.7):
```bash
python -m thesis_mvp.run_triplet_sampling --captions_file thesis_mvp/captions.txt --max_captions 5 --compare_temps 0.0,0.7 --runs 3
```

**Thesis-grade preliminary results** (all captions, 20 runs, deterministic vs stochastic):
```bash
python -m thesis_mvp.run_triplet_sampling --captions_file thesis_mvp/captions.txt --compare_temps 0.0,0.7 --runs 20
```

**Single temperature, probability histogram, custom output dir:**
```bash
python -m thesis_mvp.run_triplet_sampling --captions_file thesis_mvp/captions.txt --runs 10 --temperature 0.7 --plot_type prob --out_dir my_run
```

---

## Outputs

All written under `thesis_mvp/outputs/<timestamp>/` (or `--out_dir`):

| Path | Description |
|------|-------------|
| `temp_<T>/caption_<i>_edges.csv` | Per-caption, per-temperature edge table: `subject`, `predicate`, `object`, `count`, `probability`, `entropy`. |
| `temp_<T>/caption_<i>_hist_entropy.png` or `_hist_prob.png` | Histogram of edge entropies or probabilities for that caption (see `--plot_type`). |
| `summary.csv` | **One row per (caption, temperature).** Columns: `caption_id`, `caption_text`, `temperature`, `runs`, `n_unique_edges`, `global_entropy`, `mean_entropy_per_edge`, `n_predicate_disagreements`, `n_entity_variants`, `timestamp`. Use this for thesis figures: e.g. plot mean global entropy vs temperature, or compare n_predicate_disagreements across temps. |
| `global_entropy_by_temp.png` | Bar chart: temperature vs mean global entropy across captions (one plot per run). |
| `metadata.json` | Run config: `model`, `temperatures`, `runs`, `canonicalize`, `plot_type`, `timestamp`, `git_commit`, `seed`, `n_captions`. |

---

## Using summary.csv for thesis figures

- **Temperature comparison:** Group by `temperature`; plot mean `global_entropy` or mean `n_unique_edges` (with error bars if you run multiple seeds).
- **Uncertainty vs determinism:** Compare rows with `temperature=0` vs `temperature=0.7` (mean entropy, total predicate disagreements).
- **Entity canonicalization effect:** Compare `n_entity_variants` across runs with `canonicalize=true` vs `false` to show how many surface forms were merged.

---

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--captions_file` | None | Text file: one caption per line (optional; script also has a few hardcoded captions). |
| `--runs` | 10 | Number of extraction runs **K** per caption (same for all temperatures). |
| `--temperature` | 0.7 | LLM temperature when not using `--compare_temps`. |
| `--compare_temps` | None | Comma-separated list, e.g. `0.0,0.7`. Run each caption at each temperature; ignores `--temperature`. |
| `--plot_type` | entropy | Histogram per caption: `entropy` or `prob`. |
| `--max_captions` | None | Limit number of captions (for quick tests). |
| `--seed` | None | Integer stored in metadata (reproducibility). |
| `--canonicalize` | true | Apply entity/predicate canonicalization (e.g. "cell phone"→"phone"). |
| `--synonyms_file` | None | Optional CSV with columns `from,to` for extra synonym mappings. |
| `--model` | gpt-3.5-turbo | OpenAI chat model. |
| `--out_dir` | thesis_mvp/outputs/&lt;timestamp&gt; | Output directory. |

---

## Files in `thesis_mvp/`

| File | Role |
|------|------|
| `run_triplet_sampling.py` | Entrypoint: load captions, run extraction at each temperature, parse, canonicalize, build belief graph, write CSVs/plots/summary/metadata. |
| `triplet_parser.py` | Parses LLM text into `(subject, predicate, object)` tuples. |
| `belief_graph.py` | Edge counts, P(e), H(e), predicate disagreement. |
| `canonicalize.py` | Entity canonicalization (lowercase, determiners, synonyms, simple plurals) and predicate normalization. |
| `captions.txt` | Example captions (~25+); includes spatially ambiguous and containment-style sentences. |
| `README.md` | This file. |

---

## What you're looking at (interpreting results)

- **Edge (s, p, o):** One relation: subject, predicate, object. **P(e) = count/K.** **H(e)** = binary entropy (0 = certain, max at 50/50).
- **Predicate disagreement:** Same (subject, object) with different predicates across runs; top 5 per caption in console.
- **n_entity_variants (in summary.csv):** Number of subject/object slots changed by canonicalization (approximate).
- **global_entropy_by_temp.png:** Higher mean entropy at higher temperature → more variability under stochastic decoding.

---

## Belief graph (short)

- **Edge** \(e = (s, p, o)\). **Count** = how many of K runs contained \(e\). **P(e) = count/K**.
- **Entropy per edge:** \(H(e) = -P(e)\log_2 P(e) - (1-P(e))\log_2(1-P(e))\).
- **Global entropy** = sum of \(H(e)\) over all edges (per caption). Compare across temperatures in `summary.csv` or `global_entropy_by_temp.png`.
