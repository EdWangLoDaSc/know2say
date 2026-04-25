# Know2Say

Code for the paper **"The Detection–Extraction Gap: Models Know the Answer Before They Can Say It."**

> **Know2Say** exposes a structural mismatch in reasoning LLMs: early in a chain-of-thought, the answer is already *recoverable* from free continuations (detection), yet forced extraction still fails on a large fraction of the same prefixes. We formalize this **detection–extraction gap** via a total-variation bound and turn it into a practical black-box early-exit policy, **BAEE**, that cuts 70–85% of serial generation while improving accuracy by 1–5 pp.

---

## Setup

```bash
git clone git@github.com:EdWangLoDaSc/know2say.git
cd know2say
pip install -r requirements.txt
```

`requirements.txt` pulls the `tinker` SDK plus `tinker_cookbook`'s runtime
dependencies (`chz`, `datasets`, `transformers`, `sympy`, `pylatexenc`, …).

### tinker-cookbook (vendored)

A frozen subset of [`tinker_cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook)
ships under [`vendor/tinker_cookbook/`](vendor/) — `paths.setup_path()`
prepends this directory to `sys.path`, so no extra install step is needed.
To use a system-installed cookbook instead, override:

```bash
export TINKER_COOKBOOK=/path/to/tinker-cookbook
```

### Tinker SDK API key

Sampling from real models requires a Thinking Machines API key (see
<https://auth.thinkingmachines.ai/sign-up>):

```bash
export TINKER_API_KEY=tk-...
```

Without this you can still re-run all offline analysis and re-render every
figure from the experiment-result JSONLs already checked in.

### Results directory

Experiment outputs go to `/tmp/tinker-examples/reasoning_theater/` by default:

```bash
export REASONING_THEATER_RESULTS=/your/preferred/results/dir
```

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                             # All Python code lives here
│   ├── paths.py                     # Path config (edit defaults here)
│   ├── plot_style.py                # Shared matplotlib palette + rcParams
│   │
│   ├── experiment.py                # Core measurement protocols
│   ├── baee.py                      # Black-box Adaptive Early Exit policy
│   ├── baee_test.py                 # Sanity tests for BAEE
│   ├── baseline_experiment.py       # Prefix-free SC baselines
│   ├── analysis.py                  # Offline analysis + LaTeX tables
│   ├── dashboard.py                 # Interactive exploration of results
│   ├── theta_frontier_figure.py     # BAEE θ sweep + Pareto frontier
│   ├── efa_suffix_ablation.py       # EFA suffix-ablation runner
│   ├── reviewer_experiments.py      # Supplementary reviewer experiments
│   │
│   ├── run_experiment.py            # Main runner (MATH / GPQA / AIME)
│   ├── run_baselines.py             # Prefix-free baselines runner
│   ├── run_humaneval.py             # HumanEval (code) experiment
│   ├── run_latency_benchmark.py     # Full-CoT vs BAEE latency
│   ├── run_suffix_ablation.py       # EFA suffix ablation (MATH)
│   ├── run_suffix_ablation_gpqa.py  # EFA suffix ablation (GPQA)
│   ├── run_prefix_perturbation.py       # Prefix perturbation (MATH, f=0.10)
│   ├── run_prefix_perturbation_f50.py   # Prefix perturbation (MATH, f=0.50)
│   ├── run_prefix_perturbation_gpqa.py  # Prefix perturbation (GPQA)
│   │
│   ├── generate_figures.py          # Main MATH-500 figures
│   ├── generate_figures_aime.py     # AIME-2024 figures
│   ├── generate_figures_gpqa.py     # GPQA figures
│   ├── generate_humaneval_figures.py
│   ├── generate_fig2_main.py        # Fig 2 — 1×4 main panel
│   ├── generate_hero.py             # Hero figure (case study + gap)
│   ├── generate_entropy.py          # Entropy dynamics
│   ├── generate_entropy_ratio.py    # Pre / post-commit entropy ratio
│   ├── generate_gap.py              # Detection–extraction gap (wrapfigure)
│   ├── generate_commitment_map.py
│   ├── generate_theater_map.py
│   ├── regenerate_figures.py        # Batch re-render main figures
│   └── regenerate_appendix_figures.py
│
├── results/                         # All experiment outputs
│   ├── analysis.json                # Aggregated stats
│   ├── qwen3_32b_thinking_full500/  # MATH-500 × 5 model configs
│   ├── qwen3_32b_no_thinking_full500/
│   ├── qwen3_8b_thinking_full500/
│   ├── qwen3_8b_no_thinking_full500/
│   ├── gpt_oss_120b_full500/
│   ├── gpqa_32b_think/              # GPQA-Diamond × 5
│   ├── gpqa_32b_nothink/
│   ├── gpqa_8b_think/
│   ├── gpqa_8b_nothink/
│   ├── gpqa_gpt_oss_120b/
│   ├── humaneval_32b_think/         # HumanEval × 4
│   ├── humaneval_32b_nothink/
│   ├── humaneval_8b_think/
│   ├── humaneval_8b_nothink/
│   ├── aime24_32b_think/             # AIME-2024 × 5
│   ├── aime24_32b_nothink/
│   ├── aime24_8b_think/
│   ├── aime24_8b_nothink/
│   ├── aime24_gpt_oss/
│   ├── baselines_gpqa_8b_nothink/    # SC-8 baselines
│   ├── baselines_gpqa_gpt_oss/
│   └── belief_runs/
│
├── data/                            # Raw benchmark data (GPQA, …)
└── vendor/tinker_cookbook/          # Frozen subset of upstream cookbook
```

> **How to run.** All scripts are invoked from inside `src/`:
> ```bash
> cd src
> python regenerate_figures.py            # re-render the main figures
> python run_experiment.py --preset 32b-think
> ```

> **Note.** This repository ships *code and experiment results only*.
> The LaTeX paper sources (NIPS + arXiv variants) and the generated figure
> bundle are kept out of the repo via `.gitignore`. Running any of the
> `generate_*.py` / `regenerate_*.py` scripts recreates `figures/` on demand.

---

## Running experiments

All experiment types are controlled by named **presets**. Use `--list` to see available options.

### Main experiments (MATH-500 / GPQA-Diamond / AIME-2024)

```bash
python run_experiment.py --list
python run_experiment.py --preset 32b-think
python run_experiment.py --preset gpqa-gpt-oss
python run_experiment.py --preset aime-32b-think

# Shard a run (e.g. problems 450–499 of a 500-problem job)
python run_experiment.py --preset 32b-think --offset 450
```

Available presets:

| Preset | Benchmark | Model | n |
|---|---|---|---|
| `32b-think` | MATH-500 | Qwen3-32B (think) | 500 |
| `32b-nothink` | MATH-500 | Qwen3-32B (no-think) | 500 |
| `8b-think` | MATH-500 | Qwen3-8B (think) | 500 |
| `8b-nothink` | MATH-500 | Qwen3-8B (no-think) | 500 |
| `gpt-oss` | MATH-500 | GPT-OSS-120B | 500 |
| `gpqa-32b-think` | GPQA-Diamond | Qwen3-32B (think) | 198 |
| `gpqa-32b-nothink` | GPQA-Diamond | Qwen3-32B (no-think) | 198 |
| `gpqa-8b-think` | GPQA-Diamond | Qwen3-8B (think) | 198 |
| `gpqa-8b-nothink` | GPQA-Diamond | Qwen3-8B (no-think) | 198 |
| `gpqa-gpt-oss` | GPQA-Diamond | GPT-OSS-120B | 198 |
| `gpqa-pilot` | GPQA-Diamond | Qwen3-32B (think) | 5 |
| `aime-32b-think` | AIME-2024 | Qwen3-32B (think) | 30 |
| `aime-32b-nothink` | AIME-2024 | Qwen3-32B (no-think) | 30 |
| `aime-8b-think` | AIME-2024 | Qwen3-8B (think) | 30 |
| `aime-8b-nothink` | AIME-2024 | Qwen3-8B (no-think) | 30 |
| `aime-gpt-oss` | AIME-2024 | GPT-OSS-120B | 30 |

### Baseline experiments

```bash
python run_baselines.py --list
python run_baselines.py --preset 32b-think
python run_baselines.py --preset gpqa-gpt-oss
```

### EFA suffix ablation

```bash
python run_suffix_ablation.py --list
python run_suffix_ablation.py --preset 32b-think
```

### Other experiments

```bash
python run_prefix_perturbation.py       # f=0.10 perturbation, MATH-500
python run_prefix_perturbation_f50.py   # f=0.50 perturbation, MATH-500
python run_prefix_perturbation_gpqa.py  # GPQA perturbation

python run_suffix_ablation_gpqa.py      # Suffix ablation on GPQA-Diamond
python run_latency_benchmark.py         # Full CoT vs BAEE latency
python run_humaneval.py                 # HumanEval code generation
```

---

## Generating figures

After running experiments, copy (or symlink) the result directories from `REASONING_THEATER_RESULTS` into the paper directory, then run the figure scripts.

```bash
# Main paper figures
python generate_figures.py          # EFA, theater map, stacked bar
python generate_fig2_main.py        # Fig 2: four-panel main figure
python generate_hero.py             # Hero figure (case study + gap)
python generate_entropy.py          # Entropy dynamics

# Benchmark-specific figures
python generate_figures_gpqa.py
python generate_figures_aime.py
python generate_humaneval_figures.py

# Supplementary / wrapfigures
python generate_gap.py
python generate_entropy_ratio.py
python generate_commitment_map.py
python generate_theater_map.py

# Batch regenerate
python regenerate_figures.py
python regenerate_appendix_figures.py
```

All figures are saved to `figures/` as both `.pdf` and `.png` at 300 dpi.

---

## Measurement protocols

The core experiment (`experiment.py`) runs four protocols at prefix checkpoints (default: 10%, 20%, …, 90% of CoT length):

| Protocol | Description |
|---|---|
| **EFA** (Early Forced Answering) | Append answer-forcing suffix at prefix; greedy-decode the answer |
| **ATLT** (Answer Token Logprob Trajectory) | Compute logprob of the correct answer token at each prefix |
| **ED** (Entropy Dynamics) | Top-k token entropy over the full CoT |
| **PSC** (Prefix Self-Consistency) | Sample N continuations from each prefix; measure agreement rate |

---

## Key concepts

- **Commitment point** — the earliest prefix fraction at which PSC ≥ 0.75 (the model reliably continues to the correct answer)
- **Post-commitment fraction** — share of CoT tokens generated *after* the commitment point (52–88% across our settings)
- **Detection–extraction gap** — PSC is high (model "knows") but EFA fails (forced extraction returns the wrong answer). Lower-bounded by the TV-distance between free and forced continuation distributions.
- **BAEE** (Black-box Adaptive Early Exit) — probes PSC at each checkpoint, exits on agreement, returns the majority answer from the continuations (no white-box access needed).

---

## Citation

```bibtex
@article{know2say_2026,
  title  = {The Detection--Extraction Gap: Models Know the Answer Before They Can Say It},
  year   = {2026},
  note   = {NeurIPS submission},
}
```
