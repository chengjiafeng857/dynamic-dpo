## Dynamic DPO (Dynamic Beta for DPO)

Train and evaluate **Direct Preference Optimization (DPO)** with a **dynamically adjusted temperature** `beta` driven by a simple risk test on the per-batch DPO margin distribution.

This repo contains:
- A **baseline DPO** trainer with fixed `beta` (`dpo_training.py`)
- A **dynamic-beta DPO** trainer (`training.py`)
- A **multi-GPU FSDP** variant for dynamic-beta training (`training-fsdp.py`)
- An end-to-end **evaluation pipeline** that generates responses, builds A/B pairs, runs an LLM judge, and summarizes win-rates (`test/run_test_pipeline.py`)

---

## Core Idea (Dynamic Beta)

Per example, compute the standard DPO “margin”:

- Let `lpπ(y|x)` be the (summed) token log-prob of a completion `y` given prompt `x` under the policy model `π`.
- Let `lpref(y|x)` be the same under a frozen reference model.

This code uses:

- `chosen_log_prob = lpπ(y⁺|x) - lpref(y⁺|x)`
- `rejected_log_prob = lpπ(y⁻|x) - lpref(y⁻|x)`
- DPO loss (per-sample): `L = -log σ(beta * (chosen_log_prob - rejected_log_prob))`

It also tracks the margin:

`M = (lpπ(y⁺|x) - lpπ(y⁻|x)) - (lpref(y⁺|x) - lpref(y⁻|x))`

During training (`training.py` / `training-fsdp.py`):
1. **Warmup**: collect margins for `warmup_steps`, set an initial threshold `tau₀` to the `(1 - delta)` quantile of warmup margins.
2. **EMA threshold**: maintain `tau` as an EMA of the per-batch `(1 - delta)` quantile (so tail probability above `tau` is roughly `delta`).
3. **Risk test**: compute `p̂ = P(M ≥ tau)` and compare against a Hoeffding-style bound `delta' = delta + ε(n)`.
4. **Update beta**: adjust `beta` multiplicatively and clip to `[beta_min, beta_max]`.

Implementation lives in:
- `dpo_loss.py` (`dpo_loss`, `margin_compute`, `risk_test`, `update_beta`)
- `mean_and_var.py` (`WarmupQuantileAccumulator`, `EMAUpdate`)

---

## Project Layout

- `config_dpo.yaml`: main configuration (models, dataset, training, eval, dynamic-beta knobs)
- `training.py`: dynamic-beta DPO training (single GPU / single process)
- `dpo_training.py`: baseline fixed-beta DPO training
- `training-fsdp.py`: dynamic-beta DPO training with PyTorch FSDP (`torchrun`)
- `dataset_process_hh.py`: loads HH-RLHF, converts to `{prompt, chosen, rejected}`, builds dataloaders
- `batch_log_prob.py`: batched log-prob computation for chosen/rejected under policy & ref
- `dpo_loss.py`: DPO loss + margin/risk/beta utilities + margin logging
- `test/`: unified evaluation pipeline + docs
  - `test/run_test_pipeline.py`: generate → pair → judge → summarize
  - `test/test.sh`: convenience wrapper around the unified pipeline
  - `test/README.md`: detailed evaluation docs
- `archive/`: older versions of the evaluation scripts

---

## Setup

### Python & deps

`pyproject.toml` targets Python `>= 3.13` and includes `torch`, `transformers`, `datasets`, `wandb`, and (optionally for judging) `openai`.

If you use `uv`:
```bash
uv sync
```

Or with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Credentials (optional)

- **Weights & Biases**: set `WANDB_API_KEY` if you want logging.
- **OpenAI judging** (evaluation step 3): set `OPENAI_API_KEY`.
- **Hugging Face gated models** (e.g. Llama): ensure you can download the model (login/token + license acceptance as needed).

---

## Configuration (`config_dpo.yaml`)

Top-level keys you’ll likely touch:

- `policy_name`, `ref_name`: Hugging Face model IDs (or local paths)
- `precision`: `bf16` is supported in training scripts

### Dataset (`dataset`)

Uses `datasets.load_dataset()`; default is HH-RLHF:

- `dataset_name`: e.g. `Anthropic/hh-rlhf`
- `subset`: HF split selector, e.g. `train[:20%]`
- `val_ratio`, `seed`, `max_len`

### Training (`dpo_training`)

Shared knobs:
- `epochs`, `batch_size`, `learning_rate`
- `warmup_steps`: for dynamic-beta warmup (and also used in baseline to delay optimizer stepping)
- `max_grad_norm`, `log_steps`
- `save_dpo_dir`: output dir for baseline DPO model
- `save_dir`: output dir for dynamic-beta model

### Dynamic beta (`risk_test`, `beta_update`)

- `risk_test.delta`: target tail probability
- `risk_test.eplison_0`: confidence parameter used in the bound (spelled `eplison_0` in code/config)
- `risk_test.lambda`: EMA momentum for `tau`

- `beta_update.beta_0`: initial beta
- `beta_update.alpha`, `beta_update.gamma`: update aggressiveness
- `beta_update.beta_min`, `beta_update.beta_max`: clipping bounds

### Margin logging (`margin_log`)

- `margin_log.log_dir`: used by `training.py` and `training-fsdp.py`
- `margin_log.dpo_log_dir`: used by `dpo_training.py`

Margin logs include per-step `.npy` dumps and a JSONL with summary stats.

### Evaluation (`test`)

Used by `test/run_test_pipeline.py`:
- generation params: `test_num`, `max_new_tokens`, `temperature`, `top_p`, `batch_size`
- output paths: `*_out_dir`, `summary_output`
- judge: `judge_model` (also overridable by CLI)

---

## Training

### 1) Baseline DPO (fixed beta)

Runs `dpo_training.py` and saves to `dpo_training.save_dpo_dir`:

```bash
uv run python dpo_training.py --config config_dpo.yaml
```

### 2) Dynamic-beta DPO (single process)

Runs `training.py` and saves to `dpo_training.save_dir`:

```bash
uv run python training.py --config config_dpo.yaml
```

This writes:
- margin logs under `margin_log.log_dir/`
- risk/beta trace to `risk_test_and_beta_log.jsonl` (repo root)

### 3) Dynamic-beta DPO with FSDP (multi-GPU)

```bash
torchrun --nproc_per_node 4 training-fsdp.py --config config_dpo.yaml
```

Notes:
- FSDP training uses NCCL and expects CUDA GPUs.
- `dataset_process_hh.build_train_val_distributed()` reads `dataset.num_works` (num dataloader workers). Add it to your YAML when using FSDP, e.g.:
  ```yaml
  dataset:
    num_works: 2
  ```

---

## Evaluation (A/B judging pipeline)

The recommended entrypoint is the unified pipeline:

```bash
uv run python test/run_test_pipeline.py --config config_dpo.yaml
```

Or via the convenience wrapper:

```bash
./test/test.sh
```

Pipeline steps:
1. Generate responses for `{ref, dpo, dynamic_dpo}`
2. Build randomized A/B pairs
3. Judge pairs with an OpenAI model (if `openai` installed + `OPENAI_API_KEY` set)
4. Summarize win/tie/invalid rates

Outputs are written under `eval_outputs/` (gitignored by default).

More details: `test/README.md`.

---

## Common Pitfalls

- **OOM during generation/eval**: reduce `test.batch_size` and/or `test.max_new_tokens`.
- **Tokenizer padding**: scripts set `pad_token = eos_token` if missing (decoder-only models).
- **Gated model downloads**: `meta-llama/*` may require HF auth + license acceptance.
- **Judging requires network access**: step 3 calls OpenAI APIs; run with `--skip-judge` to generate/pair only.
