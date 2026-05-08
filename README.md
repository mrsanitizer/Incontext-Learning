# Incontext Learning (ICL)

This repository implements a modular **In-Context Learning (ICL)** pipeline for code correctness estimation. It uses a configurable **encoder** to find similar training examples by embedding similarity, then feeds those examples as few-shot context into an **estimator LLM** that predicts whether a given code snippet is functionally correct (`yes` / `no`).

## Features

- **Few-Shot ICL & Zero-Shot:** Run experiments with `k=0` (zero-shot) through `k=N` (N examples per class). All values of `k` can be evaluated in a single run.
- **Configurable Encoders:** Supports any HuggingFace `AutoModel`-compatible encoder for computing prompt/code embeddings.
- **Embedding Cache:** Embeddings are cached as `.npz` files per `(encoder, dataset)` pair — encode once, reuse across all experiments.
- **Memory Efficient:** Sequential model loading (encoder freed before LLM loads), CPU offloading, and optional 4-bit quantization for consumer hardware.

---

## Setup & Installation

1. Create a Python virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** Exact versions in `requirements.txt` are tested and verified. For a different CUDA version, install PyTorch manually from [pytorch.org](https://pytorch.org) first, then install the rest.

---

## Running the Pipeline

The main entry point is `fewshot-cqe/run_icl_test.py`, driven by a JSON config file.

```bash
cd fewshot-cqe
python run_icl_test.py --config configs/test_default.json
```

---

## Configuration File Reference

Edit `fewshot-cqe/configs/test_default.json` before running. Every key is explained below.

```json
{
  "input_train_file": "../../data/mbpp-codestral-22b.jsonl",
  "input_test_file":  "../../data/mbjp-codellamacut-7b.jsonl",
  "output_dir":       "../output/icl/my_experiment_name/",

  "k_values":  [0, 1, 2],
  "p_values":  [0.0, 0.5, 1.0],
  "balanced":  true,
  "filter":    true,

  "model_name":        "Qwen/Qwen2.5-Coder-3B-Instruct",
  "torch_dtype":       "float16",
  "use_4bit":          false,
  "use_cpu_offload":   true,
  "max_gpu_memory_gb": 7,
  "max_prompt_tokens": 4096,

  "encoder_name":       "microsoft/codebert-base",
  "embedding_cache_dir":"../embeddings_cache",
  "log_prompts":        true
}
```

### Data Paths

| Key | Description |
|---|---|
| `input_train_file` | Path to the **training pool** JSONL — the source from which few-shot examples are retrieved. The paper used `mbpp-codestral-22b.jsonl`. Paths are resolved relative to the config file location. |
| `input_test_file` | Path to the **test dataset** JSONL to evaluate on (e.g. HumanEval, MBJP variants). |
| `output_dir` | Directory where all output files are saved. Created automatically if it does not exist. |

### ICL Hyper-parameters

| Key | Type | Description |
|---|---|---|
| `k_values` | `list[int]` | List of `k` values to sweep over in a single run. `k=0` is zero-shot (no examples). `k=1` means 1 correct + 1 incorrect example. Each value in the list produces a separate set of output files. Example: `[0, 1, 2, 3, 4, 5]` runs zero-shot then k=1 then k=2 then k=3 and so on till 5. |
| `p_values` | `list[float]` | Similarity weight between prompt and code embeddings. `p=1.0` = match purely on **problem description similarity**. `p=0.0` = match purely on **code similarity**. `p=0.5` = equal mix. Only applies when `k > 0`. |
| `balanced` | `bool` | If `true`, selects exactly `k` correct **and** `k` incorrect examples from the training pool for each test instance. If `false`, picks the top `2k` most similar examples regardless of correctness label. |
| `filter` | `bool` | If `true`, enforces that no two selected examples share the same problem description (prevents the LLM from seeing the same problem twice in the context). |

### Estimator LLM Settings

| Key | Type | Description |
|---|---|---|
| `model_name` | `string` | HuggingFace model ID for the **estimator LLM**. This is the causal language model that reads the prompt and predicts `yes`/`no`. See supported models below. |
| `torch_dtype` | `string` | Floating point precision. Options: `"float16"` (recommended for GPU), `"bfloat16"`, `"float32"`. |
| `use_4bit` | `bool` | If `true`, loads the LLM in 4-bit quantization via `bitsandbytes`. Cuts VRAM usage. Requires `bitsandbytes` installed. |
| `use_cpu_offload` | `bool` | If `true`, uses `device_map="auto"` with a memory cap. Layers that don't fit on GPU are offloaded to CPU RAM. Slower but allows running larger models on limited VRAM. |
| `max_gpu_memory_gb` | `int` | GPU memory cap in GB used when `use_cpu_offload` is `true`. Set to slightly less than your total VRAM (e.g., `7` for an 8GB card) to leave headroom. |
| `max_prompt_tokens` | `int` or `null` | Maximum token length of the prompt sent to the LLM. Prompts exceeding this are **truncated** and flagged as errors. **Set to `null` to disable truncation entirely** (recommended for unlimited VRAM — see section below). |

NOTE: Setting `max_gpu_memory_gb` is only for loading the model layers into memory. It does not include the active calculations of attention layers during inference, so you should set it decently lower than your total VRAM to avoid OOM errors depending on the model you are importing. Similarly, `use_cpu_offload` will offload the model weights and not the inference memory (which is temporary). 

#### Supported Estimator LLMs (`model_name`)

Any HuggingFace causal language model that supports next-token prediction works. Tested models:

| Model | VRAM (float16) | Notes |
|---|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | ~3GB | Fastest, lowest quality |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | ~6GB | Good balance  |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | ~14GB | Higher quality |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | ~28GB | Best quality |
Below are not tested but should work:
| `meta-llama/Llama-3.2-3B-Instruct` | ~6GB | Alternative family |
| `deepseek-ai/deepseek-coder-1.3b-instruct` | ~3GB | Compact alternative |

> Any instruction-tuned causal LM from HuggingFace should work. The model must output `yes` or `no` as its first token given the prompt.

### Encoder Settings

| Key | Type | Description |
|---|---|---|
| `encoder_name` | `string` | HuggingFace model ID for the **embedding encoder**. Used only when `k > 0` to compute similarity between test instances and training examples. See supported encoders below. |
| `embedding_cache_dir` | `string` | Directory for cached `.npz` embedding files. Embeddings are computed once per `(encoder, dataset)` pair and reused. |

#### Supported Encoders (`encoder_name`)

| Model | Embedding Dim | Architecture | Notes |
|---|---|---|---|
| `microsoft/codebert-base` | 768 | BERT encoder | Default. RoBERTa-based, strong on code. |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | Decoder-based | Larger embedding dim, newer model. |

> **Important:** Switching encoders between runs invalidates old embedding caches. The pipeline auto-detects dimension mismatches and recomputes automatically.

### Logging

| Key | Type | Description |
|---|---|---|
| `log_prompts` | `bool` | If `true`, saves the **full raw text** of every prompt sent to the LLM into `prompts_k_{k}_p_{p}.txt`. Very useful for debugging which examples were selected. Disable to save disk space. |

---

## Token Limit & Truncation

### The Problem

The `max_prompt_tokens` setting controls how many tokens can be in a single prompt. For ICL with large `k` values or datasets with long code snippets, prompts can easily exceed this limit. When this happens:

1. The prompt is **silently truncated** (the LLM only sees the first `max_prompt_tokens` tokens — it loses the end of the prompt including the code to evaluate).
2. The instance is marked as a `TRUNCATED` error in the live log and `error_log.txt`.
3. **After 5 consecutive errors in a single `k`/`p` combination, the entire combination is aborted** with the message:
   ```
   [ABORT] Too many errors (5) for k=1, p=0.5. Skipping the rest of this combination to save time!
   ```

This abort is a safety mechanism — if your token limit is too low for your dataset, it will bail out rather than waste hours producing useless truncated results.

### Solution: Disable the Token Limit

If you have **no VRAM constraints** (e.g., cloud GPU, A100, H100), you can completely remove the token limit by setting `max_prompt_tokens` to `null` in the config:

```json
"max_prompt_tokens": null
```

With `null`, the pipeline passes the **entire prompt** to the LLM without any truncation. No TRUNCATED errors, no aborts. The `tkns` column in the live output will still show you the actual token count for monitoring.

> **Warning:** With `null`, very large prompts (e.g. k=4 with long code) can cause **CUDA Out-Of-Memory** if your GPU does not have sufficient VRAM. The OOM is caught gracefully and also counts toward the 5-error abort.

### Recommended Settings by Hardware

| Setup | Recommended `max_prompt_tokens` |
|---|---|
| 8GB VRAM (laptop GPU) | `4096` |
| 16GB VRAM | `8192` |
| 24GB+ VRAM | `16384` or `null` |
| Cloud GPU (A100 80GB, etc.) | `null` (no limit) |

Note: This highly depends on the estimator model used and the test dataset one is using.

---

## Output & Results

All outputs are saved inside `output_dir/balanced/` (when `balanced: true`).

| File | Description |
|---|---|
| `correct_instances.jsonl` | Training pool instances where `result=1`, with their embeddings. Used for balanced example selection. |
| `incorrect_instances.jsonl` | Training pool instances where `result=0`, with their embeddings. |
| `live_results_k_{k}_p_{p}.jsonl` | **Failsafe backup.** One JSON record per evaluated instance, written immediately. If the pipeline crashes after hours, all processed data is preserved here. |
| `similarity_results_k_{k}_p_{p}.jsonl` | Final predictions: `task_id`, `code_key`, `logits_yes`, `logits_no`, `score`. Used for metric computation. |
| `avg_scores_k_{k}_p_{p}.jsonl` | Per-task average score across all code candidates. Used for Task-Local nDCG. |
| `ndcg_results.json` | Final evaluation metrics: Global nDCG and Local Average nDCG. Updated after every completed `k`/`p` combination. |
| `prompts_k_{k}_p_{p}.txt` | *(if `log_prompts: true`)* Full raw prompt text sent to the LLM for every instance. |
| `error_log.txt` | *(created only on errors)* Details of every truncation or OOM event, including the exact prompt text that caused it. |

### Live Terminal Output

```
============================================================
  ICL (k=1, p=0.5)
============================================================
    [1/500]  task_45                   code_1   | score=0.8500  P(yes)=0.8500  P(no)=0.1500 | gt=PASS pred=yes [OK] tkns=1240
    [2/500]  task_45                   code_2   | score=0.3800  P(yes)=0.3800  P(no)=0.6200 | gt=FAIL pred=no  [OK] tkns=1265
    [3/500]  task_45                   code_3   | score=0.9100  P(yes)=0.9100  P(no)=0.0900 | gt=FAIL pred=yes [XX] tkns=1201
      => [ERROR] TRUNCATED: Prompt was 8020 tokens (Limit: 4096)

==================================================
  ICL Test Results
==================================================
  k_1_p_0.5
    Global  nDCG : 0.841234
    Local   nDCG : 0.810567
--------------------------------------------------
```

- `[OK]` — model prediction matches ground truth
- `[XX]` — model prediction is wrong
- `tkns` — actual token count of the prompt (before truncation if a limit is set)
- Lines prefixed with `=> [ERROR]` are also logged to `error_log.txt`

### Error Log Format

```text
################################################################################
ERROR: TRUNCATED: Prompt was 8020 tokens (Limit: 4096)
k=1, p=1.0 | Task: MBJP/493 | Code: code_1
################################################################################

You are an experienced software engineer...
[Full prompt text follows so you can inspect exactly what was too long]
```

---

## Available Datasets

The `data/` directory contains the following pre-built JSONL datasets:

| File | Description |
|---|---|
| `mbpp-codestral-22b.jsonl` | MBPP problems with code generated by Codestral-22B. **Used as training pool.** |
| `mbjp-codestral-22b.jsonl` | MBJP (multilingual) problems with Codestral-22B code. |
| `mbjp-codellama-7b.jsonl` | MBJP problems with CodeLlama-7B code. |
| `mbjp-codellamacut-7b.jsonl` | Cut/reduced version of the MBJP CodeLlama dataset. |
| `heval-codestral-22b.jsonl` | HumanEval problems with Codestral-22B code. |
| `heval-codellama-7b.jsonl` | HumanEval problems with CodeLlama-7B code. |

Each JSONL file has rows with: `task_id`, `prompt`,`solution`, `code_1`...`code_10`, `results` (list of 0/1) as columns.
