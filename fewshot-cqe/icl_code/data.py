"""
Data loading, embedding computation (with caching), and instance management.

Each row in the JSONL dataset has:
  - task_id, prompt, solution
  - code_1 ... code_10  (generated code candidates)
  - results            (list of 0/1 per code candidate)

We convert each (task, code_i) pair into a ProblemInstance carrying
pre-computed encoder embeddings for downstream similarity lookup.

Embedding caching:
  Embeddings are saved as .npz files keyed by (encoder_name, dataset_name).
  This means you encode MBPP once with a given encoder and reuse it across
  all test datasets without re-computing.
"""

import json
import os
import hashlib

import numpy as np
import pandas as pd
import torch


# ── ProblemInstance ──────────────────────────────────────────────────────

class ProblemInstance:
    """A single (problem, code_candidate) pair with metadata & embeddings."""

    __slots__ = (
        "task_id", "prompt", "code_key", "code",
        "result", "prompt_embedding", "code_embedding",
    )

    def __init__(self, task_id, prompt, code_key, code, result,
                 prompt_embedding, code_embedding):
        self.task_id = task_id
        self.prompt = prompt
        self.code_key = code_key
        self.code = code
        self.result = result
        self.prompt_embedding = prompt_embedding
        self.code_embedding = code_embedding


# ── Data I/O ────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """Read a JSONL dataset into a DataFrame."""
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return pd.DataFrame(data)


def split_train_dev(training_data: pd.DataFrame, split_ratio: float = 0.1):
    """
    Deterministic 90/10 task-level split.

    Returns (train_df, dev_df).  The dev split is not used in test mode
    but we still split so the training pool matches paper methodology.
    """
    unique_task_ids = training_data["task_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_task_ids)
    split_idx = int(len(unique_task_ids) * (1 - split_ratio))
    train_ids = set(unique_task_ids[:split_idx])
    train_df = training_data[training_data["task_id"].isin(train_ids)]
    dev_df = training_data[~training_data["task_id"].isin(train_ids)]
    return train_df, dev_df


# ── Embeddings ──────────────────────────────────────────────────────────

def get_cls_embedding(text: str, tokenizer, model) -> np.ndarray:
    """
    Return a single embedding vector for *text* using the appropriate
    pooling strategy for the model family:
      - BERT/RoBERTa-based encoders (CodeBERT, GraphCodeBERT, UniXcoder):
            CLS token at index 0  →  768-dim
      - Decoder-based embedding models (Qwen3-Embedding):
            Last non-padding token (via attention_mask)  →  1024-dim
    """
    # Ensure pad token is set (decoder models like Qwen usually lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use model-specific max length: Qwen keeps 4096, encoder models use their config max.
    model_type = getattr(model.config, "model_type", "")
    is_qwen = "qwen" in model_type.lower()
    config_max = getattr(model.config, "max_position_embeddings", 512)
    max_len = 4096 if is_qwen else min(512, int(config_max))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)

    # Determine the model device robustly. Some HF model wrappers don't expose
    # a single `model.device` (they may use `hf_device_map`). Prefer the
    # device of the first parameter when available; fall back to CPU.
    model_device = None
    try:
        first_param = next(model.parameters())
        model_device = first_param.device
    except Exception:
        model_device = getattr(model, "device", None)

    if model_device is not None:
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # Safety: truncate if tokenized sequence still exceeds model's positional limit.
    max_pos_embed = getattr(model.config, "max_position_embeddings", None)
    if max_pos_embed is not None:
        seq_len = inputs["input_ids"].shape[1]
        if seq_len > int(max_pos_embed):
            print(
                f"  [WARN] Truncating input from {seq_len} to "
                f"max_position_embeddings={max_pos_embed}"
            )
            for k in list(inputs.keys()):
                inputs[k] = inputs[k][:, :int(max_pos_embed)]

    with torch.no_grad():
        outputs = model(**inputs)

    # Choose pooling strategy
    is_decoder = getattr(model.config, "is_decoder", False)
    is_qwen    = "qwen" in getattr(model.config, "model_type", "").lower()

    if is_decoder or is_qwen:
        # Decoder / Qwen3-Embedding: use attention_mask to find the TRUE last
        # non-padding token.  This is correct for both left- and right-padded
        # inputs (though Qwen tokenizer should already use left-padding).
        if not hasattr(model, "_pooling_printed"):
            print(
                f"  [INFO] {getattr(model.config, 'model_type', 'model')}: "
                "last-token pooling (attention_mask-aware)."
            )
            model._pooling_printed = True
        attention_mask = inputs["attention_mask"]          # (1, seq_len)
        # sequence_lengths: index of the last real (non-pad) token
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size = outputs.last_hidden_state.shape[0]
        token_emb = outputs.last_hidden_state[
            torch.arange(batch_size, device=outputs.last_hidden_state.device),
            sequence_lengths,
        ]  # (batch, hidden)
        return token_emb.squeeze(0).float().cpu().numpy()
    else:
        # BERT: CLS token is at index 0
        if not hasattr(model, "_pooling_printed"):
            print(
                f"  [INFO] {getattr(model.config, 'model_type', 'model')}: "
                "CLS-token pooling (index 0)."
            )
            model._pooling_printed = True
        return outputs.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()


#Embedding cache 
def _cache_filename(encoder_short_name: str, dataset_path: str) -> str:
    """
    Build a descriptive cache filename from encoder name and dataset.

    Format: {encoder}_{dataset_basename}.npz
    Example: codebert-base_mbpp-codestral-22b.npz
             graphcodebert-base_heval-codellama-7b.npz
    """
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    return f"{encoder_short_name}_{dataset_name}.npz"


def _text_key(text: str) -> str:
    """Stable hash for use as an npz dict key (npz keys must be valid identifiers)."""
    return "h_" + hashlib.md5(text.encode("utf-8")).hexdigest()


def save_embeddings_cache(prompt_embs: dict, code_embs: dict,
                          cache_path: str) -> None:
    """Save prompt and code embedding dicts to a single .npz file."""
    arrays = {}
    # Store prompt embeddings with prefix 'p_'
    prompt_keys = []
    for text, emb in prompt_embs.items():
        key = _text_key(text)
        arrays[f"p_{key}"] = emb
        prompt_keys.append(f"{key}|||{text}")

    # Store code embeddings with prefix 'c_'
    code_keys = []
    for text, emb in code_embs.items():
        key = _text_key(text)
        arrays[f"c_{key}"] = emb
        code_keys.append(f"{key}|||{text}")

    # Store the text->key mapping so we can reconstruct
    arrays["__prompt_keys__"] = np.array(prompt_keys, dtype=object)
    arrays["__code_keys__"] = np.array(code_keys, dtype=object)

    cache_dir_path = os.path.dirname(cache_path)
    if cache_dir_path:
        os.makedirs(cache_dir_path, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    print(f"  [CACHE] Saved embeddings -> {cache_path}")
    print(f"          {len(prompt_embs)} prompts, {len(code_embs)} codes")


def load_embeddings_cache(cache_path: str):
    """
    Load embeddings from cache.  Returns (prompt_embs, code_embs) dicts
    keyed by original text.
    """
    data = np.load(cache_path, allow_pickle=True)

    prompt_embs = {}
    for entry in data["__prompt_keys__"]:
        key, text = str(entry).split("|||", 1)
        prompt_embs[text] = data[f"p_{key}"]

    code_embs = {}
    for entry in data["__code_keys__"]:
        key, text = str(entry).split("|||", 1)
        code_embs[text] = data[f"c_{key}"]

    print(f"  [CACHE] Loaded embeddings <- {cache_path}")
    print(f"          {len(prompt_embs)} prompts, {len(code_embs)} codes")
    return prompt_embs, code_embs


def _get_expected_embedding_dim(model) -> int:
    """Return the hidden size of *model* (used for stale-cache dimension checks)."""
    for attr in ("hidden_size", "d_model", "embed_dim"):
        val = getattr(model.config, attr, None)
        if val is not None:
            return int(val)
    # Fallback: infer from first forward pass on a dummy token
    return -1


def compute_embeddings_cached(df: pd.DataFrame, tokenizer, model,
                              encoder_short_name: str, dataset_path: str,
                              cache_dir: str):
    """
    Compute embeddings with caching.

    If a cache file exists for this (encoder, dataset) pair, load it.
    Otherwise compute fresh embeddings and save the cache.

    If the cached embeddings have a different dimension than the current
    encoder (e.g. switching between Qwen3-Embedding 1024-dim and
    CodeBERT 768-dim), the stale cache is ignored and recomputed.

    Missing texts (e.g. test prompts not in a train cache) are computed
    on the fly and merged.

    Returns (prompt_embeddings, code_embeddings) dicts keyed by text.
    """
    cache_file = _cache_filename(encoder_short_name, dataset_path)
    cache_path = os.path.join(cache_dir, cache_file)

    prompt_embs = {}
    code_embs = {}

    # Try loading existing cache
    if os.path.isfile(cache_path):
        loaded_prompt, loaded_code = load_embeddings_cache(cache_path)

        # ── Stale-cache dimension guard ─────────────────────────────────────
        # When the user switches encoder (e.g. Qwen 1024-dim <-> CodeBERT 768-dim)
        # the cached .npz is for a DIFFERENT encoder even though the filename
        # matches.  Detect this by comparing the first cached vector's dimension
        # against the current model's hidden_size and recompute if they differ.
        expected_dim = _get_expected_embedding_dim(model)
        stale = False
        if expected_dim > 0 and loaded_prompt:
            sample_emb = next(iter(loaded_prompt.values()))
            if sample_emb.shape[-1] != expected_dim:
                print(
                    f"  [WARN] Stale cache detected: cached dim={sample_emb.shape[-1]} "
                    f"but current encoder dim={expected_dim}. "
                    f"Discarding cache and recomputing."
                )
                stale = True

        if not stale:
            prompt_embs = loaded_prompt
            code_embs = loaded_code
        # else: leave prompt_embs / code_embs as empty dicts → full recompute

    # Compute any missing embeddings
    computed_new = 0

    for _, row in df.iterrows():
        prompt = row["prompt"]
        if prompt not in prompt_embs:
            prompt_embs[prompt] = get_cls_embedding(prompt, tokenizer, model)
            computed_new += 1

        for i in range(1, 11):
            code_key = f"code_{i}"
            if code_key in row and not pd.isna(row[code_key]):
                code = row[code_key]
                if code not in code_embs:
                    code_embs[code] = get_cls_embedding(code, tokenizer, model)
                    computed_new += 1

    if computed_new > 0:
        print(f"  Computed {computed_new} new embeddings")
        save_embeddings_cache(prompt_embs, code_embs, cache_path)
    else:
        print(f"  All embeddings loaded from cache (0 new computations)")

    return prompt_embs, code_embs


def compute_embeddings(df: pd.DataFrame, tokenizer, model):
    """
    Compute embeddings without caching (legacy interface).

    Returns (prompt_embeddings, code_embeddings) dicts keyed by text.
    """
    prompt_embeddings = {}
    code_embeddings = {}

    for _, row in df.iterrows():
        prompt = row["prompt"]
        if prompt not in prompt_embeddings:
            prompt_embeddings[prompt] = get_cls_embedding(prompt, tokenizer, model)

        for i in range(1, 11):
            code_key = f"code_{i}"
            if code_key in row and not pd.isna(row[code_key]):
                code = row[code_key]
                if code not in code_embeddings:
                    code_embeddings[code] = get_cls_embedding(code, tokenizer, model)

    return prompt_embeddings, code_embeddings


def transform_data(df: pd.DataFrame, prompt_embeddings: dict,
                   code_embeddings: dict) -> list:
    """Expand DataFrame rows into a flat list of ProblemInstances."""
    instances = []
    for _, row in df.iterrows():
        task_id = row["task_id"]
        prompt = row["prompt"]
        for i in range(1, 11):
            code_key = f"code_{i}"
            if code_key in row and not pd.isna(row[code_key]):
                code = row[code_key]
                instances.append(ProblemInstance(
                    task_id, prompt, code_key, code,
                    row["results"][i - 1],
                    prompt_embeddings[prompt],
                    code_embeddings[code],
                ))
    return instances


# Persistence helpers

def _convert_ndarray(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def save_instances_by_result(instances: list, output_dir: str):
    """Split instances into correct/incorrect and save as JSONL."""
    os.makedirs(output_dir, exist_ok=True)
    correct = [i for i in instances if i.result == 1]
    incorrect = [i for i in instances if i.result == 0]
    _save_instance_list(correct, os.path.join(output_dir, "correct_instances.jsonl"))
    _save_instance_list(incorrect, os.path.join(output_dir, "incorrect_instances.jsonl"))


def _save_instance_list(instances: list, path: str):
    with open(path, "w") as f:
        for inst in instances:
            obj = {
                "task_id": inst.task_id,
                "prompt": inst.prompt,
                "code_key": inst.code_key,
                "code": inst.code,
                "result": inst.result,
                "prompt_embedding": _convert_ndarray(inst.prompt_embedding),
                "code_embedding": _convert_ndarray(inst.code_embedding),
            }
            json.dump(obj, f, default=_convert_ndarray)
            f.write("\n")
