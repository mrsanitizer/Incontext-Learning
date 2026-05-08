"""
Configuration management for ICL test runs.

Handles loading JSON configs, CLI override merging, path normalisation,
and validation -- all in one place so the rest of the pipeline never
touches raw dicts.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ICLConfig:
    """Typed, validated configuration for a single ICL test run."""

    # ── Data paths ──────────────────────────────────────────────────────
    input_train_file: str = ""        # MBPP pool used for few-shot example selection
    input_test_file: str = ""         # HumanEval / MBJP / etc. test file
    output_dir: str = "output/icl"

    # ── ICL hyper-parameters ────────────────────────────────────────────
    k_values: List[int] = field(default_factory=lambda: [1, 2])
    p_values: List[float] = field(default_factory=lambda: [1.0]) # Weight for prompt vs code similarity (1=prompt only, 0=code only)
    balanced: bool = True             # Balanced example selection (equal correct / incorrect)
    filter: bool = True               # Filter to unique problem statements in examples

    # ── Estimator LLM settings ──────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    torch_dtype: str = "float16"
    use_4bit: bool = False
    use_cpu_offload: bool = True
    max_gpu_memory_gb: int = 7        # Tuned for 4070 Mobile 8GB VRAM
    max_prompt_tokens: Optional[int] = 4096  # Set to null in JSON to disable truncation (unlimited VRAM)

    # ── Encoder settings ────────────────────────────────────────────────
    encoder_name: str = "microsoft/codebert-base"   # Encoder model for embeddings
    embedding_cache_dir: str = "embeddings_cache"   # Dir for cached .npz files

    # ── Logging ─────────────────────────────────────────────────────────
    log_prompts: bool = True          # Save full prompt text for each instance

    def to_dict(self):
        return asdict(self)


# ── Loading helpers ─────────────────────────────────────────────────────

def _resolve_path(base_dir: str, maybe_relative: str) -> str:
    """Resolve a path that may be relative to *base_dir*."""
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.normpath(os.path.join(base_dir, maybe_relative))


def load_config(config_path: str, cli_overrides: Optional[dict] = None) -> ICLConfig:
    """
    Build an ICLConfig from a JSON file, optionally patched with CLI overrides.

    Parameters
    ----------
    config_path : str
        Path to a JSON config file.
    cli_overrides : dict, optional
        Key-value pairs that take precedence over the JSON file.
        Keys must match ICLConfig field names.

    Returns
    -------
    ICLConfig
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path, "r") as f:
        raw = json.load(f)

    # Merge CLI overrides (they win over the file)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                raw[key] = value

    # Resolve relative paths against config file location
    for path_key in ("input_train_file", "input_test_file", "output_dir",
                     "embedding_cache_dir"):
        if path_key in raw and raw[path_key]:
            raw[path_key] = _resolve_path(config_dir, raw[path_key])

    cfg = ICLConfig(**{k: v for k, v in raw.items() if k in ICLConfig.__dataclass_fields__})
    validate_config(cfg)
    return cfg


def validate_config(cfg: ICLConfig) -> None:
    """Raise early on missing / invalid settings."""
    if not cfg.input_train_file:
        raise ValueError("input_train_file is required")
    if not cfg.input_test_file:
        raise ValueError("input_test_file is required")
    if not isinstance(cfg.k_values, list) or not cfg.k_values:
        raise ValueError("k_values must be a non-empty list of ints")
    if not isinstance(cfg.p_values, list) or not cfg.p_values:
        raise ValueError("p_values must be a non-empty list of floats")
    if not os.path.isfile(cfg.input_train_file):
        raise FileNotFoundError(f"Training file not found: {cfg.input_train_file}")
    if not os.path.isfile(cfg.input_test_file):
        raise FileNotFoundError(f"Test file not found: {cfg.input_test_file}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.embedding_cache_dir, exist_ok=True)
