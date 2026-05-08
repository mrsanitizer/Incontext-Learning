"""
Model loading for the ICL pipeline.

Two models are used:
  1. Encoder   - for embedding prompts/code to find similar examples.
                 Default is CodeBERT but any HuggingFace encoder model works.
  2. Estimator LLM - the causal LM that judges functional correctness.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from .config import ICLConfig


# Encoder (embedding model) 

def load_encoder(encoder_name: str = "microsoft/codebert-base"):
    """
    Load any HuggingFace encoder model for embedding-based similarity.

    Supports:
      - microsoft/codebert-base       (RoBERTa encoder, CLS pooling, 768-dim)
      - microsoft/graphcodebert-base  (RoBERTa encoder, CLS pooling, 768-dim)
      - microsoft/unixcoder-base      (RoBERTa encoder, CLS pooling, 768-dim)
      - Qwen/Qwen3-Embedding-0.6B     (Decoder, last-token pooling, 1024-dim)
      - Salesforce/codet5p-110m-embedding
      - Any AutoModel-compatible encoder

    Parameters
    ----------
    encoder_name : str
        HuggingFace model identifier.

    Returns
    -------
    (tokenizer, model, encoder_short_name)
        encoder_short_name is used for cache file naming.
    """
    print(f"  Loading encoder: {encoder_name}")

    # Qwen3-Embedding (and similar decoder-based embedding models) require
    # padding_side='left' so that index -1 is always the last REAL token,
    # regardless of padding.  BERT/RoBERTa-based models are unaffected because
    # they use the CLS token at index 0. 
    _encoder_lower = encoder_name.lower()
    _is_qwen_embed = "qwen" in _encoder_lower
    _padding_side = "left" if _is_qwen_embed else "right"

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        trust_remote_code=True,
        padding_side=_padding_side,
    )
    model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    # Build a short name for cache files: "microsoft/codebert-base" -> "codebert-base"
    short_name = encoder_name.split("/")[-1].lower().replace(" ", "-")

    return tokenizer, model, short_name


# Estimator LLM 

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP.get(name, torch.float16)


def load_llm(cfg: ICLConfig):
    """
    Load the estimator LLM specified in *cfg*.

    Returns (tokenizer, model, device).
    Memory-safe defaults are tuned for a laptop 4070 Mobile 8 GB VRAM:
      - CPU offload via device_map="auto"
      - max_memory capped at cfg.max_gpu_memory_gb
    """
    model_name = cfg.model_name
    torch_dtype = _parse_dtype(cfg.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        quantization_config = None
        if cfg.use_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                print(f"4-bit quantization unavailable ({exc}). Falling back.")

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            if cfg.use_cpu_offload:
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {
                    0: f"{cfg.max_gpu_memory_gb}GiB",
                    "cpu": "10GiB",
                }
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, "hf_device_map"):
        model.to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model, device
