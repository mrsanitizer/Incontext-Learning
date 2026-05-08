"""
LLM inference for code correctness estimation.

Supports two modes:
  - ICL (few-shot): builds prompt with similar examples from training pool
  - Zero-shot (baseline): no examples, just the instruction + problem + code

Builds the prompt, feeds it to the estimator LLM, and extracts
P(yes) vs P(no) from the next-token logits.
"""

import torch


# Prompt templates 

ICL_TEMPLATE = """\
You are an experienced software engineer. Your task is to check the functional correctness of code for the given problem statement. Generate 'yes' if the code is functionally correct (i.e., code meets the problem's requirements) otherwise generate 'no'. 
Refer to the examples below on functional correctness for code evaluation.

{examples}

Evaluate the following code snippet.

Problem Description: 
{problem}

Code Snippet: 
{code}

Functionally Correct:

"""

ZEROSHOT_TEMPLATE = """\
You are an experienced software engineer. Your task is to check the functional correctness of code for the given problem statement. Generate 'yes' if the code is functionally correct (i.e., code meets the problem's requirements) otherwise generate 'no'. 

Problem Description: 
{problem}

Code Snippet: 
{code}

Functionally Correct:
"""


def format_examples(top_k_correct, top_k_incorrect):
    """
    Build the few-shot examples block from selected similar instances.

    Parameters
    ----------
    top_k_correct : list[(float, ProblemInstance)]
    top_k_incorrect : list[(float, ProblemInstance)]
    """
    text = ""
    for i, (_, inst) in enumerate(top_k_correct + top_k_incorrect, 1):
        label = "yes" if inst.result == 1 else "no"
        text += (
            f"Example {i}:\n"
            f" Problem Description: {inst.prompt}\n"
            f" Code snippet: \n {inst.code}\n  \n"
            f" Functionally Correct: {label}\n\n"
        )
    return text


def build_prompt(problem, code, examples_text=None):
    """
    Build the full prompt string.

    If examples_text is None or empty, uses zero-shot template.
    Otherwise uses ICL (few-shot) template.

    Returns the complete prompt string.
    """
    if examples_text:
        return ICL_TEMPLATE.format(
            problem=problem, code=code, examples=examples_text,
        )
    else:
        return ZEROSHOT_TEMPLATE.format(problem=problem, code=code)


# Token-ID helpers 

def _get_first_token_ids(tokenizer, label):
    """Get candidate token IDs for 'yes' or 'no' (incl. capitalised variants)."""
    variants = [label, f" {label}", label.capitalize(), f" {label.capitalize()}"]
    ids = []
    for v in variants:
        encoded = tokenizer.encode(v, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return list(dict.fromkeys(ids))  # deduplicate, preserve order


# Core inference 

def get_correctness_score(problem, code, examples_text,
                          tokenizer, model, device, max_prompt_tokens):
    """
    Run a single forward pass and return (pos_prob, neg_prob, full_prompt).

    max_prompt_tokens:
      - Positive int  : truncate prompt to this many tokens; flag as TRUNCATED if exceeded.
      - None or 0     : no limit — full prompt is passed to the model as-is.
                        Use this when you have unlimited VRAM (cloud GPUs, etc.).

    Returns (None, None, prompt, error_status, raw_tokens) on OOM.
    The full_prompt is always returned for logging purposes.
    """
    prompt = build_prompt(problem, code, examples_text)
    pos_ids = _get_first_token_ids(tokenizer, "yes")
    neg_ids = _get_first_token_ids(tokenizer, "no")

    raw_tokens = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    error_status = None

    # Determine whether a token limit is active
    limit_active = max_prompt_tokens is not None and int(max_prompt_tokens) > 0

    if limit_active and raw_tokens > int(max_prompt_tokens):
        error_status = f"TRUNCATED: Prompt was {raw_tokens} tokens (Limit: {max_prompt_tokens})"

    if limit_active:
        tokenized = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=int(max_prompt_tokens),
        )
    else:
        # No limit: pass the full prompt without truncation
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)

    # Decode back into text to capture exactly what the LLM saw
    actual_prompt_seen = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

    if not hasattr(model, "hf_device_map"):
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

    try:
        with torch.inference_mode():
            outputs = model(**tokenized, use_cache=False) #use_cache is false since we don't want to save the k-v cache of previous tokens
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(
                logits.detach().float().cpu(), dim=-1
            )

        if not pos_ids or not neg_ids:
            return None, None, actual_prompt_seen, error_status, raw_tokens

        pos_prob = max(probs[0, tid].item() for tid in pos_ids)
        neg_prob = max(probs[0, tid].item() for tid in neg_ids)
        return pos_prob, neg_prob, actual_prompt_seen, error_status, raw_tokens

    except torch.cuda.OutOfMemoryError:
        print("  [WARN] Skipping instance due to CUDA OOM.")
        torch.cuda.empty_cache()
        return None, None, actual_prompt_seen, "OOM: GPU ran out of memory", raw_tokens


def logits_to_score(pos_prob, neg_prob):
    """Convert P(yes)/P(no) into a single correctness score in [0, 1]."""
    if pos_prob is None or neg_prob is None:
        return 0.0
    denom = pos_prob + neg_prob
    if denom <= 0.0:
        return 0.0
    return pos_prob / denom
