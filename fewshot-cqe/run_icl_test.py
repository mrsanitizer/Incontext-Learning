#!/usr/bin/env python
"""
run_icl_test.py  --  Modular ICL / Zero-shot test runner
=========================================================

Runs In-Context Learning (ICL) or zero-shot baseline code-correctness
estimation on a test set using a configurable encoder + estimator LLM.

Modes
-----
  k >= 1  : ICL (few-shot) with k examples per class
  k = 0   : Zero-shot baseline (no examples, just instruction)

Usage
-----
# Set values in test_default.json
python run_icl_test.py --config configs/test_default.json

# Zero-shot baseline only, overriding k_values to contain only 0
python run_icl_test.py --config configs/test_default.json --k 0

# ICL + zero-shot together
python run_icl_test.py --config configs/test_default.json --k 0 1 2

# Different encoder model
python run_icl_test.py --config configs/test_default.json \
    --encoder "microsoft/graphcodebert-base"

# Different estimator LLM
python run_icl_test.py --config configs/test_default.json \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Different test dataset
python run_icl_test.py --config configs/test_default.json \
    --test-data "../data/mbjp-codellama-7b.jsonl"

# Combine overrides
python run_icl_test.py --config configs/test_default.json \
    --k 0 1 3 --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --encoder "microsoft/unixcoder-base" --p 0.5
"""

import argparse
import sys
import time
from collections import defaultdict

import pandas as pd

from icl.config import load_config
from icl.models import load_encoder, load_llm
from icl.data import (
    load_data, split_train_dev, compute_embeddings_cached,
    transform_data, save_instances_by_result,
)
from icl.similarity import select_examples
from icl.inference import format_examples, get_correctness_score, logits_to_score
from icl.evaluation import (
    compute_ndcg, save_similarity_results, save_average_scores,
    save_ndcg_results, display_results,
)
from icl.logging import LiveLogger


# ── CLI ─────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="ICL / zero-shot test evaluation for code quality estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True,
                   help="Path to JSON config file")

    # ── quick overrides (all optional) ──
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="Override k_values (e.g. --k 0 1 2 3). k=0 = zero-shot")
    p.add_argument("--model", type=str, default=None,
                   help="Override estimator LLM model name")
    p.add_argument("--encoder", type=str, default=None,
                   help="Override encoder model (e.g. microsoft/graphcodebert-base)")
    p.add_argument("--test-data", type=str, default=None,
                   help="Override input_test_file path")
    p.add_argument("--train-data", type=str, default=None,
                   help="Override input_train_file path")
    p.add_argument("--p", type=float, nargs="+", default=None,
                   help="Override p_values (prompt-vs-code similarity weight, e.g. --p 0 0.5 1)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory")
    p.add_argument("--max-prompt-tokens", type=int, default=None,
                   help="Override max prompt token length")
    p.add_argument("--no-log-prompts", action="store_true",
                   help="Disable saving full prompt text (saves disk space)")
    return p


def cli_overrides_from_args(args):
    """Build a dict of non-None CLI overrides to merge into the config."""
    overrides = {}
    if args.k is not None:
        overrides["k_values"] = args.k
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.encoder is not None:
        overrides["encoder_name"] = args.encoder
    if args.test_data is not None:
        overrides["input_test_file"] = args.test_data
    if args.train_data is not None:
        overrides["input_train_file"] = args.train_data
    if args.p is not None:
        overrides["p_values"] = args.p
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_prompt_tokens is not None:
        overrides["max_prompt_tokens"] = args.max_prompt_tokens
    if args.no_log_prompts:
        overrides["log_prompts"] = False
    return overrides


# ── Main pipeline ───────────────────────────────────────────────────────

def run_test(cfg):
    """
    Full test pipeline:
      1. Load data (training pool + test set)
      2. Load models (encoder + estimator LLM)
      3. Compute/load cached embeddings
      4. For each k: score every test instance, compute nDCG
         - k=0: zero-shot (no examples)
         - k>=1: ICL with k examples per class
      5. Save & display results (live per-instance + final summary)
    """
    t0 = time.time()
    has_icl = any(k > 0 for k in cfg.k_values)
    has_zeroshot = 0 in cfg.k_values

    # ── 1. Data ──
    print(f"[1/5] Loading data ...")
    print(f"  Training pool : {cfg.input_train_file}")
    print(f"  Test set      : {cfg.input_test_file}")
    training_data = load_data(cfg.input_train_file)
    training_data, _ = split_train_dev(training_data)   # use train split only
    testing_data = load_data(cfg.input_test_file)
    print(f"  Training tasks: {len(training_data)}")
    print(f"  Test tasks    : {len(testing_data)}")

    # ── 2. Embeddings (Sequential Loading) ──
    # We load the encoder, compute the embeddings, and then immediately delete
    # the encoder from the GPU to free up VRAM before loading the massive LLM.
    train_instances = None
    test_instances_with_embs = None
    enc_short = None

    if has_icl:
        print(f"\n[2/5] Loading encoder: {cfg.encoder_name} ...")
        enc_tok, enc_model, enc_short = load_encoder(cfg.encoder_name)

    if has_icl:
        print(f"\n[3/5] Computing embeddings (encoder: {enc_short}) ...")

        # Cache training embeddings separately (encode MBPP once, reuse)
        print(f"  Training pool embeddings:")
        train_prompt_embs, train_code_embs = compute_embeddings_cached(
            training_data, enc_tok, enc_model,
            enc_short, cfg.input_train_file, cfg.embedding_cache_dir,
        )

        # Cache test embeddings separately (per test dataset)
        print(f"  Test set embeddings:")
        test_prompt_embs, test_code_embs = compute_embeddings_cached(
            testing_data, enc_tok, enc_model,
            enc_short, cfg.input_test_file, cfg.embedding_cache_dir,
        )

        # Merge for transform_data (train instances need train embeddings,
        # test instances need test embeddings; prompts might overlap)
        all_prompt_embs = {**train_prompt_embs, **test_prompt_embs}
        all_code_embs = {**train_code_embs, **test_code_embs}

        train_instances = transform_data(training_data, all_prompt_embs, all_code_embs)
        test_instances_with_embs = transform_data(testing_data, all_prompt_embs, all_code_embs)
        print(f"  Train instance count: {len(train_instances)}")
        print(f"  Test  instance count: {len(test_instances_with_embs)}")

        if cfg.balanced:
            save_instances_by_result(train_instances, cfg.output_dir)
            
        # Free Encoder from GPU to save memory for LLM
        print("  Freeing Encoder from GPU VRAM...")
        del enc_tok
        del enc_model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[2/5] Encoder not needed (zero-shot only)")
        print(f"\n[3/5] Embeddings not needed (zero-shot only)")

    print(f"\n[4/5] Loading estimator LLM: {cfg.model_name} ...")
    llm_tok, llm_model, device = load_llm(cfg)
    print(f"  Device: {device}")

    # For zero-shot, we still need test instances (without embeddings)
    if has_zeroshot and test_instances_with_embs is None:
        # Build minimal test instances for zero-shot (no embeddings needed)
        import numpy as np
        dummy_emb = np.zeros(1)
        from icl.data import ProblemInstance
        test_instances_zeroshot = []
        for _, row in testing_data.iterrows():
            for i in range(1, 11):
                code_key = f"code_{i}"
                if code_key in row and not pd.isna(row[code_key]):
                    test_instances_zeroshot.append(ProblemInstance(
                        row["task_id"], row["prompt"], code_key, row[code_key],
                        row["results"][i - 1], dummy_emb, dummy_emb,
                    ))
    else:
        test_instances_zeroshot = test_instances_with_embs

    # ── 4. Score & evaluate per k ──
    mode_str = "ICL + zero-shot" if (has_icl and has_zeroshot) else \
               "zero-shot" if has_zeroshot else "ICL"
    print(f"\n[5/5] Running {mode_str} evaluation for k = {cfg.k_values} ...")
    ndcg_results = {}
    filter_unique = cfg.filter

    for k in cfg.k_values:
        is_zeroshot = (k == 0)
        
        # Zero-shot runs once (p is irrelevant). ICL runs for each p in p_values.
        p_loop = [None] if is_zeroshot else cfg.p_values
        
        for p in p_loop:
            mode_label = "ZERO-SHOT" if is_zeroshot else f"ICL (k={k}, p={p})"
            print(f"\n{'=' * 60}")
            print(f"  {mode_label}")
            print(f"{'=' * 60}")

            # Pick the right test instances
            instances = test_instances_zeroshot if is_zeroshot else test_instances_with_embs

            global_rels = []
            global_scores = []
            task_locals = defaultdict(list)
            predictions = []

            with LiveLogger(cfg.output_dir, k, p, log_prompts=cfg.log_prompts) as logger:
                logger.set_total(len(instances))
                error_count = 0

                for idx, inst in enumerate(instances):
                    # Example selection (only for ICL, not zero-shot)
                    examples_text = None
                    if not is_zeroshot:
                        top_correct, top_incorrect = select_examples(
                            inst, train_instances, k, p,
                            filter_unique, cfg.balanced,
                        )
                        examples_text = format_examples(top_correct, top_incorrect)

                    pos_prob, neg_prob, full_prompt, error_status, raw_tokens = get_correctness_score(
                        inst.prompt, inst.code, examples_text,
                        llm_tok, llm_model, device, cfg.max_prompt_tokens,
                    )
                    score = logits_to_score(pos_prob, neg_prob)

                    # Live log
                    logger.log_instance(
                        inst.task_id, inst.code_key,
                        pos_prob, neg_prob, score, inst.result,
                        prompt_text=full_prompt, error_status=error_status,
                        raw_tokens=raw_tokens
                    )

                    predictions.append({
                        "task_id": inst.task_id,
                        "code_key": inst.code_key,
                        "logits_yes": pos_prob,
                        "logits_no": neg_prob,
                        "score": score,
                    })
                    global_rels.append(inst.result)
                    global_scores.append(score)
                    task_locals[inst.task_id].append((inst.result, score))
                    
                    if error_status:
                        error_count += 1
                        if error_count >= 5:
                            print(f"\n[ABORT] Too many errors ({error_count}) for k={k}, p={p}. Skipping the rest of this combination to save time!")
                            break

            # Truncate if mismatch (safety)
            if len(global_rels) > len(global_scores):
                global_rels = global_rels[:len(global_scores)]

            save_similarity_results(predictions, cfg.output_dir, k, p)
            save_average_scores(predictions, cfg.output_dir, k, p)
            
            res_key = f"k_{k}_p_{p}" if p is not None else f"k_{k}"
            ndcg_results[res_key] = compute_ndcg(global_rels, global_scores, task_locals)

            # Persist nDCG results after every completed k/p combination so
            # partial runs produce usable output even if later combinations fail.
            save_ndcg_results(ndcg_results, cfg.output_dir)
            print(f"  Saved intermediate nDCG results for {res_key} -> {cfg.output_dir}/ndcg_results.json")
            display_results({res_key: ndcg_results[res_key]})

    save_ndcg_results(ndcg_results, cfg.output_dir)
    display_results(ndcg_results)

    elapsed = time.time() - t0
    print(f"Done in {elapsed / 60:.1f} min.  Results saved to: {cfg.output_dir}")


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()
    overrides = cli_overrides_from_args(args)
    cfg = load_config(args.config, cli_overrides=overrides)

    # Print effective config
    print("=" * 60)
    print("  ICL Test Configuration")
    print("=" * 60)
    for field_name, val in cfg.to_dict().items():
        print(f"  {field_name:22s} : {val}")
    print("=" * 60)

    has_zs = 0 in cfg.k_values
    has_icl = any(k > 0 for k in cfg.k_values)
    if has_zs and has_icl:
        print("  Mode: ICL + zero-shot baseline")
    elif has_zs:
        print("  Mode: Zero-shot baseline only")
    else:
        print("  Mode: ICL (few-shot)")
    print()

    run_test(cfg)


if __name__ == "__main__":
    main()
