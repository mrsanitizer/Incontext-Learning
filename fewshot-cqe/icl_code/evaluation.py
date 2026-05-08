"""
Evaluation metrics and result persistence for ICL test runs.

Computes global and per-task nDCG scores, saves prediction files,
and prints a summary table.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import ndcg_score


# nDCG computation 

def compute_ndcg(global_relevance, global_scores, task_local_ndcgs):
    """
    Compute global nDCG and average per-task (local) nDCG.

    Parameters
    ----------
    global_relevance : list[int]
    global_scores : list[float]
    task_local_ndcgs : dict[str, list[(int, float)]]

    Returns
    -------
    dict  with keys 'global_nDCG' and 'local_avg_nDCG'
    """
    if len(global_relevance) < 2:
        print(f"  [WARN] Not enough instances to compute global nDCG (got {len(global_relevance)}). Returning 0.0.")
        return {"global_nDCG": 0.0, "local_avg_nDCG": 0.0}

    global_ndcg = ndcg_score([global_relevance], [global_scores])

    local_values = []
    for task_id, pairs in task_local_ndcgs.items():
        rels, preds = zip(*pairs)
        rels = rels[:len(preds)]
        if len(rels) > 1:
            local_values.append(ndcg_score([rels], [preds]))
        else:
            print(f"  Task with < 2 items: {task_id}")
            local_values.append(0.0)

    return {
        "global_nDCG": global_ndcg,
        "local_avg_nDCG": float(np.mean(local_values)) if local_values else 0.0,
    }


# Persistence 

def _as_serialisable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def save_similarity_results(predicted_results, output_dir, k, p=None):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_k_{k}_p_{p}" if p is not None else f"_k_{k}"
    path = os.path.join(output_dir, f"similarity_results{suffix}.jsonl")
    with open(path, "w") as f:
        for r in predicted_results:
            json.dump({k_: _as_serialisable(v) for k_, v in r.items()}, f)
            f.write("\n")


def save_average_scores(predicted_results, output_dir, k, p=None):
    task_scores = defaultdict(list)
    for r in predicted_results:
        task_scores[r["task_id"]].append(r["score"])
    avg = {tid: float(np.mean(scores)) for tid, scores in task_scores.items()}

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_k_{k}_p_{p}" if p is not None else f"_k_{k}"
    path = os.path.join(output_dir, f"avg_scores{suffix}.jsonl")
    with open(path, "w") as f:
        for tid, score in avg.items():
            json.dump({"task_id": tid, "score": _as_serialisable(score)}, f)
            f.write("\n")


def save_ndcg_results(ndcg_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ndcg_results.json")
    with open(path, "w") as f:
        json.dump(ndcg_results, f, indent=4)


def display_results(ndcg_results):
    """Pretty-print the nDCG table."""
    print("\n" + "=" * 50)
    print("  ICL Test Results")
    print("=" * 50)
    for key, res in ndcg_results.items():
        print(f"  {key}")
        print(f"    Global  nDCG : {res['global_nDCG']:.6f}")
        print(f"    Local   nDCG : {res['local_avg_nDCG']:.6f}")
        print("-" * 50)
    print()
