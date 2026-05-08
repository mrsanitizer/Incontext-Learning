"""
Similarity-based few-shot example selection.

For each test instance we rank the training pool by a weighted combination
of prompt-embedding and code-embedding cosine similarity:

    sim_total = p * sim_prompt + (1 - p) * sim_code

Then we pick the top-k most similar examples.  If balanced=True, we
separately pick k correct and k incorrect examples.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_top_k_similarities(test_instance, pool_instances, k, p,
                                  filter_unique=False):
    """
    Return the k most-similar instances from pool_instances.

    Parameters
    ----------
    test_instance : ProblemInstance
    pool_instances : list[ProblemInstance]
    k : int
    p : float   - weight for prompt similarity (0 <= p <= 1)
    filter_unique : bool
        If True, enforce at most one example per unique problem statement.
    """
    if not pool_instances:
        return []

    test_prompt = test_instance.prompt_embedding.reshape(1, -1)
    test_code   = test_instance.code_embedding.reshape(1, -1)

    pool_prompt_embs = np.array([inst.prompt_embedding for inst in pool_instances])
    pool_code_embs   = np.array([inst.code_embedding   for inst in pool_instances])

    # Dimension sanity check 
    # Catches the case where test embeddings and pool embeddings were produced
    # by different encoders (e.g. test cached with Qwen 1024-dim, pool with
    # CodeBERT 768-dim) which would cause a cryptic numpy broadcast error.
    if test_prompt.shape[-1] != pool_prompt_embs.shape[-1]:
        raise ValueError(
            f"Embedding dimension mismatch: test prompt has dim "
            f"{test_prompt.shape[-1]} but pool has dim "
            f"{pool_prompt_embs.shape[-1]}. "
            f"This usually means the embedding cache is stale from a "
            f"different encoder. Delete the relevant .npz files in "
            f"embeddings_cache/ and re-run."
        )

    prompt_sims = cosine_similarity(test_prompt, pool_prompt_embs).flatten()
    code_sims   = cosine_similarity(test_code,   pool_code_embs  ).flatten()

    total_sims = p * prompt_sims + (1 - p) * code_sims

    pairs = sorted(zip(total_sims, pool_instances),
                   key=lambda x: x[0], reverse=True)

    if filter_unique:
        seen_prompts = set()
        selected = []
        for sim, inst in pairs:
            if inst.prompt not in seen_prompts:
                selected.append((sim, inst))
                seen_prompts.add(inst.prompt)
            if len(selected) >= k:
                break
        return selected

    return pairs[:k]


def select_examples(test_instance, train_instances, k, p,
                    filter_unique, balanced):
    """
    High-level example selector.

    Returns (top_k_correct, top_k_incorrect).
    When balanced=False, top_k_incorrect is [].
    """
    if balanced:
        correct = [i for i in train_instances if i.result == 1]
        incorrect = [i for i in train_instances if i.result == 0]
        top_correct = calculate_top_k_similarities(
            test_instance, correct, k, p, filter_unique)
        top_incorrect = calculate_top_k_similarities(
            test_instance, incorrect, k, p, filter_unique)
        return top_correct, top_incorrect
    else:
        top_2k = calculate_top_k_similarities(
            test_instance, train_instances, 2 * k, p, filter_unique)
        return top_2k, []
