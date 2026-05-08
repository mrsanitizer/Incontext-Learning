"""
Live logging for ICL test runs.

Streams per-instance results to both console and a JSONL log file
in real-time, and optionally saves the full prompt text for each
question in a readable format.
"""

import json
import os
import sys
from datetime import datetime


class LiveLogger:
    """
    Streams results live to console + file as each instance is scored.

    Files created in output_dir:
      - live_results_k_{k}.jsonl   : one JSON line per scored instance
      - prompts_k_{k}.txt         : full readable prompts (if log_prompts=True)
    """

    def __init__(self, output_dir: str, k, p=None, log_prompts: bool = True):
        self.output_dir = output_dir
        self.k = k
        self.p = p
        self.log_prompts = log_prompts
        self._count = 0
        self._total = 0

        os.makedirs(output_dir, exist_ok=True)

        suffix = f"_k_{k}_p_{p}" if p is not None else f"_k_{k}"
        
        # JSONL live results file (append mode so we don't lose data on crash)
        results_path = os.path.join(output_dir, f"live_results{suffix}.jsonl")
        self._results_file = open(results_path, "a", encoding="utf-8")

        # Prompt text log
        self._prompt_file = None
        if log_prompts:
            prompt_path = os.path.join(output_dir, f"prompts{suffix}.txt")
            self._prompt_file = open(prompt_path, "w", encoding="utf-8")

    def set_total(self, total: int):
        self._total = total

    def log_instance(self, task_id, code_key, pos_prob, neg_prob, score, ground_truth,
                     prompt_text=None, error_status=None, raw_tokens=None):
        """Log a single scored instance (live to console + file)."""
        self._count += 1

        record = {
            "index": self._count,
            "task_id": task_id,
            "code_key": code_key,
            "logits_yes": pos_prob,
            "logits_no": neg_prob,
            "score": score,
            "ground_truth": ground_truth,
            "correct_prediction": (score >= 0.5) == (ground_truth == 1),
            "error": error_status,
            "raw_tokens": raw_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        # Write to JSONL file immediately
        self._results_file.write(json.dumps(record) + "\n")
        self._results_file.flush()

        # Console output (compact, one line per instance)
        gt_label = "PASS" if ground_truth == 1 else "FAIL"
        pred_label = "yes" if score >= 0.5 else "no"
        match = "OK" if record["correct_prediction"] else "XX"
        progress = f"[{self._count}/{self._total}]" if self._total else f"[{self._count}]"

        token_info = f" tkns={raw_tokens}" if raw_tokens is not None else ""
        print(
            f"    {progress} {task_id:25s} {code_key:8s} | "
            f"score={score:.4f}  P(yes)={pos_prob or 0:.4f}  P(no)={neg_prob or 0:.4f} | "
            f"gt={gt_label} pred={pred_label} [{match}]{token_info}"
        )
        if error_status:
            print(f"      => [ERROR] {error_status}")
            error_log_path = os.path.join(self.output_dir, "error_log.txt")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'#' * 80}\n")
                f.write(f"ERROR: {error_status}\n")
                f.write(f"k={self.k}, p={self.p} | Task: {task_id} | Code: {code_key}\n")
                f.write(f"{'#' * 80}\n\n")
                if prompt_text:
                    f.write(prompt_text + "\n\n")
        sys.stdout.flush()

        # Write full prompt text
        if self._prompt_file and prompt_text:
            self._prompt_file.write(
                f"{'=' * 80}\n"
                f"INSTANCE {self._count}: {task_id} / {code_key}\n"
                f"Ground Truth: {gt_label} | Score: {score:.6f} | "
                f"P(yes): {pos_prob or 0:.6f} | P(no): {neg_prob or 0:.6f}\n"
                f"{'=' * 80}\n\n"
                f"{prompt_text}\n\n"
            )
            self._prompt_file.flush()

    def close(self):
        """Flush and close all file handles."""
        if self._results_file:
            self._results_file.close()
        if self._prompt_file:
            self._prompt_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
