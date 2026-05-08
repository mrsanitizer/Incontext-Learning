#!/usr/bin/env python3
"""
token_report.py

Parse `live_results*.jsonl` files, report highest token counts and
produce plots (max-per-file bar chart + combined histogram).

Usage:
  python scripts/token_report.py --output-dir ../output/icl/heval-codestral/balanced

Outputs:
  - token_summary.csv
  - max_per_file.png
  - combined_hist.png
  - hist_<filename>.png for each input file
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def read_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', '-o', required=True,
                   help='Directory containing live_results*.jsonl')
    p.add_argument('--pattern', default='live_results*.jsonl',
                   help='Filename glob pattern (default: live_results*.jsonl)')
    p.add_argument('--top-n', type=int, default=20,
                   help='Top-N instances to print per-file')
    args = p.parse_args()

    glob_path = os.path.join(args.output_dir, args.pattern)
    files = sorted(glob.glob(glob_path))
    if not files:
        print(f'No files found for: {glob_path}')
        return

    all_records = []
    summary = []

    for fpath in files:
        rows = read_jsonl(fpath)
        entries = []
        for r in rows:
            rt = r.get('raw_tokens')
            if rt is None:
                continue
            try:
                rt_int = int(rt)
            except Exception:
                continue
            entries.append({
                'raw_tokens': rt_int,
                'task_id': r.get('task_id'),
                'code_key': r.get('code_key'),
            })

        toks = [e['raw_tokens'] for e in entries]

        if toks:
            mx = max(toks)
            mean = sum(toks) / len(toks)
            # find the entry (first occurrence) with max tokens
            max_entry = max(entries, key=lambda e: e['raw_tokens'])
            max_index = next((i for i, e in enumerate(entries, 1) if e['raw_tokens'] == mx), None)
            max_task = max_entry.get('task_id')
            max_code = max_entry.get('code_key')
        else:
            mx = 0
            mean = 0
            max_index = None
            max_task = None
            max_code = None

        summary.append({
            'file': os.path.basename(fpath),
            'path': fpath,
            'count': len(toks),
            'max_tokens': mx,
            'mean_tokens': mean,
            'max_index': max_index,
            'max_task_id': max_task,
            'max_code_key': max_code,
        })

        # add to combined table
        for i, e in enumerate(entries, 1):
            all_records.append({
                'file': os.path.basename(fpath),
                'index': i,
                'raw_tokens': e['raw_tokens'],
                'task_id': e.get('task_id'),
                'code_key': e.get('code_key'),
            })

        # per-file histogram
        if toks:
            plt.figure(figsize=(8, 4))
            plt.hist(toks, bins=50, color='C0', alpha=0.8)
            plt.axvline(mx, color='red', linestyle='--', label=f'max={mx}')
            plt.title(f'raw_tokens distribution — {os.path.basename(fpath)}')
            plt.xlabel('raw_tokens')
            plt.ylabel('count')
            plt.legend()
            out_hist = os.path.join(args.output_dir, f"hist_{os.path.basename(fpath)}.png")
            plt.tight_layout()
            plt.savefig(out_hist)
            plt.close()

    df_summary = pd.DataFrame(summary).sort_values('max_tokens', ascending=False)
    df_all = pd.DataFrame(all_records)

    # save CSV summary
    summary_csv = os.path.join(args.output_dir, 'token_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f'Wrote summary -> {summary_csv}')

    # bar chart of max per file
    plt.figure(figsize=(10, 4))
    plt.bar(df_summary['file'], df_summary['max_tokens'], color='C1')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('max raw_tokens')
    plt.title('Max raw_tokens per live_results file')
    plt.tight_layout()
    out_bar = os.path.join(args.output_dir, 'max_per_file.png')
    plt.savefig(out_bar)
    plt.close()
    print(f'Wrote bar chart -> {out_bar}')

    # combined histogram
    if not df_all.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(df_all['raw_tokens'], bins=100, color='C2', alpha=0.8)
        plt.title('Combined raw_tokens distribution')
        plt.xlabel('raw_tokens')
        plt.ylabel('count')
        plt.tight_layout()
        out_comb = os.path.join(args.output_dir, 'combined_hist.png')
        plt.savefig(out_comb)
        plt.close()
        print(f'Wrote combined histogram -> {out_comb}')

    # print top instances per file
    for row in df_summary.itertuples():
        print(f"{row.file}: count={row.count}  max={row.max_tokens}  mean={row.mean_tokens:.1f}")


if __name__ == '__main__':
    main()
