#!/usr/bin/env python3
"""Simple JSONL to CSV converter."""

import csv
import json
from pathlib import Path

# === SET YOUR INPUT FILE PATH HERE ===
INPUT_FILE_PATH = r".\output\icl\Error_mbjp-codellama-qwen3b-16bit-qwen0.6\balanced\live_results_k_2_p_0.0.jsonl" 
# =====================================

def main() -> None:
    input_path = Path(INPUT_FILE_PATH)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        raise SystemExit(1)

    # Automatically create output path in the same folder with .csv extension
    output_path = input_path.with_suffix('.csv')

    rows = []
    columns = set()

    print(f"Reading from: {input_path}...")
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
            columns.update(obj.keys())

    fieldnames = sorted(columns)

    print(f"Writing to: {output_path}...")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Success! Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    main()