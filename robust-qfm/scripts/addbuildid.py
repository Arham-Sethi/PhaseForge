#!/usr/bin/env python3
import csv

CSV_IN  = "style.csv"
CSV_OUT = "style.csv"   # overwrite in-place; change to "style_with_base.csv" if you want a backup

def infer_num_corruptions(rows):
    # Find distance between first two 'clean' rows → that's the cycle length
    clean_idxs = [i for i, r in enumerate(rows) if r['env'] == 'clean']
    if len(clean_idxs) < 2:
        raise RuntimeError("Couldn't infer cycle length (need at least two 'clean' rows).")
    n = clean_idxs[1] - clean_idxs[0]
    if n <= 0:
        raise RuntimeError("Invalid cycle length inferred.")
    return n

def main():
    with open(CSV_IN, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Make sure required columns exist
    for col in ("filepath","env","param","severity"):
        if col not in reader.fieldnames:
            raise ValueError(f"Missing column '{col}' in {CSV_IN}")

    num_corruptions = infer_num_corruptions(rows)
    print(f"[INFO] Inferred num_corruptions per image: {num_corruptions}")

    # Add base_id = row_index // num_corruptions
    for i, r in enumerate(rows):
        r["base_id"] = str(i // num_corruptions)

    # Write back (adding the new column at the end)
    fieldnames = reader.fieldnames + (["base_id"] if "base_id" not in reader.fieldnames else [])
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote '{CSV_OUT}' with base_id column.")

if __name__ == "__main__":
    main()
