#!/usr/bin/env python3
import os
import json
from collections import Counter

def save_mappings(vocab, label2id, id2label, out_dir="saved_vocab_gqa"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) vocab.txt
    with open(os.path.join(out_dir, "vocab.txt"), "w") as f:
        for ans in vocab:
            f.write(ans.replace("\n", " ") + "\n")

    # 2) label2id.json
    with open(os.path.join(out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    # 3) id2label.json
    with open(os.path.join(out_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)

def main():
    ann_file = "./gqa/ann.json"   # now reading from gqa/ann.json
    out_dir  = "gqa"

    # 1) Load all annotations
    with open(ann_file, "r") as f:
        data = json.load(f)

    # 2) Collect & count every answer string (normalized)
    all_answers = [
        rec["answer"].strip().lower()
        for rec in data
        if isinstance(rec.get("answer"), str) and rec["answer"].strip()
    ]
    freq = Counter(all_answers)

    # 3) Use all unique answers, ordered by descending frequency
    vocab    = [ans for ans, _ in freq.most_common()]
    label2id = {ans: idx for idx, ans in enumerate(vocab)}
    id2label = {idx: ans for ans, idx in label2id.items()}

    # 4) Save to disk
    save_mappings(vocab, label2id, id2label, out_dir)
    print(f"Saved {len(vocab)} labels into '{out_dir}/'")

if __name__ == "__main__":
    main()
