#!/usr/bin/env python3
import os
import json
from collections import Counter

def save_mappings(vocab, label2id, id2label, out_dir="saved_vocab_art"):
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
    ann_file = "./art/ann.json"  # Your GQA formatted annotation
    out_dir  = "art"
    topk     = 8000  # Adjust this as needed

    # 1) Load all annotations
    with open(ann_file, "r") as f:
        data = json.load(f)

    # 2) Collect & count answers
    all_answers = []
    for rec in data:
        if isinstance(rec["answer"], str):
            txt = rec["answer"].strip().lower()
            if txt:
                all_answers.append(txt)

    freq = Counter(all_answers)

    # 3) Create vocab & mappings
    vocab    = [a for a, _ in freq.most_common(topk)]
    label2id = {a: i for i, a in enumerate(vocab)}
    id2label = {i: a for a, i in label2id.items()}

    # 4) Save
    save_mappings(vocab, label2id, id2label, out_dir)
    print(f"Saved top-{topk} vocab into '{out_dir}'")

if __name__ == "__main__":
    main()
