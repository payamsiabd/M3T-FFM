#!/usr/bin/env python3
import os, json, random
from collections import Counter, defaultdict
from PIL import Image
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
ann_file      = "./gqa_vqa_formatted.json"
img_dir       = "/Users/payamabd/Documents/Research/AAAI/code/FedDAT-main/src/train/data/vg/VG_100K/"
out_dir       = "gqa"
labels_to_use = None    # None → pick top-K; or set to a list e.g. ["pipe","white","yes"]
top_k_labels  = 20      # if labels_to_use is None, pick this many most frequent
per_label_max = 300     # up to this many samples per label

# ─── LOAD & COUNT ────────────────────────────────────────────────────────────────
records = json.load(open(ann_file))
answers = [r["answer"].strip().lower() for r in records]
freq    = Counter(answers)

# ─── PICK LABELS ─────────────────────────────────────────────────────────────────
if labels_to_use is None:
    labels_to_use = [label for label,_ in freq.most_common(top_k_labels)]
else:
    labels_to_use = [l.strip().lower() for l in labels_to_use]

print(f"Selected {len(labels_to_use)} labels:", labels_to_use)

# ─── GROUP BY LABEL ──────────────────────────────────────────────────────────────
by_label = defaultdict(list)
for rec in records:
    ans = rec["answer"].strip().lower()
    if ans in labels_to_use:
        by_label[ans].append(rec)

# ─── SAMPLE SUBSET ──────────────────────────────────────────────────────────────
os.makedirs(f"{out_dir}/train", exist_ok=True)

subset = []
for label in labels_to_use:
    recs = by_label[label]
    n    = min(len(recs), per_label_max)
    chosen = random.sample(recs, n)
    subset.extend(chosen)
    print(f"Label {label!r}: {len(recs)} available, {n} chosen")

random.shuffle(subset)
print(f"Total in subset: {len(subset)}")

# ─── WRITE ANNOTATIONS ───────────────────────────────────────────────────────────

with open(f"{out_dir}/ann.json", "w") as f:
    json.dump(subset, f)


# ─── COPY IMAGES ────────────────────────────────────────────────────────────────
for rec in tqdm(subset, desc="Copying images"):
    src = os.path.join(img_dir, rec["image_id"])
    dst = os.path.join(out_dir, "train", rec["image_id"])
    Image.open(src).save(dst)

print("Done! Balanced dataset ready in", out_dir)
