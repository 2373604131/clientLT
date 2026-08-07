"""Experiment 2 (part b): CLIP-weak tail zero-shot probe.

Purpose
-------
Before committing the method's CV landing point to a fine-grained natural-image
domain, we must rule out the "data-limited" dead-end: if frozen CLIP already
scores the tail classes highly out of the box, then any tail gain we report is
about *scarcity*, not *new visual concepts*, and reviewers will say the setting
is trivial. Conversely, if CLIP zero-shot on the low-accuracy classes is poor,
those classes are genuinely new visual concepts -> real learnable gain -> a
defensible CV contribution.

What it does
------------
Runs the *original* frozen CLIP (no LoRA, no prompt tuning) zero-shot over an
ImageFolder-style dataset, computes per-class accuracy, ranks classes, and
reports the bottom `tail_ratio` fraction (the CLIP-weak tail) vs the top
fraction (the CLIP-strong head). Emits a verdict.

Usage (run in the GPU training env, NOT the plain python shell)
---------------------------------------------------------------
python scripts/probe_clip_weak_tail.py --data-root /path/to/dataset/test --backbone ViT-B/16 --output-csv output/clip_weak_probe.csv

- --data-root points to a directory whose sub-folders are classes (ImageFolder).
- --classnames-file (optional) maps folder order to readable prompt names, one
  per line, in sorted-folder order; use when folders are codes (e.g. n01234567).
"""

import argparse
import csv
import os

import torch

import clip


def parse_args():
    p = argparse.ArgumentParser(description="CLIP-weak tail zero-shot probe")
    p.add_argument("--data-root", required=True,
                   help="ImageFolder-style dir (sub-folders = classes), e.g. the test split")
    p.add_argument("--backbone", default="ViT-B/16",
                   help="CLIP backbone name (match the repo's config, default ViT-B/16)")
    p.add_argument("--classnames-file", default="",
                   help="optional file: one readable class name per line, in sorted-folder order")
    p.add_argument("--prompt-template", default="a photo of a {}.",
                   help="zero-shot prompt template with a single {} slot")
    p.add_argument("--tail-ratio", type=float, default=0.2,
                   help="bottom fraction (by zero-shot acc) treated as the CLIP-weak tail")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-csv", default="output/clip_weak_probe.csv")
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="cap eval images per class for a quick smoke test (0 = no cap)")
    return p.parse_args()


def load_readable_names(folder_classes, classnames_file):
    """Map ImageFolder class order to readable prompt names."""
    if not classnames_file:
        return [c.replace("_", " ") for c in folder_classes]
    with open(classnames_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != len(folder_classes):
        raise ValueError(
            f"--classnames-file has {len(names)} names but the dataset has "
            f"{len(folder_classes)} class folders; they must match 1:1 in sorted-folder order."
        )
    return [n.replace("_", " ") for n in names]


@torch.no_grad()
def build_text_features(model, readable_names, template, device):
    prompts = [template.format(name) for name in readable_names]
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # (num_classes, dim)


@torch.no_grad()
def evaluate_zero_shot(model, loader, text_features, num_classes, device, limit_per_class):
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes
    seen = [0] * num_classes
    logit_scale = model.logit_scale.exp()
    for images, labels in loader:
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        preds = logits.argmax(dim=1).cpu()
        for label, pred in zip(labels.tolist(), preds.tolist()):
            if limit_per_class and seen[label] >= limit_per_class:
                continue
            seen[label] += 1
            total[label] += 1
            if label == pred:
                correct[label] += 1
    per_class_acc = [
        (100.0 * correct[c] / total[c]) if total[c] > 0 else float("nan")
        for c in range(num_classes)
    ]
    return per_class_acc, total


def summarize(per_class_acc, totals, readable_names, tail_ratio):
    """Rank by zero-shot accuracy; return (rows, tail_mean, head_mean, verdict)."""
    valid = [(c, a) for c, a in enumerate(per_class_acc) if a == a]  # drop NaN
    ranked = sorted(valid, key=lambda t: t[1])  # ascending: weakest first
    n = len(ranked)
    n_tail = max(1, int(round(n * tail_ratio)))
    tail = ranked[:n_tail]
    head = ranked[n_tail:]
    tail_mean = sum(a for _, a in tail) / len(tail)
    head_mean = sum(a for _, a in head) / len(head) if head else float("nan")

    rows = []
    tail_ids = {c for c, _ in tail}
    for rank, (c, a) in enumerate(ranked):
        rows.append({
            "rank": rank,
            "class_id": c,
            "class_name": readable_names[c],
            "zero_shot_acc": round(a, 2),
            "n_eval": totals[c],
            "group": "clip_weak_tail" if c in tail_ids else "clip_strong_head",
        })

    # Verdict thresholds: below 40% tail zero-shot -> genuinely CLIP-weak (good).
    # Above 70% -> domain already solved by CLIP (data-limited dead-end).
    if tail_mean < 40.0:
        verdict = "CLIP-WEAK TAIL (good landing point: tail = new visual concepts, learnable gain)"
    elif tail_mean > 70.0:
        verdict = "CLIP-STRONG TAIL (dead-end: domain already solved, tail gain would be scarcity-only)"
    else:
        verdict = "MIXED (borderline; inspect per-class rows, consider a harder domain)"
    return rows, tail_mean, head_mean, verdict


def main():
    args = parse_args()
    from torchvision import datasets
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP backbone {args.backbone} on {device} ...")
    model, preprocess = clip.load(args.backbone, device=device, jit=False)
    model.eval()

    dataset = datasets.ImageFolder(args.data_root, transform=preprocess)
    folder_classes = dataset.classes
    num_classes = len(folder_classes)
    print(f"Dataset: {args.data_root}  classes={num_classes}  images={len(dataset)}")

    readable_names = load_readable_names(folder_classes, args.classnames_file)
    text_features = build_text_features(model, readable_names, args.prompt_template, device)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    per_class_acc, totals = evaluate_zero_shot(
        model, loader, text_features, num_classes, device, args.limit_per_class
    )

    rows, tail_mean, head_mean, verdict = summarize(
        per_class_acc, totals, readable_names, args.tail_ratio
    )

    overall = sum(a for a in per_class_acc if a == a) / max(
        sum(1 for a in per_class_acc if a == a), 1
    )
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "class_id", "class_name", "zero_shot_acc", "n_eval", "group"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 68)
    print(f"CLIP zero-shot probe  (backbone={args.backbone}, tail_ratio={args.tail_ratio})")
    print(f"  overall macro acc      : {overall:6.2f}%")
    print(f"  CLIP-strong head mean  : {head_mean:6.2f}%")
    print(f"  CLIP-weak tail mean    : {tail_mean:6.2f}%")
    print(f"  head - tail gap        : {head_mean - tail_mean:6.2f} pp")
    print("  weakest 10 classes:")
    for row in rows[:10]:
        print(f"    [{row['zero_shot_acc']:5.1f}%] {row['class_name']}  (n={row['n_eval']})")
    print(f"\n  VERDICT: {verdict}")
    print(f"  full per-class CSV -> {args.output_csv}")
    print("=" * 68)


if __name__ == "__main__":
    main()
