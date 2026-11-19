"""Prepare public NoRA annotations in MS-SWIFT's messages format."""

import argparse
import json
from pathlib import Path
import random

from nora.adapters import validate_annotation
from nora.data import DATASET_ID, DATASET_REVISION, index_rows, load_references, read_rows
from nora.media import download_media, media_path


def split_train_val(samples, val_ratio, seed):
    # Keep the existing split rule: shuffle indices, then preserve input order.
    if len(samples) < 2 or not 0 < val_ratio < 1:
        raise ValueError("Need at least two training rows and 0 < val_ratio < 1")
    count = min(max(1, round(len(samples) * val_ratio)), len(samples) - 1)
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[:count])
    return ([row for i, row in enumerate(samples) if i not in selected],
            [row for i, row in enumerate(samples) if i in selected])


def sample(row, media_root, prompt, *, target):
    clip_id = row["clip_id"]
    image = media_path(media_root, clip_id, "frames")
    if not image.is_file():
        raise ValueError(f"Missing pre-action montage: {clip_id}")
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": "<image>\n" + prompt["user"].format(clip_id=clip_id)},
    ]
    if target:
        annotation = {key: row[key] for key in ("clip_id", "facts", "reasons", "actions")}
        validate_annotation(annotation)
        messages.append({"role": "assistant", "content": json.dumps(annotation, ensure_ascii=False)})
    return {"clip_id": clip_id, "messages": messages, "images": [str(image)]}


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="Local training annotation JSONL; use with --test")
    parser.add_argument("--test", type=Path, help="Local test reference JSONL; use with --train")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--download-media", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.train) != bool(args.test):
        parser.error("Supply both --train and --test, or neither")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    train = read_rows(args.train) if args.train else load_references("train")
    test = read_rows(args.test) if args.test else load_references("test")
    if not train or not test:
        parser.error("Train and test annotations must both be nonempty")
    if set(index_rows(train)) & set(index_rows(test)):
        parser.error("Train and test clip IDs overlap")
    prompt = json.loads(Path(__file__).with_name("prompt.json").read_text())
    if args.download_media:
        report = download_media(train + test, output=args.media_root, media="frames")
        if report["invalid"]:
            parser.error(f"{report['invalid']} media downloads failed; retry before preparing data")
    training, validation = split_train_val(
        [sample(row, args.media_root, prompt, target=True) for row in train],
        args.val_ratio, args.seed,
    )
    testing = [sample(row, args.media_root, prompt, target=False) for row in test]
    args.output.mkdir(parents=True)
    for name, rows in (("train", training), ("val", validation), ("test", testing), ("references", test)):
        write_jsonl(args.output / f"{name}.jsonl", rows)
    membership = {
        "source": "local files" if args.train else DATASET_ID,
        "dataset_revision": None if args.train else DATASET_REVISION,
        "split_seed": args.seed, "validation_fraction": args.val_ratio,
        "prompt": prompt,
        "train": [row["clip_id"] for row in training],
        "validation": [row["clip_id"] for row in validation],
        "test": [row["clip_id"] for row in testing],
    }
    (args.output / "splits.json").write_text(json.dumps(membership, indent=2) + "\n")
    print(f"Prepared {len(training)} train, {len(validation)} validation, {len(testing)} test inputs")


if __name__ == "__main__":
    main()
