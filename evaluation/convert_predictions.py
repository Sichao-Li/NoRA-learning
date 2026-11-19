"""Convert MS-SWIFT JSON responses without reconstructing or repairing content."""

import argparse
import json
from pathlib import Path

from nora.adapters import validate_annotation
from nora.data import index_rows, read_rows


def input_key(row):
    messages = row.get("messages", [])
    return tuple(message["content"] for message in messages if message.get("role") == "user")


def parse_response(response, clip_id):
    if not isinstance(response, str) or not response.strip():
        raise ValueError("missing_response")
    text = response.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0] in ("```", "```json") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("invalid_json") from None
    if not isinstance(payload, dict):
        raise ValueError("annotation_not_object")
    if payload.get("clip_id") != clip_id:
        raise ValueError("prediction_clip_id_mismatch")
    try:
        validate_annotation(payload)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("invalid_annotation_structure_or_links") from None
    if not payload["actions"]:
        raise ValueError("no_candidate_actions")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Prepared test inputs, without gold answers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; choose a new directory")
    expected = index_rows(read_rows(args.inputs))
    identities = {input_key(row): key for key, row in expected.items()}
    if not expected or () in identities or len(identities) != len(expected):
        parser.error("Test inputs must have unique, nonempty user prompts")
    predictions, failures, seen = [], [], set()
    for row in read_rows(args.input):
        key = input_key(row)
        clip_id = identities.get(key) if key else row.get("clip_id")
        if clip_id not in expected or (row.get("clip_id") not in (None, clip_id)):
            parser.error("Unknown or conflicting input identity; row-order matching is not supported")
        if clip_id in seen:
            parser.error(f"Duplicate response for clip_id: {clip_id}")
        seen.add(clip_id)
        response = row.get("response")
        try:
            if row.get("status") == "failed":
                raise ValueError("generation_failed")
            if row.get("finish_reason") not in (None, "stop"):
                raise ValueError("incomplete_generation")
            payload = parse_response(response, clip_id)
            predictions.append(payload)
        except ValueError as exc:
            failures.append({"clip_id": clip_id, "code": str(exc), "response": response})
            predictions.append({"clip_id": clip_id, "status": "failed", "error": str(exc)})
    for clip_id in expected.keys() - seen:
        failures.append({"clip_id": clip_id, "code": "missing_response"})
    args.output.mkdir(parents=True)
    for name, rows in (("predictions", predictions), ("failures", failures)):
        with (args.output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    valid = sum(row.get("status") != "failed" for row in predictions)
    summary = {"expected": len(expected), "received": len(seen), "valid": valid,
               "invalid": len(seen) - valid, "missing": len(expected) - len(seen)}
    (args.output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
