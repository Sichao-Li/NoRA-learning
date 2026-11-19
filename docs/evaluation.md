# Evaluation

## NoRA

This repository uses the public NoRA evaluator as a dependency. Its action,
fact, and support scoring code is not copied or reimplemented here.

The prediction launcher saves MS-SWIFT response records. Convert their JSON
responses into NoRA annotation records:

```bash
uv run python evaluation/convert_predictions.py \
  --input runs/raw.jsonl --inputs data/prepared/test.jsonl \
  --output runs/raw-annotations
```

The converter takes clip identity from the returned input, never from row order.
It accepts an annotation object or JSON enclosed in a single code fence; it
does not rewrite facts, infer links, select the first action, or use reference
answers to repair a response. Missing, invalid, and failed responses are kept
separate from usable predictions. Failed candidates and safe error codes are
saved in `failures.jsonl`. A conversion with no scoreable responses fails.

```bash
uv run --extra scorer nora evaluate \
  --predictions runs/raw-annotations/predictions.jsonl \
  --references data/prepared/references.jsonl \
  --output runs/raw-scores --progress
```

Repeat for SFT. Compare runs on their common valid clips:

```bash
uv run nora compare runs/raw-scores runs/sft-scores
```

Pass all eight run directories for the four-size comparison. Use the same
reference and scorer settings across runs. Report coverage with the scores;
valid but wrong answers remain in the denominator. No chosen-action score
should be inferred when the model did not explicitly choose an action.

The semantic scorer downloads on first use; after caching, `--offline` is
available. These commands do not use a paid reconstruction API. For free-form
responses instead of JSON, use the separate reconstruction workflow documented
in the [NoRA model guide](https://github.com/mint-philosophy/NoRA-benchmark/blob/main/docs/models.md).

The public evaluator scores all submitted candidate actions. The paper's saved
learning scores used an earlier fixed reference and supported-action conversion.
New public-format evaluations must therefore be reported separately, not added
to the paper's historical leaderboard.

## Transfer Benchmarks

Use each benchmark's own tasks and evaluator. The included result tables are
the paper's saved comparisons, not new judgments. This repository does not
bundle separate forks of every external benchmark or their restricted inputs.

| Benchmark | Measure and interpretation |
| --- | --- |
| [MoReBench](https://github.com/morebench/morebench) | Public dilemmas; weighted score plus positive fulfillment and negative avoidance. Paper judgments use GPT-OSS-20B. |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Attack success; separate behavior-classified and copyright cases, then functional and semantic categories. |
| [XSTest](https://github.com/paul-rottger/xstest) | Safe compliance and unsafe refusal, reported together. |
| [MSSBench](https://openreview.net/forum?id=I9bEi6LNgt) | Safe/unsafe chat and embodied accuracy. Paper judgments use substituted GPT-OSS-20B, not a validated equivalent of the original judge. |
| ETHICS, MoralChoice, MoralBench, NormLens, BBQ, IFEval | Supplementary comparisons with the task-specific scope described in the paper. |

For cross-prompt comparisons, use cases valid for all systems under both prompts.
NoRA-style prompting is not native model thinking. Keep macro scores distinct
from pooled accuracy and keep refusal gains alongside safe-request costs.
