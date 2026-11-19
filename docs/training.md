# Training

## Reported Study

All four Qwen3.5 sizes use the same reported settings:

| Setting | Value |
| --- | --- |
| Supervision source | GPT-5.4 annotations from LLMSilver-1230 |
| Mode | MS-SWIFT full SFT |
| Epochs / learning rate | 3 / 0.00001 |
| Per-device batch / gradient accumulation | 1 / 16 |
| Precision / sequence limit | BF16 / 4,096 tokens |
| Image-token limit | 1,024 |
| Attention / warmup ratio | SDPA / 0.05 |
| Gradient checkpointing / length grouping | Enabled |
| Validation split | 5% of the selected subset, split seed 42 |
| Evaluation / saving | Every 50 steps, retain two checkpoints |

LLMSilver-1230 is the source pool, not a verified count of examples optimized
over. The retained records do not establish the exact selected clip membership,
training seed, optimizer/scheduler configuration, or checkpoint-selection rule.
The paper's targets also included missing-context and retrospective records.
See the manuscript's training appendix for the scope of the reported experiment.

## New Runs with the Public Dataset

`prepare_data.py` uses the public NoRA annotations and a pre-action montage
per clip. It builds train and validation examples from LLMSilver only and keeps
HumanGold for testing. Test inputs contain no reference answers. The script
saves the split membership and test references alongside the prepared files.

```bash
uv run python training/prepare_data.py \
  --media-root data/media --download-media --output data/prepared
```

If media are already available, omit `--download-media`. Each clip directory
must contain `frame_all_prev.jpg`. Source-media access and licensing conditions
still apply. Local annotation JSONL files can be supplied with `--train` and
`--test`; pass both files together.

The public-format target contains facts, reasons, and actions with supporting
links. It does not restore retrospective fields or negative-only actions.
Its prompt is in `training/prompt.json`. This is a public-dataset workflow,
not a claim to reproduce the historical targets or the exact paper scores.

## Fine-tuning and Prediction

On a GPU machine:

```bash
uv sync --extra training
uv run --extra training bash training/train.sh \
  Qwen/Qwen3.5-2B data/prepared runs/2b-sft
```

Replace the model with the 4B, 9B, or 27B checkpoint as needed. GPU memory
requirements depend on model size and device count; the command does not
reserve cluster resources. Cluster-specific account names and PBS directives
are deliberately not included.

The launcher retains the existing training command's settings. Other options
use the installed MS-SWIFT defaults and are recorded by that framework. In
particular, `tuner_type=full` does not by itself specify which visual modules
are frozen. Pass such choices explicitly if they are part of your new study:

```bash
uv run --extra training bash training/train.sh \
  Qwen/Qwen3.5-2B data/prepared runs/2b-sft \
  --seed 42 --freeze_vit false --freeze_aligner false
```

This example makes new choices explicit; it does not assert that those were
the original experiment's settings. Keep the generated training arguments,
split membership, package versions, and selected checkpoint with your run.
The dependency pin here is for this companion workflow, not a recovered
historical environment.

Select an actual full-checkpoint directory after training, then use identical
test inputs and generation settings for RAW and SFT:

```bash
uv run --extra training bash training/predict.sh \
  Qwen/Qwen3.5-2B data/prepared/test.jsonl runs/raw.jsonl
uv run --extra training bash training/predict.sh \
  /path/to/full-checkpoint data/prepared/test.jsonl runs/sft.jsonl
```

The prediction launcher uses greedy decoding and disables native thinking.
This is distinct from a prompt asking for facts and reasons. Change it only
as an explicitly reported condition. Existing output files are not overwritten.

Continue with the [evaluation guide](evaluation.md).

Framework reference: [MS-SWIFT command-line arguments](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
