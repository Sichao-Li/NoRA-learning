# Normative Reasoning from Humans to Vision-Language Models

Code and results for the learning and transfer study extending
[NoRA](https://github.com/mint-philosophy/NoRA-benchmark).
We fine-tune Qwen3.5 at 2B, 4B, 9B, and 27B on NoRA annotations and compare
each original checkpoint (RAW) with its fine-tuned counterpart (SFT) on NoRA
and on external moral and safety benchmarks.

## Workflow

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Steps 2 and 3 need a GPU.

```bash
git clone https://github.com/Sichao-Li/NoRA-learning.git
cd NoRA-learning
uv sync --extra training --extra scorer
```

**1. Prepare data.** Train and validation examples come from LLMSilver;
HumanGold is kept for testing.

```bash
uv run python training/prepare_data.py \
  --media-root data/media --download-media --output data/prepared
```

**2. Fine-tune.** Replace the model name for the 4B, 9B, or 27B size.

```bash
uv run --extra training bash training/train.sh \
  Qwen/Qwen3.5-2B data/prepared runs/2b-sft
```

**3. Predict.** RAW and SFT use the same test inputs and generation settings.

```bash
uv run --extra training bash training/predict.sh \
  Qwen/Qwen3.5-2B data/prepared/test.jsonl runs/raw.jsonl
uv run --extra training bash training/predict.sh \
  /path/to/checkpoint data/prepared/test.jsonl runs/sft.jsonl
```

**4. Score.** Convert the responses and score them with the NoRA evaluator.
Repeat for SFT, then compare the two runs on their shared valid clips.

```bash
uv run python evaluation/convert_predictions.py \
  --input runs/raw.jsonl --inputs data/prepared/test.jsonl \
  --output runs/raw-annotations
uv run --extra scorer nora evaluate \
  --predictions runs/raw-annotations/predictions.jsonl \
  --references data/prepared/references.jsonl \
  --output runs/raw-scores --progress
uv run nora compare runs/raw-scores runs/sft-scores
```

This workflow uses the public NoRA annotations and evaluator. The scores
reported in the paper are saved in [`results/`](results/).

## Training Settings

All four sizes use the same settings, set in [`training/train.sh`](training/train.sh).

| Setting | Value |
| --- | --- |
| Supervision | GPT-5.4 annotations from LLMSilver-1230 |
| Mode | MS-SWIFT full SFT |
| Epochs / learning rate | 3 / 1e-5 |
| Batch size / gradient accumulation | 1 / 16 |
| Precision / sequence limit | BF16 / 4,096 tokens |
| Image-token limit | 1,024 |
| Warmup ratio | 0.05 |
| Validation split | 5%, seed 42 |

## Transfer Benchmarks

The same RAW and SFT checkpoints are evaluated with each benchmark's own
tasks and evaluator:
[MoReBench](https://github.com/morebench/morebench),
[HarmBench](https://github.com/centerforaisafety/HarmBench),
[XSTest](https://github.com/paul-rottger/xstest), and
[MSSBench](https://openreview.net/forum?id=I9bEi6LNgt), with supplementary
comparisons on ETHICS, MoralChoice, MoralBench, NormLens, BBQ, and IFEval.

## Results

The reported scores are in [`results/`](results/README.md) and the plots are
in [`figures/`](figures/). To redraw the plots from the saved scores:

```bash
uv sync
make figures
```

## Checkpoints

The fine-tuned (SFT) checkpoints are on Hugging Face. Each model card gives
the loading code and the prompt.

| Size | Model |
| --- | --- |
| 2B | [Qwen3.5-2B-NoRA](https://huggingface.co/Lester1996/Qwen3.5-2B-NoRA) |
| 4B | [Qwen3.5-4B-NoRA](https://huggingface.co/Lester1996/Qwen3.5-4B-NoRA) |
| 9B | [Qwen3.5-9B-NoRA](https://huggingface.co/Lester1996/Qwen3.5-9B-NoRA) |
| 27B| [Qwen3.5-27B-NoRA](https://huggingface.co/Lester1996/Qwen3.5-27B-NoRA) | 

The RAW models are the original
[Qwen3.5](https://huggingface.co/Qwen) checkpoints.

## Repository

```text
training/        Data preparation and MS-SWIFT training and prediction commands
evaluation/      Conversion of saved responses for the NoRA scorer
results/         Reported scores, coverage, and saved intervals
figures/         Plots of the reported learning and transfer results
```

The NoRA [dataset](https://huggingface.co/datasets/MINTLABJHUANU/NoRA) and
scorer are maintained in their original repositories. Benchmark videos and
raw model responses are not distributed here.
