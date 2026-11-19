# Normative Reasoning from Humans to Vision-Language Models

Code and results for the learning and transfer study extending
[NoRA](https://github.com/mint-philosophy/NoRA-benchmark).
We study Qwen3.5 at 2B, 4B, 9B, and 27B, comparing each original checkpoint
(RAW) with its fine-tuned counterpart (SFT).

[Results](results/README.md) |
[Training](docs/training.md) |
[Evaluation](docs/evaluation.md) |
[Dataset](https://huggingface.co/datasets/MINTLABJHUANU/NoRA)

## Findings

- On the shared valid NoRA subset, SFT improves action alignment, factual
  grounding, support binding, and soft and Hungarian reasonableness at every size.
- MoReBench gains occur in selected aspects of written responses, not uniformly
  in the overall weighted score.
- Lower harmful compliance on HarmBench accompanies more refusal of harmless
  requests on XSTest. The sources of the HarmBench reduction differ by size.
- MSSBench gains concentrate in unsafe visual chat under NoRA-style prompting,
  with losses on safe cases and weak unsafe embodied performance.

These are comparisons on shared scoreable cases, not full-test estimates of
general normative competence. The [results guide](results/README.md) explains
the reference, prompts, evaluators, and denominators.

## Repository

```text
results/         Reported scores, coverage, and saved intervals
training/        Data preparation and MS-SWIFT training/inference commands
evaluation/      Conversion of saved responses for the NoRA scorer
figures/         Plots of the reported learning and transfer results
docs/            Training and evaluation guides
```

The NoRA dataset and scorer are maintained in their original repositories.
This repository does not duplicate them or distribute benchmark videos,
model checkpoints, or raw model responses.

## Install

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). From this repository:

```bash
uv sync
```

This creates a repository-local `.venv`. Training and semantic scoring have
separate optional dependencies:

```bash
uv sync --extra training --extra scorer
```

Training requires a suitable GPU environment. The commands below are for new
runs using the public annotations; they do not recreate the paper's historical
training subset or replace its reported scores.

## Train and Evaluate

1. Prepare annotations and pre-action montages:

   ```bash
   uv run python training/prepare_data.py --media-root data/media \
     --download-media --output data/prepared
   ```

2. Fine-tune a model, then generate RAW and SFT responses with the same inputs:

   ```bash
   uv run --extra training bash training/train.sh \
     Qwen/Qwen3.5-2B data/prepared runs/2b-sft
   uv run --extra training bash training/predict.sh \
     Qwen/Qwen3.5-2B data/prepared/test.jsonl runs/2b-raw.jsonl
   uv run --extra training bash training/predict.sh \
     /path/to/full-checkpoint data/prepared/test.jsonl runs/2b-sft.jsonl
   ```

3. Convert the JSON responses and score them:

   ```bash
   uv run python evaluation/convert_predictions.py \
     --input runs/2b-raw.jsonl --inputs data/prepared/test.jsonl \
     --output runs/2b-raw-annotations
   uv run --extra scorer nora evaluate \
     --predictions runs/2b-raw-annotations/predictions.jsonl \
     --references data/prepared/references.jsonl --output runs/2b-raw-scores
   ```

Repeat step 3 for SFT, then compare the same valid clips:

```bash
uv run nora compare runs/2b-raw-scores runs/2b-sft-scores
```

The guides describe checkpoint selection, input formats, failed responses,
and the distinction between new runs and the paper results. Evaluation does
not use a paid reconstruction model when predictions already contain JSON.

## Plot the Reported Results

```bash
make figures
```

This reads the included score tables and writes PDF and PNG plots to
`build/figures/`. It does not run models, call judges, or calculate new scores
or uncertainty intervals.

## Citation and Use

The manuscript is **Normative Reasoning from Humans to Vision-Language Models**.
Citation metadata is provided in [CITATION.cff](CITATION.cff). For the underlying
benchmark, also cite the [NoRA paper](https://arxiv.org/abs/2606.04806).

The [MIT license](LICENSE) covers the code. NoRA annotations and source media have their own licenses
and access conditions; this repository does not change them.
