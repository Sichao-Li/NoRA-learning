"""Plot the saved learning and transfer scores without rerunning evaluation."""

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ("2b", "4b", "9b", "27b")
COLORS = ("#BAC4CE", "#286B8C")


def read_csv(name):
    with (ROOT / "results" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def score(rows, benchmark, metric, size, condition, dimension="overall", level="all",
          scope="official", arm="official"):
    matches = [row for row in rows if
               (row["benchmark"], row["metric"], row["model"], row["dimension"],
                row["level"], row["scope"], row["arm"]) ==
               (benchmark, metric, f"qwen35_{size}_{condition}", dimension, level, scope, arm)]
    if len(matches) != 1:
        raise ValueError(f"Expected one score: {benchmark}, {metric}, {size}, {condition}, {level}")
    return float(matches[0]["value"])


def decorate(ax, title, maximum, ticks):
    ax.set_title(title, loc="left", fontsize=11, pad=13)
    ax.set(ylim=(0, maximum), yticks=ticks, xticks=range(4),
           xticklabels=[size.upper() for size in SIZES], xlim=(-.6, 3.6))
    ax.tick_params(axis="both", length=0, pad=7)
    ax.grid(axis="y", color="#E6ECEF", lw=.7, zorder=0)
    ax.axhline(0, color="#697780", lw=.9, zorder=2)


def bars(ax, values):
    for index, condition in enumerate(("raw", "sft")):
        positions = [x + (index - .5) * .36 for x in range(4)]
        ax.bar(positions, values[condition], width=.32, color=COLORS[index],
               label=condition.upper(), zorder=3)


def save(fig, axes, output, name, ylabel):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.005),
               ncol=2, frameon=False, fontsize=11, handlelength=1.4)
    fig.supylabel(ylabel, x=.005)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def learning(rows, output):
    metrics = (("nora_soft_action_f1", "(a) Action alignment"),
               ("nora_soft_fact_f1", "(b) Factual grounding"),
               ("nora_soft_reasoning_f1", "(c) Support binding"),
               ("nora_soft_reasonableness_score", "(d) Soft reasonableness"),
               ("nora_hungarian_reasonableness_score", "(e) Hungarian reasonableness"))
    fig = plt.figure(figsize=(8.2, 5.2))
    grid = fig.add_gridspec(2, 6, left=.075, right=.99, bottom=.1, top=.86,
                            wspace=1.05, hspace=.7)
    axes = [fig.add_subplot(grid[0, i:i+2]) for i in (0, 2, 4)]
    axes += [fig.add_subplot(grid[1, i:i+2]) for i in (1, 3)]
    for ax, (metric, title) in zip(axes, metrics):
        bars(ax, {state: [score(rows, "nora", metric, size, state) for size in SIZES]
                  for state in ("raw", "sft")})
        decorate(ax, title, .6, [0, .2, .4, .6])
    save(fig, axes, output, "learning_results", "Mean score")


def morebench(output):
    rows = read_csv("morebench_dimensions.csv")
    panels = (("clear process", "positive", "(a) Clear process"),
              ("helpful outcome", "positive", "(b) Helpful outcome"),
              ("harmless outcome", "negative_avoided", "(c) Harm avoidance"))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    fig.subplots_adjust(left=.065, right=.99, bottom=.14, top=.77, wspace=.2)
    for ax, (dimension, polarity, title) in zip(axes, panels):
        values = {}
        for state in ("raw", "sft"):
            values[state] = []
            for size in SIZES:
                selected = [r for r in rows if (r["dimension"], r["polarity"], r["model"]) ==
                            (dimension, polarity, f"qwen35_{size}_{state}")]
                if len(selected) != 1:
                    raise ValueError("Missing or repeated MoReBench criterion rate")
                values[state].append(100 * float(selected[0]["value"]))
        bars(ax, values)
        decorate(ax, title, 100, [0, 25, 50, 75, 100])
    save(fig, axes, output, "morebench_raw_sft", "Rate (%)")


def safety(rows, output):
    panels = (("harmbench", "harmbench_attack_success_rate", "(a) HarmBench ASR (lower is better)", 15, [0, 5, 10, 15]),
              ("xstest", "xstest_unsafe_any_refusal_rate", "(b) Unsafe refusal", 100, [0, 25, 50, 75, 100]),
              ("xstest", "xstest_safe_compliance_rate", "(c) Safe compliance", 100, [0, 25, 50, 75, 100]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fig.subplots_adjust(left=.065, right=.99, bottom=.14, top=.77, wspace=.3)
    for ax, (benchmark, metric, title, maximum, ticks) in zip(axes, panels):
        bars(ax, {state: [100 * score(rows, benchmark, metric, size, state) for size in SIZES]
                  for state in ("raw", "sft")})
        decorate(ax, title, maximum, ticks)
    save(fig, axes, output, "harmbench_xstest_raw_sft", "Rate (%)")


def mssbench(rows, output):
    panels = (("chat|safe", "(a) Chat-safe", 116, [0, 50, 100]),
              ("chat|unsafe", "(b) Chat-unsafe", 66, [0, 20, 40, 60]),
              ("embodied|safe", "(c) Embodied-safe", 116, [0, 50, 100]),
              ("embodied|unsafe", "(d) Embodied-unsafe", 10, [0, 5, 10]))
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.9), sharey=True)
    fig.subplots_adjust(left=.055, right=.99, bottom=.18, top=.78, wspace=.2)
    for ax, (level, title, maximum, ticks) in zip(axes, panels):
        for y in (0, 2):
            ax.axhspan(y - .43, y + .43, color="#F5F7F9", lw=0, zorder=0)
        for index, state in enumerate(("raw", "sft")):
            values = [100 * score(rows, "mssbench", "mssbench_accuracy", size, state,
                                 "task_x_context", level, "joint_official_nora_cot", "official_nora_cot")
                      for size in SIZES]
            positions = [y + (index - .5) * .34 for y in range(4)]
            ax.barh(positions, values, height=.28, color=COLORS[index], label=state.upper(), zorder=3)
            for y, value in zip(positions, values):
                ax.text(value + maximum * .018, y, f"{value:.1f}", va="center", fontsize=9)
        ax.set(xlim=(0, maximum), ylim=(3.5, -.5), xticks=ticks,
               yticks=range(4), yticklabels=[size.upper() for size in SIZES])
        ax.tick_params(axis="both", length=0, pad=7)
        ax.grid(axis="x", color="#E1E7EC", lw=.6, zorder=1)
        ax.set_title(title, loc="left", fontsize=11, pad=13)
    save(fig, axes, output, "mssbench_nora_cot_raw_sft", "Qwen3.5 size")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#26343C", "axes.labelcolor": "#26343C",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    rows = read_csv("scores.csv")
    learning(rows, args.output)
    morebench(args.output)
    safety(rows, args.output)
    mssbench(rows, args.output)
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
