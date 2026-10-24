#!/usr/bin/env python
"""Create the figures used in the README and paper-style report."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from neosim.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, SimulationConfig
from neosim.simulator.engine import NeoSimulator

BG = "#08111f"
PANEL = "#0e1c2e"
TEXT = "#e6f0fb"
MUTED = "#8ba4bf"
GRID = "#1d3148"
PINK = "#ff729b"
CYAN = "#58e3c2"
BLUE = "#62a8ff"
AMBER = "#f6c96a"
VIOLET = "#b49bff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.dpi": 160,
        "savefig.dpi": 160,
    }
)


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.35, linewidth=0.6)


def box(ax, x, y, w, h, title, subtitle, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor=color,
        linewidth=0.0,
        alpha=0.16,
    )
    ax.add_patch(patch)
    edge = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="none",
        edgecolor=color,
        linewidth=1.2,
        alpha=0.7,
    )
    ax.add_patch(edge)
    ax.text(
        x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT
    )
    ax.text(x + w / 2, y + h * 0.3, subtitle, ha="center", va="center", fontsize=7.6, color=MUTED)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, "NeoSim 2.0 architecture", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(
        0.02,
        0.905,
        "One conditional flow-matching transformer replaces four per-signal GANs",
        fontsize=8.5,
        color=MUTED,
    )

    box(ax, 0.02, 0.48, 0.16, 0.3, "Public data", "PreMo · PICS · legacy", BLUE)
    box(ax, 0.23, 0.48, 0.17, 0.3, "Window builder", "120 min · 1-min grid\nmissing masks + events", CYAN)
    box(ax, 0.45, 0.48, 0.19, 0.3, "NeoFlow", "4.3M transformer\nrectified flow · CFG", PINK)
    box(ax, 0.69, 0.48, 0.14, 0.3, "Physiology", "response kernels\nthermoregulation", AMBER)
    box(ax, 0.87, 0.48, 0.11, 0.3, "Outputs", "vitals + events\nJSON · CSV · FHIR", VIOLET)

    for x1, x2 in [(0.18, 0.23), (0.40, 0.45), (0.64, 0.69), (0.83, 0.87)]:
        arrow(ax, x1, 0.63, x2, 0.63)

    ax.text(0.02, 0.32, "Conditioning", fontsize=8, color=MUTED, fontweight="bold")
    chips = [
        ("static profile", 0.02, 0.17),
        ("event flags", 0.18, 0.17),
        ("known prefix", 0.33, 0.17),
        ("time / circadian", 0.48, 0.17),
    ]
    for label, x, y in chips:
        patch = FancyBboxPatch(
            (x, y),
            0.14,
            0.09,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=PANEL,
            edgecolor=GRID,
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + 0.07, y + 0.045, label, ha="center", va="center", fontsize=7.4, color=MUTED)
    arrow(ax, 0.55, 0.26, 0.55, 0.47, CYAN)

    ax.text(0.69, 0.36, "The four old GANs", fontsize=8, color=PINK, fontweight="bold")
    old = ["heart", "respiration", "bradycardia", "tachycardia"]
    for i, label in enumerate(old):
        x = 0.69 + i * 0.07
        ax.text(x, 0.3, label, fontsize=6.8, color=MUTED, ha="center")
        ax.plot([x - 0.025, x + 0.025], [0.265, 0.285], color=PINK, linewidth=1.2)
    ax.text(0.93, 0.36, "→ one model", fontsize=8, color=CYAN, fontweight="bold", ha="right")
    fig.savefig(FIGURES_DIR / "architecture.png", bbox_inches="tight")
    plt.close(fig)


def training_curve() -> None:
    path = RESULTS_DIR / "training_history.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    epochs = np.arange(1, len(data["train"]) + 1)
    ax.plot(epochs, data["train"], color=CYAN, linewidth=1.8, label="train")
    ax.plot(epochs, data["val"], color=PINK, linewidth=1.8, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("flow-matching loss")
    ax.set_title("NeoFlow training convergence")
    ax.legend(frameon=False, labelcolor=TEXT)
    _clean(ax)
    fig.savefig(FIGURES_DIR / "training_curve.png", bbox_inches="tight")
    plt.close(fig)


def samples() -> None:
    path = RESULTS_DIR / "synthetic_eval.npz"
    if not path.exists():
        return
    data = np.load(path)
    xs = data["premo_x"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    for i in range(3):
        row = xs[i]
        t = np.arange(120)
        axes[0, i].plot(t, row[:, 0], color=PINK, linewidth=1.2)
        axes[0, i].set_ylim(60, 220)
        axes[0, i].set_ylabel("HR (bpm)", color=PINK)
        axes[0, i].set_title(f"Generated stay {i + 1}", color=TEXT)
        _clean(axes[0, i])
        axes[1, i].plot(t, row[:, 2], color=CYAN, linewidth=1.1)
        axes[1, i].set_ylim(55, 101)
        axes[1, i].set_ylabel("SpO$_2$ (%)", color=CYAN)
        axes[1, i].set_xlabel("minutes")
        _clean(axes[1, i])
    fig.suptitle("Two-hour synthetic vital-sign windows", color=TEXT, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "samples.png", bbox_inches="tight")
    plt.close(fig)


def distributions() -> None:
    if not (RESULTS_DIR / "synthetic_eval.npz").exists():
        return
    synth = np.load(RESULTS_DIR / "synthetic_eval.npz")
    premo = np.load(PROCESSED_DIR / "premo_test.npz")
    pics = np.load(PROCESSED_DIR / "pics_test.npz")
    panels = [
        (
            "Heart rate",
            premo["x"][:, :, 0][premo["mask"][:, :, 0] > 0],
            synth["premo_x"][:, :, 0].ravel(),
            PINK,
            "bpm",
        ),
        (
            "Respiratory rate",
            pics["x"][:, :, 1][pics["mask"][:, :, 1] > 0],
            synth["pics_x"][:, :, 1].ravel(),
            BLUE,
            "breaths/min",
        ),
        (
            "SpO$_2$",
            premo["x"][:, :, 2][premo["mask"][:, :, 2] > 0],
            synth["premo_x"][:, :, 2].ravel(),
            CYAN,
            "%",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (title, real, fake, color, unit) in zip(axes, panels, strict=False):
        bins = np.linspace(min(real.min(), fake.min()), max(real.max(), fake.max()), 45)
        ax.hist(real, bins=bins, density=True, color=color, alpha=0.38, label="real")
        ax.hist(fake, bins=bins, density=True, histtype="step", color=color, linewidth=1.8, label="NeoFlow")
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_yticks([])
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=7.5)
        _clean(ax)
    fig.suptitle("Held-out distributions versus generated cohorts", color=TEXT, fontweight="bold", y=1.03)
    fig.savefig(FIGURES_DIR / "distributions.png", bbox_inches="tight")
    plt.close(fig)


def regimes() -> None:
    sim = NeoSimulator(allow_fallback=True)
    regimes = [
        ("stable_preterm", 30.0, 1250, 0.2, "Stable"),
        ("bradycardia", 26.0, 820, 0.7, "Bradycardia"),
        ("apnea_of_prematurity", 28.0, 1050, 0.6, "Apnoea"),
        ("rds", 27.0, 900, 0.75, "RDS"),
        ("sepsis", 34.0, 2100, 0.65, "Sepsis"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), sharey=True)
    for ax, (condition, ga, weight, severity, label) in zip(axes, regimes, strict=False):
        result = sim.simulate(
            SimulationConfig(
                condition=condition,
                gestational_age_weeks=ga,
                birth_weight_g=weight,
                severity=severity,
                shift_hours=4,
                interval_minutes=1,
                seed=11,
            )
        )
        t = np.arange(len(result.arrays["hr"]))
        ax.plot(t, result.arrays["hr"], color=PINK, linewidth=1.0)
        ax.set_ylim(55, 220)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("min")
        _clean(ax)
        ax2 = ax.twinx()
        ax2.plot(t, result.arrays["spo2"], color=CYAN, linewidth=0.9, alpha=0.9)
        ax2.set_ylim(55, 101)
        ax2.set_yticks([])
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    axes[0].set_ylabel("HR (bpm)")
    fig.suptitle(
        "Clinical regime conditioning (HR pink, SpO$_2$ cyan)", color=TEXT, fontweight="bold", y=1.04
    )
    fig.savefig(FIGURES_DIR / "regimes.png", bbox_inches="tight")
    plt.close(fig)


def evaluation_summary() -> None:
    path = RESULTS_DIR / "evaluation.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    labels, ks, auc = [], [], []
    for dataset in ("premo", "pics"):
        for channel, metrics in data[dataset].items():
            if isinstance(metrics, dict) and "distribution" in metrics:
                labels.append(f"{dataset.upper()} {channel.upper()}")
                ks.append(metrics["distribution"]["ks"])
        if "discriminative_auc" in data[dataset]:
            auc.append((dataset.upper(), data[dataset]["discriminative_auc"]["auc"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [2.3, 1]})
    colors = [PINK, BLUE, CYAN, VIOLET, AMBER][: len(labels)]
    ax1.barh(labels, ks, color=colors, alpha=0.8)
    ax1.set_xlim(0, max(ks) * 1.2 if ks else 1)
    ax1.set_xlabel("KS distance (lower is better)")
    ax1.set_title("Marginal distribution fidelity")
    _clean(ax1)
    if auc:
        names = [a[0] for a in auc]
        values = [0.5 if np.isnan(a[1]) else a[1] for a in auc]
        ax2.bar(names, values, color=[CYAN, VIOLET], alpha=0.85, width=0.55)
        ax2.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("AUC (0.5 = indistinguishable)")
        ax2.set_title("Real-vs-synthetic classifier")
        _clean(ax2)
    fig.savefig(FIGURES_DIR / "evaluation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    print("architecture")
    training_curve()
    print("training_curve")
    samples()
    print("samples")
    distributions()
    print("distributions")
    regimes()
    print("regimes")
    evaluation_summary()
    print("evaluation")
    print(f"figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
