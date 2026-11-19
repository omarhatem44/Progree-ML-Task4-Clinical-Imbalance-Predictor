"""Figures for Task 4, drawn in a plain journal style: no gridlines, no titles
inside the image, boxed and lettered panels, and series labelled directly."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

MODELS = ("SVM", "Random Forest", "XGBoost")
STRATEGIES = ("None", "Class weights", "SMOTE")

MODEL_COLORS = {"SVM": "#0072B5", "Random Forest": "#20854E", "XGBoost": "#BC3C29"}
STRATEGY_COLORS = {"None": "#808080", "Class weights": "#0072B5", "SMOTE": "#E18727"}
STRATEGY_TEXT = {"None": "unadjusted", "Class weights": "class weights", "SMOTE": "SMOTE"}
GREY = "#808080"
RED = "#BC3C29"
WIDTH = {"single": 3.35, "onehalf": 5.0, "double": 7.0}

SHORT_LABELS = {
    "LB": "LB (baseline FHR)",
    "AC": "AC (accelerations)",
    "FM": "FM (fetal movements)",
    "UC": "UC (uterine contractions)",
    "DL": "DL (light decelerations)",
    "DS": "DS (severe decelerations)",
    "DP": "DP (prolonged decelerations)",
    "ASTV": "ASTV (abnormal STV, % of time)",
    "MSTV": "MSTV (mean STV)",
    "ALTV": "ALTV (abnormal LTV, % of time)",
    "MLTV": "MLTV (mean LTV)",
    "Width": "Width (histogram width)",
    "Min": "Min (histogram minimum)",
    "Max": "Max (histogram maximum)",
    "Nmax": "Nmax (histogram peaks)",
    "Nzeros": "Nzeros (histogram zeros)",
    "Mode": "Mode (histogram mode)",
    "Mean": "Mean (histogram mean)",
    "Median": "Median (histogram median)",
    "Variance": "Variance (histogram variance)",
    "Tendency": "Tendency (histogram tendency)",
}


def config_label(model: str, strategy: str) -> str:
    return f"{model}, {STRATEGY_TEXT[strategy]}"


def use_style(base: float = 8.0) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": base,
        "axes.labelsize": base + 0.5,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


class Panels:
    """A figure split into equal boxed cells, one lettered panel per cell."""

    def __init__(self, nrows=1, ncols=1, width="double", panel_height=2.6):
        self.nrows, self.ncols = nrows, ncols
        self.w_in = WIDTH[width] if isinstance(width, str) else float(width)
        self.h_in = nrows * panel_height
        self.fig = plt.figure(figsize=(self.w_in, self.h_in))
        for i in range(nrows * ncols):
            x0, y0, cw, ch = self.cell(i)
            self.fig.add_artist(Rectangle((x0, y0), cw, ch, transform=self.fig.transFigure,
                                          fill=False, lw=0.8, edgecolor="black", zorder=0))

    def cell(self, i):
        r, c = divmod(i, self.ncols)
        cw, ch = 1.0 / self.ncols, 1.0 / self.nrows
        return c * cw, 1.0 - (r + 1) * ch, cw, ch

    def axes(self, i, left=0.17, right=0.07, bottom=0.18, top=0.20):
        x0, y0, cw, ch = self.cell(i)
        return self.fig.add_axes([x0 + left * cw, y0 + bottom * ch,
                                  (1 - left - right) * cw, (1 - top - bottom) * ch])

    def label(self, i, letter, title="", size=9):
        x0, y0, cw, ch = self.cell(i)
        px, py = 0.13 / self.w_in, 0.16 / self.h_in
        self.fig.text(x0 + px, y0 + ch - py, letter, weight="bold", size=10, ha="left", va="top")
        if title:
            self.fig.text(x0 + px + 0.19 / self.w_in, y0 + ch - py, title,
                          weight="bold", size=size, ha="left", va="top")


def _tidy(ax, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, weight="bold")


def _finish(fig, path):
    if path is not None:
        fig.savefig(path, dpi=300, format="png", metadata={"Software": None})
    return fig


def _text_block(ax, x, y0, lines, ha, dy=0.066):
    """Stack (text, colour, weight) lines upward from y0, in axes coordinates."""
    for k, (text, color, weight) in enumerate(reversed(lines)):
        ax.text(x, y0 + k * dy, text, transform=ax.transAxes, ha=ha, va="bottom",
                color=color, fontsize=6.8, weight=weight)


# --------------------------------------------------------------------------
# Figure 1: class distribution
# --------------------------------------------------------------------------

def plot_class_distribution(nsp_counts: dict, path=None):
    use_style()
    names = ["Normal", "Suspect", "Pathologic"]
    counts = [nsp_counts[1], nsp_counts[2], nsp_counts[3]]
    total = sum(counts)
    fig = plt.figure(figsize=(WIDTH["single"], 2.8))
    ax = fig.add_axes([0.21, 0.30, 0.74, 0.64])
    ax.bar(range(3), counts, width=0.62, color=[GREY, GREY, RED], edgecolor="none")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.015 * max(counts), f"{c:,}\n({100 * c / total:.1f}%)",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, max(counts) * 1.25)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    neg = counts[0] + counts[1]
    for x0, x1, text in [(-0.31, 1.31, f"Negative class\n(n={neg:,})"),
                         (1.69, 2.31, f"Positive class\n(n={counts[2]:,})")]:
        ax.plot([x0, x1], [-0.14, -0.14], transform=trans, color="black", lw=0.6, clip_on=False)
        ax.text((x0 + x1) / 2, -0.18, text, transform=trans, ha="center", va="top", fontsize=7)
    _tidy(ax, None, "No. of CTG Records")
    return _finish(fig, path)


# --------------------------------------------------------------------------
# Figure 2: cross-validation arrays
# --------------------------------------------------------------------------

def plot_cv_folds(summaries: dict, final_key, floor: float, path=None):
    use_style()
    layout = []
    for m in MODELS:
        layout.append(("head", m))
        layout += [("row", (m, s)) for s in STRATEGIES]
    n = len(layout)
    ys = [n - 1 - i for i in range(n)]
    P = Panels(1, 2, "double", 3.4)
    specs = [("recall", "A", "Recall (Sensitivity)", "Recall in Each Validation Fold"),
             ("precision", "B", "Precision (Positive Predictive Value)",
              "Precision in Each Validation Fold")]
    for j, (metric, letter, title, xlabel) in enumerate(specs):
        ax = P.axes(j, left=0.31 if j == 0 else 0.07, right=0.25, bottom=0.15, top=0.12)
        P.label(j, letter, title)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        values = []
        for (kind, item), y in zip(layout, ys):
            if kind == "head":
                continue
            s = summaries[item]
            folds = np.asarray(s["cv_folds"][metric])
            mean, sd = s["cv_mean"][metric], s["cv_sd"][metric]
            values.extend(folds)
            c = STRATEGY_COLORS[item[1]]
            ax.scatter(folds, y + np.linspace(-0.22, 0.22, folds.size), s=11,
                       facecolors="white", edgecolors=c, linewidths=0.8, zorder=3, clip_on=False)
            ax.errorbar(mean, y, xerr=sd, fmt="D", ms=4.2, color=c, mfc=c, mec=c,
                        elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)
            ax.text(1.04, y, f"{mean:.3f} ± {sd:.3f}", transform=trans, ha="left", va="center",
                    fontsize=7, weight="bold" if item == final_key else "normal")
        ax.text(1.04, ys[0], "Mean ± SD", transform=trans, ha="left", va="center",
                fontsize=7, weight="bold")
        lo = np.floor((min(values) - 0.02) * 20) / 20
        if metric == "precision":
            lo = min(lo, floor - 0.05)
            ax.axvline(floor, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.text(floor, ys[0], " Floor", color=GREY, fontsize=7, ha="left", va="center")
        ax.set_xlim(lo, 1.0)
        ax.set_ylim(-0.7, n - 0.4)
        if j == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([item if kind == "head" else item[1] for kind, item in layout])
            for tick, (kind, _) in zip(ax.get_yticklabels(), layout):
                if kind == "head":
                    tick.set_fontweight("bold")
        else:
            ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        _tidy(ax, xlabel)
    return _finish(P.fig, path)


# --------------------------------------------------------------------------
# Figure 3: ROC and precision-recall curves on the test set
# --------------------------------------------------------------------------

def plot_test_curves(curves: dict, tests: dict, labels: dict, prevalence: float, path=None):
    use_style()
    P = Panels(1, 2, "double", 3.4)

    ax = P.axes(0, left=0.17, right=0.06, bottom=0.15, top=0.13)
    P.label(0, "A", "ROC Curve, Test Set")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    lines = [("AUC (95% CI)", "black", "bold")]
    for m in MODELS:
        c, col, t = curves[m], MODEL_COLORS[m], tests[m]
        ax.plot(c["fpr"], c["tpr"], color=col, lw=1.4, zorder=3)
        op = c["operating_point"]
        ax.plot(op["fpr"], op["tpr"], "o", ms=4.5, color=col, mec="white", mew=0.7, zorder=5, clip_on=False)
        lo, hi = t["roc_auc_ci95"]
        lines += [(labels[m], col, "bold"), (f"{t['roc_auc']:.3f} ({lo:.3f}–{hi:.3f})", col, "normal")]
    _text_block(ax, 0.97, 0.04, lines, "right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    _tidy(ax, "1 − Specificity", "Sensitivity")
    # Magnified top-left corner, where the three curves differ
    ins = ax.inset_axes([0.40, 0.57, 0.46, 0.36])
    for m in MODELS:
        c, col = curves[m], MODEL_COLORS[m]
        ins.plot(c["fpr"], c["tpr"], color=col, lw=1.2, zorder=3)
        op = c["operating_point"]
        ins.plot(op["fpr"], op["tpr"], "o", ms=4, color=col, mec="white", mew=0.6, zorder=5, clip_on=False)
    ins.set_xlim(0, 0.15)
    ins.set_ylim(0.8, 1.005)
    ins.set_xticks([0, 0.05, 0.10, 0.15])
    ins.set_yticks([0.8, 0.9, 1.0])
    ins.tick_params(labelsize=6, length=2.5)
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)

    ax = P.axes(1, left=0.17, right=0.06, bottom=0.15, top=0.13)
    P.label(1, "B", "Precision–Recall Curve, Test Set")
    ax.axhline(prevalence, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.03, prevalence + 0.012, f"Prevalence {100 * prevalence:.1f}%", color=GREY,
            fontsize=6.8, ha="left", va="bottom",
            transform=blended_transform_factory(ax.transAxes, ax.transData))
    lines = [("Average precision (95% CI)", "black", "bold")]
    for m in MODELS:
        c, col, t = curves[m], MODEL_COLORS[m], tests[m]
        ax.plot(c["recall"], c["precision"], color=col, lw=1.4, drawstyle="steps-post", zorder=3)
        op = c["operating_point"]
        ax.plot(op["tpr"], op["precision"], "o", ms=4.5, color=col, mec="white", mew=0.7, zorder=5, clip_on=False)
        lo, hi = t["average_precision_ci95"]
        lines += [(labels[m], col, "bold"),
                  (f"{t['average_precision']:.3f} ({lo:.3f}–{hi:.3f})", col, "normal")]
    _text_block(ax, 0.03, 0.17, lines, "left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    _tidy(ax, "Recall (Sensitivity)", "Precision")
    return _finish(P.fig, path)


# --------------------------------------------------------------------------
# Figure 4: confusion matrices on the test set
# --------------------------------------------------------------------------

def plot_confusion_matrices(tests: dict, labels: dict, path=None):
    use_style()
    cmap = LinearSegmentedColormap.from_list("rate", ["#FFFFFF", "#0072B5"])
    P = Panels(1, 3, "double", 2.6)
    names = ["Not\npathologic", "Pathologic"]
    for j, m in enumerate(MODELS):
        t = tests[m]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        rates = cm / cm.sum(axis=1, keepdims=True)
        ax = P.axes(j, left=0.33, right=0.06, bottom=0.27, top=0.17)
        P.label(j, "ABC"[j], labels[m], size=8)
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for r in range(2):
            for c in range(2):
                col = "white" if rates[r, c] > 0.6 else "black"
                ax.text(c, r - 0.1, f"{cm[r, c]:,}", ha="center", va="center",
                        fontsize=9, weight="bold", color=col)
                ax.text(c, r + 0.22, f"{100 * rates[r, c]:.1f}%", ha="center", va="center",
                        fontsize=7, color=col)
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=0.6, edgecolor="black"))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(names)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(names)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predicted Class", weight="bold")
        ax.set_ylabel("True Class", weight="bold")
    return _finish(P.fig, path)


# --------------------------------------------------------------------------
# Figure 5: permutation importance of the final model
# --------------------------------------------------------------------------

def plot_importance(imp, color, top: int = 10, path=None):
    use_style()
    d = imp.head(top).iloc[::-1]
    fig = plt.figure(figsize=(WIDTH["onehalf"], 3.0))
    ax = fig.add_axes([0.37, 0.16, 0.58, 0.80])
    y = np.arange(len(d))
    ax.barh(y, d["importance_mean"], xerr=d["importance_sd"], height=0.62, color=color,
            edgecolor="none", error_kw=dict(elinewidth=0.9, capsize=2.2, capthick=0.9, ecolor="black"))
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS[f] for f in d["feature"]])
    ax.tick_params(axis="y", length=0)
    left = min(0.0, float((d["importance_mean"] - d["importance_sd"]).min()))
    ax.set_xlim(left, None)
    ax.axvline(0, color="black", lw=0.8)
    ax.spines["left"].set_visible(False)
    _tidy(ax, "Decrease in Average Precision after Permutation")
    return _finish(fig, path)
