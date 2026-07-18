"""Render thesis figures and LaTeX table fragments from the analysis outputs.

Reads only data/signatures/*.csv; writes to out/figures/ and out/tables/.
--only restricts to a comma-separated subset of {e1, e2, e3, correlations};
note the mvb heatmaps pool e1+e3 and are rendered when either is selected.

  mvb_{channel}.pdf          per-model MF x MF JSD heatmaps (Figs 5.3, 5.6, 5.7)
  perturbation_boxplot.pdf   per-topic perturbation biases (Fig 5.4)
  moral_bias.tex             moral vs non-moral by domain (Table 5.1)
  cross_model_rho.tex        Table 5.2
  cross_domain_rho.tex       Table 5.3
  cross_channel_rho.tex      Table 5.4
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm import BASE_DIR, DATA_DIR

SIGNATURES_DIR = DATA_DIR / "signatures"
FIGURES_DIR = BASE_DIR / "out" / "figures"
TABLES_DIR = BASE_DIR / "out" / "tables"


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# -------------------------------------------------------------------- figures

def mvb_heatmaps(mvb: pd.DataFrame) -> None:
    mvb = mvb[(mvb["family"] == "mvb") & (mvb["scheme"] == "raw")]
    for channel, group in mvb.groupby("channel"):
        models = sorted(group["model"].unique())
        foundations = sorted(set(group["partition_a"]) | set(group["partition_b"]))
        n = len(foundations)
        fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4))
        vmax = group["jsd"].max()
        for ax, model in zip(np.atleast_1d(axes), models):
            matrix = np.full((n, n), np.nan)
            for r in group[group["model"] == model].itertuples():
                i, j = foundations.index(r.partition_a), foundations.index(r.partition_b)
                matrix[max(i, j), min(i, j)] = r.jsd
            ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax)
            for r in group[group["model"] == model].itertuples():
                i, j = foundations.index(r.partition_a), foundations.index(r.partition_b)
                ax.text(min(i, j), max(i, j), f"{r.jsd:.2f}{stars(r.p_bonferroni)}",
                        ha="center", va="center", fontsize=6, color="white")
            ax.set_xticks(range(n), foundations, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(n), foundations, fontsize=6)
            ax.set_title(model, fontsize=8)
        fig.suptitle(f"Moral value bias signatures — {channel}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"mvb_{channel}.pdf")
        plt.close(fig)
        print(f"mvb_{channel}.pdf")


def perturbation_boxplot(e2: pd.DataFrame) -> None:
    wide = e2[e2["scope"] == "wide"]
    narrow = e2[e2["scope"] == "8sa"]
    comparisons = list(dict.fromkeys(wide["partition_b"]))
    models = sorted(wide["model"].unique())
    fig, ax = plt.subplots(figsize=(1.9 * len(comparisons) + 1, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    width = 0.8 / len(models)
    for ci, comparison in enumerate(comparisons):
        for mi, model in enumerate(models):
            values = wide[(wide["partition_b"] == comparison) & (wide["model"] == model)]["jsd"]
            x = ci + (mi - len(models) / 2 + 0.5) * width
            ax.boxplot(values, positions=[x], widths=width * 0.85,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors[mi], alpha=0.5), showfliers=False)
            ax.scatter(np.full(len(values), x), values, s=8, color=colors[mi], zorder=3)
        median_8sa = narrow[narrow["partition_b"] == comparison]["jsd"].median()
        ax.hlines(median_8sa, ci - 0.42, ci + 0.42, colors="red", linestyles="dashed")
    ax.set_xticks(range(len(comparisons)), comparisons)
    ax.set_ylabel("JSD vs. baseline")
    ax.legend(handles=[plt.Line2D([], [], color=colors[i], lw=4, label=m)
                       for i, m in enumerate(models)], fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_boxplot.pdf")
    plt.close(fig)
    print("perturbation_boxplot.pdf")


# --------------------------------------------------------------------- tables

def moral_bias_table(e1: pd.DataFrame) -> None:
    mb = e1[e1["family"] == "moral_bias"]
    lines = [
        r"\begin{tabular}{lrrrl rrrl}", r"\toprule",
        r" & \multicolumn{4}{c}{Political} & \multicolumn{4}{c}{Personal-Situational} \\",
        r" & $\bar{s}_M$ & $\bar{s}_{\neg M}$ & $\Delta\bar{s}$ & JSD"
        r" & $\bar{s}_M$ & $\bar{s}_{\neg M}$ & $\Delta\bar{s}$ & JSD \\",
        r"\midrule",
    ]
    for model, group in mb.groupby("model"):
        cells = []
        for channel in ("political", "personal"):
            r = group[group["channel"] == channel].iloc[0]
            cells += [f"{r.mean_a:.2f}", f"{r.mean_b:.2f}",
                      f"{r.mean_a - r.mean_b:+.2f}", f".{f'{r.jsd:.3f}'[2:]}{stars(r.p_bonferroni)}"]
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "moral_bias.tex").write_text("\n".join(lines) + "\n")
    print("moral_bias.tex")


def rho_cell(r) -> str:
    return f"{r.rho:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"


def correlation_tables(corr: pd.DataFrame) -> None:
    # Table 5.2: cross-model matrix per channel
    lines = [r"\begin{tabular}{l" + "r" * 4 + "}", r"\toprule"]
    for channel, group in corr[corr["comparison"] == "cross_model"].groupby("channel"):
        models = sorted(set(group["unit_a"]) | set(group["unit_b"]))
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\textit{{{channel}}}}} \\",
                  " & " + " & ".join(models[:-1]) + r" \\"]
        for m2 in models[1:]:
            cells = []
            for m1 in models[:-1]:
                hit = group[(group["unit_a"] == m1) & (group["unit_b"] == m2)]
                cells.append(f"{hit.iloc[0].rho:+.3f}" if len(hit) else "")
            lines.append(f"{m2} & " + " & ".join(cells) + r" \\")
    pooled = corr[corr["comparison"] == "cross_model_pooled"].iloc[0]
    lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{Pooled mean $\rho$ = {rho_cell(pooled)}}} \\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "cross_model_rho.tex").write_text("\n".join(lines) + "\n")

    # Table 5.3: cross-domain per model
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Model & $\rho$ \\", r"\midrule"]
    for r in corr[corr["comparison"] == "cross_domain"].itertuples():
        lines.append(f"{r.unit_a} & {r.rho:+.3f} \\\\")
    pooled = corr[corr["comparison"] == "cross_domain_pooled"].iloc[0]
    lines += [r"\midrule", rf"Pooled & {rho_cell(pooled)} \\\\", r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "cross_domain_rho.tex").write_text("\n".join(lines) + "\n")

    # Table 5.4: cross-channel pairs
    lines = [r"\begin{tabular}{lrl}", r"\toprule",
             r"Channel pair & $\rho$ & 95\% CI \\", r"\midrule"]
    for r in corr[corr["comparison"] == "cross_channel"].itertuples():
        dagger = r"$^\dagger$" if r.ci_lo * r.ci_hi > 0 else ""
        lines.append(
            rf"{r.unit_a} $\leftrightarrow$ {r.unit_b}{dagger} & {r.rho:+.3f}"
            rf" & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "cross_channel_rho.tex").write_text("\n".join(lines) + "\n")
    print("cross_model_rho.tex, cross_domain_rho.tex, cross_channel_rho.tex")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated experiment names (e1, e2, e3, correlations)")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else {"e1", "e2", "e3", "correlations"}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if only & {"e1", "e3"}:
        # each channel is a separate heatmap, so pooling whatever exists is safe
        mvb_heatmaps(pd.concat(
            pd.read_csv(SIGNATURES_DIR / f"{e}.csv")
            for e in ("e1", "e3") if (SIGNATURES_DIR / f"{e}.csv").exists()
        ))
    if "e1" in only:
        moral_bias_table(pd.read_csv(SIGNATURES_DIR / "e1.csv"))
    if "e2" in only:
        perturbation_boxplot(pd.read_csv(SIGNATURES_DIR / "e2.csv"))
    if "correlations" in only:
        correlation_tables(pd.read_csv(SIGNATURES_DIR / "correlations.csv"))
