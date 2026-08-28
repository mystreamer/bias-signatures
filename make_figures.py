"""Render thesis figures and LaTeX table fragments from the analysis outputs.

Reads data/signatures/*.csv (the cleveland stage instead reads
data/responses/e1/ and data/components/issues.json); writes to out/figures/
and out/tables/.
--only restricts to a comma-separated subset of {e1, e2, e3, correlations, cleveland}.
--export copies everything in out/ into thesis/figures/bias_signatures/.

  mvb_{experiment}.pdf       MF x MF JSD heatmaps, models x EMs (Figs 5.3, 5.6, 5.7)
  perturbation_boxplot.pdf   per-topic perturbation biases (Fig 5.4)
  stance_cleveland_{political,personal}.pdf  per-model mean 8SA stance on the curated E2 topics
  {mfq30,mfq2}_per_foundation_model_likert.pdf  E3 Likert counts (appendix)
  moral_bias.tex             moral vs non-moral by domain (Table 5.1)
  stance_bin_counts.tex      E1 stance bin counts by domain and issue type (Table A.1)
  sanctity_stance_distribution.tex  E1 personal sanctity stance bin counts (appendix)
  perturbation_biases.tex    per-(topic, model) perturbation JSDs (appendix)
  cross_model_rho.tex        Table 5.2
  cross_domain_rho.tex       Table 5.3
  cross_em_rho.tex           Table 5.4
"""

from __future__ import annotations

import json
import shutil
import textwrap

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constructs import INVALID, MFQ2_SUPPORT, MFQ30_SUPPORT, REFUSAL
from llm import BASE_DIR, DATA_DIR
from make_signatures import load_responses

SIGNATURES_DIR = DATA_DIR / "signatures"
FIGURES_DIR = BASE_DIR / "out" / "figures"
TABLES_DIR = BASE_DIR / "out" / "tables"
EXPORT_DIR = BASE_DIR.parent / "thesis" / "figures" / "bias_signatures"

# Figures are included at width=\textwidth (448.13pt, scrreprt at 12pt).
# Rendering at true print width makes figure points equal points on the page.
TEXTWIDTH_IN = 448.13095 / 72.27
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
})


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# thesis display conventions: row order and names as hand-set in chapter 5
MODEL_DISPLAY = {"llama-3.3-70b": "Llama-3.3-70B", "kimi-k2.5": "Kimi-K2.5",
                 "llama-3.1-8b": "Llama-3.1-8B", "qwen-2.5-72b": "Qwen-72B",
                 "deepseek-r1": "DeepSeek-R1"}
MODEL_ORDER = list(MODEL_DISPLAY)
EM_DISPLAY = {"political": "Task political", "personal": "Task personal",
              "mfq30": "MFQ-30", "mfq2": "MFQ-2"}
STARS_NOTE = r"$*\,p<0.05$, $**\,p<0.01$, $***\,p<0.001$ (permutation test)"


# -------------------------------------------------------------------- figures

def mvb_heatmaps(mvb: pd.DataFrame, experiment: str) -> None:
    mvb = mvb[(mvb["family"] == "mvb") & (mvb["scheme"] == "raw")]
    ems = sorted(mvb["em"].unique(), key=lambda c: (c != "political", c))
    models = sorted(mvb["model"].unique())
    vmax = mvb["jsd"].max()
    fig, axes = plt.subplots(len(models), len(ems),
                             figsize=(TEXTWIDTH_IN, 1.6 * len(models) + 0.5),
                             squeeze=False, layout="constrained")
    im = None
    for ci, em in enumerate(ems):
        group = mvb[mvb["em"] == em]
        foundations = sorted(set(group["partition_a"]) | set(group["partition_b"]))
        n = len(foundations)
        for mi, model in enumerate(models):
            ax = axes[mi, ci]
            cells = group[group["model"] == model]
            # lower triangle only: drop the empty first row / last column
            matrix = np.full((n - 1, n - 1), np.nan)
            for r in cells.itertuples():
                i, j = foundations.index(r.partition_a), foundations.index(r.partition_b)
                matrix[max(i, j) - 1, min(i, j)] = r.jsd
            im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax)
            for r in cells.itertuples():
                i, j = foundations.index(r.partition_a), foundations.index(r.partition_b)
                ax.text(min(i, j), max(i, j) - 1,
                        f"{r.jsd:.3f}"[1:] + "\n" + stars(r.p_bonferroni),
                        ha="center", va="center", fontsize=7,
                        color="white" if r.jsd > 0.65 * vmax else "black")
            ax.set_yticks(range(n - 1), foundations[1:], fontsize=8)
            if mi == len(models) - 1:
                ax.set_xticks(range(n - 1), foundations[:-1],
                              rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticks([])
            if mi == 0:
                ax.set_title(em.capitalize(), fontsize=11)
            if ci == 0:
                ax.set_ylabel(model, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.35, label="Jensen–Shannon divergence")
    fig.savefig(FIGURES_DIR / f"mvb_{experiment}.pdf")
    plt.close(fig)
    print(f"mvb_{experiment}.pdf")


DOMAIN_MARKERS = {"political": "o", "personal": "^"}


def perturbation_boxplot(e2: pd.DataFrame) -> None:
    wide = e2[e2["scope"] == "wide"]
    narrow = e2[e2["scope"] == "8sa"]
    comparisons = list(dict.fromkeys(wide["partition_b"]))
    models = sorted(wide["model"].unique())
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 4.6), layout="constrained")
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    width = 0.8 / len(models)
    for ci, comparison in enumerate(comparisons):
        for mi, model in enumerate(models):
            cell = wide[(wide["partition_b"] == comparison) & (wide["model"] == model)]
            values = cell["jsd"]
            x = ci + (mi - len(models) / 2 + 0.5) * width
            ax.boxplot(values, positions=[x], widths=width * 0.85,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors[mi], alpha=0.5), showfliers=False)
            for domain, marker in DOMAIN_MARKERS.items():
                v = cell[cell["em"] == domain]["jsd"]
                ax.scatter(np.full(len(v), x), v, s=8, color=colors[mi],
                           marker=marker, zorder=3)
        median_8sa = narrow[narrow["partition_b"] == comparison]["jsd"].median()
        ax.hlines(median_8sa, ci - 0.42, ci + 0.42, colors="red", linestyles="dashed")
    ax.set_xticks(range(len(comparisons)), comparisons, fontsize=10)
    ax.set_ylabel("Perturbed vs. baseline (JSD)")
    handles = [plt.Line2D([], [], color=colors[i], lw=4, label=m)
               for i, m in enumerate(models)]
    handles += [plt.Line2D([], [], marker=m, color="grey", lw=0, markersize=5, label=lbl)
                for lbl, m in (("political", "o"), ("personal-situational", "^"))]
    fig.legend(handles=handles, fontsize=10, ncols=3, loc="outside lower center")
    fig.savefig(FIGURES_DIR / "perturbation_boxplot.pdf")
    plt.close(fig)
    print("perturbation_boxplot.pdf")


def likert_facets() -> None:
    """Per-(foundation, model) Likert response counts for the E3 appendix
    figures: one panel per foundation, grouped bars per model, viridis-coded
    bins. Same row filter as signatures_e3 (no refusal/invalid, no control
    items), so these counts underlie the mvb_e3 JSDs."""
    responses = load_responses("e3")
    models = sorted(responses)
    for questionnaire, support in (("mfq30", MFQ30_SUPPORT), ("mfq2", MFQ2_SUPPORT)):
        counts = {}  # (foundation, model) -> per-bin count Series
        for model, df in responses.items():
            kept = df[~df["label"].isin([REFUSAL, INVALID])
                      & (df["control"] == False)  # noqa: E712
                      & (df["questionnaire"] == questionnaire)]
            foundation = kept["mf"].map(lambda raw: (json.loads(raw) or [None])[0])
            for f, g in kept.groupby(foundation):
                counts[(f, model)] = g["label"].value_counts().reindex(support, fill_value=0)
        foundations = sorted({f for f, _ in counts})
        n_cols, n_rows = 2, (len(foundations) + 1) // 2
        colors = plt.cm.viridis(np.linspace(0, 1, len(support)))
        width = 0.8 / len(support)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(TEXTWIDTH_IN, 2.5 * n_rows + 0.8),
                                 squeeze=False, layout="constrained")
        x = np.arange(len(models))
        for ax, f in zip(axes.flat, foundations):
            for i, bin_ in enumerate(support):
                heights = [counts[(f, m)][bin_] if (f, m) in counts else 0 for m in models]
                ax.bar(x + (i - (len(support) - 1) / 2) * width, heights,
                       width, color=colors[i])
            ax.set_xticks(x, models, rotation=30, ha="right", fontsize=9)
            ax.set_title(f, fontsize=11)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", lw=0.3, alpha=0.4)
            ax.set_axisbelow(True)
        for row in axes:
            row[0].set_ylabel("Responses", fontsize=10)
        for ax in axes.flat[len(foundations):]:
            ax.set_visible(False)
        handles = [mpatches.Patch(color=colors[i], label=s) for i, s in enumerate(support)]
        fig.legend(handles=handles, loc="outside lower center", ncols=len(support),
                   fontsize=9, title_fontsize=10,
                   title=f"Likert (1 = lowest, {len(support)} = highest)")
        fig.savefig(FIGURES_DIR / f"{questionnaire}_per_foundation_model_likert.pdf")
        plt.close(fig)
        print(f"{questionnaire}_per_foundation_model_likert.pdf")


# --------------------------------------------------------------------- tables

def moral_bias_table(e1: pd.DataFrame) -> None:
    mb = e1[e1["family"] == "moral_bias"].set_index(["model", "em"])
    lines = [
        r"\begin{tabular}{l c c c c c c c c}", r"\toprule",
        r" & \multicolumn{4}{c}{Political} & \multicolumn{4}{c}{Personal-Situational} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r" & $\bar{s}_{M}$ & $\bar{s}_{\neg M}$ & $\Delta\bar{s}$ & JSD"
        r" & $\bar{s}_{M}$ & $\bar{s}_{\neg M}$ & $\Delta\bar{s}$ & JSD \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = []
        for em in ("political", "personal"):
            r = mb.loc[(model, em)]
            cells += [f"{r.mean_a:.2f}", f"{r.mean_b:.2f}",
                      f"${r.mean_a - r.mean_b:+.2f}$",
                      f".{f'{r.jsd:.3f}'[2:]}{stars(r.p_bonferroni)}"]
        lines.append(f"{MODEL_DISPLAY[model]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule",
              r"\multicolumn{9}{l}{\footnotesize " + STARS_NOTE + r"} \\",
              r"\end{tabular}"]
    (TABLES_DIR / "moral_bias.tex").write_text("\n".join(lines) + "\n")
    print("moral_bias.tex")


def sanctity_table() -> None:
    """Appendix sanctity table: absolute stance bin counts per model on the
    personal-situational items tagged sanctity-degradation. The refusal column
    folds in every non-numeric label, as in stance_bin_counts_table."""
    bins = [str(b) for b in range(1, 6)]
    lines = [r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r"Model & 1 & 2 & 3 & 4 & 5 & refusal & $n$ \\", r"\midrule"]
    responses = load_responses("e1")
    for model in MODEL_ORDER:
        df = responses[model]
        sub = df[(df["domain"] == "personal") & (df["morally_relevant"] == True)  # noqa: E712
                 & df["mf"].map(lambda raw: any("sanctity" in f for f in json.loads(raw)))]
        numeric = [int((sub["label"] == b).sum()) for b in bins]
        cells = numeric + [len(sub) - sum(numeric), len(sub)]
        lines.append(f"{MODEL_DISPLAY[model]} & " + " & ".join(map(str, cells)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "sanctity_stance_distribution.tex").write_text("\n".join(lines) + "\n")
    print("sanctity_stance_distribution.tex")


def stance_bin_counts_table() -> None:
    """Table A.1: absolute E1 stance bin counts per (domain, issue type,
    model). The refusal column folds in every non-numeric label."""
    responses = load_responses("e1")
    bins = [str(b) for b in range(1, 6)]
    blocks = [("political", True, "Political Essays --- moral"),
              ("political", False, "Political Essays --- nonmoral"),
              ("personal", True, "Personal-Situational Essays --- moral"),
              ("personal", False, "Personal-Situational Essays --- nonmoral")]
    lines = [r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r"Model & 1 & 2 & 3 & 4 & 5 & refusal & $n$ \\", r"\midrule"]
    for bi, (domain, moral, title) in enumerate(blocks):
        if bi:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\")
        for model, df in responses.items():
            sub = df[(df["domain"] == domain) & (df["morally_relevant"] == moral)]
            numeric = [int((sub["label"] == b).sum()) for b in bins]
            cells = numeric + [len(sub) - sum(numeric), len(sub)]
            lines.append(f"{model} & " + " & ".join(map(str, cells)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "stance_bin_counts.tex").write_text("\n".join(lines) + "\n")
    print("stance_bin_counts.tex")


def perturbation_table(e2: pd.DataFrame) -> None:
    """Tabular companion to the perturbation boxplot: same wide-scope cells,
    footer row = the pooled 8sa medians drawn as its dashed red lines."""
    wide = e2[e2["scope"] == "wide"]
    narrow = e2[e2["scope"] == "8sa"]
    comparisons = list(dict.fromkeys(wide["partition_b"]))
    models = sorted(wide["model"].unique())
    cells = wide.set_index(["topic", "model", "partition_b"])["jsd"]

    def fmt(jsd: float) -> str:
        return "--" if pd.isna(jsd) else f"{jsd:.3f}"

    lines = [
        r"\begin{tabular}{ll" + "r" * len(comparisons) + "}", r"\toprule",
        "Topic & Model & " + " & ".join(comparisons) + r" \\", r"\midrule",
    ]
    for ti, topic in enumerate(sorted(wide["topic"].unique())):
        if ti:
            lines.append(r"\addlinespace")
        for mi, model in enumerate(models):
            row = " & ".join(fmt(cells.get((topic, model, c), np.nan)) for c in comparisons)
            lines.append(f"{topic if mi == 0 else ''} & {model} & {row} \\\\")
    medians = " & ".join(fmt(narrow[narrow["partition_b"] == c]["jsd"].median())
                         for c in comparisons)
    lines += [r"\midrule", r"\multicolumn{2}{l}{8SA-only median} & " + medians + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "perturbation_biases.tex").write_text("\n".join(lines) + "\n")
    print("perturbation_biases.tex")


def correlation_tables(corr: pd.DataFrame) -> None:
    # Table 5.2: cross-model matrix per EM, political block first as in §5.2
    cm = corr[corr["comparison"] == "cross_model"]
    models = sorted(set(cm["unit_a"]) | set(cm["unit_b"]))
    lines = [r"\begin{tabular}{l r r r r}", r"\toprule",
             " & " + " & ".join(models[:-1]) + r" \\", r"\midrule"]
    for bi, (em, title) in enumerate((("political", "Political"),
                                      ("personal", "Personal-Situational"))):
        if bi:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\")
        group = cm[cm["em"] == em]
        for m2 in models[1:]:
            cells = []
            for m1 in models[:-1]:
                hit = group[(group["unit_a"] == m1) & (group["unit_b"] == m2)]
                cells.append(f"${hit.iloc[0].rho:+.3f}$" if len(hit) else "")
            lines.append(f"{m2} & " + " & ".join(cells) + r" \\")
    pooled = corr[corr["comparison"] == "cross_model_pooled"].iloc[0]
    lines += [r"\midrule",
              rf"\multicolumn{{5}}{{l}}{{\footnotesize Aggregated (mean) $\rho = {pooled.rho:+.3f}$.}} \\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "cross_model_rho.tex").write_text("\n".join(lines) + "\n")

    # Table 5.3: cross-domain per model
    lines = [r"\begin{tabular}{l r}", r"\toprule", r"Model & $\rho$ \\", r"\midrule"]
    for r in corr[corr["comparison"] == "cross_domain"].itertuples():
        lines.append(rf"{r.unit_a} & ${r.rho:+.3f}$ \\")
    pooled = corr[corr["comparison"] == "cross_domain_pooled"].iloc[0]
    lines += [r"\midrule",
              rf"Pooled & ${pooled.rho:+.3f}$, 95\% CI $[{pooled.ci_lo:+.3f}, {pooled.ci_hi:+.3f}]$ \\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES_DIR / "cross_domain_rho.tex").write_text("\n".join(lines) + "\n")

    # Table 5.4: cross-EM pairs, task domains before the questionnaires
    pair_order = [("political", "personal"), ("political", "mfq30"),
                  ("political", "mfq2"), ("personal", "mfq30"),
                  ("personal", "mfq2"), ("mfq30", "mfq2")]
    ce = corr[corr["comparison"] == "cross_em"]
    lines = [r"\begin{tabular}{l r c}", r"\toprule",
             r"Elicitation Method pair & $\rho$ & 95\% CI \\", r"\midrule"]
    for a, b in pair_order:
        hit = ce[ce["unit_a"].isin((a, b)) & ce["unit_b"].isin((a, b))].iloc[0]
        dagger = r"\textsuperscript{$\dagger$}" if hit.ci_lo * hit.ci_hi > 0 else ""
        lines.append(
            rf"{EM_DISPLAY[a]} $\leftrightarrow$ {EM_DISPLAY[b]}{dagger}"
            rf" & ${hit.rho:+.3f}$ & $[{hit.ci_lo:+.3f}, {hit.ci_hi:+.3f}]$ \\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{3}{l}{\footnotesize $\dagger$ CI excludes zero.} \\",
              r"\end{tabular}"]
    (TABLES_DIR / "cross_em_rho.tex").write_text("\n".join(lines) + "\n")
    print("cross_model_rho.tex, cross_domain_rho.tex, cross_em_rho.tex")


# legacy cleveland-dot style (src/analysis/render_viz.py), fonts/PDF adjusted
STANCE_COLORS = {"1": "#2ecc71", "2": "#a8e6a3", "3": "#f5c518",
                 "4": "#f0a070", "5": "#e74c3c", REFUSAL: "#95a5a6"}
STANCE_NAMES = {"1": "Fully pro", "2": "Overwhelmingly pro", "3": "Neutral",
                "4": "Overwhelmingly con", "5": "Fully con", REFUSAL: "Refusal"}
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def cleveland_chart(curated: dict[str, str] | None = None, save: bool = True,
                    responses: bool = False, data: dict | None = None) -> list:
    """Per-model mean stance on the curated E2 topics, from E1's 8SA responses
    (E1 is 8SA-only). One figure per domain, one panel per topic (same grid
    layout as the Likert facets); dot colour = modal label; refusals excluded
    from means. Pass curated={topic: domain} to chart an arbitrary selection.
    responses=True superimposes every individual numeric response as a small
    jittered marker coloured by its own stance label, behind the mean.
    data={model: labelled df} overrides the E1 source (e.g. an E2 perturbation
    slice). Returns the figures; save=False skips the PDF write (and leaves them
    open, so they render inline when called from an interactive window)."""
    if curated is None:
        components = json.loads((DATA_DIR / "components" / "issues.json").read_text())
        curated = {c["text"]: c["features"]["domain"] for c in components
                   if c["features"].get("controversial")}
    cells = {}  # (topic, model) -> (mean stance, modal label, numeric labels)
    for model, df in (load_responses("e1") if data is None else data).items():
        df = df[df["topic"].isin(curated) & (df["label"] != INVALID)]
        for topic, g in df.groupby("topic"):
            values = pd.to_numeric(g["label"], errors="coerce").dropna()
            if len(values):
                cells[(topic, model)] = (values.mean(), g["label"].mode()[0], values)
    models = sorted({m for _, m in cells})
    handles = [mpatches.Patch(color=STANCE_COLORS[k], label=STANCE_NAMES[k]) for k in STANCE_COLORS]
    handles += [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], color="grey", lw=0,
                           markersize=6, label=m) for i, m in enumerate(models)]
    figs = []
    for domain in ("political", "personal"):
        topics = sorted((t for t, d in curated.items() if d == domain),
                        key=lambda t: np.mean(
                            [v[0] for (tt, _), v in cells.items() if tt == t]))
        if not topics:
            continue
        n_cols, n_rows = 2, (len(topics) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(TEXTWIDTH_IN, 1.7 * n_rows + 1.2),
                                 squeeze=False, layout="constrained")
        for pi, (ax, topic) in enumerate(zip(axes.flat, topics)):
            for mi, model in enumerate(models):
                if (topic, model) in cells:
                    mean, modal, values = cells[(topic, model)]
                    if responses:
                        jitter = (np.arange(len(values)) - (len(values) - 1) / 2) * 0.18
                        ax.scatter(values, mi + jitter, s=14, alpha=0.8, zorder=2,
                                   marker=MARKERS[mi % len(MARKERS)],
                                   color=[STANCE_COLORS[str(int(v))] for v in values])
                    ax.scatter(mean, mi, color=STANCE_COLORS.get(modal, "#cccccc"),
                               marker=MARKERS[mi % len(MARKERS)], s=55, zorder=3,
                               edgecolors="white", linewidths=0.4)
            ax.set_title(textwrap.fill(topic, 38), fontsize=10)
            ax.set_yticks(range(len(models)),
                          models if pi % n_cols == 0 else [""] * len(models), fontsize=9)
            ax.set_ylim(-0.5, len(models) - 0.5)
            ax.invert_yaxis()
            ax.set_xlim(0.5, 5.5)
            ax.set_xticks([1, 2, 3, 4, 5], ["pro", "", "neutral", "", "con"], fontsize=9)
            ax.axvline(3, color="grey", lw=0.5, ls="--", alpha=0.6)
            ax.grid(axis="x", lw=0.3, alpha=0.4)
            ax.set_axisbelow(True)
        for ax in axes.flat[len(topics):]:
            ax.set_visible(False)
        fig.legend(handles=handles, loc="outside lower center", ncols=3,
                   fontsize=9, title="Stance / Model", title_fontsize=10)
        if save:
            fig.savefig(FIGURES_DIR / f"stance_cleveland_{domain}.pdf")
            plt.close(fig)
            print(f"stance_cleveland_{domain}.pdf")
        figs.append(fig)
    return figs


def export_data() -> None:
    """Copy everything in out/ into thesis/figures/bias_signatures/, where the
    thesis sources it alongside the other figure directories."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted([*FIGURES_DIR.glob("*"), *TABLES_DIR.glob("*")]):
        shutil.copy2(src, EXPORT_DIR / src.name)
        print(f"exported {src.name}")


if __name__ == "__main__":
    import argparse

    matplotlib.use("Agg")  # headless rendering for the CLI; notebooks keep their inline backend

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated experiment names (e1, e2, e3, correlations, cleveland)")
    parser.add_argument("--export", action="store_true",
                        help="copy out/ artefacts into thesis/figures/bias_signatures/")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else {"e1", "e2", "e3", "correlations", "cleveland"}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for experiment in ("e1", "e3"):
        if experiment in only and (SIGNATURES_DIR / f"{experiment}.csv").exists():
            mvb_heatmaps(pd.read_csv(SIGNATURES_DIR / f"{experiment}.csv"), experiment)
    if "e1" in only:
        moral_bias_table(pd.read_csv(SIGNATURES_DIR / "e1.csv"))
        stance_bin_counts_table()
        sanctity_table()
    if "e3" in only:
        likert_facets()
    if "e2" in only:
        e2 = pd.read_csv(SIGNATURES_DIR / "e2.csv")
        perturbation_boxplot(e2)
        perturbation_table(e2)
    if "correlations" in only:
        correlation_tables(pd.read_csv(SIGNATURES_DIR / "correlations.csv"))
    if "cleveland" in only:
        cleveland_chart()
    if args.export:
        export_data()
