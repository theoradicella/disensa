"""The thesis figures, as PDFs, in one style. Needs matplotlib (`pip install disensa[analysis]`).

Every figure function takes tables or loaded runs, saves `<out>/<name>.pdf` and writes the
numbers it plots to `<out>/<name>.csv` next to it (the table view of the figure).

Colours follow a fixed order, never cycled, and are never the only cue: every series also has
its own line style or marker and a legend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from disensa.analysis.summary import clusters, convergence_step

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# fixed categorical order (validated: CVD and normal-vision separation pass on white)
COLORS = {
    "llm": "#2a78d6",
    "degroot": "#eb6834",
    "bounded_confidence": "#1baf7a",
    "friedkin_johnsen": "#eda100",
    "nobody_moves": "#8a8984",
}
STYLES = {
    "llm": "-",
    "degroot": "--",
    "bounded_confidence": "-.",
    "friedkin_johnsen": ":",
    "nobody_moves": (0, (1, 3)),
}
MARKERS = {
    "llm": "o",
    "degroot": "s",
    "bounded_confidence": "^",
    "friedkin_johnsen": "D",
    "nobody_moves": "x",
}
NAMES = {
    "llm": "LLM agents",
    "degroot": "DeGroot",
    "bounded_confidence": "Bounded confidence",
    "friedkin_johnsen": "Friedkin–Johnsen",
    "nobody_moves": "Nobody moves",
}
TOPOLOGY = {
    "erdos_renyi": "Random",
    "watts_strogatz": "Small-world",
    "barabasi_albert": "Scale-free",
}


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,  # editable text in the PDF
            "savefig.bbox": "tight",
        }
    )
    return pyplot


def _save(fig, data: pd.DataFrame, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    data.to_csv(out / f"{name}.csv", index=False)
    import matplotlib.pyplot as pyplot

    pyplot.close(fig)
    return out / f"{name}.pdf"


def _note(ax, text: str) -> None:
    """A muted line under the plot, right-aligned (sample sizes, settings)."""
    ax.annotate(
        text,
        (1, 0),
        xycoords="axes fraction",
        xytext=(0, -30),
        textcoords="offset points",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=7.5,
    )


def _final(path) -> np.ndarray:
    from disensa.runs import load_run

    return load_run(path).opinion_matrix()[-1]


# ---------------------------------------------------------------- numeric reference figures


def bc_clusters_vs_epsilon(runs: pd.DataFrame, out: Path, delta: float = 0.05) -> Path | None:
    """Bounded confidence: number of final clusters against ε, mean ± sd over seeds.

    Numeric clusters are point masses, so δ = 0.05 counts them exactly.
    """
    bc = runs[runs["rule"] == "bounded_confidence"].copy()
    if bc.empty:
        return None
    bc["clusters"] = [clusters(_final(p), delta) for p in bc["path"]]
    data = (
        bc.groupby(["mu", "epsilon"])["clusters"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "seeds"})
    )
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(4.6, 3.0))
    for k, (mu, g) in enumerate(data.groupby("mu")):
        g = g.sort_values("epsilon")
        sd = g["std"].fillna(0)
        color = (
            list(COLORS.values())[k % 4]
            if data["mu"].nunique() > 1
            else COLORS["bounded_confidence"]
        )
        ax.fill_between(g["epsilon"], g["mean"] - sd, g["mean"] + sd, color=color, alpha=0.18, lw=0)
        ax.plot(g["epsilon"], g["mean"], color=color, marker="o", ms=3, label=f"μ = {mu:g}")
    ax.set_xlabel("confidence bound ε")
    ax.set_ylabel("clusters at the end")
    ax.set_title("Bounded confidence: clusters against ε", loc="left")
    ax.set_ylim(bottom=0)
    if data["mu"].nunique() > 1:
        ax.legend()
    _note(
        ax, f"mean ± sd over {int(data['seeds'].max())} seeds · clusters split at gaps > {delta:g}"
    )
    return _save(fig, data, out, "bc_clusters_vs_epsilon")


def fj_polarization_vs_stubbornness(runs: pd.DataFrame, out: Path) -> Path | None:
    """Friedkin–Johnsen: final polarization against the stubbornness λ, mean ± sd over seeds."""
    from disensa.metrics import polarization

    fj = runs[runs["rule"] == "friedkin_johnsen"].copy()
    if fj.empty:
        return None
    fj["polarization"] = [polarization(_final(p)) for p in fj["path"]]
    data = (
        fj.groupby("stubbornness")["polarization"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "seeds"})
        .sort_values("stubbornness")
    )
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(4.6, 3.0))
    c = COLORS["friedkin_johnsen"]
    sd = data["std"].fillna(0)
    ax.fill_between(
        data["stubbornness"], data["mean"] - sd, data["mean"] + sd, color=c, alpha=0.2, lw=0
    )
    ax.plot(data["stubbornness"], data["mean"], color=c, marker="D", ms=3)
    ax.set_xscale("log")
    ax.set_xlabel("stubbornness λ")
    ax.set_ylabel("polarization at the end")
    ax.set_title("Friedkin–Johnsen: polarization against λ", loc="left")
    ax.set_ylim(bottom=0)
    _note(ax, f"mean ± sd over {int(data['seeds'].max())} seeds")
    return _save(fig, data, out, "fj_polarization_vs_stubbornness")


def degroot_convergence_vs_topology(
    runs: pd.DataFrame, out: Path, tol: float = 1e-3
) -> Path | None:
    """DeGroot: convergence step per topology, mean ± sd over seeds (non-converged counted)."""
    from disensa.runs import load_run

    dg = runs[runs["rule"] == "degroot"].copy()
    if dg.empty:
        return None
    dg["convergence_step"] = [
        convergence_step(load_run(p).opinion_matrix(), tol) for p in dg["path"]
    ]
    rows = []
    for topo, g in dg.groupby("topology"):
        conv = g["convergence_step"].dropna().astype(float)
        rows.append(
            {
                "topology": topo,
                "mean": conv.mean() if len(conv) else np.nan,
                "std": conv.std() if len(conv) > 1 else 0.0,
                "runs": len(g),
                "not_converged": int(g["convergence_step"].isna().sum()),
            }
        )
    data = pd.DataFrame(rows)
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(data))
    c = COLORS["degroot"]
    ax.errorbar(x, data["mean"], yerr=data["std"], fmt="s", color=c, ms=6, capsize=3, lw=1.2)
    for i, r in data.iterrows():
        note = f"{r['runs']} runs" + (
            f", {r['not_converged']} not converged" if r["not_converged"] else ""
        )
        ax.annotate(
            note,
            (i, r["mean"]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=MUTED,
            fontsize=7.5,
        )
    ax.set_xticks(x, [TOPOLOGY.get(t, t) for t in data["topology"]])
    ax.set_xlim(-0.5, len(data) - 0.2)
    ax.set_ylabel(f"convergence step (tol {tol:g})")
    ax.set_title("DeGroot: steps to consensus by topology", loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    return _save(fig, data, out, "degroot_convergence_vs_topology")


# ---------------------------------------------------------------- LLM runs against the rules


def topic_runs(run, fits: pd.DataFrame, out: Path, name: str, measure: str = "self") -> Path:
    """One LLM run: its agents' trajectories, and its final distribution against the
    best-fit (Wasserstein) simulation of each rule on the same world."""
    from matplotlib.ticker import MaxNLocator

    from disensa.analysis.fit import best_dynamics, simulate_numeric, world
    from disensa.analysis.summary import opinion_matrix

    X = opinion_matrix(run, measure)
    graph, x0 = world(run)
    steps = len(X) - 1
    pyplot = plt()
    fig, (a, b) = pyplot.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
    t = np.arange(steps + 1)
    for i in range(X.shape[1]):
        a.plot(t, X[:, i], color=COLORS["llm"], alpha=0.18, lw=0.8)
    a.plot(t, X.mean(axis=1), color=INK, lw=1.6, label="mean")
    a.set_ylim(-2.1, 2.1)
    a.set_xlabel("step")
    a.set_ylabel("opinion")
    scorer = measure.split("/", 1)[-1]
    a.set_title(
        "LLM agents, one line each" + ("" if measure == "self" else f"\nrated by {scorer}"),
        loc="left",
    )
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.legend(loc="lower right")
    edges = np.arange(-2.25, 2.26, 0.5)  # bins centred on the whole and half numbers
    rows = [
        {"series": "llm", "bin_mid": m, "count": c}
        for m, c in zip((edges[:-1] + edges[1:]) / 2, np.histogram(X[-1], edges)[0], strict=True)
    ]
    b.hist(X[-1], edges, color=COLORS["llm"], alpha=0.35, label=NAMES["llm"])
    for rule in ("degroot", "bounded_confidence", "friedkin_johnsen"):
        if not ((fits["rule"] == rule) & (fits["measure"] == "wasserstein")).any():
            continue
        dyn = best_dynamics(fits, rule)
        sim = simulate_numeric(graph, x0, dyn, steps)[-1]
        counts = np.histogram(sim, edges)[0]
        rows += [
            {"series": rule, "bin_mid": m, "count": c}
            for m, c in zip((edges[:-1] + edges[1:]) / 2, counts, strict=True)
        ]
        params = {
            "bounded_confidence": f" (ε={dyn.epsilon:.2g}, μ={dyn.mu:.2g})",
            "friedkin_johnsen": f" (λ={dyn.stubbornness:.3g})",
        }.get(rule, "")
        b.step(
            edges,
            np.append(counts, counts[-1]),
            where="post",
            color=COLORS[rule],
            ls=STYLES[rule],
            lw=1.4,
            label=NAMES[rule] + params,
        )
    b.set_xlabel("opinion at the end")
    b.set_ylabel("agents")
    b.set_title("Final opinions vs best-fit rules", loc="left")
    b.legend(fontsize=7, loc="upper left")
    b.grid(axis="x", visible=False)
    return _save(fig, pd.DataFrame(rows), out, name)


def fit_distances(fits: pd.DataFrame, out: Path, measure: str = "wasserstein") -> Path | None:
    """Per topic: each rule's best distance to the LLM runs (mean over its runs), with the
    "nobody moves" baseline. `fits` needs a `topic` column (see analyze)."""
    f = fits[fits["measure"] == measure]
    if f.empty:
        return None
    data = f.groupby(["topic", "rule"])["distance"].agg(["mean", "std", "count"]).reset_index()
    topics = list(dict.fromkeys(data["topic"]))
    order = ["nobody_moves", "degroot", "bounded_confidence", "friedkin_johnsen"]
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(5.6, 0.55 * len(topics) + 1.2))
    for k, rule in enumerate(order):
        g = data[data["rule"] == rule].set_index("topic").reindex(topics)
        y = np.arange(len(topics)) + (k - 1.5) * 0.15
        ax.errorbar(
            g["mean"],
            y,
            xerr=g["std"].fillna(0),
            fmt=MARKERS[rule],
            color=COLORS[rule],
            ms=5,
            capsize=2,
            lw=1,
            label=NAMES[rule],
        )
    ax.set_yticks(np.arange(len(topics)), topics)
    ax.invert_yaxis()
    ax.set_xlabel({"wasserstein": "Wasserstein distance, final opinions"}.get(measure, measure))
    ax.set_xlim(left=0)
    ax.set_title("How close the best numeric rule gets, by topic", loc="left")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.22 - 0.02 * len(topics)))
    ax.grid(axis="y", visible=False)
    return _save(fig, data, out, f"fit_distances_{measure}")


def response_curves(curve: pd.DataFrame, out: Path, name: str, title: str) -> Path:
    """The micro response curve: mean Δ and P(move) against d, with each rule's prediction."""
    c = curve[curve["n"] > 0]
    pyplot = plt()
    fig, (a, b) = pyplot.subplots(1, 2, figsize=(7.2, 2.9))
    a.axhline(0, color=MUTED, lw=0.6)
    a.errorbar(
        c["d_mid"],
        c["mean_delta"],
        yerr=c["se_delta"].fillna(0),
        fmt="o-",
        color=COLORS["llm"],
        ms=4,
        capsize=2,
        label=NAMES["llm"],
    )
    b.plot(c["d_mid"], c["p_move"], "o-", color=COLORS["llm"], ms=4, label=NAMES["llm"])
    for rule in ("degroot", "bounded_confidence", "friedkin_johnsen"):
        if f"pred[{rule}]" in c:
            a.plot(
                c["d_mid"],
                c[f"pred[{rule}]"],
                color=COLORS[rule],
                ls=STYLES[rule],
                label=NAMES[rule],
            )
            b.plot(
                c["d_mid"],
                c[f"pmove[{rule}]"],
                color=COLORS[rule],
                ls=STYLES[rule],
                label=NAMES[rule],
            )
    a.set_xlabel("d: mean heard opinion − own opinion")
    a.set_ylabel("Δ: change of own opinion")
    a.set_title("How far agents move", loc="left")
    b.set_xlabel("d: mean heard opinion − own opinion")
    b.set_ylabel("share of agents that moved")
    b.set_ylim(-0.02, 1.02)
    b.set_title("How often agents move", loc="left")
    a.legend(fontsize=7, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    return _save(fig, curve, out, name)
