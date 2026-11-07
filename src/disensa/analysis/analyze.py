"""`disensa analyze`: the tables and figures of a set of runs, in one folder.

With `measure` set to a scorer model, every LLM-run result (summaries, fits, figures,
response curves) reads that scorer's ratings instead of the self-ratings (see
summary.opinion_matrix); LLM runs it has not scored are skipped, with a note.

  runs.csv          every run found, config fields flattened (load.load_runs)
  summary.csv       per-run summary (summary.summarize_run)
  fits.csv          LLM runs: per rule and distance, the best numeric parameters (fit.py)
  response.csv      LLM runs: the micro response curve per topic (response.py)
  *.pdf + *.csv     the figures, each with the numbers it plots

Figures are made for whatever the runs allow: bounded-confidence runs give clusters vs ε,
Friedkin–Johnsen runs polarization vs λ, DeGroot runs convergence by topology, LLM runs the
per-run, per-topic and fit figures. Topics are the LLM runs' statements, labelled T1, T2, …
in the figures (the full statements are in topics.csv).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from disensa.analysis import fit as fitting
from disensa.analysis.load import load_runs
from disensa.analysis.response import response_curve
from disensa.analysis.summary import DELTA, summarize_run
from disensa.config import NumericDynamics


def _median_dynamics(fits: pd.DataFrame, rule: str) -> NumericDynamics:
    """One parameter set for a topic: the median of its runs' best (Wasserstein) fits."""
    f = fits[(fits["rule"] == rule) & (fits["measure"] == "wasserstein")]
    if rule == "bounded_confidence":
        return NumericDynamics(rule=rule, epsilon=f["epsilon"].median(), mu=f["mu"].median())
    if rule == "friedkin_johnsen":
        return NumericDynamics(rule=rule, stubbornness=f["stubbornness"].median())
    return NumericDynamics(rule=rule)


def analyze(paths, out: Path, measure: str = "self", delta: float = DELTA, echo=print) -> dict:
    from disensa.analysis import figures
    from disensa.runs import load_run

    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(paths)
    if runs.empty:
        raise ValueError("no finished runs found")
    runs.to_csv(out / "runs.csv", index=False)
    made: list[str] = []

    summaries = []
    for p, engine in zip(runs["path"], runs["engine"], strict=True):
        m = measure if engine == "llm" else "self"  # numeric runs have no scorer
        try:
            s = summarize_run(load_run(p), delta, measure=m)
        except ValueError as e:  # this LLM run was not scored by `measure`
            echo(f"skipped {p}: {e}")
            continue
        s["moved_by_step"] = json.dumps(s["moved_by_step"])
        s["rounding"] = json.dumps(s["rounding"])
        summaries.append({"path": p, **s})
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    echo(f"{len(runs)} runs: runs.csv, summary.csv")

    for fn in (
        figures.bc_clusters_vs_epsilon,
        figures.fj_polarization_vs_stubbornness,
        figures.degroot_convergence_vs_topology,
    ):
        if path := fn(runs, out):
            made.append(path.name)

    llm = runs[runs["engine"] == "llm"]
    if not llm.empty:
        statements = list(dict.fromkeys(llm["statement"]))
        topics = {s: f"T{i}" for i, s in enumerate(statements, 1)}
        pd.DataFrame({"topic": list(topics.values()), "statement": statements}).to_csv(
            out / "topics.csv", index=False
        )
        all_fits, curves = [], []
        for _, r in llm.iterrows():
            run = load_run(r["path"])
            try:
                best, _ = fitting.fit_run(run, delta=delta, measure=measure)
            except ValueError:
                continue  # not scored by `measure`: reported above
            best.insert(0, "topic", topics[r["statement"]])
            best.insert(0, "path", r["path"])
            all_fits.append(best)
            # the folder name keeps it unique: several runs can share topic, seed and replicate
            name = f"run_{topics[r['statement']]}_{Path(r['path']).name}"
            made.append(figures.topic_runs(run, best, out, name, measure).name)
        if not all_fits:
            raise ValueError(f"no LLM run could be read with --measure {measure}")
        fits = pd.concat(all_fits, ignore_index=True)
        fits.to_csv(out / "fits.csv", index=False)
        if path := figures.fit_distances(fits, out):
            made.append(path.name)
        for statement, label in topics.items():
            group = llm[(llm["statement"] == statement) & llm["path"].isin(fits["path"])]
            if group.empty:
                continue
            rules = [
                _median_dynamics(fits[fits["topic"] == label], rule)
                for rule in ("degroot", "bounded_confidence", "friedkin_johnsen")
            ]
            try:
                curve = response_curve([load_run(p) for p in group["path"]], measure, rules)
            except ValueError as e:
                echo(f"{label}: no response curve ({e})")
                continue
            curve.insert(0, "topic", label)
            curves.append(curve)
            title = f"{label}: {statement[:70]}{'…' if len(statement) > 70 else ''}"
            made.append(figures.response_curves(curve, out, f"response_{label}", title).name)
        if curves:
            pd.concat(curves, ignore_index=True).to_csv(out / "response.csv", index=False)
        echo(f"{len(llm)} LLM runs over {len(topics)} topics: fits.csv, response.csv, topics.csv")
    for m in made:
        echo(f"  {m}")
    return {"runs": len(runs), "figures": made}
