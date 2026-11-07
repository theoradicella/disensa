"""Reading many runs together: the numbers and figures of the thesis results.

  load       load_runs(paths) -> one row per run, config fields flattened
  summary    per-run summaries: clusters, convergence, drift, movement, rounding
  fit        which numeric rule, with which parameters, an LLM run resembles (macro)
  response   how much agents move against how far the opinions they heard were (micro)
  figures    the thesis figures, as PDFs (needs matplotlib: `pip install disensa[analysis]`)

LLM beliefs are coarse (mostly whole or half numbers), so the comparisons are between
distributions, cluster counts, direction and frequency of moves, never exact trajectories.
Everything except `figures` works without the optional dependencies, e.g. in a notebook.
"""

from disensa.analysis.fit import fit_run, simulate_numeric
from disensa.analysis.load import find_run_dirs, load_runs
from disensa.analysis.response import response_curve, response_observations
from disensa.analysis.summary import clusters, convergence_step, summarize_run, wasserstein

__all__ = [
    "clusters",
    "convergence_step",
    "find_run_dirs",
    "fit_run",
    "load_runs",
    "response_curve",
    "response_observations",
    "simulate_numeric",
    "summarize_run",
    "wasserstein",
]
