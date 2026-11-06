"""Advice on a config before it runs: settings that are allowed but probably not what you want.

Each warning names the setting, says what will happen, and points to the docs section that
explains why. `disensa run` prints them; the lab shows them under the step they belong to.
Nothing here changes a run: impossible settings are rejected by `config.py` instead.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

from pydantic import BaseModel, ValidationError

from disensa import OPINION_MAX, OPINION_MIN
from disensa.config import (
    ExperimentConfig,
    InitialOpinions,
    InteractionsConfig,
    LLMDynamics,
    NetworkConfig,
    NumericDynamics,
)

CONNECTED_TARGET = 0.85  # warn when fewer Erdős–Rényi draws than this are connected
SMALL_N = 20


@dataclass(frozen=True)
class Check:
    field: str  # dotted config path, e.g. "network.p"
    message: str
    docs: str  # anchor on the docs page, e.g. "choosing-p"
    docs_title: str
    level: str = "warning"  # warning | error

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- network


def er_connected_probability(n: int, p: float) -> float:
    """Chance that an Erdős–Rényi draw is connected.

    Near the threshold a G(N, p) graph is connected exactly when it has no isolated node, and
    the number of isolated nodes is close to Poisson with mean λ = N (1 - p)^(N-1)
    (Erdős & Rényi 1960; Network Science, ch. 3). So P(connected) ≈ exp(-λ).
    """
    return math.exp(-n * (1 - p) ** (n - 1))


def er_min_p(n: int, target: float = CONNECTED_TARGET) -> float:
    """Smallest p (rounded up to 0.01) with P(connected) ≥ target."""
    p = 1 - (-math.log(target) / n) ** (1 / (n - 1))
    return math.ceil(p * 100) / 100


def mean_degree(cfg: NetworkConfig) -> float:
    n = cfg.n
    if cfg.topology == "erdos_renyi":
        return cfg.p * (n - 1)
    if cfg.topology == "watts_strogatz":
        return float(cfg.k)
    return 2 * cfg.m * (n - cfg.m) / n  # NetworkX starts from a star on m + 1 nodes


def network_checks(cfg: NetworkConfig) -> list[Check]:
    out: list[Check] = []
    n = cfg.n

    if cfg.topology == "erdos_renyi":
        k = mean_degree(cfg)
        prob = er_connected_probability(n, cfg.p)
        if prob < CONNECTED_TARGET:
            fail = (1 - prob) ** 100 > 0.05
            out.append(
                Check(
                    "network.p",
                    f"With p = {cfg.p:g} and {n} agents (⟨k⟩ ≈ {k:.1f}), only about "
                    f"{prob:.0%} of draws are connected. "
                    + (
                        "The run will probably fail after 100 redraws. "
                        if fail
                        else "Disconnected draws are redrawn, so the networks you get are a "
                        "filtered sample, not plain random graphs. "
                    )
                    + f"Use p ≥ {er_min_p(n):.2f}.",
                    "choosing-p",
                    "Network topologies › Choosing p",
                )
            )

    elif cfg.topology == "watts_strogatz":
        k, beta = cfg.k, cfg.beta
        if k == 2:
            out.append(
                Check(
                    "network.k",
                    "With k = 2 an agent's two neighbours are never linked to each other, so "
                    "there are no triangles and clustering is 0: the feature this topology "
                    "exists for is missing. Use k ≥ 4.",
                    "watts-strogatz",
                    "Network topologies › Small-world",
                )
            )
        rewired = beta * n * k / 2
        if beta == 0:
            out.append(
                Check(
                    "network.beta",
                    "β = 0 is the plain ring lattice: no shortcuts, so paths are long "
                    f"(about N/2k ≈ {n / (2 * k):.0f} steps on average). "
                    "Use β ≈ 0.01–0.1 for the small-world regime.",
                    "watts-strogatz",
                    "Network topologies › Small-world",
                )
            )
        elif rewired < 3:
            out.append(
                Check(
                    "network.beta",
                    f"β = {beta:g} rewires only about {rewired:.1f} of the "
                    f"{n * k // 2} links, so the network is still almost a ring lattice "
                    f"with long paths. Use β ≥ {math.ceil(600 / (n * k)) / 100:.2f}.",
                    "watts-strogatz",
                    "Network topologies › Small-world",
                )
            )
        kept = (1 - beta) ** 3
        if beta > 0 and kept < 0.5:
            out.append(
                Check(
                    "network.beta",
                    f"With β = {beta:g} clustering falls to about {kept:.0%} of the ring's "
                    "(C ≈ (1 − β)³ C(0)), so the network is close to a random one. The "
                    "small-world regime is β ≈ 0.01–0.1.",
                    "watts-strogatz",
                    "Network topologies › Small-world",
                )
            )

    elif cfg.m == 1:
        out.append(
            Check(
                "network.m",
                "With m = 1 every new agent adds a single link, so the network is a tree: "
                "no loops, no clustering, and removing any link splits it. Use m ≥ 2.",
                "barabasi-albert",
                "Network topologies › Scale-free",
            )
        )

    if mean_degree(cfg) > (n - 1) / 2:
        param = {"erdos_renyi": "p", "watts_strogatz": "k", "barabasi_albert": "m"}
        out.append(
            Check(
                f"network.{param[cfg.topology]}",
                f"Each agent is linked to more than half of the others (⟨k⟩ ≈ "
                f"{mean_degree(cfg):.1f} of {n - 1}). The network is nearly complete, so "
                "the topology barely matters.",
                "describing-a-network",
                "Network topologies › Describing a network",
            )
        )
    if n < SMALL_N:
        out.append(
            Check(
                "network.n",
                f"With {n} agents the three topologies are hard to tell apart: there are too "
                "few nodes for hubs, clustering or short paths to show clearly.",
                "small-networks",
                "Network topologies › Small networks",
            )
        )
    return out


# ---------------------------------------------------------------- initial opinions


def _outside(mean: float, std: float) -> float:
    """Share of a normal(mean, std) outside [OPINION_MIN, OPINION_MAX]."""
    if std == 0:
        return float(not OPINION_MIN <= mean <= OPINION_MAX)
    cdf = lambda v: 0.5 * (1 + math.erf((v - mean) / (std * math.sqrt(2))))  # noqa: E731
    return cdf(OPINION_MIN) + 1 - cdf(OPINION_MAX)


def opinion_checks(cfg: InitialOpinions) -> list[Check]:
    if cfg.distribution == "uniform":
        return []
    out: list[Check] = []
    title = "Initial opinions"
    mean = abs(cfg.mean) if cfg.distribution == "bimodal" else cfg.mean
    clipped = _outside(mean, cfg.std)  # the two camps are mirror images
    if clipped > 0.1:
        out.append(
            Check(
                "opinions.std",
                f"About {clipped:.0%} of agents would start outside [−2, 2] and are clipped "
                "to the edge, so they pile up at the extremes. Lower the spread"
                + (" or move the mean toward 0." if cfg.distribution == "normal" else "."),
                "initial-opinions",
                title,
            )
        )
    if cfg.distribution == "normal" and cfg.std == 0:
        out.append(
            Check(
                "opinions.std",
                "With spread 0 every agent starts at the same opinion, so nothing can change.",
                "initial-opinions",
                title,
            )
        )
    if cfg.distribution == "bimodal" and cfg.std >= mean:
        out.append(
            Check(
                "opinions.std",
                f"The camps at ±{mean:g} have spread {cfg.std:g}, so they overlap heavily and "
                "there are not really two camps. Use a spread well below the camp position.",
                "initial-opinions",
                title,
            )
        )
    return out


# ---------------------------------------------------------------- dynamics


def numeric_checks(cfg: NumericDynamics) -> list[Check]:
    width = OPINION_MAX - OPINION_MIN
    if cfg.rule == "bounded_confidence" and cfg.epsilon >= width:
        return [
            Check(
                "dynamics.epsilon",
                f"ε = {cfg.epsilon:g} covers the whole opinion range ({width:g}), so every "
                "neighbour is always within confidence and bounded confidence behaves like "
                "plain averaging (DeGroot).",
                "algorithms",
                "Numeric algorithms",
            )
        ]
    return []


def interaction_checks(cfg: InteractionsConfig, net: NetworkConfig | None) -> list[Check]:
    if net is None or net.topology == "erdos_renyi":
        return []
    fewest = net.m if net.topology == "barabasi_albert" else net.k
    label = "have only" if net.topology == "barabasi_albert" else "have about"
    if cfg.min > fewest:
        return [
            Check(
                "dynamics.interactions.min",
                f"Interactions min is {cfg.min}, but many agents {label} {fewest} neighbours, "
                f"so they will interact with fewer than {cfg.min} per step.",
                "agents",
                "LLM agents",
            )
        ]
    return []


def model_checks(cfg: LLMDynamics) -> list[Check]:
    out: list[Check] = []
    for i, m in enumerate(cfg.models):
        if m.name.startswith("openai/") and m.api_base and not os.environ.get("OPENAI_API_KEY"):
            out.append(
                Check(
                    f"dynamics.models.{i}.name",
                    f"{m.name} with api_base {m.api_base}: LiteLLM's openai/ connector refuses "
                    "to send a request without an OPENAI_API_KEY, so every call would fail and "
                    "every agent would keep its opinion. For a vLLM server use "
                    f"hosted_vllm/{m.name.removeprefix('openai/')} instead.",
                    "vllm",
                    "Quick start › Running on a vLLM server",
                    level="error",
                )
            )
    return out


# ---------------------------------------------------------------- entry points


def replicate_check(replicate: int) -> list[Check]:
    if not replicate:
        return []
    return [
        Check(
            "replicate",
            f"replicate = {replicate} has no effect on a numeric run: the rules are "
            "deterministic, so every replicate of the same seed gives the same run.",
            "seeds",
            "How a run works › Seeds and replicates",
        )
    ]


def config_checks(cfg: ExperimentConfig) -> list[Check]:
    out = network_checks(cfg.network) + opinion_checks(cfg.opinions)
    d = cfg.dynamics
    if d.type == "numeric":
        out += numeric_checks(d) + replicate_check(cfg.replicate)
    else:
        out += interaction_checks(d.interactions, cfg.network) + model_checks(d)
    return out


def raw_checks(raw: dict) -> list[dict]:
    """Checks for a possibly incomplete config (the lab, while it is being filled in).

    Each section is validated on its own, so a missing model does not hide a network
    warning. A section that does not validate is reported as an error.
    """
    out: list[Check] = []

    def parse(model: type[BaseModel], data, prefix: str):
        try:
            return model.model_validate(data if data is not None else {})
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(p) for p in (prefix, *err["loc"]) if p != "")
                msg = err["msg"].removeprefix("Value error, ")
                out.append(Check(loc, msg, "", "", level="error"))
            return None

    net = parse(NetworkConfig, raw.get("network"), "network")
    if net:
        out += network_checks(net)
    op = parse(InitialOpinions, raw.get("opinions"), "opinions")
    if op:
        out += opinion_checks(op)
    d = raw.get("dynamics") or {}
    if d.get("type") == "numeric":
        num = parse(NumericDynamics, d, "dynamics")
        if num:
            out += numeric_checks(num)
    elif d.get("type") == "llm":
        inter = parse(InteractionsConfig, d.get("interactions"), "dynamics.interactions")
        if inter:
            out += interaction_checks(inter, net)
        if d.get("models"):
            llm = parse(LLMDynamics, d, "dynamics")
            if llm:
                out += model_checks(llm)
    return [c.to_dict() for c in out]
