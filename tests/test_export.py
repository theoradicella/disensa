import asyncio
import json
import re

from disensa.analysis import load_run
from disensa.config import ExperimentConfig
from disensa.engine import Simulation
from disensa.export import build_payload, write_viewer
from disensa.runs import RunWriter, new_run_dir


def _run(tmp_path, **sim):
    cfg = ExperimentConfig.model_validate(
        {
            "name": "v",
            "seed": 1,
            "network": {"topology": "barabasi_albert", "n": 15, "m": 2},
            "simulation": {"days": 3, **sim},
            "models": [{"name": "fake/a"}, {"name": "fake/stubborn"}],
            "scorer": {"name": "fake/scorer"},
            "llm": {"cache_dir": str(tmp_path / "c"), "max_concurrency": 32},
            "output_dir": str(tmp_path / "runs"),
        }
    )
    run_dir = new_run_dir(cfg)
    writer = RunWriter(run_dir)
    asyncio.run(Simulation(cfg).run(writer))
    writer.close()
    return run_dir


def test_payload_shape(tmp_path):
    p = build_payload(load_run(_run(tmp_path)))
    assert len(p["agents"]) == 15 and len(p["positions"]) == 15
    assert len(p["opinions"]) == 4 and len(p["posts"]) == 4
    assert all(0 <= c <= 1 for xy in p["positions"] for c in xy)
    assert p["models"] == ["fake/a", "fake/stubborn"]
    assert "mean_opinion[fake/a]" in p["metrics"][0]
    json.dumps(p, allow_nan=False)  # strict JSON: no NaN sneaks through


def test_viewer_embeds_data_safely(tmp_path):
    run_dir = _run(tmp_path)
    out = write_viewer(run_dir)
    html = out.read_text()
    assert "/*__DISENSA_DATA__*/null" not in html
    m = re.search(r"window.DISENSA_DATA = (.*?);\n", html, re.S)
    assert json.loads(m.group(1))["name"] == "v"
    # model text containing </script> must not break out of the script tag
    assert html.count("</script>") == html.count("<script")
    assert 'src="/d3.min.js"' not in html  # d3 inlined, so the file works offline
