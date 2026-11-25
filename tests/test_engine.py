import json
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from station.engine import Station

SAMPLES = Path(__file__).resolve().parents[1] / "station" / "samples"


def sample(type_id, label="good"):
    info = json.loads((SAMPLES / type_id / "samples.json").read_text())
    f = next(s["file"] for s in info if (s["label"] == "good") == (label == "good"))
    return Image.open(SAMPLES / type_id / f)


@pytest.fixture(scope="module")
def st():
    return Station()


def test_types_lists_bundled(st):
    ids = [t["id"] for t in st.types()]
    assert "bottle" in ids and len(ids) == 15


def test_inspect_returns_verdict(st):
    r = st.inspect(sample("bottle"), "bottle")
    assert r["verdict"] in ("PASS", "FAIL")
    assert r["overlay_png"][:4] == b"\x89PNG"
    assert r["ms"] > 0 and r["threshold"] > 0


def test_defect_scores_higher_than_good(st):
    assert st.inspect(sample("bottle", "defect"), "bottle")["score"] > st.inspect(sample("bottle"), "bottle")["score"]


def test_unknown_type(st):
    with pytest.raises(KeyError):
        st.inspect(sample("bottle"), "no_such_part")


def test_greyscale_input(st):
    r = st.inspect(sample("bottle").convert("L"), "bottle")
    assert r["verdict"] in ("PASS", "FAIL")


def test_teach_needs_five(st):
    with pytest.raises(ValueError):
        st.teach("washer", [sample("bottle")] * 3)


def test_teach_builds_new_type(tmp_path):
    st = Station(banks_dir=tmp_path)
    base = sample("bottle")
    imgs = [base, ImageOps.mirror(base), ImageOps.flip(base), base.rotate(90), base.rotate(180), base.rotate(270)]
    t = st.teach("My bottle", imgs)
    assert t["id"] in [x["id"] for x in st.types()] and t["n_train"] == 6 and t["seconds"] > 0
    assert st.inspect(base, t["id"])["verdict"] in ("PASS", "FAIL")


def test_teach_does_not_overwrite_existing_type(tmp_path):
    st = Station(banks_dir=tmp_path)
    base = sample("bottle")
    imgs = [base, ImageOps.mirror(base), ImageOps.flip(base), base.rotate(90), base.rotate(180)]
    first = st.teach("Bottle", imgs)
    second = st.teach("Bottle", imgs)
    assert first["id"] != second["id"] and len(st.types()) == 2
