import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from station.server import create_app

SAMPLES = Path(__file__).resolve().parents[1] / "station" / "samples"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(db_path=tmp_path_factory.mktemp("db") / "log.db"))


def sample_bytes(type_id="bottle"):
    f = json.loads((SAMPLES / type_id / "samples.json").read_text())[0]["file"]
    return (SAMPLES / type_id / f).read_bytes()


def test_page_and_types(client):
    assert client.get("/").status_code == 200
    assert any(t["id"] == "bottle" for t in client.get("/api/types").json())


def test_inspect_sample_is_logged(client):
    before = client.get("/api/log").json()["counts"]["total"]
    r = client.post("/api/inspect", data={"type": "bottle"}, files={"image": ("a.jpg", sample_bytes(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["verdict"] in ("PASS", "FAIL")
    assert r.json()["overlay"].startswith("data:image/png;base64,")
    assert client.get("/api/log").json()["counts"]["total"] == before + 1


def test_corrupt_upload_is_400(client):
    r = client.post("/api/inspect", data={"type": "bottle"}, files={"image": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_unknown_type_is_404(client):
    r = client.post("/api/inspect", data={"type": "nope"}, files={"image": ("a.jpg", sample_bytes(), "image/jpeg")})
    assert r.status_code == 404


def test_teach_needs_five(client):
    files = [("images", (f"{i}.jpg", sample_bytes(), "image/jpeg")) for i in range(3)]
    assert client.post("/api/teach", data={"name": "washer"}, files=files).status_code == 400


def test_csv_and_design(client):
    csv = client.get("/api/log.csv").text
    assert csv.splitlines()[0].startswith("time,type")
    assert "chosen" in client.get("/api/design").json()
