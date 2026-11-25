"""NetraQC Station web server."""
import base64
import csv
import io
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from station import db
from station.engine import HERE, Station

RESULTS = HERE.parent / "results"


def read_image(data):
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not an image we can read.")


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def create_app(db_path=HERE / "log.db", station=None):
    st = station or Station()
    db.init(db_path)
    app = FastAPI(title="NetraQC Station")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.mount("/samples", StaticFiles(directory=HERE / "samples"), name="samples")
    app.mount("/results", StaticFiles(directory=RESULTS), name="results")

    @app.get("/")
    def page():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/api/types")
    def types():
        return st.types()

    @app.get("/api/samples/{type_id}")
    def samples(type_id: str):
        info = load_json(HERE / "samples" / type_id / "samples.json")
        if info is None:
            return []
        return [{**s, "url": f"/samples/{type_id}/{s['file']}"} for s in info]

    @app.post("/api/inspect")
    def inspect(type: str = Form(...), image: UploadFile = File(...), label: str = Form("")):
        im = read_image(image.file.read())
        try:
            r = st.inspect(im, type)
        except KeyError:
            raise HTTPException(404, f"Unknown part type: {type}")
        png = r.pop("overlay_png")
        db.add(db_path, {**r, "file": image.filename, "label": label})
        return {**r, "overlay": "data:image/png;base64," + base64.b64encode(png).decode()}

    @app.post("/api/teach")
    def teach(name: str = Form(...), images: list[UploadFile] = File(...)):
        if len(images) < 5:
            raise HTTPException(400, "Need at least 5 good images.")
        try:
            return st.teach(name, [read_image(f.file.read()) for f in images])
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/log")
    def log():
        return {"rows": db.rows(db_path), "counts": db.counts(db_path)}

    @app.get("/api/log.csv")
    def log_csv():
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=db.COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(db.rows(db_path, limit=100000))
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=netraqc_log.csv"})

    @app.get("/api/design")
    def design():
        d = load_json(RESULTS / "design" / "design_summary.json") or {}
        return {"chosen": d.get("current_default"), "heavy": d.get("heavy_reference"), "smallest": d.get("frugal_pick"),
                "search": load_json(RESULTS / "design" / "search_eval.json"),
                "edge": load_json(RESULTS / "edge" / "edge_summary.json"),
                "all_categories": load_json(RESULTS / "all_categories" / "aggregate.json"),
                "threshold": {k: v for k, v in (load_json(RESULTS / "threshold" / "summary.json") or {}).items() if k != "per_category"},
                "weak_types": load_json(RESULTS / "weak_types" / "summary.json"),
                "types": st.types()}

    return app
