"""Start NetraQC Station: python app.py, then open http://localhost:8000"""
import argparse
import webbrowser

import uvicorn

from station.server import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        webbrowser.open(f"http://localhost:{a.port}")
    uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")
