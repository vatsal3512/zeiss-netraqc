"""Download one MVTec AD category from a Hugging Face mirror (n good train images + full test set).

MVTec AD is CC BY-NC-SA 4.0, research use only (Bergmann et al., CVPR 2019).

    python download_mvtec.py --category metal_nut --n-train 20 --out ../netraqc_data
"""
import argparse
import csv
import io
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://huggingface.co/datasets/TheoM55/mvtec_anomaly_detection/resolve/main/"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="metal_nut")
    ap.add_argument("--n-train", type=int, default=20, help="how many defect-free training images to fetch")
    ap.add_argument("--out", default="../netraqc_data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(io.StringIO(fetch(BASE + "metadata.csv").decode("utf-8"))))
    rows = [r for r in rows if r["object"] == a.category]
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["path"])
    random.Random(a.seed).shuffle(train)
    train = train[: a.n_train]
    test = [r for r in rows if r["split"] == "test"]

    jobs = []
    for r in train + test:
        jobs.append(("images/" + r["path"], Path(a.out) / r["path"]))
        if r["mask_path"]:
            jobs.append(("masks/" + r["mask_path"], Path(a.out) / r["mask_path"]))

    def get(job):
        remote, local = job
        if local.exists():
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(fetch(BASE + remote))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, jobs))
    print(f"{a.category}: {len(train)} train (good only), {len(test)} test images -> {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
