"""Run the default detector on all 15 MVTec AD categories, same settings everywhere.

    python run_all_categories.py --data ../netraqc_data --out results/all_categories
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from netraqc_patchcore import FeatureExtractor, anomaly_map, coreset, load, overlay, patches

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
              "tile", "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def run(cat, root, model, a, out):
    train_paths = sorted((root / "train" / cat / "good").glob("*.png"))[: a.n_train]
    feats = [model(load(p, a.size)[1]) for p in train_paths]
    per_img = [patches(f) for f in feats]
    bank = coreset(torch.cat(per_img), a.coreset)
    loo = []
    for i in range(len(per_img)):
        others = coreset(torch.cat(per_img[:i] + per_img[i + 1:]), a.coreset, seed=i + 1)
        loo.append(float(anomaly_map(feats[i], others, a.size).max()))
    thr = max(loo)

    rows, lat, pix_y, pix_s, keep = [], [], [], [], {}
    for p in sorted((root / "test" / cat).rglob("*.png")):
        if p.stem.endswith("_mask"):
            continue
        defect = p.parent.name
        t0 = time.perf_counter()
        im, x = load(p, a.size)
        m = anomaly_map(model(x), bank, a.size)
        lat.append(time.perf_counter() - t0)
        mp = root / "test" / cat / defect / (p.stem + "_mask.png")
        mask = (np.asarray(Image.open(mp).convert("L").resize((a.size, a.size), Image.NEAREST)) > 127) \
            if mp.exists() else np.zeros((a.size, a.size), bool)
        pix_y.append(mask.ravel())
        pix_s.append(m.ravel())
        rows.append({"file": f"{defect}/{p.name}", "defect": defect, "score": float(m.max())})
        keep[rows[-1]["file"]] = (im, m)

    y = np.array([r["defect"] != "good" for r in rows])
    s = np.array([r["score"] for r in rows])
    pred = s > thr
    pix_s = np.concatenate(pix_s)
    res = {
        "category": cat, "kind": "texture" if cat in TEXTURES else "object",
        "train_images_good_only": len(train_paths), "test_images": len(rows),
        "image_auroc": round(float(roc_auc_score(y, s)), 4),
        "pixel_auroc": round(float(roc_auc_score(np.concatenate(pix_y), pix_s)), 4),
        "threshold": round(thr, 4),
        "caught": int((pred & y).sum()), "defective": int(y.sum()),
        "false_alarms": int((pred & ~y).sum()), "good": int((~y).sum()),
        "latency_ms_mean": round(1000 * float(np.mean(lat[1:])), 1),
    }

    # save heatmap of the top scoring defect
    vmin, vmax = float(np.percentile(pix_s, 50)), float(np.percentile(pix_s, 99.9))
    top = max((r for r in rows if r["defect"] != "good"), key=lambda r: r["score"])
    im, m = keep[top["file"]]
    overlay(im, m, vmin, vmax, out_size=400).save(out / "heatmaps" / f"{cat}.png")
    (out / "scores" / f"{cat}.json").write_text(json.dumps(rows, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--out", default="results/all_categories")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    root, out = Path(a.data), Path(a.out)
    for d in ("heatmaps", "scores"):
        (out / d).mkdir(parents=True, exist_ok=True)
    model = FeatureExtractor("resnet18")

    summary_p = out / "summary.json"
    done = {r["category"]: r for r in json.loads(summary_p.read_text())} if summary_p.exists() else {}
    for cat in a.categories:
        if cat in done:
            continue
        r = run(cat, root, model, a, out)
        done[cat] = r
        summary_p.write_text(json.dumps([done[c] for c in CATEGORIES if c in done], indent=2))
        print(f"{cat:11s} img {r['image_auroc']:.3f}  pix {r['pixel_auroc']:.3f}  caught {r['caught']}/{r['defective']}"
              f"  FA {r['false_alarms']}/{r['good']}  {r['latency_ms_mean']} ms", flush=True)

    rs = [done[c] for c in CATEGORIES if c in done]
    if len(rs) == len(CATEGORIES):
        agg = {
            "mean_image_auroc": round(float(np.mean([r["image_auroc"] for r in rs])), 4),
            "mean_pixel_auroc": round(float(np.mean([r["pixel_auroc"] for r in rs])), 4),
            "mean_image_auroc_objects": round(float(np.mean([r["image_auroc"] for r in rs if r["kind"] == "object"])), 4),
            "mean_image_auroc_textures": round(float(np.mean([r["image_auroc"] for r in rs if r["kind"] == "texture"])), 4),
            "caught": f"{sum(r['caught'] for r in rs)}/{sum(r['defective'] for r in rs)}",
            "false_alarms": f"{sum(r['false_alarms'] for r in rs)}/{sum(r['good'] for r in rs)}",
            "note": "Laptop CPU, 4 threads. One fixed design and threshold rule for all categories; no per-category tuning.",
        }
        (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
        print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
