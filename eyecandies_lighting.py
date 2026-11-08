"""Lighting experiments on Eyecandies (image_0 box light, image_1-4 single lights, image_5 all lights).

1. does the all-lights image equal a sum of the single-light images?
2. one detector per light, combine with max of score/threshold, score all 63 light subsets

    python eyecandies_lighting.py --root ../eyecandies_data/Eyecandies/Lollipop --out results/eyecandies
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

from netraqc_patchcore import MEAN, STD, FeatureExtractor, anomaly_map, coreset, patches

CROP = None  # (top, bottom, left, right) in source pixels, set from --crop


def load(path, size):
    im = Image.open(path).convert("RGB")
    if CROP:
        im = im.crop((CROP[2], CROP[0], CROP[3], CROP[1]))
    t = torch.from_numpy(np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0)
    return im, (t.permute(2, 0, 1).unsqueeze(0) - MEAN) / STD


LIGHTS = {0: "box", 1: "cam1", 2: "cam2", 3: "cam3", 4: "cam4", 5: "all-cam"}


def additivity(test_dir, n=10):
    """least squares fit of image_5 from images 1-4, returns mean relative error"""
    def img(i, k):
        return np.asarray(Image.open(test_dir / f"{i:02d}_image_{k}.png"), dtype=np.float64) / 255

    def lin(a):
        return np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)

    out = {}
    for space, f in (("8-bit", lambda a: a), ("linear sRGB", lin)):
        for model in ("sum of 4 lights", "each light weighted", "each light weighted + box"):
            errs = []
            for i in range(n):
                ks = (1, 2, 3, 4) + ((0,) if "box" in model else ())
                X = np.stack([f(img(i, k)).ravel() for k in ks], 1)
                y = f(img(i, 5)).ravel()
                m = y > 0.002
                A = X[m][:, :4].sum(1, keepdims=True) if model == "sum of 4 lights" else X[m]
                c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                errs.append(float(np.abs(y[m] - A @ c).mean() / y[m].mean()))
            out[f"{space}, {model}"] = round(float(np.mean(errs)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../eyecandies_data/Eyecandies/Lollipop")
    ap.add_argument("--out", default="results/eyecandies")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--crop", default="", help="top,bottom,left,right in source pixels")
    a = ap.parse_args()
    global CROP
    CROP = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    torch.set_num_threads(a.threads)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test_dir, train_dir = root / "test_public" / "data", root / "train" / "data"

    res = {"dataset": f"Eyecandies {root.name}, public test split", "size": a.size, "crop": a.crop or None, "n_train": a.n_train, "additivity_relative_error": additivity(test_dir)}
    print(json.dumps(res["additivity_relative_error"], indent=1), flush=True)

    ids = sorted({p.name.split("_")[0] for p in test_dir.glob("*_metadata.yaml")})
    y = np.array([yaml.safe_load((test_dir / f"{i}_metadata.yaml").read_text())["anomalous"] == 1 for i in ids])
    res["test_samples"], res["defective"] = len(ids), int(y.sum())

    model = FeatureExtractor("resnet18")
    norm = {}
    for k in LIGHTS:
        train = [train_dir / f"{i:03d}_image_{k}.png" for i in range(a.n_train)]
        feats = [model(load(p, a.size)[1]) for p in train]
        per = [patches(f) for f in feats]
        bank = coreset(torch.cat(per), a.coreset)
        thr = max(float(anomaly_map(feats[j], coreset(torch.cat(per[:j] + per[j + 1:]), a.coreset, seed=j + 1), a.size).max())
                  for j in range(len(per)))
        s = np.array([float(anomaly_map(model(load(test_dir / f"{i}_image_{k}.png", a.size)[1]), bank, a.size).max()) for i in ids])
        norm[k] = s / thr
        print(f"light {k} ({LIGHTS[k]}): AUROC {roc_auc_score(y, s):.3f}", flush=True)

    subsets = []
    for r in range(1, 7):
        for S in itertools.combinations(LIGHTS, r):
            sc = np.max(np.stack([norm[k] for k in S]), 0)
            pred = sc > 1
            subsets.append({"lights": [LIGHTS[k] for k in S], "captures": r,
                            "image_auroc": round(float(roc_auc_score(y, sc)), 4),
                            "caught": f"{int((pred & y).sum())}/{int(y.sum())}",
                            "false_alarms": f"{int((pred & ~y).sum())}/{int((~y).sum())}"})
    best = max(s["image_auroc"] for s in subsets)
    fewest = min((s for s in subsets if s["image_auroc"] >= best - 0.01), key=lambda s: (s["captures"], -s["image_auroc"]))
    by_k = lambda names: next(s for s in subsets if s["lights"] == names)
    res.update({
        "single_shot_all_lights": by_k(["all-cam"]), "single_shot_box": by_k(["box"]),
        "four_single_lights": by_k(["cam1", "cam2", "cam3", "cam4"]),
        "best_subset": max(subsets, key=lambda s: (s["image_auroc"], -s["captures"])),
        "fewest_within_0.01_of_best": fewest,
        "best_by_captures": {r: max((s for s in subsets if s["captures"] == r), key=lambda s: s["image_auroc"]) for r in range(1, 7)},
        "note": "Selection on the same 50 test samples is optimistic; the private test labels are not public.",
    })
    (out / "subsets.json").write_text(json.dumps(subsets, indent=1))
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "additivity_relative_error"}, indent=1))


if __name__ == "__main__":
    main()
