"""Small design search for the weakest MVTec types: input size x feature depth x coreset.

Picked on the val half of each test set, reported on the held-out half (same split as design_optimizer.py).

    python weak_types.py --data ../netraqc_data --out results/weak_types
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from design_optimizer import Extractor, score, split
from netraqc_patchcore import coreset, load

WEAK = ["screw", "grid", "capsule", "pill", "zipper"]
SIZES, LAYERS, CORESETS = [224, 320], ["l2", "l2+l3"], [0.10, 0.25]
DEFAULT = (224, "l2+l3", 0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--out", default="results/weak_types")
    ap.add_argument("--categories", nargs="*", default=WEAK)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Extractor("resnet18")

    res = {}
    for cat in a.categories:
        train = sorted((root / "train" / cat / "good").glob("*.png"))[: a.n_train]
        test = [p for p in sorted((root / "test" / cat).rglob("*.png")) if not p.stem.endswith("_mask")]
        val, hold = split(test)
        yv = np.array([p.parent.name != "good" for p in val])
        yh = np.array([p.parent.name != "good" for p in hold])
        rows = []
        for size, layers in itertools.product(SIZES, LAYERS):
            tr = [model(load(p, size)[1], layers) for p in train]
            fv = [model(load(p, size)[1], layers) for p in val]
            fh = [model(load(p, size)[1], layers) for p in hold]
            pts = torch.cat([f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in tr])
            for cr in CORESETS:
                bank = coreset(pts, cr)
                rows.append({"size": size, "layers": layers, "coreset": cr,
                             "val_auroc": round(float(roc_auc_score(yv, [score(f, bank, size) for f in fv])), 4),
                             "held_out_auroc": round(float(roc_auc_score(yh, [score(f, bank, size) for f in fh])), 4),
                             "bank_mb": round(bank.numel() * 4 / 1e6, 2)})
        default = next(r for r in rows if (r["size"], r["layers"], r["coreset"]) == DEFAULT)
        pick = max(rows, key=lambda r: (r["val_auroc"], -r["bank_mb"]))
        res[cat] = {"default": default, "pick": pick, "all": rows}
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        print(f"{cat}: default held-out {default['held_out_auroc']}  pick {pick['size']} {pick['layers']} {pick['coreset']} "
              f"val {pick['val_auroc']} held-out {pick['held_out_auroc']}", flush=True)


if __name__ == "__main__":
    main()
