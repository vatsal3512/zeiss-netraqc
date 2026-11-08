"""Grid search over detector design choices (backbone, input size, layers, coreset, n_train).

Scores every design on AUROC, latency and footprint. Designs are picked on a val half of the
MVTec test set and reported on the other half. design_search_eval.py compares search strategies.

    python design_optimizer.py --data ../netraqc_data --category metal_nut --out results/design
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from netraqc_patchcore import coreset, load, nn_dist

BACKBONES = ["resnet18", "wide_resnet50_2"]
SIZES = [128, 160, 192, 224, 256]
LAYERS = ["l2", "l2+l3"]
CORESETS = [0.01, 0.025, 0.05, 0.1, 0.25]
NTRAIN = [5, 10, 20]
DEFAULT = ("resnet18", 224, "l2+l3", 0.1, 20)          # the design reported elsewhere in the deck
REFERENCE = ("wide_resnet50_2", 256, "l2+l3", 0.25, 20)  # a heavy, PatchCore-paper-style design


class Extractor(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        w = {"resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
             "wide_resnet50_2": torchvision.models.Wide_ResNet50_2_Weights.DEFAULT}[name]
        net = getattr(torchvision.models, name)(weights=w)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        mb = lambda m: sum(p.numel() for p in m.parameters()) * 4 / 1e6
        self.mb = {"l2": mb(self.stem) + mb(self.layer2)}
        self.mb["l2+l3"] = self.mb["l2"] + mb(self.layer3)
        self.eval()

    @torch.no_grad()
    def forward(self, x, layers):
        f2 = self.layer2(self.stem(x))
        p2 = F.avg_pool2d(f2, 3, 1, 1)
        if layers == "l2":
            return p2
        f3 = F.avg_pool2d(self.layer3(f2), 3, 1, 1)
        return torch.cat([p2, F.interpolate(f3, size=p2.shape[-2:], mode="bilinear", align_corners=False)], 1)


def score(feat, bank, size):
    h, w = feat.shape[-2:]
    d = nn_dist(feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1]), bank).view(1, 1, h, w)
    m = F.interpolate(d, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    return float(gaussian_filter(m, sigma=4).max())


def split(test_paths, seed=0):
    """50/50 split of the test set per defect type"""
    rng, val, hold = np.random.default_rng(seed), [], []
    by_type = {}
    for p in test_paths:
        by_type.setdefault(p.parent.name, []).append(p)
    for t in sorted(by_type):
        ps = sorted(by_type[t])
        idx = rng.permutation(len(ps))
        k = len(ps) // 2
        val += [ps[i] for i in idx[:k]]
        hold += [ps[i] for i in idx[k:]]
    return val, hold


def pareto(rows, keys=("val_auroc", "latency_ms", "footprint_mb")):
    """non-dominated rows (max first key, min the rest)"""
    v = np.array([[-r[keys[0]]] + [r[k] for k in keys[1:]] for r in rows])
    keep = []
    for i in range(len(v)):
        dom = np.all(v <= v[i], 1) & np.any(v < v[i], 1)
        if not dom.any():
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--category", default="metal_nut")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/design")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_paths = sorted((root / "train" / a.category / "good").glob("*.png"))[: max(NTRAIN)]
    test_paths = [p for p in sorted((root / "test" / a.category).rglob("*.png")) if not p.stem.endswith("_mask")]
    val, hold = split(test_paths)
    y_val = np.array([p.parent.name != "good" for p in val])
    y_hold = np.array([p.parent.name != "good" for p in hold])

    rows = []
    for bb in BACKBONES:
        model = Extractor(bb)
        for size in SIZES:
            for layers in LAYERS:
                t0 = time.perf_counter()
                tr = [model(load(p, size)[1], layers) for p in train_paths]
                fv = [model(load(p, size)[1], layers) for p in val]
                fh = [model(load(p, size)[1], layers) for p in hold]
                feat_ms = 1000 * (time.perf_counter() - t0) / (len(tr) + len(fv) + len(fh))
                for cr, n in itertools.product(CORESETS, NTRAIN):
                    pts = torch.cat([f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in tr[:n]])
                    bank = coreset(pts, cr)
                    t1 = time.perf_counter()
                    sv = np.array([score(f, bank, size) for f in fv])
                    nn_ms = 1000 * (time.perf_counter() - t1) / len(fv)
                    sh = np.array([score(f, bank, size) for f in fh])
                    rows.append({
                        "backbone": bb, "size": size, "layers": layers, "coreset": cr, "n_train": n,
                        "val_auroc": round(float(roc_auc_score(y_val, sv)), 4),
                        "test_auroc": round(float(roc_auc_score(y_hold, sh)), 4),
                        "latency_ms": round(feat_ms + nn_ms, 1),
                        "footprint_mb": round(model.mb[layers] + bank.numel() * 4 / 1e6, 2),
                        "bank_mb": round(bank.numel() * 4 / 1e6, 2),
                    })
                print(f"{bb:16s} {size:3d} {layers:5s}  feat {feat_ms:6.1f} ms  best val "
                      f"{max(r['val_auroc'] for r in rows[-15:]):.3f}", flush=True)
        del model

    key = lambda r: (r["backbone"], r["size"], r["layers"], r["coreset"], r["n_train"])
    table = {key(r): r for r in rows}
    front = pareto(rows)
    front_keys = {key(rows[i]) for i in front}

    # pick on val, report on held-out
    best_val = max(r["val_auroc"] for r in rows)
    ok = [rows[i] for i in front if rows[i]["val_auroc"] >= best_val - 0.01]
    frugal = min(ok, key=lambda r: (r["footprint_mb"], r["latency_ms"]))
    fastest = min(ok, key=lambda r: (r["latency_ms"], r["footprint_mb"]))
    pick = lambda r: {k: r[k] for k in ("backbone", "size", "layers", "coreset", "n_train", "val_auroc",
                                          "test_auroc", "latency_ms", "footprint_mb", "bank_mb")}
    res = {
        "category": a.category, "grid_size": len(rows),
        "val_images": len(val), "heldout_images": len(hold),
        "true_pareto_designs": len(front),
        "best_val_auroc": best_val,
        "selection_rule": "smallest footprint on the validation Pareto front within 0.01 of the best validation AUROC",
        "frugal_pick": pick(frugal), "fastest_within_0.01": pick(fastest),
        "current_default": pick(table[DEFAULT]), "heavy_reference": pick(table[REFERENCE]),
        "cpu_threads": a.threads,
        "note": "Latency measured on the laptop CPU that ran this script, not on a Raspberry Pi.",
    }
    (out / "design_grid.json").write_text(json.dumps(rows, indent=1))
    (out / "design_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
