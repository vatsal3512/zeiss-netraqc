"""PatchCore-style anomaly detection (Roth et al., CVPR 2022) on CPU.

Memory bank and threshold come from defect-free images only. Saves metrics, heatmaps and demo frames.

    python netraqc_patchcore.py --data ../netraqc_data --category metal_nut --out results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- model
class FeatureExtractor(torch.nn.Module):
    """layer2 + layer3 features, avg pooled, upsampled to layer2 size"""

    def __init__(self, backbone: str):
        super().__init__()
        if backbone == "resnet18":
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif backbone == "wide_resnet50_2":
            net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(backbone)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)  # B x C x h x w


def load(path, size):
    im = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0)
    return im, ((t.permute(2, 0, 1).unsqueeze(0) - MEAN) / STD)


def patches(feat):
    return feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])


def coreset(x, ratio, seed=0):
    """Greedy k-center subsampling on a random projection (as in PatchCore)."""
    n = max(1, int(len(x) * ratio))
    g = torch.Generator().manual_seed(seed)
    proj = x @ torch.randn(x.shape[1], 128, generator=g)
    sel = [int(torch.randint(len(x), (1,), generator=g))]
    mind = torch.cdist(proj, proj[sel[-1]:sel[-1] + 1]).squeeze(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        sel.append(i)
        mind = torch.minimum(mind, torch.cdist(proj, proj[i:i + 1]).squeeze(1))
    return x[sel]


def nn_dist(p, bank, chunk=4096):
    out = []
    for i in range(0, len(p), chunk):
        out.append(torch.cdist(p[i:i + chunk], bank).min(1).values)
    return torch.cat(out)


def anomaly_map(feat, bank, size):
    h, w = feat.shape[-2:]
    d = nn_dist(patches(feat), bank).view(1, 1, h, w)
    m = F.interpolate(d, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(m, sigma=4)


# ---------------------------------------------------------------- visuals
def jet(x):
    """Minimal jet colormap for x in [0,1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def font(sz):
    for f in ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def overlay(im, amap, vmin, vmax, out_size=700):
    base = im.resize((out_size, out_size), Image.BILINEAR)
    a = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat = Image.fromarray(jet(a)).resize((out_size, out_size), Image.BILINEAR)
    alpha = Image.fromarray((np.clip(a * 1.6 - 0.25, 0, 0.65) * 255).astype(np.uint8)).resize((out_size, out_size))
    return Image.composite(heat, base, alpha)


def verdict_frame(im, amap, score, thr, vmin, vmax, label, size=700):
    """input | heatmap with a PASS/FAIL bar on top"""
    fail = score > thr
    W, H, bar = size * 2 + 30, size + 150, 110
    fr = Image.new("RGB", (W, H), (26, 29, 33))
    fr.paste(im.resize((size, size), Image.BILINEAR), (10, bar))
    fr.paste(overlay(im, amap, vmin, vmax, size), (size + 20, bar))
    d = ImageDraw.Draw(fr)
    col = (214, 69, 69) if fail else (46, 158, 91)
    d.rounded_rectangle((10, 18, 330, 92), 37, fill=col)
    d.text((170, 55), "FAIL" if fail else "PASS", font=font(46), fill="white", anchor="mm")
    d.text((360, 30), f"anomaly score {score:.2f}  ·  threshold {thr:.2f}", font=font(28), fill=(201, 206, 214))
    d.text((360, 66), f"ground truth: {label}", font=font(26), fill=(242, 169, 0))
    d.text((10 + size // 2, H - 20), "input", font=font(24), fill=(201, 206, 214), anchor="mm")
    d.text((size + 20 + size // 2, H - 20), "NetraQC heatmap", font=font(24), fill=(201, 206, 214), anchor="mm")
    return fr


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--category", default="metal_nut")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "wide_resnet50_2"])
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (a Raspberry Pi 5 has 4 cores)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    root, out = Path(a.data), Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    model = FeatureExtractor(a.backbone)

    # memory bank from good images
    train_paths = sorted((root / "train" / a.category / "good").glob("*.png"))[: a.n_train]
    t0 = time.perf_counter()
    feats = [model(load(p, a.size)[1]) for p in train_paths]
    per_img = [patches(f) for f in feats]
    bank = coreset(torch.cat(per_img), a.coreset)
    train_s = time.perf_counter() - t0

    # threshold = max leave-one-out score on the good images
    loo = []
    for i in range(len(per_img)):
        others = coreset(torch.cat(per_img[:i] + per_img[i + 1:]), a.coreset, seed=i + 1)
        loo.append(float(anomaly_map(feats[i], others, a.size).max()))
    thr = max(loo)

    # test
    rows, maps, lat = [], {}, []
    for p in sorted((root / "test" / a.category).rglob("*.png")):
        if p.stem.endswith("_mask"):
            continue
        defect = p.parent.name
        t1 = time.perf_counter()
        im, x = load(p, a.size)
        m = anomaly_map(model(x), bank, a.size)
        lat.append(time.perf_counter() - t1)
        mask_p = root / "test" / a.category / defect / (p.stem + "_mask.png")
        mask = (np.asarray(Image.open(mask_p).convert("L").resize((a.size, a.size), Image.NEAREST)) > 127) \
            if mask_p.exists() else np.zeros((a.size, a.size), bool)
        rows.append({"file": f"{defect}/{p.name}", "defect": defect, "score": float(m.max())})
        maps[rows[-1]["file"]] = (im, m, mask)

    y = np.array([r["defect"] != "good" for r in rows])
    s = np.array([r["score"] for r in rows])
    pred = s > thr
    per_type = {}
    for t in sorted({r["defect"] for r in rows} - {"good"}):
        sel = np.array([r["defect"] in ("good", t) for r in rows])
        per_type[t] = round(float(roc_auc_score(y[sel], s[sel])), 4)
    pix_y = np.concatenate([maps[k][2].ravel() for k in maps])
    pix_s = np.concatenate([maps[k][1].ravel() for k in maps])

    res = {
        "category": a.category, "backbone": a.backbone, "train_images_good_only": len(train_paths),
        "defect_images_used_for_training": 0,
        "image_auroc": round(float(roc_auc_score(y, s)), 4),
        "pixel_auroc": round(float(roc_auc_score(pix_y, pix_s)), 4),
        "image_auroc_by_defect": per_type,
        "threshold_from_good_parts_only": round(thr, 4),
        "at_threshold": {
            "defects_caught": f"{int((pred & y).sum())}/{int(y.sum())}",
            "detection_rate": round(float((pred & y).sum() / y.sum()), 4),
            "false_alarms_on_good": f"{int((pred & ~y).sum())}/{int((~y).sum())}",
            "accuracy": round(float((pred == y).mean()), 4),
        },
        "memory_bank_patches": int(bank.shape[0]), "memory_bank_mb": round(bank.numel() * 4 / 1e6, 2),
        "backbone_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "train_seconds": round(train_s, 1),
        "latency_ms_per_part_mean": round(1000 * float(np.mean(lat[1:])), 1),
        "cpu_threads": a.threads, "test_images": len(rows),
        "hardware_note": "Latency measured on the machine that ran this script, not on a Raspberry Pi.",
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "scores.json").write_text(json.dumps(rows, indent=2))

    # same colour scale for all heatmaps
    vmin = float(np.percentile(pix_s, 50))
    vmax = float(np.percentile(pix_s, 99.9))
    order = ["scratch", "bent", "color", "flip", "good"]
    picks = []
    for t in order:
        cands = sorted([r for r in rows if r["defect"] == t], key=lambda r: -r["score"] if t != "good" else r["score"])
        picks += cands[:3]
    reel = []
    for r in picks:
        im, m, _ = maps[r["file"]]
        fr = verdict_frame(im, m, r["score"], thr, vmin, vmax, r["defect"])
        fr.save(out / "frames" / (r["file"].replace("/", "_")))
        reel.append(fr.resize((fr.width // 2, fr.height // 2)))
        overlay(im, m, vmin, vmax).save(out / ("heat_" + r["file"].replace("/", "_")))
    reel[0].save(out / "demo_reel.gif", save_all=True, append_images=reel[1:], duration=1800, loop=0)

    # frames for the missed defects
    (out / "missed").mkdir(exist_ok=True)
    for r in rows:
        if r["defect"] != "good" and r["score"] <= thr:
            im, m, _ = maps[r["file"]]
            name = r["file"].replace("/", "_")
            verdict_frame(im, m, r["score"], thr, vmin, vmax, r["defect"]).save(out / "missed" / name)
            overlay(im, m, vmin, vmax).save(out / "missed" / ("heat_" + name))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
