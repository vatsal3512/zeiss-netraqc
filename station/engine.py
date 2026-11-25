"""Inspection engine: INT8 feature extractor + one memory bank per part type."""
import io
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
MODEL_FILES = {"l2": "netraqc_l2_int8.onnx", "l2+l3": "netraqc_l23_int8.onnx"}

# threshold rules, all computed from the leave-one-out scores of the good images
RULES = {
    "max": lambda s: s.max(),
    "mean+2sd": lambda s: s.mean() + 2 * s.std(),
    "mean+3sd": lambda s: s.mean() + 3 * s.std(),
    "mean+4sd": lambda s: s.mean() + 4 * s.std(),
    "max*1.05": lambda s: 1.05 * s.max(),
    "max*1.10": lambda s: 1.10 * s.max(),
}


def to_input(img, size):
    x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return (x.transpose(2, 0, 1)[None] - MEAN) / STD


def patches(feat):
    return feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])


def coreset(x, ratio, seed=0):
    """greedy k-center on a random projection (PatchCore); returns the chosen row indices"""
    n = max(1, int(len(x) * ratio))
    g = torch.Generator().manual_seed(seed)
    proj = x @ torch.randn(x.shape[1], 128, generator=g)
    sel = [int(torch.randint(len(x), (1,), generator=g))]
    mind = torch.cdist(proj, proj[sel[-1]:sel[-1] + 1]).squeeze(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        sel.append(i)
        mind = torch.minimum(mind, torch.cdist(proj, proj[i:i + 1]).squeeze(1))
    return torch.tensor(sel)


def anomaly_map(feat, bank, size):
    h, w = feat.shape[-2:]
    p = patches(feat)
    d = torch.cat([torch.cdist(p[i:i + 4096], bank).min(1).values for i in range(0, len(p), 4096)]).view(1, 1, h, w)
    m = F.interpolate(d, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(m, sigma=4)


def jet(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay_png(img, amap, vmin, vmax, out=512):
    base = img.convert("RGB").resize((out, out), Image.BILINEAR)
    a = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat = Image.fromarray(jet(a)).resize((out, out), Image.BILINEAR)
    alpha = Image.fromarray((np.clip(a * 1.6 - 0.25, 0, 0.65) * 255).astype(np.uint8)).resize((out, out))
    buf = io.BytesIO()
    Image.composite(heat, base, alpha).save(buf, "PNG")
    return buf.getvalue()


class Station:
    def __init__(self, models_dir=HERE / "models", banks_dir=HERE / "banks", threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.sessions = {k: ort.InferenceSession(str(Path(models_dir) / f), so, providers=["CPUExecutionProvider"])
                         for k, f in MODEL_FILES.items()}
        self.banks_dir = Path(banks_dir)
        self.banks_dir.mkdir(parents=True, exist_ok=True)
        self.banks = {p.stem: self._load(p) for p in sorted(self.banks_dir.glob("*.npz"))}

    @staticmethod
    def _load(path):
        z = np.load(path, allow_pickle=False)
        return {"bank": torch.from_numpy(z["bank"].astype(np.float32)), "threshold": float(z["threshold"]),
                "size": int(z["size"]), "layers": str(z["layers"]), "rule": str(z["rule"]), "name": str(z["name"]),
                "n_train": int(z["n_train"]), "vmin": float(z["vmin"]), "vmax": float(z["vmax"]), "loo": z["loo"]}

    def set_rule(self, type_id, rule):
        """switch the threshold rule using the stored leave-one-out scores (no rebuild)"""
        b = self.banks[type_id]
        b["rule"], b["threshold"] = rule, float(RULES[rule](b["loo"]))
        b["vmax"] = 1.5 * b["threshold"]
        self.save(type_id, b["name"], b)

    def features(self, img, size, layers):
        return torch.from_numpy(self.sessions[layers].run(None, {"image": to_input(img, size)})[0])

    def types(self):
        return [{"id": k, "name": b["name"], "size": b["size"], "layers": b["layers"], "threshold": round(b["threshold"], 3),
                 "rule": b["rule"], "n_train": b["n_train"], "bank_mb": round(b["bank"].numel() * 2 / 1e6, 2)}
                for k, b in self.banks.items()]

    def inspect(self, img, type_id):
        if type_id not in self.banks:
            raise KeyError(type_id)
        b = self.banks[type_id]
        t0 = time.perf_counter()
        amap = anomaly_map(self.features(img, b["size"], b["layers"]), b["bank"], b["size"])
        score = float(amap.max())
        ms = (time.perf_counter() - t0) * 1000
        return {"type": type_id, "score": round(score, 3), "threshold": round(b["threshold"], 3),
                "verdict": "FAIL" if score > b["threshold"] else "PASS", "ms": round(ms, 1),
                "overlay_png": overlay_png(img, amap, b["vmin"], b["vmax"])}

    def build(self, images, size=224, layers="l2+l3", ratio=0.10, rule="max"):
        """memory bank + threshold from good images only"""
        feats = [self.features(im, size, layers) for im in images]
        per = [patches(f) for f in feats]
        allp = torch.cat(per)
        origin = torch.cat([torch.full((len(p),), i) for i, p in enumerate(per)])
        idx = coreset(allp, ratio)
        bank, from_img = allp[idx], origin[idx]
        loo, normal = [], []
        for i in range(len(per)):
            # leave image i out: score it against the memory without its own patches
            m = anomaly_map(feats[i], bank[from_img != i], size)
            loo.append(float(m.max()))
            normal.append(float(np.median(m)))
        thr = float(RULES[rule](np.array(loo)))
        return {"bank": bank, "threshold": thr, "size": size, "layers": layers, "rule": rule, "loo": np.array(loo),
                "n_train": len(images), "vmin": float(np.median(normal)), "vmax": 1.5 * thr}

    def save(self, type_id, name, b):
        np.savez_compressed(self.banks_dir / f"{type_id}.npz", bank=b["bank"].numpy().astype(np.float16),
                            threshold=b["threshold"], size=b["size"], layers=b["layers"], rule=b["rule"], name=name,
                            n_train=b["n_train"], vmin=b["vmin"], vmax=b["vmax"], loo=b["loo"])
        self.banks[type_id] = self._load(self.banks_dir / f"{type_id}.npz")

    def teach(self, name, images, rule="max"):
        if len(images) < 5:
            raise ValueError("need at least 5 good images")
        t0 = time.perf_counter()
        type_id = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "part"
        self.save(type_id, name.strip() or type_id, self.build(images, rule=rule))
        return {"id": type_id, "threshold": round(self.banks[type_id]["threshold"], 3), "n_train": len(images),
                "seconds": round(time.perf_counter() - t0, 1)}
