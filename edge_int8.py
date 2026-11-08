"""Export the feature extractor to ONNX, quantise to INT8 and compare against FP32.

Calibration uses the defect-free training images only. Timing is on 4 CPU threads.

    python edge_int8.py --data ../netraqc_data --categories metal_nut screw carpet --out results/edge
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import roc_auc_score

from netraqc_patchcore import FeatureExtractor, anomaly_map, coreset, load, patches


class Calib(CalibrationDataReader):
    def __init__(self, paths, size, name):
        self.it = iter([{name: load(p, size)[1].numpy()} for p in paths])

    def get_next(self):
        return next(self.it, None)


def ort_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def evaluate(extract, root, cat, a):
    train = sorted((root / "train" / cat / "good").glob("*.png"))[: a.n_train]
    feats = [extract(load(p, a.size)[1]) for p in train]
    per = [patches(f) for f in feats]
    bank = coreset(torch.cat(per), a.coreset)
    thr = max(float(anomaly_map(feats[i], coreset(torch.cat(per[:i] + per[i + 1:]), a.coreset, seed=i + 1), a.size).max())
              for i in range(len(per)))
    ys, ss, lat = [], [], []
    for p in sorted((root / "test" / cat).rglob("*.png")):
        if p.stem.endswith("_mask"):
            continue
        x = load(p, a.size)[1]
        t0 = time.perf_counter()
        s = float(anomaly_map(extract(x), bank, a.size).max())
        lat.append(time.perf_counter() - t0)
        ys.append(p.parent.name != "good")
        ss.append(s)
    y, s = np.array(ys), np.array(ss)
    return {"image_auroc": round(float(roc_auc_score(y, s)), 4), "caught": f"{int(((s > thr) & y).sum())}/{int(y.sum())}",
            "false_alarms": f"{int(((s > thr) & ~y).sum())}/{int((~y).sum())}",
            "ms_per_image": round(1000 * float(np.median(lat)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--categories", nargs="*", default=["metal_nut"])
    ap.add_argument("--calib-category", default="metal_nut")
    ap.add_argument("--out", default="results/edge")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = FeatureExtractor("resnet18")
    fp32, pre, int8 = out / "netraqc_fp32.onnx", out / "netraqc_pre.onnx", out / "netraqc_int8.onnx"
    torch.onnx.export(model, (torch.zeros(1, 3, a.size, a.size),), str(fp32), input_names=["image"],
                      output_names=["features"], opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(pre))
    calib = sorted((root / "train" / a.calib_category / "good").glob("*.png"))[: a.n_train]
    quantize_static(str(pre), str(int8), Calib(calib, a.size, "image"), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    pre.unlink()

    s32, s8 = ort_session(fp32, a.threads), ort_session(int8, a.threads)
    engines = {
        "onnx_fp32": lambda x: torch.from_numpy(s32.run(None, {"image": x.numpy()})[0]),
        "onnx_int8": lambda x: torch.from_numpy(s8.run(None, {"image": x.numpy()})[0]),
    }
    res = {"model_mb": {"onnx_fp32": round(fp32.stat().st_size / 1e6, 1), "onnx_int8": round(int8.stat().st_size / 1e6, 1)},
           "calibration": f"{len(calib)} defect-free {a.calib_category} training images", "threads": a.threads,
           "note": "Laptop CPU limited to 4 threads; not a Raspberry Pi measurement. ms = feature extraction + memory search.",
           "categories": {}}
    for cat in a.categories:
        res["categories"][cat] = {name: evaluate(f, root, cat, a) for name, f in engines.items()}
        print(cat, json.dumps(res["categories"][cat]), flush=True)
        (out / "edge_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["model_mb"]))


if __name__ == "__main__":
    main()
