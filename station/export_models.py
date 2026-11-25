"""Export the ResNet-18 feature extractor to INT8 ONNX, one model per feature depth.

Any input size works (dynamic H, W). Calibrated on defect-free metal nut images only.

    python -m station.export_models --data ../netraqc_data
"""
import argparse
from pathlib import Path

import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from design_optimizer import Extractor
from netraqc_patchcore import load

MODELS = Path(__file__).parent / "models"


class Depth(torch.nn.Module):
    def __init__(self, ext, layers):
        super().__init__()
        self.ext, self.layers = ext, layers

    def forward(self, x):
        return self.ext(x, self.layers)


class Calib(CalibrationDataReader):
    def __init__(self, paths, size):
        self.it = iter([{"image": load(p, size)[1].numpy()} for p in paths])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--calib-category", default="metal_nut")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    ext = Extractor("resnet18")
    calib = sorted((Path(a.data) / "train" / a.calib_category / "good").glob("*.png"))[:20]
    for layers, tag in (("l2", "l2"), ("l2+l3", "l23")):
        fp32, pre, int8 = MODELS / f"tmp_{tag}.onnx", MODELS / f"tmp_{tag}_pre.onnx", MODELS / f"netraqc_{tag}_int8.onnx"
        torch.onnx.export(Depth(ext, layers).eval(), (torch.zeros(1, 3, 224, 224),), str(fp32), input_names=["image"],
                          output_names=["features"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {2: "h", 3: "w"}, "features": {2: "fh", 3: "fw"}})
        quant_pre_process(str(fp32), str(pre))
        quantize_static(str(pre), str(int8), Calib(calib, 224), quant_format=QuantFormat.QDQ, per_channel=True,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
        fp32.unlink()
        pre.unlink()
        print(int8.name, round(int8.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
