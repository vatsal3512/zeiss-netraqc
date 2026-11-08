# NetraQC demo: learns "good" from 20 parts, flags defects it has never seen

PatchCore-style anomaly detection in plain PyTorch, sized for CPU and edge devices.
No defect images are used anywhere: the model and the pass/fail threshold both come from good parts only.

## Run it (CPU is enough)

```bash
pip install -r requirements.txt
python download_mvtec.py --category metal_nut --n-train 20 --out ../netraqc_data
python netraqc_patchcore.py --data ../netraqc_data --category metal_nut --out results
```

Outputs in `results/`:

| File | What it is |
|---|---|
| `results.json` | Headline metrics (see below) |
| `scores.json` | Per-image anomaly score |
| `frames/*.png` | Input + heatmap + PASS/FAIL frames, ready for the video |
| `heat_*.png` | Heatmap overlays used on the slides |
| `demo_clip.gif` | 8-second clip for the video: scratch, bent, stain (caught), one missed scratch, one good part |
| `missed/` | Frames and heatmaps for every defect the detector missed |
| `demo_reel.gif` | 15-frame reel cycling through scratch, bent, colour, flip and good parts |

## Our runs (14–15 Oct 2026, laptop CPU, 4 threads)

MVTec AD `metal_nut`, ResNet-18 backbone, trained on **20 good images, 0 defect images**:

| Metric | Result |
|---|---|
| Image AUROC | **0.985** |
| Pixel AUROC (heatmap localisation) | **0.981** |
| AUROC by defect type | bent 1.00 · flip 1.00 · scratch 0.978 · colour 0.961 |
| Defects caught at a threshold set from good parts only | **82 / 93 (88%)** |
| False alarms on good parts | **0 / 22** |
| Memory bank | 1,568 patches · **2.4 MB** |
| Training time | 6.6 s |
| Inference | **50–75 ms per image** across two runs (laptop CPU, 4 threads; Raspberry Pi 5 not yet measured) |

Of the 11 missed defects, 9 scored within 0.17 of the threshold (2.32); the other two were 0.32 and 0.47 below it.
The threshold is deliberately conservative: it is the highest score any training image got in leave-one-out, and it gave 0 / 22 false alarms on the test set.
Lowering it trades false alarms for catches. Do not tune it on the test set.

## ML design optimiser (17 Oct 2026)

```bash
python design_optimizer.py --data ../netraqc_data --category metal_nut --out results/design   # ~40 min on a laptop CPU
python design_search_eval.py      # search strategies vs. the full grid
python plot_design_space.py       # results/design/design_space.png
```

300 detector designs: backbone (ResNet-18, WideResNet-50) × input size (128–256) × feature depth (layer2, layer2+3)
× memory size (coreset 1–25 %) × training images (5, 10, 20). Objectives: image AUROC, latency, footprint
(backbone weights used + memory bank). The MVTec test set is split 56 validation / 59 held-out; designs are chosen
on validation only.

| Design (held-out half) | AUROC | ms / image | Footprint |
|---|---|---|---|
| WideResNet-50, 256 px, layer2+3, 25 % memory | 0.992 | 362 | 131 MB |
| Our default: ResNet-18, 224 px, layer2+3, 10 % memory | 0.992 | 67 | 13.5 MB |
| Smallest within 0.01 of the best on validation: ResNet-18, 160 px, layer2, 5 % | 0.977 | 39 | 2.9 MB |

The 131 MB model buys no accuracy over the 13.5 MB default. Going below about 3 MB costs 0.015–0.025 AUROC
(the same selection rule, run with the halves swapped, picked a 2.8 MB design that scored 0.966 on the other half).

Search efficiency, as a share of the best trade-off (hypervolume) that testing all 300 designs finds, 20 seeds:

| Evaluations | Random | Optuna TPE (multivariate) | NSGA-II |
|---|---|---|---|
| 20 | 0.924 ± 0.057 | 0.948 ± 0.057 | 0.876 ± 0.121 |
| 40 | 0.960 ± 0.036 | **0.988 ± 0.005** | 0.931 ± 0.082 |
| 60 | 0.974 ± 0.027 | **0.993 ± 0.005** | 0.953 ± 0.071 |

This matters for the lighting search, where each evaluation is a physical set-up. Planned: capture each part once
under each of the 8 LEDs in RAW; because light adds linearly, any LED combination (255) and exposure can be
simulated from those captures and scored by the same optimiser, with parts cost and capture time as objectives.

## Hackathon sprint, mid-point (8 Nov 2026)

Everything below runs offline on a laptop CPU limited to 4 threads, to match a Raspberry Pi 5. No hardware was used.

```bash
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  python download_mvtec.py --category $c --n-train 20 --out ../netraqc_data; done
python run_all_categories.py --data ../netraqc_data --out results/all_categories            # step 5, ~45 min
python edge_int8.py --data ../netraqc_data --categories metal_nut screw carpet capsule --out results/edge   # step 1
python eyecandies_lighting.py --root ../eyecandies_data/Eyecandies/Lollipop --out results/eyecandies         # steps 2-4
```

**Step 5: all 15 MVTec AD categories, one fixed design, no per-category tuning.** Mean image AUROC 0.921, mean
pixel AUROC 0.962; 923 / 1,258 defective images caught and 53 / 467 defect-free images flagged. Ten of 15 categories
are at 0.95 or above. Weakest: screw 0.565, grid 0.748, capsule 0.841, pill 0.881, zipper 0.934, where defects are
tiny or parts are rotated. The leave-one-out threshold is too tight for some textures: leather ranks perfectly
(AUROC 1.000) but 15 / 32 good samples were flagged. Per-category numbers: `results/all_categories/summary.json`.

**Step 1: INT8 for edge devices.** ONNX export, static INT8 quantisation with ONNX Runtime (QDQ, per-channel
weights), calibrated on 20 defect-free metal-nut images only. Model 11.1 MB → 2.9 MB; 21–24 ms → 14–15 ms per image
on 4 threads; image AUROC moved by at most 0.004 on metal nut, screw, carpet and capsule (bottle unchanged).
`results/edge/edge_summary.json`.

**Step 2: does light add up?** On Eyecandies (Lollipop), the all-lights render cannot be predicted from the four
single-light renders: 28–57 % mean error in 8-bit or linearised sRGB, with or without per-light weights. Each render is
tone-mapped separately, so light combinations cannot be simulated from these images. A real station must save linear
RAW captures for that. `results/eyecandies/summary.json`.

**Steps 3–4: one shot vs. several lights.** At 224 px on the full frame, the detector is close to chance on
Eyecandies Lollipop (image AUROC 0.489–0.574 across all 63 combinations of the six shots). Cropped to the region the
candy occupies (rows 100–480, columns 30–480) at 320 px (`--size 320 --crop 100,480,30,480`,
`results/eyecandies_crop320/`):

| Lighting | Image AUROC |
|---|---|
| One shot, all camera lights | 0.583 |
| One shot, box lights | 0.586 |
| Each single directional light | 0.643–0.726 |
| All four directional lights (max rule, chosen in advance) | **0.735** |
| Best of 63 combinations (cam3 + cam4, picked on the test set) | 0.756 |

Directional light reveals more than a single evenly lit shot, which is NetraQC's lighting hypothesis. It is an early
signal, not proof: 50 test samples (16 defective), the best subset is chosen on the same samples, and only 3 / 16
defects are caught at the threshold.

## Next steps

- Run on the Raspberry Pi 5 and report real latency and power.
- Export to ONNX / INT8 (OpenVINO or TFLite) and compare accuracy kept against FP32.
- Add the photometric-stereo channels (surface normals from the LED ring) and test on PS-AD and MSDD.
- Run the design optimiser on the lighting (LED subset, angles, exposure); the optical self-check.

## Credits

Dataset: MVTec AD (Bergmann et al., CVPR 2019), CC BY-NC-SA 4.0, fetched from a Hugging Face mirror.
The licence forbids commercial use: we use MVTec AD only as a research benchmark. Any commercial version would be trained
and validated on our own (or properly licensed) images.
Method: PatchCore (Roth et al., CVPR 2022), re-implemented in minimal form.
