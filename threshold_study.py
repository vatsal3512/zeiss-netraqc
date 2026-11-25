"""Compare threshold rules computed from the 20 leave-one-out scores of the good images.

Rule picked on the val half of each test set, reported on the held-out half.
Test scores are reused from results/all_categories/scores (same bank, same seed).

    python threshold_study.py --data ../netraqc_data --out results/threshold
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from design_optimizer import split
from netraqc_patchcore import FeatureExtractor, anomaly_map, coreset, load, patches
from run_all_categories import CATEGORIES

RULES = {
    "max": lambda s: s.max(),
    "mean+2sd": lambda s: s.mean() + 2 * s.std(),
    "mean+3sd": lambda s: s.mean() + 3 * s.std(),
    "mean+4sd": lambda s: s.mean() + 4 * s.std(),
    "max*1.05": lambda s: 1.05 * s.max(),
    "max*1.10": lambda s: 1.10 * s.max(),
}


def loo_scores(cat, root, model, a):
    train = sorted((root / "train" / cat / "good").glob("*.png"))[: a.n_train]
    feats = [model(load(p, a.size)[1]) for p in train]
    per = [patches(f) for f in feats]
    return np.array([float(anomaly_map(feats[i], coreset(torch.cat(per[:i] + per[i + 1:]), a.coreset, seed=i + 1), a.size).max())
                     for i in range(len(per))])


def errors(thr, rows):
    bad = [r for r in rows if r["defect"] != "good"]
    good = [r for r in rows if r["defect"] == "good"]
    missed = sum(r["score"] <= thr for r in bad)
    fa = sum(r["score"] > thr for r in good)
    return {"missed": missed, "defective": len(bad), "false_alarms": fa, "good": len(good)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--scores", default="results/all_categories/scores")
    ap.add_argument("--out", default="results/threshold")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = FeatureExtractor("resnet18")

    per_cat = {}
    for cat in CATEGORIES:
        rows = {r["file"]: r for r in json.loads((Path(a.scores) / f"{cat}.json").read_text())}
        test_paths = [p for p in sorted((root / "test" / cat).rglob("*.png")) if not p.stem.endswith("_mask")]
        val, hold = split(test_paths)
        key = lambda p: f"{p.parent.name}/{p.name}"
        loo = loo_scores(cat, root, model, a)
        per_cat[cat] = {"loo": loo.round(4).tolist(), "rules": {}}
        for name, rule in RULES.items():
            thr = float(rule(loo))
            per_cat[cat]["rules"][name] = {"threshold": round(thr, 4),
                                           "val": errors(thr, [rows[key(p)] for p in val]),
                                           "held_out": errors(thr, [rows[key(p)] for p in hold])}
        print(cat, {n: per_cat[cat]["rules"][n]["val"] for n in ("max", "mean+3sd")}, flush=True)

    def balanced(name, half):
        e = [per_cat[c]["rules"][name][half] for c in CATEGORIES]
        return float(np.mean([x["missed"] / x["defective"] + x["false_alarms"] / x["good"] for x in e]) / 2)

    def totals(name, half):
        e = [per_cat[c]["rules"][name][half] for c in CATEGORIES]
        return {k: sum(x[k] for x in e) for k in ("missed", "defective", "false_alarms", "good")}

    summary = {n: {"val_balanced_error": round(balanced(n, "val"), 4), "held_out_balanced_error": round(balanced(n, "held_out"), 4),
                   "held_out": totals(n, "held_out")} for n in RULES}
    chosen = min(RULES, key=lambda n: summary[n]["val_balanced_error"])
    res = {"rules": summary, "chosen_rule": chosen, "previous_rule": "max",
           "selection": "lowest mean of miss rate and false-alarm rate over 15 val halves", "per_category": per_cat}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("chosen", chosen, json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
