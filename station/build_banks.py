"""Build the ready-made part types for the station: one memory bank per MVTec category + a few sample images.

Uses the threshold rule from results/threshold and, for the weak types, the design from results/weak_types
when it beat the default on the held-out half.

    python -m station.build_banks --data ../netraqc_data
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from run_all_categories import CATEGORIES
from station.engine import HERE, Station

RESULTS = HERE.parent / "results"


def settings():
    rule = "max"
    tp = RESULTS / "threshold" / "summary.json"
    if tp.exists():
        rule = json.loads(tp.read_text())["chosen_rule"]
    picks = {}
    wp = RESULTS / "weak_types" / "summary.json"
    if wp.exists():
        for cat, r in json.loads(wp.read_text()).items():
            if r["pick"]["held_out_auroc"] > r["default"]["held_out_auroc"]:
                picks[cat] = r["pick"]
    return rule, picks


def copy_samples(cat, root, out):
    out.mkdir(parents=True, exist_ok=True)
    test = root / "test" / cat
    chosen = [("good", p) for p in sorted((test / "good").glob("*.png"))[:3]]
    for d in sorted(x for x in test.iterdir() if x.is_dir() and x.name != "good")[:3]:
        chosen.append((d.name, sorted(p for p in d.glob("*.png") if not p.stem.endswith("_mask"))[0]))
    info = []
    for label, p in chosen:
        name = f"{label}_{p.stem}.jpg"
        Image.open(p).convert("RGB").resize((512, 512), Image.BILINEAR).save(out / name, quality=90)
        info.append({"file": name, "label": label, "source": f"MVTec AD test/{label}/{p.name}"})
    (out / "samples.json").write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../netraqc_data")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--n-train", type=int, default=20)
    a = ap.parse_args()
    root = Path(a.data)
    rule, picks = settings()
    st = Station()
    for cat in a.categories:
        pick = picks.get(cat, {"size": 224, "layers": "l2+l3", "coreset": 0.10})
        train = [Image.open(p) for p in sorted((root / "train" / cat / "good").glob("*.png"))[: a.n_train]]
        b = st.build(train, size=pick["size"], layers=pick["layers"], ratio=pick["coreset"], rule=rule)
        st.save(cat, cat.replace("_", " ").capitalize(), b)
        copy_samples(cat, root, HERE / "samples" / cat)
        print(cat, pick["size"], pick["layers"], pick["coreset"], rule, round(b["threshold"], 3), flush=True)
    (HERE / "config.json").write_text(json.dumps({"rule": rule, "weak_type_picks": picks}, indent=1))


if __name__ == "__main__":
    main()
