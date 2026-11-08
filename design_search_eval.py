"""Compare random search, TPE and NSGA-II on the design grid (hypervolume vs. full grid)."""
import json
from pathlib import Path

import numpy as np
import optuna

D = Path(__file__).parent / "results" / "design"
rows = json.loads((D / "design_grid.json").read_text())
KEYS = ("backbone", "size", "layers", "coreset", "n_train")
SPACE = {k: sorted({r[k] for r in rows}, key=str) for k in KEYS}
table = {tuple(r[k] for k in KEYS): r for r in rows}


def objs(r, split):
    return np.array([1 - r[split], np.log10(r["latency_ms"]), np.log10(r["footprint_mb"])])


def hv(points, ref, n=200_000, seed=0):
    """Monte-Carlo hypervolume (minimisation) inside the box [lo, ref]."""
    P = np.array(points)
    lo = np.array([o.min() for o in OBJ.T])
    rng = np.random.default_rng(seed)
    S = lo + rng.random((n, 3)) * (ref - lo)
    dom = np.zeros(n, bool)
    for p in P:
        dom |= np.all(S >= p, 1)
    return dom.mean()


OBJ = np.array([objs(r, "val_auroc") for r in rows])
REF = OBJ.max(0) + 1e-9
FULL = hv(OBJ, REF)


def run(strategy, budget, seed):
    if strategy == "random":
        idx = np.random.default_rng(seed).choice(len(rows), budget, replace=False)
        return [OBJ[i] for i in idx]
    sampler = {"tpe": optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, multivariate=True),
               "nsga2": optuna.samplers.NSGAIISampler(seed=seed, population_size=10)}[strategy]
    st = optuna.create_study(directions=["minimize"] * 3, sampler=sampler)
    seen = []

    def f(t):
        k = tuple(t.suggest_categorical(n, SPACE[n]) for n in KEYS)
        o = objs(table[k], "val_auroc")
        seen.append(o)
        return tuple(o)

    st.optimize(f, n_trials=budget)
    return seen


optuna.logging.set_verbosity(optuna.logging.WARNING)
out = {"full_grid_hv": 1.0, "seeds": 20, "results": {}}
for budget in (20, 40, 60):
    for s in ("random", "tpe", "nsga2"):
        v = [hv(run(s, budget, seed), REF) / FULL for seed in range(20)]
        out["results"][f"{s}@{budget}"] = {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)}
        print(s, budget, out["results"][f"{s}@{budget}"], flush=True)


# pick on one half, check on the other
def pick(sel):
    best = max(r[sel] for r in rows)
    ok = [r for r in rows if r[sel] >= best - 0.01]
    return min(ok, key=lambda r: (r["footprint_mb"], r["latency_ms"]))


out["two_fold"] = []
for sel, rep in (("val_auroc", "test_auroc"), ("test_auroc", "val_auroc")):
    r = pick(sel)
    out["two_fold"].append({"select_on": sel, "report_on": rep, **{k: r[k] for k in KEYS},
                            "selected_split_auroc": r[sel], "other_split_auroc": r[rep],
                            "latency_ms": r["latency_ms"], "footprint_mb": r["footprint_mb"]})
(D / "search_eval.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["two_fold"], indent=1))
