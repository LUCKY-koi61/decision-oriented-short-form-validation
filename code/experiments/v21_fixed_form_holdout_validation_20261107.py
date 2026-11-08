#!/usr/bin/env python3
"""v21: fixed-form, internal holdout validation for two-stage screening.

This analysis addresses a limitation of v20.  v20 held out respondents when
selecting a *configuration* (number of stage-1 items and candidate share), but
each read-out still re-drew an item subset.  Here the selection half performs
all three decisions: it picks the least-cost configuration that reaches .80
recall in every scoring unit, and it fixes one nested, recall-greedy item set
for each unit.  The evaluation half receives those item sets unchanged.

The comparison is deliberately narrow.  At the same selected configuration,
we compare the fixed recall-greedy forms with fixed correlation-greedy forms
and with 200 fixed random forms.  All performance is full-form-referenced,
not external-criterion or clinical accuracy.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "code" / "core"))
import psych_gate_r5e_tiesafe_curves as mc  # noqa: E402

OUT = PROJ / "outputs" / "validation" / "v21_fixed_form_holdout_validation_20261107.json"
SPLIT_SEED = 2026110703
TIE_SEED = 2026110704
RANDOM_SEED = 2026110705
BOOT_SEED = 2026110706
TARGET_RECALL = 0.80
N_RANDOM_FORMS = 200
N_BOOT = 1000
CANDIDATE_SHARES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
GRID = {
    "HEXACO": [1, 2, 4, 6, 10],
    "IPIP50": [1, 2, 4, 6, 10],
    "NPI40": [4, 8, 16, 24, 32, 40],
    "SD3": [1, 2, 4, 6, 9],
}


def load_raw() -> dict[str, dict]:
    """Load complete cases, retaining raw item labels for a reproducible form."""
    d = Path(os.environ.get("SHORTFORM_DATA_DIR", PROJ / "data")).resolve()
    out: dict[str, dict] = {}

    h = pd.read_csv(d / "HEXACO" / "HEXACO" / "data.csv", sep="\t", low_memory=False)
    pat = re.compile(r"^([HEXACO])([A-Za-z]+?)(\d+)$")
    hgroups: dict[str, list[str]] = {}
    for col in h.columns:
        if pat.match(col):
            hit = pat.match(col)
            assert hit is not None
            hgroups.setdefault(hit.group(1) + hit.group(2), []).append(col)
    hgroups = {name: sorted(cols) for name, cols in hgroups.items()}
    h = h.dropna(subset=[col for cols in hgroups.values() for col in cols]).reset_index(drop=True)
    out["HEXACO"] = {"units": {name: h[cols].to_numpy(float) for name, cols in hgroups.items()},
                     "item_names": hgroups, "npi_score": None}

    b = pd.read_csv(d / "BIG5" / "BIG5" / "data.csv", sep="\t")
    bcols = {name: [f"{name}{i}" for i in range(1, 11)] for name in "ENACO"}
    b = b.dropna(subset=[col for cols in bcols.values() for col in cols]).reset_index(drop=True)
    out["IPIP50"] = {"units": {name: b[cols].to_numpy(float) for name, cols in bcols.items()},
                      "item_names": bcols, "npi_score": None}

    n = pd.read_csv(d / "NPI" / "NPI" / "data.csv")
    ncols = [f"Q{i}" for i in range(1, 41)]
    n = n.dropna(subset=ncols + ["score"]).reset_index(drop=True)
    out["NPI40"] = {"units": {"NPI40": n[ncols].to_numpy(float)},
                    "item_names": {"NPI40": ncols}, "npi_score": n["score"].to_numpy(float)}

    s = pd.read_csv(d / "SD3" / "SD3" / "data.csv", sep="\t")
    scols = {name: [f"{name}{i}" for i in range(1, 10)] for name in "MNP"}
    s = s.dropna(subset=[col for cols in scols.values() for col in cols]).reset_index(drop=True)
    out["SD3"] = {"units": {name: s[cols].to_numpy(float) for name, cols in scols.items()},
                  "item_names": scols, "npi_score": None}
    return out


def fixed_top(score: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    n = len(score)
    count = int(round(share * n))
    order = np.argsort(-(score + rng.random(n) * 1e-9), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:count]] = True
    return mask


def orient_train_test(
    train_raw: np.ndarray, test_raw: np.ndarray, *, npi_train_score: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate scoring direction and standardization in training rows only."""
    if npi_train_score is not None:
        keys = np.array([
            1 if npi_train_score[train_raw[:, j] == 1].mean()
            > npi_train_score[train_raw[:, j] == 2].mean() else 2
            for j in range(train_raw.shape[1])
        ])
        train_raw = (train_raw == keys).astype(float)
        test_raw = (test_raw == keys).astype(float)
    mu = train_raw.mean(axis=0)
    sd = train_raw.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    train_z = (train_raw - mu) / sd
    test_z = (test_raw - mu) / sd
    signs = mc.pc_signs(train_z)
    signs[signs == 0] = 1.0
    return train_z * signs, test_z * signs


def rate(target: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    hits = int((target & selected).sum())
    return hits / int(target.sum()), hits / int(selected.sum())


def ci_pair(target: np.ndarray, selected: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(target)
    rec, prec = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        hit = int((target[idx] & selected[idx]).sum())
        rec.append(hit / max(int(target[idx].sum()), 1))
        prec.append(hit / max(int(selected[idx].sum()), 1))
    return {
        "recall_95ci": [round(float(np.percentile(rec, 2.5)), 3), round(float(np.percentile(rec, 97.5)), 3)],
        "stage1_precision_95ci": [round(float(np.percentile(prec, 2.5)), 3), round(float(np.percentile(prec, 97.5)), 3)],
    }


def greedy_order(train: np.ndarray, share: float, rule: str, rng: np.random.Generator) -> list[int]:
    full = train.mean(axis=1)
    target = fixed_top(full, 0.10, rng)
    selected: list[int] = []
    remaining = list(range(train.shape[1]))
    for _ in range(train.shape[1]):
        best, best_value = None, -np.inf
        for item in remaining:
            score = train[:, selected + [item]].mean(axis=1)
            if rule == "recall":
                value = rate(target, fixed_top(score, share, rng))[0]
            else:
                value = float(spearmanr(score, full).statistic)
            if value > best_value:
                best, best_value = item, value
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    return selected


def choice_for_inventory(train_units: dict[str, np.ndarray], grid: list[int], rng: np.random.Generator) -> dict:
    """Choose an all-unit operating point and fixed recall-greedy forms in train."""
    cache = {share: {name: greedy_order(unit, share, "recall", rng)
                     for name, unit in train_units.items()}
             for share in CANDIDATE_SHARES}
    candidates = []
    for k in grid:
        for share in CANDIDATE_SHARES:
            recalls = []
            for name, train in train_units.items():
                target = fixed_top(train.mean(axis=1), 0.10, rng)
                advance = fixed_top(train[:, cache[share][name][:k]].mean(axis=1), share, rng)
                recalls.append(rate(target, advance)[0])
            cost = k / next(iter(train_units.values())).shape[1] + share * (1 - k / next(iter(train_units.values())).shape[1])
            if min(recalls) >= TARGET_RECALL:
                candidates.append({"items_per_unit": k, "candidate_share": share,
                                   "total_item_cost_share": round(cost, 3),
                                   "item_saving_share": round(1 - cost, 3),
                                   "selection_recall_range": [round(min(recalls), 3), round(max(recalls), 3)],
                                   "selection_recall_median": round(float(np.median(recalls)), 3)})
    if not candidates:
        raise RuntimeError("No all-unit configuration meets the predeclared recall target")
    chosen = min(candidates, key=lambda x: (x["total_item_cost_share"], x["items_per_unit"], x["candidate_share"]))
    chosen["recall_greedy_orders"] = {name: cache[chosen["candidate_share"]][name]
                                      for name in train_units}
    return chosen


def evaluate_strategy(
    test_units: dict[str, np.ndarray], orders: dict[str, list[int]], k: int, share: float,
    rng: np.random.Generator, *, bootstrap: bool = False
) -> dict:
    per_unit, recalls, precisions, paired_flags = {}, [], [], []
    for name, test in test_units.items():
        target = fixed_top(test.mean(axis=1), 0.10, rng)
        selected = fixed_top(test[:, orders[name][:k]].mean(axis=1), share, rng)
        rec, prec = rate(target, selected)
        row = {"recall": round(rec, 3), "stage1_precision": round(prec, 3)}
        if bootstrap:
            row.update(ci_pair(target, selected, rng))
        per_unit[name] = row
        recalls.append(rec)
        precisions.append(prec)
        paired_flags.append((target, selected))
    out = {
        "per_unit": per_unit,
        "recall_median_over_units": round(float(np.median(recalls)), 3),
        "recall_range_over_units": [round(min(recalls), 3), round(max(recalls), 3)],
        "stage1_precision_median_over_units": round(float(np.median(precisions)), 3),
        "units_at_or_above_target": int(sum(x >= TARGET_RECALL for x in recalls)),
        "n_units": len(recalls),
    }
    if bootstrap:
        median_boot_recall, median_boot_precision = [], []
        for _ in range(N_BOOT):
            boot_recall, boot_precision = [], []
            for target, selected in paired_flags:
                idx = rng.integers(0, len(target), size=len(target))
                hits = int((target[idx] & selected[idx]).sum())
                boot_recall.append(hits / max(int(target[idx].sum()), 1))
                boot_precision.append(hits / max(int(selected[idx].sum()), 1))
            median_boot_recall.append(float(np.median(boot_recall)))
            median_boot_precision.append(float(np.median(boot_precision)))
        out["recall_median_95ci"] = [
            round(float(np.percentile(median_boot_recall, 2.5)), 3),
            round(float(np.percentile(median_boot_recall, 97.5)), 3),
        ]
        out["stage1_precision_median_95ci"] = [
            round(float(np.percentile(median_boot_precision, 2.5)), 3),
            round(float(np.percentile(median_boot_precision, 97.5)), 3),
        ]
    return out


def main() -> int:
    raw = load_raw()
    out = {
        "date": "2026-11-07",
        "protocol": "one deterministic 50/50 respondent split per inventory; all scoring orientation, operating-point selection, and fixed-item selection use the selection half only",
        "seeds": {"split": SPLIT_SEED, "ties": TIE_SEED, "random_forms": RANDOM_SEED, "bootstrap": BOOT_SEED},
        "target_recall": TARGET_RECALL,
        "candidate_shares": list(CANDIDATE_SHARES),
        "random_forms_per_unit": N_RANDOM_FORMS,
        "bootstrap_replicates": N_BOOT,
        "scope": "internal validation of fixed forms against same-inventory full-form top-decile references; not external-population, external-criterion, or clinical validation",
        "inventories": {},
    }
    for inv_num, (inventory, spec) in enumerate(raw.items()):
        n = next(iter(spec["units"].values())).shape[0]
        split_rng = np.random.default_rng(SPLIT_SEED + inv_num)
        perm = split_rng.permutation(n)
        tr_idx, te_idx = perm[: n // 2], perm[n // 2 :]
        train_units, test_units = {}, {}
        for name, x in spec["units"].items():
            npi_score = spec["npi_score"][tr_idx] if spec["npi_score"] is not None else None
            train_units[name], test_units[name] = orient_train_test(x[tr_idx], x[te_idx], npi_train_score=npi_score)
        tie_rng = np.random.default_rng(TIE_SEED + 100 * inv_num)
        chosen = choice_for_inventory(train_units, GRID[inventory], tie_rng)
        k, share = chosen["items_per_unit"], chosen["candidate_share"]
        recall_orders = chosen.pop("recall_greedy_orders")
        corr_orders = {name: greedy_order(train, share, "correlation", tie_rng)
                       for name, train in train_units.items()}
        test_rng = np.random.default_rng(TIE_SEED + 100 * inv_num + 1)
        recall_eval = evaluate_strategy(test_units, recall_orders, k, share, test_rng, bootstrap=True)
        corr_eval = evaluate_strategy(test_units, corr_orders, k, share, test_rng, bootstrap=True)

        random_metrics = []
        random_rng = np.random.default_rng(RANDOM_SEED + inv_num)
        for _ in range(N_RANDOM_FORMS):
            random_orders = {name: random_rng.permutation(test.shape[1]).tolist()
                             for name, test in test_units.items()}
            random_metrics.append(evaluate_strategy(test_units, random_orders, k, share, test_rng, bootstrap=False))
        random_recall = [row["recall_median_over_units"] for row in random_metrics]
        random_precision = [row["stage1_precision_median_over_units"] for row in random_metrics]
        fixed_items = {
            name: [spec["item_names"][name][j] for j in recall_orders[name][:k]]
            for name in recall_orders
        }
        out["inventories"][inventory] = {
            "n_selection": int(len(tr_idx)), "n_evaluation": int(len(te_idx)),
            "selected_configuration": chosen,
            "fixed_recall_greedy_item_sets": fixed_items,
            "fixed_recall_greedy_holdout": recall_eval,
            "fixed_correlation_greedy_holdout": corr_eval,
            "fixed_random_forms_holdout": {
                "n_forms": N_RANDOM_FORMS,
                "recall_median_of_form_medians": round(float(np.median(random_recall)), 3),
                "recall_iqr_of_form_medians": [round(float(np.percentile(random_recall, 25)), 3), round(float(np.percentile(random_recall, 75)), 3)],
                "stage1_precision_median_of_form_medians": round(float(np.median(random_precision)), 3),
                "stage1_precision_iqr_of_form_medians": [round(float(np.percentile(random_precision, 25)), 3), round(float(np.percentile(random_precision, 75)), 3)],
            },
        }
        print(inventory, json.dumps(out["inventories"][inventory]["fixed_recall_greedy_holdout"], indent=1), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
