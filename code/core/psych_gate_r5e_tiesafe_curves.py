#!/usr/bin/env python3
"""R5-1 corrected main analysis (2026-11-05).

Two defects in the frozen dose-response scripts motivate this re-run.

1. Decile rule. The frozen scripts define the top decile as "score >= the 90th
   percentile". At short dosages a short-form score takes few distinct values,
   so that rule returns more respondents than 10% (up to ~24% at one item per
   domain), and the paper's identity "recall = precision = 2J/(1+J)" fails
   because the two sets differ in size. Here the decile is fixed-size:
   round(0.1 n) respondents, ties at the boundary broken by a random jitter
   redrawn each replication, so recall = precision by construction.

2. SD3 sign orientation. The frozen r5b script called pc_signs() without the
   one-anchor orientation, so each scale's first component kept an arbitrary
   global sign before the three scales were averaged into one composite. The
   psychopathy scale entered anti-aligned. Here the SD3 replication is scored
   per scale (3 units of 9 items, as in the selection-strategy experiment),
   which sidesteps the composite and matches the text's description.

Everything else mirrors the frozen scripts: item standardization within the
unit, first-PC polarity, dosage grids, 40 replications, medians over units.

Additional analyses requested by external review:
  - non-overlapping reference: short form vs the items it did not use;
  - split-half ceiling: capture between two disjoint halves of the full form.

Output: results/psych_gate_r5e_tiesafe_curves.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from scipy.stats import multivariate_normal as mvn

ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get("SHORTFORM_DATA_DIR", ROOT / "data")).resolve()
RES = Path(os.environ.get("SHORTFORM_RESULTS_DIR", ROOT / "results")).resolve()
OUT = RES / "psych_gate_r5e_tiesafe_curves.json"
REPS = 200
SEED = 20261105


def z(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=0)
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def pc_signs(Xs: np.ndarray) -> np.ndarray:
    C = np.corrcoef(Xs.T)
    _w, v = np.linalg.eigh(C)
    v = v[:, -1]
    if v.mean() < 0:
        v = -v
    return np.sign(v)


def top_fixed(score: np.ndarray, rng: np.random.Generator, q: float = 0.9) -> np.ndarray:
    """Fixed-size top (1-q) share; ties at the boundary broken at random."""
    n = len(score)
    target = int(round((1 - q) * n))
    jittered = score + rng.random(n) * 1e-9
    order = np.argsort(-jittered, kind="stable")
    m = np.zeros(n, dtype=bool)
    m[order[:target]] = True
    return m


def top_threshold(score: np.ndarray, q: float = 0.9) -> np.ndarray:
    """The frozen rule, kept only to report how much it inflates the set."""
    return score >= np.quantile(score, q)


def capture_q(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
              q: float = 0.9) -> float:
    if np.allclose(a, b):
        return 1.0
    A, B = top_fixed(a, rng, q), top_fixed(b, rng, q)
    return float((A & B).sum() / A.sum())


def benchmark(q: float, rho: float) -> float:
    """Bivariate-normal implied recall at a two-sided (1-q) cut."""
    r = 2 * np.sin(np.pi * rho / 6)
    zq = norm.ppf(q)
    # P(X>z, Y>z) = Phi2(z,z) - (2q - 1)
    a = mvn(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([zq, zq]) - (2 * q - 1)
    j = a / (2 * (1 - q) - a)
    return float(2 * j / (1 + j))


def score_units(df: pd.DataFrame, unit_cols: dict[str, list[str]]) -> dict[str, np.ndarray]:
    out = {}
    for name, cols in unit_cols.items():
        X = df[sorted(cols)].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        out[name] = X * pc_signs(X)
    return out


def curve(units: dict[str, np.ndarray], ks: list[int], reps: int = REPS,
          seed: int = SEED) -> tuple[dict, dict]:
    """Per-unit dose-response curves and the median over units."""
    rng = np.random.default_rng(seed)
    per_unit = {}
    for name, u in units.items():
        full = z(u.mean(axis=1))
        full_pct = pd.Series(full).rank(pct=True).to_numpy()
        rows = {}
        for k in ks:
            reps_k = 1 if k >= u.shape[1] else reps
            rho, rec, jac, pct, infl = [], [], [], [], []
            rec5, rec1 = [], []
            for _ in range(reps_k):
                sel = (rng.choice(u.shape[1], size=k, replace=False)
                       if k < u.shape[1] else np.arange(u.shape[1]))
                short = z(u[:, sel].mean(axis=1))
                A = top_fixed(full, rng)
                B = A.copy() if np.allclose(short, full) else top_fixed(short, rng)
                inter = int((A & B).sum())
                rho.append(float(spearmanr(short, full).statistic))
                rec.append(inter / int(A.sum()))
                jac.append(inter / int((A | B).sum()))
                pct.append(float(full_pct[B].mean()))
                infl.append(float(top_threshold(short).mean()))
                rec5.append(capture_q(full, short, rng, 0.95))
                rec1.append(capture_q(full, short, rng, 0.99))
            rows[str(k)] = {
                "rho": round(float(np.median(rho)), 3),
                "recall": round(float(np.median(rec)), 3),
                "recall5": round(float(np.median(rec5)), 3),
                "recall1": round(float(np.median(rec1)), 3),
                "jaccard": round(float(np.median(jac)), 3),
                "mean_full_pct_of_short_decile": round(float(np.median(pct)), 3),
                "short_decile_share_under_threshold_rule": round(float(np.median(infl)), 3),
                "rho_iqr_width": round(float(np.percentile(rho, 75) - np.percentile(rho, 25)), 3),
                "recall_iqr_width": round(float(np.percentile(rec, 75) - np.percentile(rec, 25)), 3),
                "reps": reps_k,
            }
        per_unit[name] = rows
    summary = {
        k: {
            "rho": round(float(np.median([p[k]["rho"] for p in per_unit.values()])), 3),
            "recall": round(float(np.median([p[k]["recall"] for p in per_unit.values()])), 3),
            "recall5": round(float(np.median([p[k]["recall5"] for p in per_unit.values()])), 3),
            "recall1": round(float(np.median([p[k]["recall1"] for p in per_unit.values()])), 3),
            "jaccard": round(float(np.median([p[k]["jaccard"] for p in per_unit.values()])), 3),
            "mean_full_pct_of_short_decile": round(float(np.median(
                [p[k]["mean_full_pct_of_short_decile"] for p in per_unit.values()])), 3),
            "short_decile_share_under_threshold_rule": round(float(np.median(
                [p[k]["short_decile_share_under_threshold_rule"] for p in per_unit.values()])), 3),
            "max_rho_iqr_width": round(float(np.max(
                [p[k]["rho_iqr_width"] for p in per_unit.values()])), 3),
            "median_recall_iqr_width": round(float(np.median(
                [p[k]["recall_iqr_width"] for p in per_unit.values()])), 3),
        }
        for k in per_unit[next(iter(per_unit))]
    }
    return per_unit, summary


def capture(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    return capture_q(a, b, rng, 0.9)


def bootstrap_ci(units: dict[str, np.ndarray], k: int, reps: int = 200,
                 seed: int = SEED) -> dict:
    """Person-level bootstrap at one dosage: CI for median-over-units rho/recall.

    Each replicate resamples respondents with replacement, redraws the item
    subset for every unit, and records the median over units. The percentile
    interval therefore covers both sampling error in persons and draw error
    in items, which is the uncertainty a reader of the curve faces.
    """
    rng = np.random.default_rng(seed)
    names = list(units)
    n = units[names[0]].shape[0]
    boot_rho, boot_rec = [], []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        rhos, recs = [], []
        for name in names:
            v = units[name][idx]
            full = z(v.mean(axis=1))
            sel = rng.choice(v.shape[1], size=min(k, v.shape[1]), replace=False)
            short = z(v[:, sel].mean(axis=1))
            rhos.append(float(spearmanr(short, full).statistic))
            recs.append(capture(full, short, rng))
        boot_rho.append(float(np.median(rhos)))
        boot_rec.append(float(np.median(recs)))
    q = lambda x: [round(float(np.percentile(x, 2.5)), 3),
                   round(float(np.percentile(x, 97.5)), 3)]
    return {"k": k, "reps": reps,
            "rho": round(float(np.median(boot_rho)), 3), "rho_ci95": q(boot_rho),
            "recall": round(float(np.median(boot_rec)), 3), "recall_ci95": q(boot_rec)}


def complement_reference(units: dict[str, np.ndarray], k: int = 2,
                         reps: int = REPS, seed: int = SEED) -> dict:
    """Short form vs the items it did not use.

    For each unit and replication: score the k selected items and the
    remaining items separately; correlate them; and compute the capture of
    each score's top decile by the other. Overlap with the full form is
    mechanical, so this is the conservative half of the same question.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for name, u in units.items():
        rhos, c_short_of_rest, c_rest_of_short = [], [], []
        for _ in range(reps):
            sel = rng.choice(u.shape[1], size=k, replace=False)
            rest = np.setdiff1d(np.arange(u.shape[1]), sel)
            s_short = z(u[:, sel].mean(axis=1))
            s_rest = z(u[:, rest].mean(axis=1))
            rhos.append(float(spearmanr(s_short, s_rest).statistic))
            c_short_of_rest.append(capture(s_rest, s_short, rng))
            c_rest_of_short.append(capture(s_short, s_rest, rng))
        rows[name] = {
            "rho_short_vs_rest": round(float(np.median(rhos)), 3),
            "recall_of_rest_decile_by_short": round(float(np.median(c_short_of_rest)), 3),
            "recall_of_short_decile_by_rest": round(float(np.median(c_rest_of_short)), 3),
        }
    summary = {key: round(float(np.median([r[key] for r in rows.values()])), 3)
               for key in next(iter(rows.values()))}
    return {"per_unit": rows, "median_over_units": summary}


def split_half_ceiling(units: dict[str, np.ndarray], reps: int = REPS,
                       seed: int = SEED) -> dict:
    """Capture between two disjoint halves of the full form.

    Each replication splits a unit's items into two halves, scores each half,
    and computes how much of one half's top decile the other half recovers.
    This is the ceiling any short form built from disjoint items faces at that
    length, and the amount that shortening per se must add on top of it.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for name, u in units.items():
        m = u.shape[1]
        vals, rhos = [], []
        for _ in range(reps):
            perm = rng.permutation(m)
            half = m // 2
            a, b = perm[:half], perm[half:]
            if len(a) == 0 or len(b) == 0:
                continue
            s_a = z(u[:, a].mean(axis=1))
            s_b = z(u[:, b].mean(axis=1))
            vals.append(capture(s_a, s_b, rng))
            rhos.append(float(spearmanr(s_a, s_b).statistic))
        rows[name] = {
            "half_length_share": round(half / m, 3),
            "rho_between_halves": round(float(np.median(rhos)), 3),
            "recall_between_halves": round(float(np.median(vals)), 3),
        }
    summary = {key: round(float(np.median([r[key] for r in rows.values()])), 3)
               for key in ("rho_between_halves", "recall_between_halves")}
    return {"per_unit": rows, "median_over_units": summary}


def sensitivity_flagged(units: dict[str, np.ndarray], flagged: set[str],
                        cols_by_unit: dict[str, list[str]], k: int = 2,
                        reps: int = REPS, seed: int = SEED) -> dict:
    """Drop polarity-flagged items inside each unit, keep k per unit."""
    rng = np.random.default_rng(seed)
    rows = {}
    for name, u in units.items():
        keep = [j for j, c in enumerate(sorted(cols_by_unit[name])) if c not in flagged]
        if len(keep) < k:
            continue
        v = u[:, keep]
        full = z(v.mean(axis=1))
        rec, rho = [], []
        for _ in range(reps):
            sel = rng.choice(v.shape[1], size=k, replace=False)
            short = z(v[:, sel].mean(axis=1))
            rec.append(capture(full, short, rng))
            rho.append(float(spearmanr(short, full).statistic))
        rows[name] = {
            "items_retained": len(keep),
            "rho": round(float(np.median(rho)), 3),
            "recall": round(float(np.median(rec)), 3),
        }
    summary = {key: round(float(np.median([r[key] for r in rows.values()])), 3)
               for key in ("items_retained", "rho", "recall")}
    return {"per_unit": rows, "median_over_units": summary}


def main() -> int:
    out: dict = {"date": "2026-11-05", "reps": REPS, "seed": SEED,
                 "decile_rule": "fixed size round(0.1 n), random tie-break; "
                                "recall = precision by construction",
                 "benchmark_recall_by_rho_and_cut": {
                     str(q): {str(r): round(benchmark(q, r), 3)
                              for r in (0.60, 0.70, 0.77, 0.80, 0.82, 0.83, 0.85, 0.90, 0.92)}
                     for q in (0.90, 0.95, 0.99)},
                 "protocol_notes": [
                     "HEXACO: 24 facet units x 10 items; IPIP-50: 5 domain units x 10; "
                     "NPI-40: one unit; SD3: 3 scale units x 9 (scored per scale).",
                     "Inventory summaries are medians over units, as in the manuscript.",
                 ]}

    # --- HEXACO ---
    h = pd.read_csv(D / "HEXACO" / "HEXACO" / "data.csv", sep="\t", low_memory=False)
    pat = re.compile(r"^([HEXACO])([A-Za-z]+?)(\d+)$")
    hgroups: dict[str, list[str]] = {}
    for c in h.columns:
        m = pat.match(c)
        if m:
            hgroups.setdefault(m.group(1) + m.group(2), []).append(c)
    h = h.dropna(subset=[c for cs in hgroups.values() for c in cs])
    hex_units = score_units(h, hgroups)
    hp, hs = curve(hex_units, [1, 2, 4, 6, 10])
    out["HEXACO"] = {"n": int(len(h)), "per_unit": hp, "median_over_units": hs,
                     "complement_20pct": complement_reference(hex_units),
                     "split_half": split_half_ceiling(hex_units),
                     "bootstrap_k2": bootstrap_ci(hex_units, k=2)}

    # --- IPIP-50 ---
    b = pd.read_csv(D / "BIG5" / "BIG5" / "data.csv", sep="\t")
    ipip_cols = {d: [f"{d}{i}" for i in range(1, 11)] for d in "ENACO"}
    b = b.dropna(subset=[c for cs in ipip_cols.values() for c in cs])
    ipip_units = score_units(b, ipip_cols)
    ip, isum = curve(ipip_units, [1, 2, 4, 6, 10])
    out["IPIP50"] = {"n": int(len(b)), "per_unit": ip, "median_over_units": isum,
                     "complement_20pct": complement_reference(ipip_units),
                     "split_half": split_half_ceiling(ipip_units),
                     "bootstrap_k2": bootstrap_ci(ipip_units, k=2)}

    # --- NPI-40 ---
    npi = pd.read_csv(D / "NPI" / "NPI" / "data.csv")
    items = [f"Q{i}" for i in range(1, 41)]
    npi = npi.dropna(subset=items + ["score"])
    key = {q: (1 if npi.loc[npi[q] == 1, "score"].mean() > npi.loc[npi[q] == 2, "score"].mean()
               else 2) for q in items}
    Y = np.column_stack([np.where(npi[q].to_numpy() == key[q], 1.0, 0.0) for q in items])
    npi_units = {"NPI40": Y}
    np_, ns = curve(npi_units, [4, 8, 16, 24, 32, 40])
    out["NPI40"] = {"n": int(len(npi)), "per_unit": np_, "median_over_units": ns,
                    "complement_20pct": complement_reference(npi_units, k=8),
                    "split_half": split_half_ceiling(npi_units),
                    "bootstrap_k8": bootstrap_ci(npi_units, k=8)}

    # --- SD3 (per scale) ---
    s = pd.read_csv(D / "SD3" / "SD3" / "data.csv", sep="\t")
    sd_cols = {d: [f"{d}{i}" for i in range(1, 10)] for d in "MNP"}
    s = s.dropna(subset=[c for cs in sd_cols.values() for c in cs])
    sd_units = score_units(s, sd_cols)
    sp, ss = curve(sd_units, [1, 2, 4, 6, 9])
    out["SD3"] = {"n": int(len(s)), "per_unit": sp, "median_over_units": ss,
                  "complement_20pct": complement_reference(sd_units),
                  "split_half": split_half_ceiling(sd_units),
                  "bootstrap_k2": bootstrap_ci(sd_units, k=2)}

    # --- flagged-item sensitivity (HEXACO, IPIP) ---
    pol = json.loads((RES / "psych_gate_r5b_polarity_npi_sd3_facet.json").read_text())
    for name, unit_map, col_map in (("HEXACO", hex_units, hgroups),
                                    ("IPIP50", ipip_units, ipip_cols)):
        flagged = {f["item"] for f in
                   pol["polarity_validation"][name]["flagged"]["text_pc_mismatch"]}
        out[name]["sensitivity_flagged_k2"] = sensitivity_flagged(
            unit_map, flagged, col_map, k=2)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    for label in ("HEXACO", "IPIP50", "NPI40", "SD3"):
        print(f"== {label} (median over units)")
        for k, v in out[label]["median_over_units"].items():
            print(f" k={k:>2}: rho={v['rho']:.3f} recall={v['recall']:.3f} "
                  f"J={v['jaccard']:.3f} pct={v['mean_full_pct_of_short_decile']:.3f} "
                  f"|B_old|={v['short_decile_share_under_threshold_rule']:.3f}")
        print("  complement:", out[label]["complement_20pct"]["median_over_units"])
        print("  split-half:", out[label]["split_half"]["median_over_units"])
        if "sensitivity_flagged_k2" in out[label]:
            print("  flagged:", out[label]["sensitivity_flagged_k2"]["median_over_units"])
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
