"""Node 1, the Data Steward: profile the data and score it against fixed thresholds.

Stage 1 (generic checks) runs before task framing. Stage 2 (`target_audit`) runs right
after the Selector has fixed the target, because class balance and target leakage cannot
be judged without one. Both stages write into one scorecard; the gate fails if any check fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .structure import detect_entity_time


def _check(cid, label, status, value, threshold, detail):
    return {"id": cid, "label": label, "status": status, "value": value, "threshold": threshold, "detail": detail}


def _numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def profile(df: pd.DataFrame, exclude=()) -> dict:
    """Stage 1. Returns scorecard checks + findings the later nodes act on."""
    n, m = df.shape
    checks, findings = [], {"drop": [], "structure": None}

    # rows
    st = "fail" if n < config.MIN_ROWS_FAIL else "warn" if n < config.MIN_ROWS_WARN else "pass"
    checks.append(_check("rows", "Sample size", st, int(n),
                         f">= {config.MIN_ROWS_WARN} (fail < {config.MIN_ROWS_FAIL})",
                         f"{n:,} rows x {m} columns"))

    # missingness
    miss = df.isna().mean()
    worst = miss.sort_values(ascending=False)
    wv = float(worst.iloc[0]) if len(worst) else 0.0
    st = "fail" if wv > config.MISSING_FAIL else "warn" if wv > config.MISSING_WARN else "pass"
    bad = [f"{c} ({v:.0%})" for c, v in worst.items() if v > config.MISSING_WARN][:6]
    checks.append(_check("missing", "Missing values", st, round(wv, 4),
                         f"<= {config.MISSING_WARN:.0%} per column (fail > {config.MISSING_FAIL:.0%})",
                         ("worst columns: " + ", ".join(bad)) if bad else "no column above threshold"))

    # duplicates
    dup = float(df.duplicated().mean())
    checks.append(_check("duplicates", "Duplicate rows", "warn" if dup > config.DUPLICATE_WARN else "pass",
                         round(dup, 4), f"<= {config.DUPLICATE_WARN:.0%}", f"{int(df.duplicated().sum())} duplicate rows"))

    # constants
    num = [c for c in _numeric_cols(df) if c not in exclude]
    const = [c for c in num if df[c].nunique(dropna=True) <= 1]
    for c in const:
        findings["drop"].append({"col": c, "reason": "constant column (zero information)"})
    checks.append(_check("constant", "Constant columns", "warn" if const else "pass", len(const), "0",
                         ("dropped: " + ", ".join(const)) if const else "none"))

    # identifiers / high cardinality
    ident = []
    for c in df.columns:
        if c in exclude or c in const:
            continue
        nu = df[c].nunique(dropna=True)
        is_obj = not pd.api.types.is_numeric_dtype(df[c])
        if nu == n and n > 1 and (is_obj or pd.api.types.is_integer_dtype(df[c])):
            ident.append((c, "one unique value per row (row identifier)"))
        elif is_obj and nu > config.HIGH_CARDINALITY:
            ident.append((c, f"{nu} distinct categories (> {config.HIGH_CARDINALITY}), identifier-like"))
    for c, why in ident:
        findings["drop"].append({"col": c, "reason": why})
    checks.append(_check("identifiers", "Identifier columns", "warn" if ident else "pass", len(ident), "0",
                         ("dropped: " + ", ".join(c for c, _ in ident)) if ident else "none"))

    # structure (run-to-failure entity + time counter)
    struct = detect_entity_time(df)
    findings["structure"] = struct
    checks.append(_check(
        "structure", "Time-series structure", "pass",
        (struct["entity_col"] + " / " + struct["time_col"]) if struct else "tabular", "n/a",
        struct["evidence"] if struct else "no entity/time counter found: treated as independent rows"))

    # collinearity + VIF over informative numeric columns
    dropped = {d["col"] for d in findings["drop"]}
    skip = set(exclude) | dropped
    if struct:
        skip |= {struct["entity_col"], struct["time_col"]}
    feats = [c for c in num if c not in skip]
    if len(feats) >= 2:
        X = df[feats].fillna(df[feats].median())
        corr = np.nan_to_num(np.corrcoef(X.to_numpy(dtype=float), rowvar=False))
        iu = np.triu_indices(len(feats), 1)
        pairs = [(feats[i], feats[j], corr[i, j]) for i, j in zip(*iu) if abs(corr[i, j]) >= config.COLLINEAR_R]
        vif = np.diag(np.linalg.pinv(corr + 1e-9 * np.eye(len(feats))))
        vmax = float(np.max(vif))
        top = sorted(zip(feats, vif), key=lambda p: -p[1])[:3]
        st = "warn" if (pairs or vmax > config.VIF_WARN) else "pass"
        checks.append(_check(
            "collinearity", "Multicollinearity", st, round(vmax, 2),
            f"|r| < {config.COLLINEAR_R}, VIF <= {config.VIF_WARN:g}",
            f"{len(pairs)} pairs with |r| >= {config.COLLINEAR_R}; max VIF {vmax:.1f} ({top[0][0]}). "
            "Tree models tolerate this, so it is reported, not blocking."))
    else:
        checks.append(_check("collinearity", "Multicollinearity", "pass", 0, "n/a", "fewer than 2 numeric features"))

    # stationarity / drift via KS
    if feats:
        if struct:
            rank = df.groupby(struct["entity_col"])[struct["time_col"]].rank(pct=True)
            a, b = df[rank <= 0.2], df[rank > 0.8]
            what = "early-life (first 20%) vs late-life (last 20%) of each unit"
        else:
            half = len(df) // 2
            a, b = df.iloc[:half], df.iloc[half:]
            what = "first half vs second half of rows"
        shifted = []
        for c in feats:
            x, y = a[c].dropna().to_numpy(), b[c].dropna().to_numpy()
            if len(x) > 5 and len(y) > 5:
                d, p = stats.ks_2samp(x, y)
                if d >= config.KS_D_DRIFT and p < config.KS_ALPHA:
                    shifted.append((c, float(d)))
        share = len(shifted) / len(feats)
        top = ", ".join(f"{c} (D={d:.2f})" for c, d in sorted(shifted, key=lambda p: -p[1])[:4])
        if struct:
            checks.append(_check(
                "stationarity", "Distribution shift (KS test)", "pass", f"{len(shifted)}/{len(feats)}",
                f"D >= {config.KS_D_DRIFT}, p < {config.KS_ALPHA}",
                f"{what}: {len(shifted)} of {len(feats)} sensors shift" + (f" ({top})" if top else "") +
                ". Expected for degradation data: this is the signal the models will learn."))
        else:
            checks.append(_check(
                "stationarity", "Distribution shift (KS test)", "warn" if share > 0.5 else "pass",
                f"{len(shifted)}/{len(feats)}", f"D >= {config.KS_D_DRIFT}, p < {config.KS_ALPHA}",
                f"{what}: {len(shifted)} of {len(feats)} features shift" + (f" ({top})" if top else "")))
    return {"checks": checks, "findings": findings}


def target_audit(df: pd.DataFrame, y: pd.Series, task: str, exclude=()) -> dict:
    """Stage 2. Needs the target: completeness, class balance, and leakage guard."""
    checks, drop = [], []
    ymiss = float(pd.isna(y).mean())
    checks.append(_check("target_missing", "Target completeness", "fail" if ymiss > 0 else "pass", round(ymiss, 4), "0",
                         "target has no missing values" if ymiss == 0 else f"{ymiss:.1%} of target values missing"))
    if task == "classification":
        share = float(min(y.mean(), 1 - y.mean()))
        st = "fail" if share < config.IMBALANCE_FAIL else "warn" if share < config.IMBALANCE_WARN else "pass"
        checks.append(_check(
            "imbalance", "Class balance", st, round(share, 4),
            f">= {config.IMBALANCE_WARN:.0%} (fail < {config.IMBALANCE_FAIL:.1%})",
            f"minority class = {share:.2%} of rows" +
            ("; ranking metric switches to average precision and classes are weighted" if st != "pass" else "")))
        leaks = []
        for c in df.columns:
            if c in exclude or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            v = df[c].dropna().unique()
            if len(v) == 2 and set(np.round(v, 6)) <= {0, 1}:
                mask = df[c] == 1
                if mask.sum() >= config.LEAKAGE_SUPPORT:
                    prec = float(y[mask].mean())
                    if prec >= config.LEAKAGE_PRECISION:
                        leaks.append((c, prec, int(mask.sum())))
        for c, prec, sup in leaks:
            drop.append({"col": c, "reason": f"target leakage: P(target=1 | {c}=1) = {prec:.0%} over {sup} rows"})
        checks.append(_check(
            "leakage", "Target-leakage guard", "warn" if leaks else "pass", len(leaks),
            f"P(target | flag) < {config.LEAKAGE_PRECISION:.0%}",
            ("dropped: " + ", ".join(f"{c} ({p:.0%})" for c, p, _ in leaks)) if leaks
            else "no flag column predicts the target near-perfectly"))
    return {"checks": checks, "drop": drop}


def scorecard(checks: list) -> dict:
    counts = {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "warn", "fail")}
    passed = counts["fail"] == 0
    score = round(100 * (counts["pass"] + 0.5 * counts["warn"]) / max(len(checks), 1))
    verdict = "PASS" if passed and not counts["warn"] else "PASS WITH WARNINGS" if passed else "FAIL: pipeline halted"
    return {"checks": checks, "counts": counts, "passed": passed, "score": score, "verdict": verdict}
