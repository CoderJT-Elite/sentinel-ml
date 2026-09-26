// Re-scores the champion model in the browser from the flat tree arrays that sentinel/whatif.py exports.
// Same walk as score_margin() in Python, so the two can be compared on any row.
//   XGBoost compares in float32 and goes left when x < threshold; LightGBM compares in double and goes left when x <= threshold.
//   A missing value takes the tree's `m` child.

export function scoreMargin(model, x) {
  const xgb = model.kind === "xgboost";
  let total = 0;
  for (const t of model.trees) {
    let node = 0;
    while (t.s[node] !== -1) {
      let v = x[t.s[node]];
      if (v == null || Number.isNaN(v)) node = t.m[node];
      else {
        if (xgb) v = Math.fround(v);
        node = (xgb ? v < Math.fround(t.c[node]) : v <= t.c[node]) ? t.l[node] : t.r[node];
      }
    }
    total += t.c[node];
  }
  return total;
}

export function scoreRisk(model, x) {
  const z = model.base + scoreMargin(model, x);
  return 1 / (1 + Math.exp(-z));
}

// The row with some features moved. `edits` maps a feature index to its new value.
export function withEdits(x, edits) {
  const out = x.slice();
  for (const [i, v] of Object.entries(edits)) out[+i] = v;
  return out;
}

// Cost of alerting at or above `thr`, counted on a fleet where every unit has one final reading and a known outcome.
export function fleetCost(fleet, thr, costMiss, costFalseAlarm) {
  let miss = 0, falseAlarm = 0;
  for (const f of fleet) {
    const alert = f.risk >= thr;
    if (f.actual >= 0.5 && !alert) miss++;
    else if (f.actual < 0.5 && alert) falseAlarm++;
  }
  return { miss, falseAlarm, total: miss * costMiss + falseAlarm * costFalseAlarm };
}

// The threshold with the lowest cost on this fleet. Ties go to the higher threshold (fewer alerts).
export function cheapestThreshold(fleet, costMiss, costFalseAlarm) {
  const cands = [...new Set(fleet.map((f) => f.risk))].sort((a, b) => a - b);
  let best = null;
  for (const t of cands) {
    const c = fleetCost(fleet, t, costMiss, costFalseAlarm);
    if (best === null || c.total <= best.total) best = { thr: t, ...c };
  }
  return best;
}
