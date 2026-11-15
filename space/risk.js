// Browser/Node implementation of the exported logistic regression.
// Must stay numerically identical to the scikit-learn pipeline
// (checked by tests/test_web_model.py).

export const HIGH_RISK = 0.6;
export const MEDIUM_RISK = 0.35;

/** One-hot encode `commit_type` and order the values like the training columns. */
export function toVector(model, features) {
  return model.columns.map((column) => {
    if (column.startsWith("commit_type_")) {
      return features.commit_type === column.slice("commit_type_".length) ? 1 : 0;
    }
    return Number(features[column] ?? model.defaults[column] ?? 0);
  });
}

/** Probability (0..1) that the commit introduces a bug. */
export function predictRisk(model, features) {
  const x = toVector(model, features);
  let logit = model.intercept;
  for (let i = 0; i < x.length; i++) {
    const z = (Math.log1p(x[i]) - model.mean[i]) / model.scale[i];
    logit += model.coef[i] * z;
  }
  return 1 / (1 + Math.exp(-logit));
}

/** Same rule as commit_risk.predict.risk_level: classify the displayed (rounded) value. */
export function riskLevel(risk) {
  const shown = Math.round(risk * 100) / 100;
  if (shown >= HIGH_RISK) return "HIGH";
  return shown >= MEDIUM_RISK ? "MEDIUM" : "LOW";
}

/** Share (0..1) of historical commits with a lower predicted risk. */
export function percentile(model, risk) {
  const below = model.risk_percentiles.filter((p) => p < risk).length;
  return Math.min(1, Math.max(0, (below - 1) / 100));
}
