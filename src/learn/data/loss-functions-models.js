export const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
export function regressionPenalty(residual, method) {
  if (method === 'mse') return residual ** 2;
  if (method === 'mae') return Math.abs(residual);
  return Math.abs(residual) <= 1 ? residual ** 2 / 2 : Math.abs(residual) - .5;
}
export function regressionSlope(residual, method) {
  if (method === 'mse') return 2 * residual;
  if (method === 'mae') return Math.sign(residual); // selected subgradient 0 at equality
  return Math.max(-1, Math.min(1, residual));
}
export function fittedConstant(values, method) {
  if (method === 'mse') return mean(values);
  const sorted = [...values].sort((a, b) => a - b);
  if (method === 'mae') return sorted[Math.floor(sorted.length / 2)]; // UI fixes seven observations
  let lo = sorted[0], hi = sorted.at(-1);
  for (let i = 0; i < 80; i++) {
    const middle = (lo + hi) / 2;
    if (values.reduce((sum, value) => sum + regressionSlope(middle - value, 'huber'), 0) > 0) hi = middle;
    else lo = middle;
  }
  return (lo + hi) / 2;
}
export function focalTerm(probability, target, gamma, alpha = null) {
  const pt = target ? probability : 1 - probability;
  const weight = alpha === null ? 1 : target ? alpha : 1 - alpha;
  const loss = -weight * (1 - pt) ** gamma * Math.log(pt);
  const slope = weight * (target ? -1 : 1) * (1 - pt) ** gamma * ((1 - pt) - gamma * pt * Math.log(pt));
  return { loss, slope };
}
export function confusionAt(probabilities, labels, threshold) {
  const result = { tp: 0, fp: 0, fn: 0, tn: 0 };
  probabilities.forEach((p, i) => { result[labels[i] ? p >= threshold ? 'tp' : 'fn' : p >= threshold ? 'fp' : 'tn']++; });
  return result;
}
export function tripletGeometry(points, margin, squared = true) {
  const distance = (a, b) => { const d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2; return squared ? d2 : Math.sqrt(d2); };
  const positive = distance(points[0], points[1]);
  const candidates = points.slice(2).map((point, i) => {
    const negative = distance(points[0], point);
    return { id: i, distance: negative, loss: Math.max(0, positive - negative + margin), kind: negative <= positive ? 'hard' : negative < positive + margin ? 'semi-hard' : 'easy' };
  });
  const eligible = candidates.filter(row => row.kind === 'semi-hard').sort((a, b) => a.distance - b.distance || a.id - b.id);
  return { positive, candidates, selected: eligible.length ? eligible[0].id : null };
}
export function candidateCompetition(scores, temperature) {
  if (!(temperature > 0)) throw new RangeError('Temperature must be positive');
  const logits = scores.map(score => score / temperature), maximum = Math.max(...logits);
  const weights = logits.map(logit => Math.exp(logit - maximum)), total = weights.reduce((a, b) => a + b, 0);
  return { logits, probabilities: weights.map(w => w / total), loss: maximum + Math.log(total) - logits[0] };
}
