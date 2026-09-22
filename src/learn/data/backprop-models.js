// Exact small investigations; training records are separate, frozen CPU observations.
export const fitDefaults = { weight: 1, bias: 0, rate: 0.1, target1: 1, target2: 3 };

export function lineFit({ weight, bias, rate, target1, target2 }) {
  const targets = [target1, target2];
  const rows = [1, 2].map((x, index) => {
    const prediction = weight * x + bias;
    const residual = prediction - targets[index];
    return { x, target: targets[index], prediction, residual, weightContribution: residual * x, biasContribution: residual };
  });
  const weightGradient = rows.reduce((sum, row) => sum + row.weightContribution, 0);
  const biasGradient = rows.reduce((sum, row) => sum + row.biasContribution, 0);
  const nextWeight = weight - rate * weightGradient;
  const nextBias = bias - rate * biasGradient;
  const updated = rows.map(row => ({ ...row, nextPrediction: nextWeight * row.x + nextBias }));
  const loss = rows.reduce((sum, row) => sum + row.residual ** 2, 0) / 2;
  const nextLoss = updated.reduce((sum, row) => sum + (row.nextPrediction - row.target) ** 2, 0) / 2;
  const delta = nextLoss - loss;
  const equal = Math.abs(delta) <= 1e-12 + 1e-10 * Math.max(Math.abs(loss), Math.abs(nextLoss));
  return { rows: updated, weightGradient, biasGradient, nextWeight, nextBias, loss, nextLoss, delta, change: equal ? 'unchanged' : delta < 0 ? 'decreased' : 'increased' };
}

export function sharedSquare(x, coefficient) {
  const square = x * x;
  const firstPath = 2 * x;
  const secondPath = coefficient * 2 * x;
  return { x, coefficient, square, scaled: coefficient * square, loss: square + coefficient * square,
    uGradient: 1 + coefficient, slotContribution: (1 + coefficient) * x,
    firstPath, secondPath, gradient: 2 * x * (1 + coefficient) };
}

export function finiteDifference({ kind = 'sine', point = 1, offset = 0, exponent = -3 }) {
  const step = 10 ** exponent;
  const fn = kind === 'linear' ? x => offset + x : kind === 'square' ? x => x * x : Math.sin;
  const analytic = kind === 'linear' ? 1 : kind === 'square' ? 2 * point : Math.cos(point);
  const left = point - step;
  const right = point + step;
  const lower = fn(left);
  const upper = fn(right);
  const estimate = (upper - lower) / (2 * step);
  const absoluteError = Math.abs(estimate - analytic);
  return { kind, point, offset, exponent, step, left, right, lower, upper, estimate, analytic, absoluteError,
    relativeError: analytic === 0 ? null : absoluteError / Math.abs(analytic), coordinatesCoalesce: left === right,
    evaluationsCoalesce: lower === upper };
}

export const broadcastSensitivity = [[1, 2], [3, 4], [5, 6]];
export const biasGradient = [0, 1].map(column => broadcastSensitivity.reduce((sum, row) => sum + row[column], 0));
export const directionalProducts = (() => {
  const jacobian = [[0.7, 0.3], [Math.cos(0.3), 0], [0, 1.4]];
  const direction = [1, 2];
  const weighting = [1, -1, 2];
  const jvp = jacobian.map(row => row.reduce((sum, value, i) => sum + value * direction[i], 0));
  const vjp = [0, 1].map(column => jacobian.reduce((sum, row, i) => sum + row[column] * weighting[i], 0));
  return { jacobian, direction, weighting, jvp, vjp, left: jvp.reduce((sum, value, i) => sum + value * weighting[i], 0), right: vjp.reduce((sum, value, i) => sum + value * direction[i], 0) };
})();

// Structural schedule, not measured bytes. Slots are explicitly unique activation states.
export const checkpointSchedule = [
  { title: 'Forward complete', boundaries: [0, 4, 8], regenerated: [], reverse: [], note: 'Compute operations 1–8; retain states 0, 4 and 8, and discard interior states 1–3 and 5–7.' },
  { title: 'Recompute second segment', boundaries: [0, 4, 8], regenerated: [5, 6, 7], reverse: [], note: 'Start from saved state 4. Re-run operations 5–7; state 8 is already saved. Local pullbacks now have states 4–8.' },
  { title: 'Reverse second segment', boundaries: [0, 4], regenerated: [], reverse: [8, 7, 6, 5], note: 'Apply pullbacks 8, 7, 6, 5 in this order. Accumulate sensitivity at state 4, then release states 5–8.' },
  { title: 'Recompute first segment', boundaries: [0, 4], regenerated: [1, 2, 3], reverse: [], note: 'Start from saved state 0. Re-run operations 1–3. Local pullbacks have states 0–4.' },
  { title: 'Reverse first segment', boundaries: [0], regenerated: [], reverse: [4, 3, 2, 1], note: 'Apply pullbacks 4, 3, 2, 1. The result is the sensitivity at input state 0.' },
];
