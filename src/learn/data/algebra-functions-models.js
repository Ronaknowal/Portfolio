// Deliberately bounded teaching models. No arbitrary expression evaluation.
export function algebraNumber(value) {
  if (!Number.isFinite(value)) throw new RangeError("A finite value is required.");
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 100000) return value.toExponential(4);
  return String(Number(value.toFixed(5)));
}
function bounded(value, low, high, name) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be between ${low} and ${high}.`);
  }
}
function onGrid(value, denominator, name) {
  if (!Number.isInteger(value * denominator)) throw new RangeError(`${name} must use the displayed ${1 / denominator} steps.`);
}
export function parseEquationDraft(draft) {
  const values = ["a", "b", "c"].map(key => {
    const text = String(draft[key]).trim();
    if (!/^[+-]?\d+$/.test(text)) throw new RangeError("Use whole-number coefficients from −30 to 30.");
    const value = Number(text);
    bounded(value, -30, 30, key);
    return value;
  });
  return equationState(...values);
}
export function equationState(a, b, c) {
  for (const [key, value] of Object.entries({
    a,
    b,
    c
  })) {
    bounded(value, -30, 30, key);
    if (!Number.isInteger(value)) throw new RangeError("Coefficients must be integers.");
  }
  const rhs = c - b;
  const kind = a === 0 ? rhs === 0 ? "all" : "none" : "one";
  const solution = kind === "one" ? rhs / a : null;
  const frames = [{
    left: `${a}x + (${b})`,
    right: String(c),
    action: "Original equation. Find every x that makes this true."
  }, {
    left: `${a}x`,
    right: String(rhs),
    action: `Subtract ${b} from both sides. This operation is reversible.`
  }, kind === "one" ? {
    left: "x",
    right: `${rhs}/${a}`,
    action: `Divide both sides by ${a}, which is nonzero. The fraction is exact.`
  } : {
    left: "0",
    right: String(rhs),
    action: kind === "all" ? "Always true: every real x works. Do not divide by zero." : "Never true: no x works. Do not divide by zero."
  }];
  return {
    a,
    b,
    c,
    rhs,
    kind,
    solution,
    frames
  };
}
export function functionProbeState(kind, x, restricted = false) {
  if (!["affine", "square", "reciprocal"].includes(kind)) throw new RangeError("Choose a displayed function.");
  bounded(x, -4, 4, "Input");
  onGrid(x, 4, "Input");
  if (kind === "reciprocal" && x !== 0 && Math.abs(x) < 0.000001) throw new RangeError("Nonzero probe magnitude must be at least 0.000001.");
  const allowed = !(kind === "reciprocal" && x === 0) && !(kind === "square" && restricted && x < 0);
  const y = !allowed ? null : kind === "affine" ? 2 * x + 1 : kind === "square" ? x * x : 1 / x;
  const preimages = !allowed ? [] : kind === "square" && !restricted && x !== 0 ? [-Math.abs(x), Math.abs(x)] : [x];
  return {
    kind,
    x,
    y,
    allowed,
    preimages,
    restricted: kind === "square" && restricted
  };
}
export function compositionState(x) {
  bounded(x, -4, 4, "Input");
  onGrid(x, 4, "Input");
  const affine = 2 * x + 1;
  const square = x * x;
  return {
    x,
    affine,
    square,
    squareAfterAffine: affine * affine,
    affineAfterSquare: 2 * square + 1,
    recovered: (affine - 1) / 2
  };
}
export function quadraticState(h, k) {
  bounded(h, -2, 4, "Vertex horizontal coordinate");
  bounded(k, -9, 4, "Vertex vertical coordinate");
  onGrid(h, 2, "Vertex horizontal coordinate");
  onGrid(k, 2, "Vertex vertical coordinate");
  const roots = k > 0 ? [] : k === 0 ? [h] : [h - Math.sqrt(-k), h + Math.sqrt(-k)];
  return {
    h,
    k,
    b: -2 * h,
    c: h * h + k,
    discriminant: -4 * k,
    roots
  };
}
export function growthState(rate, step, factor = 3) {
  bounded(rate, -0.5, 1, "Per-period fractional rate");
  if (![-0.5, -0.2, 0, 0.1, 0.2, 0.5, 1].includes(rate)) throw new RangeError("Choose one of the displayed per-period rates.");
  bounded(step, 0, 8, "Period");
  if (!Number.isInteger(step)) throw new RangeError("Choose a whole observation period.");
  bounded(factor, 0.125, 8, "Target / initial quantity");
  const multiplier = 1 + rate;
  const rows = Array.from({
    length: 9
  }, (_, t) => ({
    t,
    additive: 100 + 20 * t,
    growth: 100 * multiplier ** t,
    change: t === 0 ? null : 100 * rate * multiplier ** (t - 1)
  }));
  const crossing = rate === 0 ? factor === 1 ? 0 : null : Math.log(factor) / Math.log1p(rate);
  return {
    rate,
    step,
    factor,
    multiplier,
    rows,
    active: rows[step],
    crossing,
    future: crossing !== null && crossing >= 0
  };
}
export function logarithmState(base, exponent) {
  if (![0.5, 2, 10].includes(base)) throw new RangeError("Choose base 0.5, 2 or 10.");
  bounded(exponent, -3, 3, "Exponent");
  onGrid(exponent, 4, "Exponent");
  return {
    base,
    exponent,
    value: base ** exponent,
    increasing: base > 1,
    ticks: [-3, -2, -1, 0, 1, 2, 3].map(power => ({
      power,
      value: base ** power
    }))
  };
}
