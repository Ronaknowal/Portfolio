// Bounded teaching models. Exact rational traces are distinct from measured Number arithmetic.
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
  return value;
}
export function formatConditioning(value, digits = 6) {
  if (value === null) return 'not defined';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 1e7) return value.toExponential(digits);
  return Number(value.toPrecision(digits + 1)).toString();
}
function gcd(a, b) {
  a = a < 0n ? -a : a;
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
}
function rational(numerator, denominator = 1n) {
  if (denominator === 0n) throw new RangeError('Zero rational denominator.');
  if (denominator < 0n) [numerator, denominator] = [-numerator, -denominator];
  const divisor = gcd(numerator, denominator);
  return [numerator / divisor, denominator / divisor];
}
function subtract([a, b], [c, d]) {
  return rational(a * d - c * b, b * d);
}
function add([a, b], [c, d]) {
  return rational(a * d + c * b, b * d);
}
function absolute([a, b]) {
  return [a < 0n ? -a : a, b];
}
function asNumber([a, b]) {
  return Number(a) / Number(b);
}
function asText([a, b]) {
  return b === 1n ? String(a) : `${a}/${b}`;
}
export function storedNumberFraction(value) {
  if (!Number.isFinite(value)) throw new RangeError('Use a finite stored number.');
  if (value === 0) return [0n, 1n];
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  const sign = bits >> 63n ? -1n : 1n;
  const exponent = Number(bits >> 52n & 2047n);
  const fraction = bits & (1n << 52n) - 1n;
  const significand = exponent === 0 ? fraction : fraction + (1n << 52n);
  const shift = (exponent === 0 ? -1022 : exponent - 1023) - 52;
  return shift >= 0 ? rational(sign * (significand << BigInt(shift))) : rational(sign * significand, 1n << BigInt(-shift));
}
function integerSqrt(value) {
  if (value < 0n) throw new RangeError('Negative square root.');
  if (value < 2n) return value;
  let root = 1n << BigInt(Math.ceil(value.toString(2).length / 2));
  while (true) {
    const next = root + value / root >> 1n;
    if (next >= root) return root;
    root = next;
  }
}
export function roundingCellState(bin = 0, halfStep = 1) {
  integer(bin, 0, 1, 'Exponent bin');
  integer(halfStep, 0, 16, 'Half-step');
  const spacing = 2 ** bin / 8;
  const significand = 8 + halfStep / 2;
  const lower = Math.floor(significand);
  const upper = Math.ceil(significand);
  const chosen = significand % 1 === 0 ? lower : lower % 2 === 0 ? lower : upper;
  return {
    spacing,
    input: significand * spacing,
    rounded: chosen * spacing,
    lower: lower * spacing,
    upper: upper * spacing,
    halfway: lower !== upper,
    grid: Array.from({
      length: 9
    }, (_, i) => (8 + i) * spacing)
  };
}
export function cancellationState(exponent = 54, sign = 1) {
  integer(exponent, 0, 60, 'Dyadic exponent');
  if (![0, 1, -1].includes(sign)) throw new RangeError('Sign must be −1, 0 or 1.');
  const input = sign * 2 ** -exponent;
  const sum = 1 + input;
  const root = Math.sqrt(sum);
  const direct = root - 1;
  const repaired = input / (root + 1);
  const bits = 160;
  const scaled = (1n << BigInt(exponent)) + BigInt(sign) << BigInt(2 * bits - exponent);
  const floor = integerSqrt(scaled);
  const denominator = 1n << BigInt(bits);
  const lower = rational(floor - denominator, denominator);
  const upper = rational(floor + (floor * floor === scaled ? 0n : 1n) - denominator, denominator);
  const reference = (asNumber(lower) + asNumber(upper)) / 2;
  const errors = value => {
    const exactStored = storedNumberFraction(value);
    const differenceLow = subtract(exactStored, upper);
    const differenceHigh = subtract(exactStored, lower);
    const low = asNumber(differenceLow);
    const high = asNumber(differenceHigh);
    const absoluteLow = low <= 0 && high >= 0 ? 0 : Math.min(Math.abs(low), Math.abs(high));
    const absoluteHigh = Math.max(Math.abs(low), Math.abs(high));
    return {
      absoluteLow,
      absoluteHigh,
      relativeUpper: reference === 0 ? null : absoluteHigh / Math.min(Math.abs(asNumber(lower)), Math.abs(asNumber(upper)))
    };
  };
  return {
    input,
    sum,
    root,
    direct,
    repaired,
    reference,
    referenceLower: asText(lower),
    referenceUpper: asText(upper),
    directError: errors(direct),
    repairedError: errors(repaired),
    condition: input === 0 || input === -1 ? null : (root + 1) / (2 * root)
  };
}
export function measurementSensitivityState(exponent = 4, perturbation = 1, singular = false) {
  integer(exponent, 1, 7, 'Separation exponent');
  integer(perturbation, -2, 2, 'Reading perturbation');
  if (typeof singular !== 'boolean') throw new TypeError('Singular state must be boolean.');
  const epsilon = singular ? 0 : 2 ** -exponent;
  const delta = perturbation / 256;
  if (singular) return {
    epsilon,
    delta,
    solution: null,
    status: delta === 0 ? 'nonunique' : 'inconsistent'
  };
  const shift = delta / epsilon;
  return {
    epsilon,
    delta,
    solution: [1 - shift, 1 + shift],
    status: 'unique',
    condition: (2 + epsilon) ** 2 / epsilon,
    inputRelative: Math.abs(delta) / (2 + epsilon),
    outputRelative: Math.abs(shift),
    bound: (2 + epsilon) * Math.abs(shift)
  };
}
export function backwardWitnessState(power = 6, scaled = false) {
  integer(power, 1, 9, 'Small-row exponent');
  if (typeof scaled !== 'boolean') throw new TypeError('Scaling selection must be boolean.');
  const small = 10 ** -power;
  const row = scaled ? 1 : small;
  return {
    small,
    row,
    matrix: [[1, 0], [0, row]],
    rhs: [1, row],
    approximate: [1, 0],
    residual: [0, row],
    forwardError: 1,
    normwise: row / 2,
    componentwise: 1,
    changedEntry: row / 2,
    changedRhs: row / 2,
    condition: 1 / row,
    displayedReadingUncertainty: scaled ? 0.01 : small * 0.01,
    amountUncertainty: 0.01,
    jointBound: 2
  };
}
export const summationPresets = Object.freeze({
  cancellation: Object.freeze([1e16, 1, -1e16]),
  reordered: Object.freeze([1e16, -1e16, 1]),
  positive: Object.freeze([2 ** 53, 1, 1, 1, 1]),
  changed: Object.freeze([1e16, 3, -1e16]),
  zero: Object.freeze([1, -1])
});
export function summationState(preset = 'cancellation') {
  if (!Object.hasOwn(summationPresets, preset)) throw new RangeError('Choose a listed summation preset.');
  const values = [...summationPresets[preset]];
  let exact = [0n, 1n];
  let absoluteSum = [0n, 1n];
  let total = 0;
  let correction = 0;
  let kahanTotal = 0;
  let kahanCorrection = 0;
  const steps = values.map((value, index) => {
    exact = add(exact, storedNumberFraction(value));
    absoluteSum = add(absoluteSum, absolute(storedNumberFraction(value)));
    const before = total;
    const updated = total + value;
    const lost = subtract(add(storedNumberFraction(before), storedNumberFraction(value)), storedNumberFraction(updated));
    correction += Math.abs(total) >= Math.abs(value) ? total - updated + value : value - updated + total;
    total = updated;
    const adjusted = value - kahanCorrection;
    const nextKahan = kahanTotal + adjusted;
    kahanCorrection = nextKahan - kahanTotal - adjusted;
    kahanTotal = nextKahan;
    return {
      index,
      value,
      before,
      updated,
      lost: asText(lost),
      correction,
      exactPrefix: asText(exact)
    };
  });
  function tree(start, end) {
    if (end - start === 1) return {
      start,
      end,
      value: values[start],
      children: []
    };
    const middle = Math.floor((start + end) / 2);
    const children = [tree(start, middle), tree(middle, end)];
    return {
      start,
      end,
      value: children[0].value + children[1].value,
      children
    };
  }
  const balanced = tree(0, values.length);
  const u = 2 ** -53;
  const gamma = (values.length - 1) * u / (1 - (values.length - 1) * u);
  return {
    values,
    steps,
    exact: asText(exact),
    naive: total,
    neumaier: total + correction,
    kahan: kahanTotal,
    balanced,
    gammaBound: gamma * asNumber(absoluteSum),
    condition: exact[0] === 0n ? null : asNumber(absoluteSum) / Math.abs(asNumber(exact))
  };
}
export function propagationState(q = 0.5, mode = 'constant', steps = 12) {
  if (![-0.5, 0.5, 0.9, 1, 1.1].includes(q)) throw new RangeError('Choose a listed multiplier.');
  if (!['constant', 'alternating', 'pulse'].includes(mode)) throw new RangeError('Choose a listed disturbance.');
  integer(steps, 1, 24, 'Number of steps');
  let error = 0;
  let bound = 0;
  const frames = [{
    step: 0,
    error,
    bound,
    disturbance: 0
  }];
  for (let step = 1; step <= steps; step += 1) {
    const disturbance = mode === 'pulse' ? step === 1 ? 0.01 : 0 : 0.01 * (mode === 'alternating' ? (-1) ** step : 1);
    error = q * error + disturbance;
    bound = Math.abs(q) * bound + (mode === 'pulse' && step > 1 ? 0 : 0.01);
    frames.push({
      step,
      error,
      bound,
      disturbance
    });
  }
  return {
    q,
    mode,
    frames
  };
}
export function unstableRefinementState(steps = 8) {
  if (![4, 8, 16, 32].includes(steps)) throw new RangeError('Use 4, 8, 16 or 32 steps.');
  const h = 1 / steps;
  const frames = Array.from({
    length: steps + 1
  }, (_, step) => ({
    step,
    time: step * h,
    error: (2 ** step - 1) * h ** 2,
    stableError: step === 0 ? 0 : h ** 2
  }));
  return {
    h,
    steps,
    initialError: h ** 2,
    finalError: frames.at(-1).error,
    frames
  };
}
