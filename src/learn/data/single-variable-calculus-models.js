// Bounded analytic models for the single-variable calculus investigations.
// These functions evaluate declared formulas; they do not execute learner code.
function boundedNumber(value, name, minimum, maximum) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be finite and between ${minimum} and ${maximum}.`);
  }
}
function option(value, name, allowed) {
  if (!allowed.includes(value)) throw new RangeError(`Unknown ${name}.`);
}
function positiveInteger(value, name, minimum, maximum) {
  boundedNumber(value, name, minimum, maximum);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
}
export function motionPosition(time) {
  boundedNumber(time, 'Time', 0, 4);
  return time * (time - 3) ** 2;
}
export function motionVelocity(time) {
  boundedNumber(time, 'Time', 0, 4);
  return 3 * (time - 1) * (time - 3);
}
export function motionRate(baseTime, increment) {
  boundedNumber(baseTime, 'Base time', 1, 3);
  boundedNumber(increment, 'Time increment', -1, 1);
  if (Math.abs(increment) < 0.0001) {
    throw new RangeError('Use a nonzero increment with magnitude at least 0.0001.');
  }
  const position = motionPosition(baseTime);
  const nextPosition = motionPosition(baseTime + increment);
  const velocity = motionVelocity(baseTime);
  // Algebraically simplified quotient avoids subtracting nearby positions.
  const secant = velocity + (3 * baseTime - 6) * increment + increment ** 2;
  return {
    baseTime,
    increment,
    position,
    nextPosition,
    velocity,
    secant,
    acceleration: 6 * baseTime - 12,
    predictedChange: velocity * increment,
    exactChange: secant * increment,
    remainder: (3 * baseTime - 6) * increment ** 2 + increment ** 3
  };
}
function decimalRatio(value) {
  const [mantissa, exponent = '0'] = String(value).split('e');
  const [whole, fractional = ''] = mantissa.split('.');
  const decimalShift = Number(exponent) - fractional.length;
  return {
    numerator: BigInt(whole + fractional) * 10n ** BigInt(Math.max(0, decimalShift)),
    denominator: 10n ** BigInt(Math.max(0, -decimalShift))
  };
}
function rationalText(numerator, denominator) {
  let divisor = numerator;
  let remainder = denominator;
  while (remainder !== 0n) {
    [divisor, remainder] = [remainder, divisor % remainder];
  }
  if (divisor < 0n) divisor = -divisor;
  return `${numerator / divisor}/${denominator / divisor}`;
}
export function limitGuarantee(kind, epsilon, delta) {
  option(kind, 'limit example', ['smooth', 'hole', 'jump']);
  boundedNumber(epsilon, 'Requested error', 0.01, 1);
  boundedNumber(delta, 'Input radius', 0.001, 0.5);
  const smoothSupremum = delta * (4 + delta);
  const supremum = smoothSupremum + (kind === 'jump' ? 2 : 0);
  // Controls represent the displayed decimal inputs. Compare their exact ratios:
  // an open strip with supremum equal to epsilon satisfies the strict guarantee.
  const radiusRatio = decimalRatio(delta);
  const errorRatio = decimalRatio(epsilon);
  const jump = kind === 'jump' ? 2n : 0n;
  const radiusNumerator = radiusRatio.numerator;
  const radiusDenominator = radiusRatio.denominator;
  const maximumNumerator = radiusNumerator * (4n * radiusDenominator + radiusNumerator) + jump * radiusDenominator ** 2n;
  const guaranteed = maximumNumerator * errorRatio.denominator <= errorRatio.numerator * radiusDenominator ** 2n;
  let witness = null;
  if (!guaranteed) {
    // Approach the boundary from inside by d=delta*(1-2^-k). The exact
    // arithmetic certifies an interior counterexample even at a rounding edge.
    let scale = 2n;
    let numerator;
    let denominator;
    let errorNumerator;
    let found = false;
    for (let attempt = 0; attempt < 256; attempt += 1) {
      numerator = radiusNumerator * (scale - 1n);
      denominator = radiusDenominator * scale;
      errorNumerator = numerator * (4n * denominator + numerator) + jump * denominator ** 2n;
      if (errorNumerator * errorRatio.denominator >= errorRatio.numerator * denominator ** 2n) {
        found = true;
        break;
      }
      scale *= 2n;
    }
    if (!found) throw new Error('Could not certify an interior witness within the bounded decimal contract.');
    const displacement = Number(numerator) / Number(denominator);
    const input = 2 + displacement;
    const output = input ** 2 + (kind === 'jump' ? 2 : 0);
    witness = {
      input,
      output,
      displacement,
      error: output - 4,
      inputExact: rationalText(2n * denominator + numerator, denominator),
      displacementExact: rationalText(numerator, denominator),
      errorExact: rationalText(errorNumerator, denominator ** 2n),
      resolvedInFloat: displacement < delta && Math.abs(input - 2) < delta && output - 4 >= epsilon
    };
  }
  return {
    kind,
    epsilon,
    delta,
    limitCandidate: 4,
    centerValue: kind === 'hole' ? 6 : 4,
    leftLimit: 4,
    rightLimit: kind === 'jump' ? 6 : 4,
    supremum,
    guaranteed,
    witness,
    sufficientRadius: Math.min(1, epsilon / 5)
  };
}
export function compositionChange(input, increment) {
  boundedNumber(input, 'Input', -1, 1);
  boundedNumber(increment, 'Input increment', -0.5, 0.5);
  const intermediate = 3 * input + 1;
  const intermediateChange = 3 * increment;
  const output = intermediate ** 2;
  const nextOutput = (intermediate + intermediateChange) ** 2;
  const derivative = 6 * intermediate;
  return {
    input,
    increment,
    intermediate,
    intermediateChange,
    output,
    nextOutput,
    innerDerivative: 3,
    outerDerivative: 2 * intermediate,
    derivative,
    predictedChange: derivative * increment,
    exactChange: nextOutput - output,
    remainder: 9 * increment ** 2
  };
}
export function extremaValue(kind, input) {
  option(kind, 'extrema example', ['motion', 'inflection', 'cusp']);
  boundedNumber(input, 'Input', 0, 4);
  if (kind === 'motion') return motionPosition(input);
  if (kind === 'inflection') return (input - 2) ** 3;
  return Math.abs(input - 2);
}
function extremaDerivative(kind, input) {
  if (kind === 'motion') return motionVelocity(input);
  if (kind === 'inflection') return 3 * (input - 2) ** 2;
  return input === 2 ? null : Math.sign(input - 2);
}
function exactExtremaValue(kind, input) {
  const {
    numerator,
    denominator
  } = decimalRatio(input);
  if (kind === 'motion') return {
    numerator: numerator * (numerator - 3n * denominator) ** 2n,
    denominator: denominator ** 3n
  };
  const centered = numerator - 2n * denominator;
  if (kind === 'inflection') return {
    numerator: centered ** 3n,
    denominator: denominator ** 3n
  };
  return {
    numerator: centered < 0n ? -centered : centered,
    denominator
  };
}
function compareRationals(left, right) {
  const difference = left.numerator * right.denominator - right.numerator * left.denominator;
  return difference < 0n ? -1 : difference > 0n ? 1 : 0;
}
export function extremaCandidates(kind, left, right) {
  option(kind, 'extrema example', ['motion', 'inflection', 'cusp']);
  boundedNumber(left, 'Left endpoint', 0, 4);
  boundedNumber(right, 'Right endpoint', 0, 4);
  if (left >= right) throw new RangeError('The left endpoint must precede the right endpoint.');
  const criticalPoints = kind === 'motion' ? [1, 3] : [2];
  const inputs = [...new Set([left, right, ...criticalPoints.filter(point => point >= left && point <= right)])].sort((a, b) => a - b);
  const candidates = inputs.map(input => {
    const reasons = [];
    if (input === left) reasons.push('left endpoint');
    if (input === right) reasons.push('right endpoint');
    if (criticalPoints.includes(input)) reasons.push(kind === 'cusp' ? 'derivative undefined' : 'derivative zero');
    return {
      input,
      value: extremaValue(kind, input),
      derivative: extremaDerivative(kind, input),
      reasons
    };
  });
  const exactValues = inputs.map(input => exactExtremaValue(kind, input));
  const minimumIndex = exactValues.reduce((best, value, index) => compareRationals(value, exactValues[best]) < 0 ? index : best, 0);
  const maximumIndex = exactValues.reduce((best, value, index) => compareRationals(value, exactValues[best]) > 0 ? index : best, 0);
  return {
    kind,
    left,
    right,
    candidates: candidates.map((candidate, index) => ({
      ...candidate,
      valueExact: rationalText(exactValues[index].numerator, exactValues[index].denominator),
      isMinimum: compareRationals(exactValues[index], exactValues[minimumIndex]) === 0,
      isMaximum: compareRationals(exactValues[index], exactValues[maximumIndex]) === 0
    })),
    minimum: candidates[minimumIndex].value,
    maximum: candidates[maximumIndex].value,
    intervals: inputs.slice(0, -1).map((start, index) => {
      const end = inputs[index + 1];
      return {
        start,
        end,
        // The critical-point partition contains no derivative zero in its open
        // pieces. Classify their signs without a midpoint that can round away.
        sign: kind === 'inflection' ? 1 : kind === 'cusp' ? end <= 2 ? -1 : 1 : end <= 1 || start >= 3 ? 1 : -1
      };
    })
  };
}
export function motionAccumulation(upper, panelCount, method) {
  boundedNumber(upper, 'Upper time', 0.25, 4);
  positiveInteger(panelCount, 'Panel count', 1, 128);
  option(method, 'rectangle method', ['left', 'midpoint', 'right']);
  const width = upper / panelCount;
  const offset = {
    left: 0,
    midpoint: 0.5,
    right: 1
  }[method];
  const panels = Array.from({
    length: panelCount
  }, (_, index) => {
    const start = index * width;
    const sample = (index + offset) * width;
    const height = motionVelocity(sample);
    return {
      start,
      end: (index + 1) * width,
      sample,
      height,
      signedContribution: height * width
    };
  });
  const signedEstimate = panels.reduce((sum, panel) => sum + panel.signedContribution, 0);
  const distanceEstimate = panels.reduce((sum, panel) => sum + Math.abs(panel.signedContribution), 0);
  const boundaries = [0, ...[1, 3].filter(point => point < upper), upper];
  const exactDistance = boundaries.slice(1).reduce((sum, endpoint, index) => sum + Math.abs(motionPosition(endpoint) - motionPosition(boundaries[index])), 0);
  const exactDisplacement = motionPosition(upper);
  return {
    upper,
    panelCount,
    method,
    width,
    panels,
    signedEstimate,
    distanceEstimate,
    exactDisplacement,
    exactDistance,
    signedError: signedEstimate - exactDisplacement,
    endpointVelocity: motionVelocity(upper)
  };
}
export function exponentialRate(rate, period) {
  boundedNumber(rate, 'Continuous rate', -0.8, 0.8);
  boundedNumber(period, 'Period', 0.25, 2);
  const time = 2;
  const initialAmount = 10;
  const currentAmount = initialAmount * Math.exp(rate * time);
  const fraction = Math.expm1(rate * period);
  const exactChange = currentAmount * fraction;
  return {
    rate,
    period,
    time,
    initialAmount,
    currentAmount,
    futureAmount: currentAmount + exactChange,
    instantaneousRate: rate * currentAmount,
    intervalAverageRate: exactChange / period,
    perPeriodFraction: fraction,
    averageRelativeRate: fraction / period,
    tangentPrediction: currentAmount * (1 + rate * period)
  };
}
function factorial(integer) {
  let value = 1;
  for (let factor = 2; factor <= integer; factor += 1) value *= factor;
  return value;
}
export function taylorApproximation(kind, degree, input) {
  option(kind, 'Taylor example', ['exp', 'log']);
  positiveInteger(degree, 'Degree', 0, 12);
  boundedNumber(input, 'Input', kind === 'exp' ? -2 : -0.9, kind === 'exp' ? 2 : 1.5);
  let polynomial = kind === 'exp' ? 1 : 0;
  let power = 1;
  for (let order = 1; order <= degree; order += 1) {
    power *= input;
    polynomial += kind === 'exp' ? power / factorial(order) : (-1) ** (order + 1) * power / order;
  }
  const exactValue = kind === 'exp' ? Math.exp(input) : Math.log1p(input);
  const derivativeMaximum = kind === 'exp' ? Math.exp(Math.max(0, input)) : factorial(degree) / Math.min(1, 1 + input) ** (degree + 1);
  const remainderBound = derivativeMaximum * Math.abs(input) ** (degree + 1) / factorial(degree + 1);
  return {
    kind,
    degree,
    input,
    polynomial,
    exactValue,
    signedError: polynomial - exactValue,
    remainderBound,
    insideOpenLogSeriesInterval: kind === 'log' ? Math.abs(input) < 1 : null
  };
}
export function improperPowerIntegral(kind, power, cutoffExponent) {
  option(kind, 'improper integral', ['tail', 'endpoint']);
  boundedNumber(power, 'Power', 0, 3);
  boundedNumber(cutoffExponent, 'Cutoff exponent', 0, 6);
  const logCutoff = cutoffExponent * Math.LN10;
  const exponent = 1 - power;
  const cutoff = Math.exp((kind === 'tail' ? 1 : -1) * logCutoff);
  const truncated = exponent === 0 ? logCutoff : kind === 'tail' ? Math.expm1(exponent * logCutoff) / exponent : -Math.expm1(-exponent * logCutoff) / exponent;
  const converges = kind === 'tail' ? power > 1 : power < 1;
  const total = converges ? 1 / Math.abs(exponent) : null;
  const missing = converges ? total * Math.exp((kind === 'tail' ? 1 : -1) * exponent * logCutoff) : null;
  return {
    kind,
    power,
    cutoffExponent,
    cutoff,
    truncated,
    converges,
    total,
    missing
  };
}
