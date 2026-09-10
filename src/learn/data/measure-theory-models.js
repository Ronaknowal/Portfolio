// Bounded, deterministic probability-space teaching models.
export const DIE_OUTCOMES = Object.freeze([1, 2, 3, 4, 5, 6]);
export const DIE_LOSSES = Object.freeze([0, 2, 4, 4, 8, 12]);
export const INFORMATION_PARTITIONS = Object.freeze({
  none: {
    label: 'No face information',
    cells: [[1, 2, 3, 4, 5, 6]]
  },
  parity: {
    label: 'Only odd or even',
    cells: [[1, 3, 5], [2, 4, 6]]
  },
  pairs: {
    label: 'Which adjacent pair',
    cells: [[1, 2], [3, 4], [5, 6]]
  },
  full: {
    label: 'Exact face',
    cells: [[1], [2], [3], [4], [5], [6]]
  }
});
function bounded(value, low, high, name, integer = false) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high || integer && !Number.isInteger(value)) {
    throw new RangeError(name + ' must be ' + (integer ? 'an integer' : 'a finite number') + ' between ' + low + ' and ' + high + '.');
  }
  return value;
}
function chooseOption(value, choices, name) {
  if (!choices.includes(value)) throw new RangeError('Unknown ' + name + '.');
  return value;
}
function partition(id) {
  chooseOption(id, Object.keys(INFORMATION_PARTITIONS), 'information partition');
  return INFORMATION_PARTITIONS[id];
}
export function measureNumber(value, digits = 4) {
  if (value === null) return 'undefined here';
  if (value === Infinity) return '+∞';
  if (value === -Infinity) return '−∞';
  if (!Number.isFinite(value)) return 'undefined';
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
export function informationState(partitionId = 'pairs', eventMask = 3) {
  const selected = partition(partitionId);
  bounded(eventMask, 0, 63, 'Event mask', true);
  const event = DIE_OUTCOMES.filter(face => Boolean(eventMask & 1 << face - 1));
  const cells = selected.cells.map(faces => {
    const included = faces.filter(face => event.includes(face));
    return {
      faces,
      included,
      split: included.length > 0 && included.length < faces.length
    };
  });
  const algebra = Array.from({
    length: 2 ** cells.length
  }, (_, mask) => {
    const faces = cells.flatMap((cell, index) => mask & 1 << index ? cell.faces : []);
    return faces.sort((a, b) => a - b);
  });
  return {
    partitionId,
    label: selected.label,
    eventMask,
    event,
    cells,
    algebra,
    observable: cells.every(cell => !cell.split),
    ambientProbability: event.length / 6,
    observableEventCount: algebra.length
  };
}
export function preimageState(threshold = 1) {
  bounded(threshold, 0, 3, 'Output threshold');
  const rows = DIE_OUTCOMES.map(face => {
    const value = Math.max(face - 3, 0);
    return {
      face,
      value,
      selected: value <= threshold
    };
  });
  const law = [0, 1, 2, 3].map(value => ({
    value,
    probability: rows.filter(row => row.value === value).length / 6
  }));
  return {
    threshold,
    rows,
    law,
    preimage: rows.filter(row => row.selected).map(row => row.face),
    probability: rows.filter(row => row.selected).length / 6,
    mean: rows.reduce((sum, row) => sum + row.value / 6, 0)
  };
}
export function mixedCdf(x, atomWeight = .25) {
  bounded(x, -10, 10, 'CDF coordinate');
  bounded(atomWeight, 0, 1, 'Atom probability');
  if (x < 0) return 0;
  if (x >= 1) return 1;
  return atomWeight + (1 - atomWeight) * x;
}
export function mixedMeasureState(atomWeight = .25, lower = 0, upper = .5) {
  bounded(atomWeight, 0, 1, 'Atom probability');
  bounded(lower, -.25, 1.25, 'Lower endpoint');
  bounded(upper, -.25, 1.25, 'Upper endpoint');
  if (lower > upper) throw new RangeError('Lower endpoint must not exceed the upper endpoint.');
  const continuousLength = Math.max(0, Math.min(1, upper) - Math.max(0, lower));
  const atomIncluded = lower <= 0 && upper >= 0;
  const atomMass = atomIncluded ? atomWeight : 0;
  const continuousMass = (1 - atomWeight) * continuousLength;
  return {
    atomWeight,
    lower,
    upper,
    atomIncluded,
    atomMass,
    continuousLength,
    continuousMass,
    probability: atomMass + continuousMass,
    continuousDensity: 1 - atomWeight,
    mean: (1 - atomWeight) / 2,
    secondMoment: (1 - atomWeight) / 3,
    hasLebesgueDensity: atomWeight === 0,
    cdfAtLower: mixedCdf(lower, atomWeight),
    cdfBeforeLower: lower === 0 ? 0 : mixedCdf(lower, atomWeight),
    cdfAtUpper: mixedCdf(upper, atomWeight)
  };
}
export function cantorCoverState(level = 3) {
  bounded(level, 0, 6, 'Cantor construction level', true);
  let intervals = [{
    low: 0,
    high: 1
  }];
  for (let step = 0; step < level; step += 1) {
    intervals = intervals.flatMap(({
      low,
      high
    }) => {
      const third = (high - low) / 3;
      return [{
        low,
        high: low + third
      }, {
        low: high - third,
        high
      }];
    });
  }
  const eachMass = 2 ** -level;
  return {
    level,
    intervals: intervals.map(interval => ({
      ...interval,
      mass: eachMass
    })),
    intervalCount: 2 ** level,
    totalLength: (2 / 3) ** level,
    eachLength: 3 ** -level,
    eachMass,
    totalProbability: 1,
    approximationDensity: (3 / 2) ** level
  };
}
export function simpleIntegralState(level = 2) {
  bounded(level, 0, 6, 'Value refinement level', true);
  const count = 2 ** level;
  const steps = Array.from({
    length: count
  }, (_, index) => {
    const low = Math.sqrt(index / count);
    const high = Math.sqrt((index + 1) / count);
    const value = index / count;
    return {
      index,
      low,
      high,
      value,
      mass: high - low,
      contribution: value * (high - low)
    };
  });
  const lowerIntegral = steps.reduce((sum, step) => sum + step.contribution, 0);
  return {
    level,
    count,
    steps,
    lowerIntegral,
    upperIntegral: lowerIntegral + 1 / count,
    exactIntegral: 1 / 3,
    error: 1 / 3 - lowerIntegral,
    uniformErrorBound: 1 / count
  };
}
export const LIMIT_MODES = Object.freeze({
  spike: 'Shrinking support, unchanged integral',
  bounded: 'One common integrable bound',
  increasing: 'Increasing truncations'
});
export function limitFunction(mode, n, x) {
  chooseOption(mode, Object.keys(LIMIT_MODES), 'limit example');
  bounded(n, 1, 64, 'Sequence index', true);
  bounded(x, 0, 1, 'Domain coordinate');
  if (mode === 'spike') return x > 0 && x < 1 / n ? n : 0;
  if (mode === 'bounded') return x ** n;
  return x === 0 ? n : Math.min(n, 1 / Math.sqrt(x));
}
export function limitIntegralState(mode = 'spike', n = 4, x = .25) {
  const value = limitFunction(mode, n, x);
  const integral = mode === 'spike' ? 1 : mode === 'bounded' ? 1 / (n + 1) : 2 - 1 / n;
  const pointwiseLimit = mode === 'increasing' ? x === 0 ? Infinity : 1 / Math.sqrt(x) : mode === 'bounded' && x === 1 ? 1 : 0;
  const integralOfLimit = mode === 'increasing' ? 2 : 0;
  const limitOfIntegrals = mode === 'spike' ? 1 : integralOfLimit;
  const special = mode === 'increasing' ? 1 / (n * n) : 1 / n;
  // Include the exact truncation corner and geometric points near zero.
  const coordinates = [...new Set([...Array.from({
    length: 201
  }, (_, index) => index / 200), ...Array.from({
    length: 65
  }, (_, index) => special * (index / 64) ** 2), special, x])].sort((a, b) => a - b);
  return {
    mode,
    n,
    x,
    value,
    integral,
    pointwiseLimit,
    integralOfLimit,
    limitOfIntegrals,
    yMaximum: mode === 'bounded' ? 1 : n,
    corner: special,
    points: coordinates.map(coordinate => ({
      x: coordinate,
      y: limitFunction(mode, n, coordinate)
    }))
  };
}
export function jointCellState(column = 2, row = 1, scaleX = 2, scaleY = 3) {
  bounded(column, 0, 3, 'Cell column', true);
  bounded(row, 0, 3, 'Cell row', true);
  bounded(scaleX, .5, 3, 'Horizontal coordinate scale');
  bounded(scaleY, .5, 3, 'Vertical coordinate scale');
  const x0 = column / 4,
    x1 = (column + 1) / 4;
  const y0 = row / 4,
    y1 = (row + 1) / 4;
  const jacobian = scaleX * scaleY;
  const area = (x1 - x0) * (y1 - y0);
  const centerDensity = 4 * ((x0 + x1) / 2) * ((y0 + y1) / 2);
  const probability = (x1 * x1 - x0 * x0) * (y1 * y1 - y0 * y0);
  return {
    column,
    row,
    scaleX,
    scaleY,
    x0,
    x1,
    y0,
    y1,
    jacobian,
    area,
    centerDensity,
    probability,
    u0: scaleX * x0,
    u1: scaleX * x1,
    v0: scaleY * y0,
    v1: scaleY * y1,
    transformedArea: jacobian * area,
    transformedDensity: centerDensity / jacobian,
    marginalCellProbability: x1 * x1 - x0 * x0,
    cells: Array.from({
      length: 16
    }, (_, index) => {
      const c = index % 4,
        r = Math.floor(index / 4);
      return {
        column: c,
        row: r,
        probability: (2 * c + 1) * (2 * r + 1) / 256
      };
    })
  };
}
export function signedArrayState(rows = 4, columns = 5) {
  bounded(rows, 1, 12, 'Array rows', true);
  bounded(columns, 1, 13, 'Array columns', true);
  const values = Array.from({
    length: rows
  }, (_, i) => Array.from({
    length: columns
  }, (_, j) => j === i ? 1 : j === i + 1 ? -1 : 0));
  const rowSums = values.map(row => row.reduce((sum, value) => sum + value, 0));
  const columnSums = Array.from({
    length: columns
  }, (_, column) => values.reduce((sum, row) => sum + row[column], 0));
  return {
    rows,
    columns,
    values,
    rowSums,
    columnSums,
    finiteSum: rowSums.reduce((sum, value) => sum + value, 0),
    finiteAbsoluteSum: values.flat().reduce((sum, value) => sum + Math.abs(value), 0)
  };
}
export function parseLosses(text) {
  if (typeof text !== 'string') throw new RangeError('Enter six comma-separated losses.');
  const fields = text.split(',').map(field => field.trim());
  if (fields.length !== 6 || fields.some(field => field === '')) {
    throw new RangeError('Enter exactly six losses, separated by commas.');
  }
  const values = fields.map(Number);
  for (const value of values) bounded(value, -100, 100, 'Each loss');
  return values;
}
function samplingWeights(id) {
  chooseOption(id, ['fair', 'missing-six'], 'sampling model');
  return id === 'fair' ? Array(6).fill(1 / 6) : [.2, .2, .2, .2, .2, 0];
}
export function conditionalMeanState(partitionId = 'pairs', samplingId = 'fair', values = DIE_LOSSES, nullValue = 0) {
  const selected = partition(partitionId);
  const weights = samplingWeights(samplingId);
  bounded(nullValue, -100, 100, 'Chosen null-cell value');
  if (!Array.isArray(values) || values.length !== 6) throw new RangeError('Supply six losses.');
  for (const value of values) bounded(value, -100, 100, 'Each loss');
  const cells = selected.cells.map(faces => {
    const mass = faces.reduce((sum, face) => sum + weights[face - 1], 0);
    const weightedLoss = faces.reduce((sum, face) => sum + weights[face - 1] * values[face - 1], 0);
    return {
      faces,
      mass,
      weightedLoss,
      prediction: mass === 0 ? nullValue : weightedLoss / mass
    };
  });
  const rows = DIE_OUTCOMES.map(face => {
    const cell = cells.find(item => item.faces.includes(face));
    const residual = values[face - 1] - cell.prediction;
    return {
      face,
      loss: values[face - 1],
      weight: weights[face - 1],
      prediction: cell.prediction,
      residual,
      weightedSquare: weights[face - 1] * residual * residual
    };
  });
  const mean = rows.reduce((sum, row) => sum + row.weight * row.loss, 0);
  const risk = rows.reduce((sum, row) => sum + row.weightedSquare, 0);
  const variance = rows.reduce((sum, row) => sum + row.weight * (row.loss - mean) ** 2, 0);
  const predictionVariance = rows.reduce((sum, row) => sum + row.weight * (row.prediction - mean) ** 2, 0);
  return {
    partitionId,
    samplingId,
    nullValue,
    values: [...values],
    weights,
    cells,
    rows,
    mean,
    risk,
    variance,
    predictionVariance,
    predictionMean: rows.reduce((sum, row) => sum + row.weight * row.prediction, 0),
    hasNullCell: cells.some(cell => cell.mass === 0)
  };
}
export const TARGET_MEASURES = Object.freeze({
  uniform: {
    label: 'Uniform target',
    weights: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
  },
  lossHeavy: {
    label: 'More weight on higher-loss outcomes',
    weights: [0, 0, 1 / 6, 1 / 6, 1 / 3, 1 / 3]
  },
  last: {
    label: 'All target mass on face 6',
    weights: [0, 0, 0, 0, 0, 1]
  }
});
export function reweightState(sourceId = 'fair', targetId = 'lossHeavy') {
  const source = samplingWeights(sourceId);
  chooseOption(targetId, Object.keys(TARGET_MEASURES), 'target measure');
  const target = TARGET_MEASURES[targetId].weights;
  const rows = DIE_OUTCOMES.map(face => {
    const p = source[face - 1],
      q = target[face - 1];
    return {
      face,
      p,
      q,
      loss: DIE_LOSSES[face - 1],
      unsupported: p === 0 && q > 0,
      ratio: p > 0 ? q / p : q === 0 ? 0 : null
    };
  });
  const supported = rows.every(row => !row.unsupported);
  return {
    sourceId,
    targetId,
    rows,
    supported,
    sourceMean: rows.reduce((sum, row) => sum + row.p * row.loss, 0),
    targetMean: rows.reduce((sum, row) => sum + row.q * row.loss, 0),
    weightedMean: supported ? rows.reduce((sum, row) => sum + row.p * row.ratio * row.loss, 0) : null,
    ratioMean: supported ? rows.reduce((sum, row) => sum + row.p * row.ratio, 0) : null,
    unsupportedMass: rows.filter(row => row.unsupported).reduce((sum, row) => sum + row.q, 0)
  };
}
