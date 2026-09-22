// Topic-owned exact calculations. Plots sample these functions, not saved knots.
export const activationNames = {
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
  relu: 'ReLU',
  leaky_relu: 'Leaky ReLU (0.1)',
  gelu: 'GELU (exact definition)',
  gelu_tanh: 'GELU (tanh approximation)',
  silu: 'SiLU',
  elu: 'ELU (α = 1)',
  mish: 'Mish'
};
export const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
export const softplus = z => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
export function normalCdf(z) {
  if (z === 0) return .5;
  const x = Math.abs(z);
  if (x > 9) return z > 0 ? 1 : 0;
  // Integral series after factoring exp(-x²/2); positive terms avoid cancellation.
  let term = x,
    sum = x;
  for (let n = 1; n < 300; n++) {
    term *= x * x / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-16) break;
  }
  const area = Math.exp(-x * x / 2) * sum / Math.sqrt(2 * Math.PI);
  return .5 + Math.sign(z) * area;
}
export function activation(name, z) {
  const s = sigmoid(z),
    t = Math.tanh(z);
  switch (name) {
    case 'sigmoid':
      return {
        value: s,
        slope: s * (1 - s)
      };
    case 'tanh':
      return {
        value: t,
        slope: 1 - t * t
      };
    case 'relu':
      return {
        value: Math.max(0, z),
        slope: z > 0 ? 1 : 0,
        corner: z === 0
      };
    case 'leaky_relu':
      return {
        value: z >= 0 ? z : .1 * z,
        slope: z > 0 ? 1 : .1,
        corner: z === 0
      };
    case 'elu':
      return {
        value: z >= 0 ? z : Math.expm1(z),
        slope: z >= 0 ? 1 : Math.exp(z)
      };
    case 'gelu':
      {
        const c = normalCdf(z);
        return {
          value: z * c,
          slope: c + z * Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI)
        };
      }
    case 'gelu_tanh':
      {
        const k = Math.sqrt(2 / Math.PI),
          tanh = Math.tanh(k * (z + .044715 * z * z * z));
        return {
          value: .5 * z * (1 + tanh),
          slope: .5 * (1 + tanh) + .5 * z * (1 - tanh * tanh) * k * (1 + 3 * .044715 * z * z)
        };
      }
    case 'silu':
      return {
        value: z * s,
        slope: s + z * s * (1 - s)
      };
    case 'mish':
      {
        const a = Math.tanh(softplus(z));
        return {
          value: z * a,
          slope: a + z * (1 - a * a) * s
        };
      }
    default:
      throw new Error(`Unknown activation: ${name}`);
  }
}
export function sensitivity(name, z, weight) {
  const result = activation(name, z),
    value = weight * result.slope;
  return {
    ...result,
    sensitivity: value,
    category: value < -1e-10 ? 'negative' : Math.abs(value) <= 1e-10 ? 'zero' : value < 1 - 1e-10 ? 'positive below 1' : 'at least 1'
  };
}
export const geometryInitial = {
  x1: 2,
  x2: -1,
  w1: 1.5,
  w2: -2,
  b: -1,
  c: 1
};
export function geometry(state, scale = 1) {
  const {
      x1,
      x2
    } = state,
    w1 = state.w1 * scale,
    w2 = state.w2 * scale,
    b = state.b * scale;
  const products = [w1 * x1, w2 * x2, b],
    score = products.reduce((a, v) => a + v, 0),
    norm = Math.hypot(w1, w2);
  return {
    products,
    score,
    norm,
    distance: norm ? score / norm : null,
    hard: Number(score > 0),
    sigmoid: sigmoid(score),
    relu: Math.max(0, score),
    foot: norm ? [x1 - score * w1 / (norm * norm), x2 - score * w2 / (norm * norm)] : null
  };
}
export function boundarySegment(w1, w2, b, lo = -4, hi = 4) {
  if (w1 === 0 && w2 === 0) return [];
  const points = [];
  const add = (x, y) => {
    if (x >= lo - 1e-9 && x <= hi + 1e-9 && y >= lo - 1e-9 && y <= hi + 1e-9 && !points.some(p => Math.hypot(p[0] - x, p[1] - y) < 1e-8)) points.push([x, y]);
  };
  if (w2 !== 0) {
    add(lo, (-b - w1 * lo) / w2);
    add(hi, (-b - w1 * hi) / w2);
  }
  if (w1 !== 0) {
    add((-b - w2 * lo) / w1, lo);
    add((-b - w2 * hi) / w1, hi);
  }
  return points.slice(0, 2);
}
export const xorInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
export function xorRows(bias = -1, coefficient = -1) {
  return xorInputs.map((x, index) => {
    const sum = x[0] + x[1],
      h1 = Math.max(0, sum),
      h2 = Math.max(0, sum + bias),
      second = coefficient * h2,
      target = index === 1 || index === 2 ? 1 : 0,
      q = h1 + second;
    return {
      id: x.join(''),
      x,
      sum,
      h1,
      h2,
      second,
      q,
      target,
      residual: q - target,
      matches: Math.abs(q - target) <= 1e-8
    };
  });
}
export const triangle = x => Math.max(0, x) - 2 * Math.max(0, x - 1) + Math.max(0, x - 2);
export function parseNumeric(text, min, max) {
  if (!String(text).trim()) return {
    error: 'Enter a finite number.'
  };
  const parts = String(text).trim().replace(/−/g, '-').split('/');
  const number = parts.length === 2 && parts.every(p => p.trim() !== '' && Number.isFinite(Number(p))) ? Number(parts[0]) / Number(parts[1]) : parts.length === 1 ? Number(parts[0]) : NaN;
  return !Number.isFinite(number) ? {
    error: 'Enter a finite number, such as −1 or −4/3.'
  } : number < min || number > max ? {
    error: `Use a value from ${min} to ${max}; the last valid result remains visible.`
  } : {
    value: number
  };
}
