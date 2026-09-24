// Small deterministic calculations for the three editorial pilot lessons.
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

export function coverageIntervals(n, critical, batch = 0) {
  const random = seededRandom(407 + batch * 97);
  const se = 10 / Math.sqrt(n);
  return Array.from({ length: 40 }, () => {
    // Exact sampling distribution of a Normal(100, 10^2) sample mean.
    const z = Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
    const mean = 100 + se * z;
    return { mean, low: mean - critical * se, high: mean + critical * se };
  });
}

function logFactorial(n) {
  let sum = 0;
  for (let i = 2; i <= n; i++) sum += Math.log(i);
  return sum;
}

// Integer shapes only; pilot controls deliberately enforce this domain.
export function betaDensity(x, a, b) {
  const logScale = logFactorial(a + b - 1) - logFactorial(a - 1) - logFactorial(b - 1);
  if (x === 0) return a === 1 ? b : 0;
  if (x === 1) return b === 1 ? a : 0;
  return Math.exp(logScale + (a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x));
}

export function betaCdf(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const n = a + b - 1;
  let sum = 0;
  for (let j = a; j <= n; j++) {
    sum += Math.exp(logFactorial(n) - logFactorial(j) - logFactorial(n - j)
      + j * Math.log(x) + (n - j) * Math.log1p(-x));
  }
  return Math.min(1, Math.max(0, sum));
}

export function betaQuantile(probability, a, b) {
  let low = 0, high = 1;
  for (let i = 0; i < 48; i++) {
    const mid = (low + high) / 2;
    if (betaCdf(mid, a, b) < probability) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}

export function bridgeGraph(weight) {
  const adjacency = Array.from({ length: 6 }, () => Array(6).fill(0));
  for (const [i, j] of [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]]) {
    adjacency[i][j] = adjacency[j][i] = 1;
  }
  adjacency[2][3] = adjacency[3][2] = weight;
  const laplacian = adjacency.map((row, i) => row.map((v, j) => i === j ? row.reduce((a, b) => a + b, 0) : -v));
  return { adjacency, laplacian, ...symmetricEigen(laplacian) };
}

// Jacobi rotations for a tiny real symmetric matrix; not a general large-matrix solver.
export function symmetricEigen(matrix) {
  const n = matrix.length;
  const a = matrix.map(row => [...row]);
  const vectors = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let iteration = 0; iteration < 500; iteration++) {
    let p = 0, q = 1, max = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (Math.abs(a[i][j]) > max) { p = i; q = j; max = Math.abs(a[i][j]); }
    }
    if (max < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(angle), s = Math.sin(angle);
    const pp = a[p][p], qq = a[q][q], pq = a[p][q];
    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const kp = a[k][p], kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
      const vp = vectors[k][p], vq = vectors[k][q];
      vectors[k][p] = c * vp - s * vq;
      vectors[k][q] = s * vp + c * vq;
    }
    a[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq;
    a[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq;
    a[p][q] = a[q][p] = 0;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map(i => Math.abs(a[i][i]) < 1e-12 ? 0 : a[i][i]),
    vectors: order.map(i => {
      const column = vectors.map(row => row[i]);
      const sign = (column.find(v => Math.abs(v) > 1e-8) ?? 1) < 0 ? -1 : 1;
      return column.map(v => v * sign);
    }),
  };
}
