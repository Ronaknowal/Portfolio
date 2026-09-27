// Bernoulli RBM: W[visible][hidden]. Exact hidden enumeration is deliberately
// bounded to ten hidden units; browser studies use eight. All sums use log space.
export const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
export const softplus = x => Math.max(0, x) + Math.log1p(Math.exp(-Math.abs(x)));
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const sum = xs => xs.reduce((a, b) => a + b, 0);
export const binaryStates = n => {
  if (!Number.isInteger(n) || n < 0 || n > 10) throw new RangeError('Exact enumeration supports 0–10 binary units.');
  return Array.from({
    length: 2 ** n
  }, (_, i) => Array.from({
    length: n
  }, (_, j) => i >> n - j - 1 & 1));
};
export function normalizeLogs(logs) {
  const max = Math.max(...logs),
    mass = logs.map(x => Math.exp(x - max)),
    total = sum(mass);
  return {
    probabilities: mass.map(x => x / total),
    logZ: max + Math.log(total)
  };
}
export const rbmDefault = () => ({
  w: [[Math.log(3)], [Math.log(3)]],
  a: [0, 0],
  b: [0]
});
export function hiddenProbabilities(v, model) {
  return model.b.map((bias, j) => sigmoid(bias + sum(v.map((x, i) => x * model.w[i][j]))));
}
export function visibleProbabilities(h, model) {
  return model.a.map((bias, i) => sigmoid(bias + dot(model.w[i], h)));
}
export function freeEnergy(v, model) {
  return -dot(v, model.a) - sum(model.b.map((bias, j) => softplus(bias + sum(v.map((x, i) => x * model.w[i][j])))));
}
export function hiddenEnumeration(model) {
  const states = binaryStates(model.b.length);
  const logits = states.map(h => model.a.map((bias, i) => bias + dot(model.w[i], h)));
  const logMass = states.map((h, j) => dot(h, model.b) + sum(logits[j].map(softplus)));
  const {
    probabilities,
    logZ
  } = normalizeLogs(logMass);
  return {
    states,
    logits,
    probabilities,
    logZ,
    visible: logits.map(row => row.map(sigmoid))
  };
}
export function tinyDistribution(model, offset = 0) {
  if (model.a.length !== 2 || model.b.length !== 1) throw new RangeError('The visible-state investigation uses two visible and one hidden unit.');
  const states = binaryStates(2),
    joint = [];
  const logMass = states.flatMap(v => [0, 1].map(h => {
    const energy = -dot(v, model.a) - h * model.b[0] - h * sum(v.map((x, i) => x * model.w[i][0])) + offset;
    joint.push({
      v,
      h,
      energy
    });
    return -energy;
  }));
  const {
    probabilities,
    logZ
  } = normalizeLogs(logMass);
  const p = states.map((_, i) => probabilities[2 * i] + probabilities[2 * i + 1]);
  const marginals = [0, 1].map(i => sum(states.map((v, s) => p[s] * v[i])));
  return {
    states,
    joint: joint.map((entry, i) => ({
      ...entry,
      probability: probabilities[i]
    })),
    p,
    marginals,
    covariance: p[3] - marginals[0] * marginals[1],
    logZ,
    hidden: sum(probabilities.filter((_, i) => i % 2))
  };
}
export function statistics(model, dataMass) {
  if (dataMass.length !== 4 || dataMass.some(x => !Number.isFinite(x) || x < 0) || sum(dataMass) <= 0) throw new RangeError('Use four nonnegative counts with a positive total.');
  const mass = dataMass.map(x => x / sum(dataMass)),
    distribution = tinyDistribution(model);
  const stats = p => {
    const ph = distribution.states.map(v => hiddenProbabilities(v, model)[0]);
    return [0, 1].map(i => sum(p.map((x, j) => x * distribution.states[j][i] * ph[j]))).concat([0, 1].map(i => sum(p.map((x, j) => x * distribution.states[j][i]))), sum(p.map((x, j) => x * ph[j])));
  };
  const positive = stats(mass),
    negative = stats(distribution.p);
  return {
    positive,
    negative,
    gradient: positive.map((x, i) => x - negative[i]),
    mass,
    logLikelihood: sum(mass.map((x, i) => x * Math.log(distribution.p[i])))
  };
}
export function updateTiny(model, gradient, rate) {
  return {
    w: model.w.map((row, i) => [row[0] + rate * gradient[i]]),
    a: model.a.map((x, i) => x + rate * gradient[2 + i]),
    b: [model.b[0] + rate * gradient[4]]
  };
}
export function tinyTransition(model) {
  const states = binaryStates(2);
  const conditional = [0, 1].map(h => {
    const pv = visibleProbabilities([h], model);
    return states.map(v => v.reduce((p, bit, i) => p * (bit ? pv[i] : 1 - pv[i]), 1));
  });
  return states.map(v => {
    const ph = hiddenProbabilities(v, model)[0];
    return states.map((_, i) => (1 - ph) * conditional[0][i] + ph * conditional[1][i]);
  });
}
export function probabilityFlow(model, initial, steps) {
  const transition = tinyTransition(model),
    equilibrium = tinyDistribution(model).p;
  let mass = [...initial];
  const traces = [];
  for (let step = 0; step <= steps; step++) {
    const incoming = transition.map((row, i) => row.map(x => x * mass[i]));
    traces.push({
      step,
      mass,
      incoming,
      tv: sum(mass.map((x, i) => Math.abs(x - equilibrium[i]))) / 2
    });
    mass = equilibrium.map((_, j) => sum(incoming.map(row => row[j])));
  }
  return {
    transition,
    equilibrium,
    traces
  };
}
export function tinyDraw(v, model, uniforms) {
  const ph = hiddenProbabilities(v, model)[0],
    hidden = Number(uniforms[0] < ph);
  const pv = visibleProbabilities([hidden], model),
    next = pv.map((p, i) => Number(uniforms[i + 1] < p));
  return {
    previous: v,
    ph,
    hidden,
    pv,
    next,
    uniforms
  };
}
export function seededRandom(seed) {
  // Mulberry32; an explicit JS stream, never claimed equal to NumPy's generator.
  let state = seed >>> 0;
  return () => {
    state = state + 0x6D2B79F5 | 0;
    let value = Math.imul(state ^ state >>> 15, 1 | state);
    value ^= value + Math.imul(value ^ value >>> 7, 61 | value);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}
export function particleTrace(v, model, steps, seed) {
  const rng = seededRandom(seed),
    trace = [];
  let current = v;
  for (let step = 0; step < steps; step++) {
    const next = tinyDraw(current, model, [rng(), rng(), rng()]);
    trace.push(next);
    current = next.next;
  }
  return trace;
}
export function persistenceTrace(model, minibatches, seed) {
  const rng = seededRandom(seed);
  let persistent = binaryStates(2);
  return minibatches.map((batch, step) => {
    const draws = batch.map(() => [rng(), rng(), rng()]);
    const cd = batch.map((v, i) => tinyDraw(v, model, draws[i]));
    const pcd = persistent.map((v, i) => tinyDraw(v, model, draws[i]));
    const old = persistent;
    persistent = pcd.map(row => row.next);
    return {
      step,
      batch,
      cd,
      pcd,
      old,
      draws
    };
  });
}
export function completion(v, observed, model, prepared = hiddenEnumeration(model)) {
  const logMass = prepared.states.map((h, j) => dot(h, model.b) + sum(prepared.logits[j].map((x, i) => observed[i] ? v[i] * x : softplus(x))));
  const posterior = normalizeLogs(logMass).probabilities;
  const probabilities = v.map((value, i) => observed[i] ? value : sum(posterior.map((p, j) => p * prepared.visible[j][i])));
  return {
    probabilities,
    posterior
  };
}
export function reconstruction(v, model) {
  return visibleProbabilities(hiddenProbabilities(v, model), model);
}
export function rbmMetrics(v, model, enumeration = hiddenEnumeration(model)) {
  const reconstructed = reconstruction(v, model);
  return {
    nll: freeEnergy(v, model) + enumeration.logZ,
    reconstructed,
    mse: sum(v.map((x, i) => (x - reconstructed[i]) ** 2)) / v.length
  };
}
export function exactSample(model, seed, count = 16, v = null, observed = null) {
  const enumeration = hiddenEnumeration(model),
    rng = seededRandom(seed);
  const posterior = v ? completion(v, observed, model, enumeration).posterior : enumeration.probabilities;
  return Array.from({
    length: count
  }, () => {
    const u = rng();
    let cumulative = 0,
      selected = posterior.length - 1;
    for (let i = 0; i < posterior.length; i++) {
      cumulative += posterior[i];
      if (u < cumulative) {
        selected = i;
        break;
      }
    }
    const pixels = enumeration.visible[selected].map((p, i) => v && observed[i] ? v[i] : Number(rng() < p));
    return {
      pixels,
      hidden: enumeration.states[selected]
    };
  });
}
