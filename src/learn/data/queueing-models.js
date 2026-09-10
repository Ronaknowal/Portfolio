/** Deterministic finite teaching models. Durations are seconds; rates are jobs/second. */
function numberInRange(value, name, low, high) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be a finite number from ${low} to ${high}.`);
  }
  return value;
}
function positiveRate(value, name) {
  return numberInRange(value, name, 1e-6, 1e6);
}
function integerInRange(value, name, low, high) {
  numberInRange(value, name, low, high);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer.`);
}
function assertFinite(values) {
  if (values.some(value => !Number.isFinite(value))) {
    throw new RangeError('This calculation exceeds the supported arithmetic range; rescale the inputs.');
  }
}
export const queueExampleJobs = [{
  id: 'A',
  arrival: 0,
  service: 3
}, {
  id: 'B',
  arrival: 1,
  service: 1
}, {
  id: 'C',
  arrival: 2,
  service: 2
}, {
  id: 'D',
  arrival: 6,
  service: 1
}];
export function fcfsTrace(jobs) {
  if (!Array.isArray(jobs) || jobs.length > 32) throw new RangeError('Use at most 32 jobs.');
  let available = 0;
  let previousArrival = 0;
  const seen = new Set();
  return jobs.map(job => {
    if (!job || typeof job.id !== 'string' || !job.id.trim() || seen.has(job.id)) {
      throw new RangeError('Each job needs a distinct nonempty text ID.');
    }
    seen.add(job.id);
    const arrival = numberInRange(job.arrival, 'Arrival', 0, 1e6);
    const service = numberInRange(job.service, 'Service duration', 0, 1e4);
    if (arrival < previousArrival) throw new RangeError('List jobs in arrival order; ties keep their listed order.');
    previousArrival = arrival;
    const start = Math.max(arrival, available);
    const departure = start + service;
    if (service > 0 && departure === start) {
      throw new RangeError('This positive service duration is too small at the chosen time origin; rescale the trace.');
    }
    available = departure;
    return {
      ...job,
      start,
      departure,
      wait: start - arrival,
      total: departure - arrival
    };
  });
}
export function occupancyWindow(jobs, horizon, boundary = 'system') {
  numberInRange(horizon, 'Observation horizon', 1e-6, 1e6);
  if (!['system', 'queue'].includes(boundary)) throw new RangeError('Choose system or waiting queue.');
  const trace = fcfsTrace(jobs);
  const finish = job => boundary === 'system' ? job.departure : job.start;
  const intervals = trace.map(job => ({
    ...job,
    finish: finish(job),
    clipped: Math.max(0, Math.min(horizon, finish(job)) - job.arrival),
    observed: job.arrival <= horizon,
    completedBoundary: finish(job) <= horizon
  }));
  const events = [...new Set([0, horizon, ...intervals.flatMap(job => [job.arrival, job.finish])])].filter(time => time >= 0 && time <= horizon).sort((a, b) => a - b);
  const segments = events.slice(0, -1).map((start, index) => {
    const end = events[index + 1];
    const occupants = intervals.filter(job => job.arrival <= start && job.finish > start).map(job => job.id);
    return {
      start,
      end,
      count: occupants.length,
      occupants,
      area: (end - start) * occupants.length
    };
  });
  const clippedArea = intervals.reduce((sum, job) => sum + job.clipped, 0);
  const stepArea = segments.reduce((sum, segment) => sum + segment.area, 0);
  const completed = intervals.filter(job => job.completedBoundary && job.observed);
  const completedResidence = completed.reduce((sum, job) => sum + job.finish - job.arrival, 0);
  const pendingArea = intervals.filter(job => !job.completedBoundary).reduce((sum, job) => sum + job.clipped, 0);
  const arrived = intervals.filter(job => job.observed).length;
  return {
    trace,
    intervals,
    segments,
    horizon,
    boundary,
    clippedArea,
    stepArea,
    completedResidence,
    pendingArea,
    meanOccupancy: clippedArea / horizon,
    arrived,
    completed: completed.length,
    completedMean: completed.length ? completedResidence / completed.length : null,
    completedRate: completed.length / horizon,
    enteredRate: arrived / horizon
  };
}
export function mm1State(arrivalRate, serviceRate, percentile = 0.99, visibleStates = 12) {
  positiveRate(arrivalRate, 'Arrival rate');
  positiveRate(serviceRate, 'Service rate');
  numberInRange(percentile, 'Percentile probability', 0.001, 0.9999);
  integerInRange(visibleStates, 'Visible state count', 2, 16);
  const rho = arrivalRate / serviceRate;
  const gap = serviceRate - arrivalRate;
  if (gap <= 0) return {
    stable: false,
    rho,
    arrivalRate,
    serviceRate,
    percentile
  };
  const idle = gap / serviceRate;
  const probabilities = Array.from({
    length: visibleStates
  }, (_, n) => idle * rho ** n);
  const tail = rho ** visibleStates;
  const meanTotal = 1 / gap;
  const meanWait = rho / gap;
  const meanNumber = arrivalRate / gap;
  const meanQueue = rho * meanNumber;
  const totalQuantile = -Math.log1p(-percentile) / gap;
  const waitQuantile = percentile <= idle ? 0 : Math.log1p((percentile - idle) / (1 - percentile)) / gap;
  assertFinite([idle, tail, meanTotal, meanWait, meanNumber, meanQueue, totalQuantile, waitQuantile]);
  return {
    stable: true,
    arrivalRate,
    serviceRate,
    percentile,
    rho,
    gap,
    idle,
    probabilities,
    tail,
    meanTotal,
    meanWait,
    meanNumber,
    meanQueue,
    totalQuantile,
    waitQuantile
  };
}
export function mm1Survival(arrivalRate, serviceRate, time) {
  numberInRange(time, 'Duration', 0, 1e6);
  const state = mm1State(arrivalRate, serviceRate);
  if (!state.stable) throw new RangeError('The infinite-buffer stationary tail requires arrival rate below service rate.');
  const total = Math.exp(-state.gap * time);
  return {
    total,
    queue: state.rho * total
  };
}
export const serviceMixtureNames = {
  constant: 'Every job takes 0.1 s',
  twoSize: 'Equal chances: 0.05 or 0.15 s',
  rareLong: '95% take 0.05 s; 5% take 1.05 s',
  exponential: 'Exponential service, mean 0.1 s'
};
export function serviceVariabilityState(arrivalRate = 8, kind = 'twoSize') {
  positiveRate(arrivalRate, 'Arrival rate');
  if (!Object.hasOwn(serviceMixtureNames, kind)) throw new RangeError('Choose a supported service distribution.');
  const mean = 0.1;
  const atoms = kind === 'constant' ? [{
    probability: 1,
    duration: mean
  }] : kind === 'twoSize' ? [{
    probability: 0.5,
    duration: 0.05
  }, {
    probability: 0.5,
    duration: 0.15
  }] : kind === 'rareLong' ? [{
    probability: 0.95,
    duration: 0.05
  }, {
    probability: 0.05,
    duration: 1.05
  }] : [];
  const secondMoment = kind === 'exponential' ? 2 * mean ** 2 : atoms.reduce((sum, atom) => sum + atom.probability * atom.duration ** 2, 0);
  const variance = kind === 'constant' ? 0 : secondMoment - mean ** 2;
  const rho = arrivalRate * mean;
  const busyResidual = secondMoment / (2 * mean);
  const timeResidual = arrivalRate * secondMoment / 2;
  const stable = rho < 1;
  return {
    kind,
    arrivalRate,
    mean,
    secondMoment,
    variance,
    rho,
    stable,
    atoms: atoms.map(atom => ({
      ...atom,
      busyShare: atom.probability * atom.duration / mean,
      triangleArea: atom.duration ** 2 / 2
    })),
    busyResidual,
    timeResidual,
    meanWait: stable ? timeResidual / (1 - rho) : null,
    meanTotal: stable ? mean + timeResidual / (1 - rho) : null
  };
}
export function pooledQueueState(arrivalRate, serviceRate, servers) {
  positiveRate(arrivalRate, 'Arrival rate');
  positiveRate(serviceRate, 'Per-server service rate');
  integerInRange(servers, 'Server count', 1, 12);
  const offeredLoad = arrivalRate / serviceRate;
  const gap = servers * serviceRate - arrivalRate;
  const rho = arrivalRate / (servers * serviceRate);
  if (gap <= 0) return {
    stable: false,
    rho,
    arrivalRate,
    serviceRate,
    servers
  };
  const weights = [1];
  for (let n = 1; n <= servers; n += 1) weights.push(weights[n - 1] * offeredLoad / n);
  const tailWeight = weights[servers] * servers * serviceRate / gap;
  const normalizer = weights.slice(0, servers).reduce((sum, value) => sum + value, 0) + tailWeight;
  const waitProbability = tailWeight / normalizer;
  const meanWait = waitProbability / gap;
  const meanTotal = meanWait + 1 / serviceRate;
  const splitTotal = servers / gap;
  const fastTotal = 1 / gap;
  assertFinite([normalizer, waitProbability, meanWait, meanTotal, splitTotal, fastTotal]);
  return {
    stable: true,
    rho,
    arrivalRate,
    serviceRate,
    servers,
    offeredLoad,
    gap,
    emptyProbability: 1 / normalizer,
    waitProbability,
    meanWait,
    meanTotal,
    meanQueue: arrivalRate * meanWait,
    splitTotal,
    fastTotal
  };
}
export function finiteBufferState(arrivalRate, serviceRate, capacity) {
  positiveRate(arrivalRate, 'Offered arrival rate');
  positiveRate(serviceRate, 'Service rate');
  integerInRange(capacity, 'Total system capacity', 1, 20);
  const rho = arrivalRate / serviceRate;
  // Bounded rates/capacity keep these weights representable, including rho=1.
  const weights = Array.from({
    length: capacity + 1
  }, (_, n) => rho ** n);
  const normalizer = weights.reduce((sum, value) => sum + value, 0);
  const probabilities = weights.map(value => value / normalizer);
  // Sum the relevant states rather than subtract two nearly equal numbers.
  const admittedProbability = probabilities.slice(0, -1).reduce((sum, value) => sum + value, 0);
  const busyProbability = probabilities.slice(1).reduce((sum, value) => sum + value, 0);
  const admittedRate = arrivalRate * admittedProbability;
  const meanNumber = probabilities.reduce((sum, value, n) => sum + n * value, 0);
  const meanQueue = probabilities.reduce((sum, value, n) => sum + Math.max(0, n - 1) * value, 0);
  const meanTotal = meanNumber / admittedRate;
  assertFinite([...probabilities, admittedRate, meanNumber, meanTotal]);
  return {
    arrivalRate,
    serviceRate,
    capacity,
    rho,
    probabilities,
    dropProbability: probabilities.at(-1),
    admittedProbability,
    admittedRate,
    busyProbability,
    meanNumber,
    meanQueue,
    meanTotal,
    meanWait: meanQueue / admittedRate
  };
}
export function queueNumber(value, digits = 4) {
  if (value === null) return 'not defined';
  if (value === 0 || Object.is(value, -0)) return '0';
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 1e5) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}
