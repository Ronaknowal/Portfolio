const scheduleKinds = new Set(['constant', 'cosine', 'linear', 'exponential', 'step', 'one-cycle', 'restart']);
function integerInRange(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}.`);
}
function numberInRange(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) throw new RangeError(`${name} must be finite and from ${minimum} to ${maximum}.`);
}
const cosineBlend = (start, end, fraction) => end + (start - end) * (1 + Math.cos(Math.PI * fraction)) / 2;
export function buildRateSchedule({
  kind = 'cosine',
  total = 24,
  warmup = 4,
  peak = 0.2,
  minimum = 0.01,
  rise = 6,
  period = 6
} = {}) {
  if (!scheduleKinds.has(kind)) throw new RangeError('Unknown schedule kind.');
  integerInRange(total, 2, 80, 'Update budget');
  if (kind === 'cosine') integerInRange(warmup, 0, total - 1, 'Warmup updates');
  if (kind === 'one-cycle') integerInRange(rise, 2, total - 1, 'One-cycle rise updates');
  if (kind === 'restart' || kind === 'step') integerInRange(period, 2, 20, 'Period');
  numberInRange(peak, 0.001, 1, 'Peak rate');
  numberInRange(minimum, 0, peak, 'Minimum rate');
  if (kind === 'exponential' && minimum === 0) throw new RangeError('A finite positive exponential decay cannot reach zero.');
  const initial = peak / 10;
  const oneCycleMinimum = initial / 100;
  return Array.from({
    length: total
  }, (_, update) => {
    let rate;
    let momentum = null;
    let phase;
    if (kind === 'constant') {
      rate = peak;
      phase = 'constant';
    } else if (kind === 'one-cycle') {
      const ascending = update < rise;
      const fraction = ascending ? update / (rise - 1) : (update - rise + 1) / (total - rise);
      rate = ascending ? cosineBlend(initial, peak, fraction) : cosineBlend(peak, oneCycleMinimum, fraction);
      momentum = ascending ? cosineBlend(0.95, 0.85, fraction) : cosineBlend(0.85, 0.95, fraction);
      phase = ascending ? 'one-cycle rise' : 'one-cycle fall';
    } else if (kind === 'restart') {
      const position = update % period;
      rate = cosineBlend(peak, minimum, position / period);
      phase = `cycle ${Math.floor(update / period) + 1}, position ${position}`;
    } else if (kind === 'step') {
      rate = peak * 0.5 ** Math.floor(update / period);
      phase = `step interval ${Math.floor(update / period) + 1}`;
    } else if (kind === 'exponential') {
      rate = peak * (minimum / peak) ** (update / (total - 1));
      phase = 'geometric decay';
    } else if (kind === 'linear') {
      rate = peak + (minimum - peak) * update / (total - 1);
      phase = 'linear decay';
    } else if (warmup > 0 && update < warmup) {
      rate = peak * (update + 1) / warmup;
      phase = 'warmup';
    } else {
      const fraction = warmup > 0 ? (update - warmup + 1) / (total - warmup) : update / (total - 1);
      rate = cosineBlend(peak, minimum, fraction);
      phase = 'cosine decay';
    }
    return {
      update,
      rate,
      momentum,
      phase
    };
  });
}
export function scheduleNoiseMoments(schedule, {
  curvature = 2,
  noise = 1,
  initialError = 3
} = {}) {
  numberInRange(curvature, 0.25, 8, 'Curvature');
  numberInRange(noise, 0, 4, 'Gradient-noise standard deviation');
  numberInRange(initialError, -5, 5, 'Initial error');
  if (!Array.isArray(schedule) || schedule.length < 1 || schedule.length > 80) throw new RangeError('Supply 1–80 rates.');
  let mean = initialError;
  let variance = 0;
  const states = [{
    completed: 0,
    mean,
    variance,
    meanSquaredError: mean * mean,
    expectedLoss: curvature * mean * mean / 2
  }];
  for (const {
    rate
  } of schedule) {
    numberInRange(rate, 0, 1, 'Rate');
    const multiplier = 1 - rate * curvature;
    mean *= multiplier;
    variance = multiplier * multiplier * variance + rate * rate * noise * noise;
    const meanSquaredError = mean * mean + variance;
    states.push({
      completed: states.length,
      rate,
      multiplier,
      mean,
      variance,
      meanSquaredError,
      expectedLoss: curvature * meanSquaredError / 2
    });
  }
  return states;
}
export const clockTargets = [1, 3, 2, 4, 0, 2, 3, 1, 4, 2, 1, 3];
export function buildScheduleClockTrace({
  accumulation = 2,
  skipSecond = true,
  policy = 'committed'
} = {}) {
  integerInRange(accumulation, 1, 3, 'Accumulation');
  if (typeof skipSecond !== 'boolean') throw new TypeError('Skip flag must be boolean.');
  if (!['committed', 'microbatch', 'advance-first'].includes(policy)) throw new RangeError('Unknown clock policy.');
  const attempts = clockTargets.length / accumulation;
  const updates = attempts - Number(skipSecond);
  // The finite schedule has exactly one rate for every intended successful update.
  const schedule = buildRateSchedule({
    total: updates,
    warmup: 1
  });
  const rates = schedule.map(state => state.rate);
  let parameter = 0;
  let pendingGradient = 0;
  let committed = 0;
  let attempt = 0;
  const states = [{
    microbatches: 0,
    parameter,
    pendingGradient,
    committed,
    attempt,
    action: 'ready',
    rate: null,
    scheduleIndex: null
  }];
  clockTargets.forEach((target, microbatch) => {
    pendingGradient += parameter - target;
    const count = microbatch + 1;
    let action = 'accumulate';
    let rate = null;
    let scheduleIndex = null;
    let appliedGradient = null;
    if (count % accumulation === 0) {
      attempt += 1;
      appliedGradient = pendingGradient / accumulation;
      if (skipSecond && attempt === 2) {
        action = 'skip this attempt';
      } else {
        scheduleIndex = policy === 'microbatch' ? microbatch : committed + Number(policy === 'advance-first');
        rate = rates[scheduleIndex] ?? null;
        if (rate === null) action = 'budget exhausted: no update';else {
          parameter -= rate * appliedGradient;
          committed += 1;
          action = 'commit update';
        }
      }
      pendingGradient = 0;
    }
    states.push({
      microbatches: count,
      target,
      parameter,
      pendingGradient,
      committed,
      attempt,
      action,
      rate,
      scheduleIndex,
      appliedGradient
    });
  });
  return {
    accumulation,
    updates,
    rates,
    states,
    policy,
    skipSecond
  };
}
export function parseValidationMetrics(text) {
  const parts = text.trim().split(/[\s,]+/).filter(Boolean);
  if (parts.length < 1 || parts.length > 24) throw new RangeError('Enter 1–24 validation losses.');
  return parts.map(part => {
    if (!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(part)) throw new RangeError('Validation losses must be numbers from 0 to 10.');
    const value = Number(part);
    numberInRange(value, 0, 10, 'Validation loss');
    return value;
  });
}
export function buildPlateauTrace(metrics, {
  patience = 1,
  threshold = 0.02,
  cooldown = 1,
  initialRate = 0.2,
  minimumRate = 0.025
} = {}) {
  if (!Array.isArray(metrics) || metrics.length < 1 || metrics.length > 24) throw new RangeError('Supply 1–24 validation losses.');
  metrics.forEach(value => numberInRange(value, 0, 10, 'Validation loss'));
  integerInRange(patience, 0, 5, 'Patience');
  integerInRange(cooldown, 0, 4, 'Cooldown');
  numberInRange(threshold, 0, 0.5, 'Absolute improvement threshold');
  numberInRange(initialRate, 0.001, 1, 'Initial rate');
  numberInRange(minimumRate, 0, initialRate, 'Rate floor');
  let best = Infinity;
  let bad = 0;
  let cooling = 0;
  let rate = initialRate;
  return metrics.map((metric, observation) => {
    const bestBefore = Number.isFinite(best) ? best : null;
    const cutoff = bestBefore === null ? null : bestBefore - threshold;
    const improved = metric < best - threshold;
    const rateBefore = rate;
    const coolingBefore = cooling;
    if (improved) {
      best = metric;
      bad = 0;
    } else bad += 1;
    if (cooling > 0) {
      cooling -= 1;
      bad = 0;
    }
    const triggered = bad > patience;
    if (triggered) {
      const proposed = Math.max(rate * 0.5, minimumRate);
      if (rate - proposed > 1e-8) rate = proposed;
      cooling = cooldown;
      bad = 0;
    }
    return {
      observation,
      metric,
      bestBefore,
      cutoff,
      improved,
      best,
      bad,
      coolingBefore,
      cooling,
      triggered,
      reduced: rate < rateBefore,
      rateBefore,
      rate
    };
  });
}
