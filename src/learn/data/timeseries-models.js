/* Time-series validation and forecasting baselines: the whole computational
 * surface of the lesson.
 *
 * Nothing in this file renders. Every number the page prints, every cell a
 * calendar lane shades and every split boundary a diagram draws is produced
 * here, so the verifier can assert the same values the browser paints. A drawn
 * split boundary is a mathematical claim about what was knowable when, and a
 * claim that only exists inside a component is a claim nobody checked.
 *
 * THE ONE PROPERTY THIS FILE EXISTS TO PROTECT
 *
 * Time-series validation exists because ordinary cross-validation leaks the
 * future. So every object here that represents a fit carries its own
 * information set, and `informationSetAudit` refuses any fit that touches an
 * observation dated later than its own origin. Three separate claims are kept
 * apart because they fail separately:
 *
 *   FEATURE CAUSALITY  a training row issued at origin s uses observation
 *                      indices no later than s − d, where d is the reporting
 *                      delay. Adding a delay must move the historical feature
 *                      snapshot backward, not merely postpone the label.
 *   LABEL MATURITY     a row with horizon h enters a fit performed at origin t
 *                      only when s + h + d <= t.
 *   FIT CLOSURE        the largest observation index any fit at origin t
 *                      touches -- every training row's features AND labels, and
 *                      the prediction row's own features -- is at most t.
 *
 * Closure follows from the other two, which is exactly why it is asserted
 * separately: it is the statement a reader can check against a drawing, and a
 * drawing that satisfies it cannot be teaching the opposite of the lesson.
 *
 * THREE KINDS OF QUANTITY, KEPT APART
 *
 *   EXACT CONSTRUCTED FIXTURES   the six-value operating cycle, the
 *     recursive/updated traces, the label-arrival tables. Small integers and
 *     exact arithmetic; no tolerance, and never described as measurements.
 *   RULES APPLIED TO REAL COUNTS  the baselines replayed on the served bike
 *     series. Arithmetic here is the lesson's own, performed in the browser.
 *   RECORDED MEASUREMENTS        the six-candidate development comparison and
 *     the locked final assessment, including every ridge fit. Those live in
 *     `timeseries-data.js`, regenerated from the frozen packet and the served
 *     dataset by scripts/verify-timeseries-data.py. No model is fitted in the
 *     browser, and this file never invents one.
 */

/* =============================================================== guard rails */

/** A refusal, not a substitute. Returning a default for a bad argument is how a
 *  wrong forecast gets drawn with full confidence. */
function demand(condition, message) {
  if (!condition) throw new RangeError(message);
}

export function checkFinite(value, label) {
  demand(typeof value === 'number' && Number.isFinite(value), `${label} must be a finite number, got ${value}`);
  return value;
}

function checkCounts(values, label, { minimum = 1 } = {}) {
  demand(Array.isArray(values) && values.length >= minimum,
    `${label} needs at least ${minimum} value${minimum === 1 ? '' : 's'}, got ${Array.isArray(values) ? values.length : values}`);
  values.forEach((value, index) => checkFinite(value, `${label}[${index}]`));
  return values;
}

function checkInteger(value, label, low, high) {
  demand(Number.isInteger(value) && value >= low && value <= high,
    `${label} must be an integer between ${low} and ${high}, got ${value}`);
  return value;
}

/* ==================================================== dates without a library
 *
 * Every date on this page is a calendar day, never a moment. All arithmetic is
 * done in UTC on whole days, so no local timezone can shift a forecast origin
 * across midnight and silently change which observations were available.
 */

const MILLISECONDS_PER_DAY = 86400000;
const WEEKDAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export function parseDay(text) {
  demand(typeof text === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(text), `a calendar day must be YYYY-MM-DD, got ${text}`);
  const value = Date.parse(`${text}T00:00:00Z`);
  demand(Number.isFinite(value), `${text} is not a real calendar day`);
  return value;
}

export function dayText(milliseconds) {
  checkFinite(milliseconds, 'a day');
  return new Date(milliseconds).toISOString().slice(0, 10);
}

/** Whole days from one calendar day to another. Negative when `to` is earlier. */
export function daysBetween(from, to) {
  const span = (parseDay(to) - parseDay(from)) / MILLISECONDS_PER_DAY;
  demand(Number.isInteger(span), `${from} to ${to} is not a whole number of days`);
  return span;
}

export function shiftDay(text, days) {
  checkInteger(days, 'a day offset', -100000, 100000);
  return dayText(parseDay(text) + days * MILLISECONDS_PER_DAY);
}

export function weekdayName(text) {
  return WEEKDAY_NAMES[new Date(parseDay(text)).getUTCDay()];
}

export function weekdayIndex(text) {
  return new Date(parseDay(text)).getUTCDay();
}

/** A short form for a compact label: "Sat 31 Dec". Never used where the full
 *  date is the claim; the exact date is always available in text nearby. */
export function shortDay(text) {
  const date = new Date(parseDay(text));
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${WEEKDAY_NAMES[date.getUTCDay()].slice(0, 3)} ${date.getUTCDate()} ${months[date.getUTCMonth()]}`;
}

/** The calendar a daily series claims to be. Returns the gaps rather than
 *  throwing, because "this CSV is not a daily calendar" is a result the lesson
 *  teaches, not an error it hides. */
export function calendarGaps(dates) {
  demand(Array.isArray(dates) && dates.length >= 1, 'a calendar needs at least one day');
  const gaps = [];
  for (let index = 1; index < dates.length; index += 1) {
    const span = daysBetween(dates[index - 1], dates[index]);
    if (span !== 1) gaps.push({ afterIndex: index - 1, from: dates[index - 1], to: dates[index], days: span });
  }
  return gaps;
}

/* ======================================================= §2 · the four rules
 *
 * Four complete prediction rules on one history, each carrying the identity of
 * the observation it copies forward. The donor index is part of the answer: a
 * seasonal-naive forecast whose donor nobody can name is a number, not a rule.
 */

/** The historical index a seasonal-naive forecast copies at horizon h.
 *
 * Zero-based over a history of length T with season length m:
 *     T − m + ((h − 1) mod m)
 * which stays inside the final observed cycle for every horizon, including
 * horizons past one full cycle. It repeats OBSERVED HISTORY; it never reads a
 * future outcome to extend itself. */
export function seasonalDonorIndex({ historyLength, period, horizon }) {
  checkInteger(historyLength, 'the history length', 1, 100000);
  checkInteger(period, 'the season length', 1, historyLength);
  checkInteger(horizon, 'the horizon', 1, 10000);
  const index = historyLength - period + ((horizon - 1) % period);
  demand(index >= 0 && index < historyLength, `the seasonal donor index ${index} fell outside the history`);
  return index;
}

/**
 * All four baselines from one origin, with their donors.
 *
 * `history` is every observation available at the origin, in calendar order;
 * its last entry is the origin's own count. Counts are non-negative, so the
 * drift rule -- the only one that can extrapolate below zero -- is clipped at
 * zero. That clip is part of the declared rule, not a display convenience, and
 * `driftClipped` records when it bound.
 */
export function baselineForecasts({ history, horizon = 7, period = 7 }) {
  checkCounts(history, 'the history', { minimum: 2 });
  checkInteger(horizon, 'the horizon', 1, 400);
  checkInteger(period, 'the season length', 1, history.length);
  const length = history.length;
  const last = history[length - 1];
  const first = history[0];
  const total = history.reduce((sum, value) => sum + value, 0);
  const mean = total / length;
  const slope = (last - first) / (length - 1);
  const leads = Array.from({ length: horizon }, (_unused, index) => index + 1);
  const seasonalDonors = leads.map(lead => seasonalDonorIndex({ historyLength: length, period, horizon: lead }));
  const driftRaw = leads.map(lead => last + lead * slope);
  return {
    horizon,
    period,
    historyLength: length,
    mean: leads.map(() => mean),
    naive: leads.map(() => last),
    seasonal: seasonalDonors.map(index => history[index]),
    drift: driftRaw.map(value => Math.max(0, value)),
    driftRaw,
    driftClipped: driftRaw.map(value => value < 0),
    seasonalDonors,
    /* The drift slope is the average change per interval across the whole
       history: six observations span five intervals, so the denominator is
       T − 1 and not T. Naming it here keeps the figure and the prose reading
       the same number. */
    driftSlope: slope,
    meanLevel: mean,
    lastLevel: last,
    firstLevel: first,
  };
}

export const BASELINE_RULES = ['mean', 'naive', 'seasonal', 'drift'];

/** The generic name of each rule, independent of any season length. */
export const RULE_LABELS = {
  mean: 'Historical mean',
  naive: 'Naive',
  seasonal: 'Seasonal naive',
  drift: 'Drift',
};

/** The six candidates of the real experiment, named as the manuscript names
 *  them. Held here rather than in the generated data module so that the module
 *  stays numeric and the verifier can assert the two agree key for key. */
export const METHOD_LABELS = {
  mean: 'Historical mean',
  naive: 'Naive',
  seasonal: 'Seasonal naive, 7 days',
  drift: 'Drift',
  ridge_expanding: 'Direct ridge, expanding',
  ridge_90: 'Direct ridge, 90 rows',
};

/** What each rule carries forward, in one phrase. Used by the figures and the
 *  labs so the two cannot drift apart. */
export const BASELINE_DESCRIPTIONS = {
  mean: 'the average of every observation in the history',
  naive: 'the most recent observed value',
  seasonal: 'the value one season earlier in the observed history',
  drift: 'the last level plus the average change per day, clipped at zero',
};

/* ==================================================== §5 · error summaries
 *
 * Scored horizons only. A horizon with no supplied outcome is UNSCORED, not
 * zero, and the denominator is reported beside every mean so a shrinking
 * sample cannot masquerade as an improving score.
 */

export function errorSummary(actual, predicted) {
  demand(Array.isArray(actual) && Array.isArray(predicted),
    'an error summary needs an outcome list and a forecast list');
  demand(actual.length === predicted.length,
    `an error summary needs one outcome slot per forecast, got ${actual.length} and ${predicted.length}`);
  const rows = predicted.map((value, index) => {
    checkFinite(value, `forecast ${index + 1}`);
    const outcome = actual[index];
    const scored = outcome !== null && outcome !== undefined && Number.isFinite(outcome);
    return {
      horizon: index + 1,
      predicted: value,
      actual: scored ? outcome : null,
      scored,
      error: scored ? outcome - value : null,
      absolute: scored ? Math.abs(outcome - value) : null,
    };
  });
  const scored = rows.filter(row => row.scored);
  const denominator = scored.length;
  const absoluteTotal = scored.reduce((sum, row) => sum + row.absolute, 0);
  const squaredTotal = scored.reduce((sum, row) => sum + row.error * row.error, 0);
  const signedTotal = scored.reduce((sum, row) => sum + row.error, 0);
  return {
    rows,
    denominator,
    /* No scored horizon means no mean. A zero here would be a score nobody
       earned, and it is the difference between "this rule was perfect" and
       "this rule was never tested". */
    mae: denominator ? absoluteTotal / denominator : null,
    rmse: denominator ? Math.sqrt(squaredTotal / denominator) : null,
    bias: denominator ? signedTotal / denominator : null,
    absoluteTotal,
  };
}

/** Pooling absolute errors across origins equals averaging equal-sized
 *  per-horizon MAEs; pooling squared errors does NOT equal averaging
 *  per-horizon RMSEs. Both routes are exposed so the lesson can say so and the
 *  verifier can check it. */
export function pooledMae(matrix) {
  demand(Array.isArray(matrix) && matrix.length > 0, 'pooling needs at least one row of absolute errors');
  const widths = new Set(matrix.map(row => row.length));
  demand(widths.size === 1, 'pooling needs every row to carry the same horizons');
  const values = matrix.flat();
  values.forEach((value, index) => checkFinite(value, `absolute error ${index}`));
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function maeByHorizon(matrix) {
  demand(Array.isArray(matrix) && matrix.length > 0, 'a horizon profile needs at least one origin');
  const widths = new Set(matrix.map(row => row.length));
  demand(widths.size === 1, 'a horizon profile needs every origin to carry the same horizons');
  const horizons = matrix[0].length;
  return Array.from({ length: horizons }, (_unused, column) =>
    matrix.reduce((sum, row) => sum + row[column], 0) / matrix.length);
}

/** Mean absolute scaled error's denominator, from TRAINING history alone.
 *  Zero scale leaves the ratio undefined; this returns null rather than
 *  substituting a tiny denominator, which would silently define a different
 *  metric. */
export function naiveErrorScale(history, seasonalPeriod = 1) {
  checkCounts(history, 'the training history', { minimum: 2 });
  checkInteger(seasonalPeriod, 'the seasonal period', 1, history.length - 1);
  const differences = [];
  for (let index = seasonalPeriod; index < history.length; index += 1) {
    differences.push(Math.abs(history[index] - history[index - seasonalPeriod]));
  }
  const total = differences.reduce((sum, value) => sum + value, 0);
  const scale = total / differences.length;
  return { scale: scale === 0 ? null : scale, rawScale: scale, terms: differences.length, differences };
}

/* ========================= §3 · what was knowable when: the information sets
 *
 * A training row is named by its ORIGIN s, the moment its forecast is issued.
 * Its features are observations at or before s − d; its label is the outcome at
 * s + h, which becomes available at s + h + d. The reporting delay d moves BOTH
 * boundaries: postponing only the label, while still reading y_s as a feature,
 * is the same leak in a different costume.
 */

/** The three latest count days available to a row issued at s under delay d.
 *  The compact lab's window; the real experiment's window is wider and is built
 *  by `directRidgeRow`. */
export function featureWindow({ origin, delay, span = 3 }) {
  checkInteger(origin, 'the row origin', -1000, 100000);
  checkInteger(delay, 'the reporting delay', 0, 60);
  checkInteger(span, 'the feature span', 1, 60);
  const latest = origin - delay;
  return Array.from({ length: span }, (_unused, index) => latest - (span - 1 - index));
}

/**
 * Which offered origins may enter a fit performed at `cutoff`.
 *
 * The rule is one inequality: s + h + d <= cutoff. Everything else the lab
 * shows -- the feature interval, its arrival days, the target day, the label's
 * arrival day -- is reported so that a learner can see WHY, not so that a
 * second rule can quietly decide anything.
 */
export function eligibilityRows({ origins, cutoff, horizon, delay, featureSpan = 3 }) {
  demand(Array.isArray(origins) && origins.length > 0, 'the eligibility table needs at least one offered origin');
  checkInteger(cutoff, 'the cutoff', 0, 100000);
  checkInteger(horizon, 'the horizon', 1, 400);
  checkInteger(delay, 'the reporting delay', 0, 60);
  return origins.map(origin => {
    checkInteger(origin, 'an offered origin', -1000, 100000);
    const features = featureWindow({ origin, delay, span: featureSpan });
    const targetDay = origin + horizon;
    const labelArrival = targetDay + delay;
    return {
      origin,
      features,
      latestFeatureDay: features[features.length - 1],
      featureArrivals: features.map(day => day + delay),
      targetDay,
      labelArrival,
      eligible: labelArrival <= cutoff,
      /* Reported, not used as a second gate. The lesson's point is that this
         follows from the label inequality for positive h and d, rather than
         being an extra unexplained exclusion. */
      originBeforeCutoff: origin < cutoff,
    };
  });
}

export function eligibleOrigins(request) {
  return eligibilityRows(request).filter(row => row.eligible).map(row => row.origin);
}

/** The splitter convention of §3, stated as arithmetic rather than copied as a
 *  constant. With the first test origin at t and training ending at t − g − 1,
 *  enforcing s + h + d <= t needs g >= h + d − 1. The minus one is the
 *  splitter's indexing convention and nothing else. */
export function splitterGap({ horizon, delay }) {
  checkInteger(horizon, 'the horizon', 1, 400);
  checkInteger(delay, 'the reporting delay', 0, 60);
  const minimumGap = horizon + delay - 1;
  return {
    horizon,
    delay,
    minimumGap,
    lastTrainOffset: minimumGap + 1,
    /* At the minimum gap the last training row's label arrives exactly at t.
       Exactly at, not before: the boundary is inclusive, and a strict reading
       would discard a legitimate row. */
    labelArrivalAtLastTrainRow: 0,
  };
}

/**
 * The audit this whole file exists for.
 *
 * `fit` declares its origin and every observation index it touches, split into
 * the training rows' features, the training rows' labels and the prediction
 * row's own features. The audit returns the violations rather than throwing, so
 * a figure can DRAW an invalid claim under a label that says it is invalid --
 * which figure 3's third lane deliberately does -- while every legitimate
 * construction is asserted to produce none.
 */
export function informationSetAudit(fit) {
  demand(fit && Number.isInteger(fit.origin), 'an information-set audit needs an integer origin');
  const delay = fit.delay ?? 0;
  checkInteger(delay, 'the reporting delay', 0, 60);
  const violations = [];
  const rows = fit.trainingRows ?? [];
  let maxTouched = -Infinity;
  const touch = index => { if (index > maxTouched) maxTouched = index; };

  rows.forEach(row => {
    demand(Number.isInteger(row.origin), 'every training row needs an integer origin');
    const features = row.features ?? [];
    features.forEach(index => {
      touch(index);
      // FEATURE CAUSALITY, against the row's OWN origin.
      if (index > row.origin - delay) {
        violations.push({
          kind: 'feature after its own origin',
          rowOrigin: row.origin,
          observation: index,
          latestAllowed: row.origin - delay,
        });
      }
    });
    if (row.label !== undefined && row.label !== null) {
      touch(row.label);
      // LABEL MATURITY, against the FIT's origin.
      if (row.label + delay > fit.origin) {
        violations.push({
          kind: 'label not yet arrived at the fit origin',
          rowOrigin: row.origin,
          observation: row.label,
          arrivesAt: row.label + delay,
          fitOrigin: fit.origin,
        });
      }
    }
  });

  (fit.predictionFeatures ?? []).forEach(index => {
    touch(index);
    if (index > fit.origin - delay) {
      violations.push({
        kind: 'prediction feature after the fit origin',
        observation: index,
        latestAllowed: fit.origin - delay,
      });
    }
  });

  const observationsTouched = rows.length || (fit.predictionFeatures ?? []).length;
  // FIT CLOSURE. Asserted in its own right because it is the claim a reader can
  // check against a drawing.
  if (observationsTouched > 0 && maxTouched > fit.origin) {
    violations.push({ kind: 'fit touches an observation later than its origin', observation: maxTouched, fitOrigin: fit.origin });
  }
  return {
    origin: fit.origin,
    delay,
    trainingRows: rows.length,
    maxObservationIndex: observationsTouched > 0 ? maxTouched : null,
    violations,
    clean: violations.length === 0,
  };
}

/* ============================================= §3 · recursive versus updated
 *
 * One deliberately simple one-step rule, next = last + step. Three lanes: a
 * fixed-origin recursive forecast that feeds its own predictions forward; a
 * legitimate updated one-day forecast that advances its cutoff and consumes a
 * new observation each evening; and the same numbers as the second lane with
 * the FIRST lane's issue label attached, which is the protocol error.
 */
/**
 * The drawn style each arrow must carry, keyed by the input it actually
 * consumes.
 *
 * Figure 3 styled arrows by LANE, so lane 1's first arrow — whose input is the
 * origin's own observed count — was drawn "input is this chain's own
 * prediction", a claim the chain cannot make about a prediction it had not yet
 * produced. `inputKind` was computed correctly per row all along and nothing
 * read it. The style is a claim about the arrow, so the arrow's own kind
 * decides it, and the first arrow of every lane gets its own style rather than
 * borrowing whichever one the rest of the lane uses.
 */
export const ARROW_STYLE_BY_INPUT_KIND = {
  observation: 'origin-observation',
  prediction: 'prediction-fed',
  'newly observed outcome': 'observation-fed',
};

export function arrowStyleFor(inputKind) {
  const style = ARROW_STYLE_BY_INPUT_KIND[inputKind];
  demand(style !== undefined, `no drawn style is defined for the input kind ${inputKind}`);
  return style;
}

export function recursiveTrace({ lastObserved, step, horizon, origin = 0 }) {
  checkFinite(lastObserved, 'the last observed count');
  checkFinite(step, 'the one-step increment');
  checkInteger(horizon, 'the horizon', 1, 40);
  const predictions = [];
  let input = lastObserved;
  const rows = [];
  for (let lead = 1; lead <= horizon; lead += 1) {
    const value = input + step;
    rows.push({ lead, inputValue: input, inputIndex: origin, inputKind: lead === 1 ? 'observation' : 'prediction', predicted: value });
    predictions.push(value);
    input = value;
  }
  return {
    kind: 'fixed-origin recursive',
    origin,
    predictions,
    rows,
    /* Only the origin's own observation ever enters. Every later input is this
       lane's own output, which is exactly why it is a legitimate multi-step
       forecast issued once. */
    audit: informationSetAudit({ origin, predictionFeatures: [origin] }),
  };
}

export function updatedTrace({ lastObserved, step, outcomes, origin = 0 }) {
  checkFinite(lastObserved, 'the last observed count');
  checkFinite(step, 'the one-step increment');
  checkCounts(outcomes, 'the revealed outcomes', { minimum: 1 });
  const predictions = [];
  const rows = [];
  let input = lastObserved;
  let inputIndex = origin;
  for (let lead = 1; lead <= outcomes.length; lead += 1) {
    const value = input + step;
    rows.push({
      lead,
      issuedAt: origin + lead - 1,
      inputValue: input,
      inputIndex,
      inputKind: lead === 1 ? 'observation' : 'newly observed outcome',
      predicted: value,
    });
    predictions.push(value);
    input = outcomes[lead - 1];
    inputIndex = origin + lead;
  }
  return {
    kind: 'advancing-origin updated',
    origin,
    predictions,
    rows,
    /* Each prediction is issued from its own advancing cutoff, so each one's
       information set is clean AT ITS OWN ORIGIN. The audit is therefore run
       per row, not once against the first origin. */
    audits: rows.map(row => informationSetAudit({ origin: row.issuedAt, predictionFeatures: [row.inputIndex] })),
  };
}

/** The same numbers as `updatedTrace`, claimed as if issued at the first
 *  origin. This is the lane figure 3 marks invalid, and the audit is what makes
 *  "invalid" a computed verdict rather than a caption. */
export function mislabelledTrace({ lastObserved, step, outcomes, origin = 0 }) {
  const updated = updatedTrace({ lastObserved, step, outcomes, origin });
  const inputs = updated.rows.map(row => row.inputIndex);
  return {
    kind: 'advancing-origin numbers, fixed-origin claim',
    origin,
    predictions: updated.predictions,
    rows: updated.rows.map(row => ({ ...row, claimedOrigin: origin, illegalInput: row.inputIndex > origin })),
    audit: informationSetAudit({ origin, predictionFeatures: inputs }),
  };
}

/* ================================================= §4 · the rolling schedule
 *
 * The real experiment's shape, as a data structure rather than a picture.
 * Every horizon has its OWN eligible maximum origin t − h − d and its own fit,
 * so a single overview that pretends horizon 1 and horizon 7 share a training
 * matrix would be drawing something the code does not do.
 */

/** The direct-model row for one training origin at one horizon: the fourteen
 *  inputs of §5, reduced here to the observation indices they read. The
 *  calendar features are deterministic and read no observation at all, which is
 *  why next Tuesday's weekday is allowed and next Tuesday's weather is not. */
export function directRidgeRow({ origin, horizon, delay = 0 }) {
  checkInteger(origin, 'a training origin', 0, 100000);
  checkInteger(horizon, 'the horizon', 1, 400);
  checkInteger(delay, 'the reporting delay', 0, 60);
  const latest = origin - delay;
  const lagIndices = [latest, latest - 1, latest - 6];
  const windowIndices = Array.from({ length: 7 }, (_unused, index) => latest - 6 + index);
  const features = [...new Set([...lagIndices, ...windowIndices])].sort((left, right) => left - right);
  return {
    origin,
    horizon,
    delay,
    lagIndices,
    windowIndices,
    features,
    calendarFeatures: ['target weekday indicators', 'sine and cosine of target day-of-year', 'elapsed calendar years'],
    label: origin + horizon,
    labelArrival: origin + horizon + delay,
  };
}

/**
 * One horizon's fit at one issue origin.
 *
 * Eligible training origins run from `firstTrainOrigin` to t − h − d inclusive.
 * A sliding window keeps the latest `window` of them; an expanding window keeps
 * all of them. The scaler is fitted on exactly these rows -- that is what
 * "every horizon fit learns its own scaler" means, and the returned row list is
 * what the verifier audits.
 */
export function horizonFit({ issueOrigin, horizon, firstTrainOrigin, delay = 0, window = null }) {
  checkInteger(issueOrigin, 'the issue origin', 0, 100000);
  checkInteger(horizon, 'the horizon', 1, 400);
  checkInteger(firstTrainOrigin, 'the first training origin', 0, 100000);
  checkInteger(delay, 'the reporting delay', 0, 60);
  demand(window === null || (Number.isInteger(window) && window >= 1),
    `a sliding window must be a positive integer or null, got ${window}`);
  const lastTrainOrigin = issueOrigin - horizon - delay;
  const all = [];
  for (let origin = firstTrainOrigin; origin <= lastTrainOrigin; origin += 1) all.push(origin);
  const kept = window === null ? all : all.slice(Math.max(0, all.length - window));
  const rows = kept.map(origin => directRidgeRow({ origin, horizon, delay }));
  const predictionRow = directRidgeRow({ origin: issueOrigin, horizon, delay });
  return {
    issueOrigin,
    horizon,
    delay,
    window,
    mode: window === null ? 'expanding' : 'sliding',
    firstTrainOrigin,
    lastTrainOrigin,
    eligibleCount: all.length,
    trainCount: rows.length,
    trainOrigins: kept,
    trainStart: kept.length ? kept[0] : null,
    trainEnd: kept.length ? kept[kept.length - 1] : null,
    rows,
    targetIndex: predictionRow.label,
    predictionFeatures: predictionRow.features,
    /* The scaler's own input set, named separately: fitting it on anything
       wider than these rows would be the classic preprocessing leak, and a
       reader should be able to point at the boundary. */
    scalerFittedOn: kept,
    audit: informationSetAudit({
      origin: issueOrigin,
      delay,
      trainingRows: rows.map(row => ({ origin: row.origin, features: row.features, label: row.label })),
      predictionFeatures: predictionRow.features,
    }),
  };
}

/** The whole rehearsal: several issue origins, every horizon at each. */
export function rollingSchedule({ issueOrigins, horizons = 7, firstTrainOrigin, delay = 0, window = null }) {
  demand(Array.isArray(issueOrigins) && issueOrigins.length > 0, 'a rolling schedule needs at least one issue origin');
  checkInteger(horizons, 'the horizon count', 1, 60);
  const origins = issueOrigins.map(issueOrigin => ({
    issueOrigin,
    fits: Array.from({ length: horizons }, (_unused, index) =>
      horizonFit({ issueOrigin, horizon: index + 1, firstTrainOrigin, delay, window })),
  }));
  const violations = origins.flatMap(entry => entry.fits.flatMap(fit => fit.audit.violations));
  return {
    issueOrigins,
    horizons,
    firstTrainOrigin,
    delay,
    window,
    mode: window === null ? 'expanding' : 'sliding',
    origins,
    fitCount: origins.length * horizons,
    violations,
    clean: violations.length === 0,
    /* Recorded per origin so the staircase can state, for each rehearsal, the
       largest observation index any of its seven fits touched. The whole claim
       of the figure is that this equals the issue origin and never exceeds it. */
    maxObservationByOrigin: origins.map(entry => ({
      issueOrigin: entry.issueOrigin,
      maxObservationIndex: entry.fits.reduce((largest, fit) =>
        Math.max(largest, fit.audit.maxObservationIndex ?? -Infinity), -Infinity),
    })),
  };
}

/* ============================================ §5 · replaying the real series */

/** Every issue origin of the declared contract, derived from the series length
 *  rather than copied: `np.arange(364, len(counts) - 7, 7)`. */
export function contractOrigins(seriesLength, { firstOrigin = 364, step = 7, horizons = 7 } = {}) {
  checkInteger(seriesLength, 'the series length', 1, 100000);
  const origins = [];
  for (let origin = firstOrigin; origin < seriesLength - horizons; origin += step) origins.push(origin);
  return origins;
}

/**
 * One forecast request on the real series: the baselines at one origin, the
 * outcomes that follow, and the errors once those outcomes are revealed.
 *
 * `counts` and `dates` are the whole served series. The history handed to the
 * rules is `counts[0 .. origin]` inclusive -- the origin's own count is
 * available because the declared replay assumption is that counts arrive at the
 * end of their recorded day. Nothing at index > origin is read to build a
 * forecast; the outcomes are attached separately and only for scoring.
 */
export function forecastRequest({ counts, dates, origin, period = 7, horizon = 7 }) {
  checkCounts(counts, 'the count series', { minimum: 2 });
  demand(Array.isArray(dates) && dates.length === counts.length,
    `the calendar has ${Array.isArray(dates) ? dates.length : 'no'} days for ${counts.length} counts`);
  checkInteger(origin, 'the forecast origin', 1, counts.length - 1);
  checkInteger(horizon, 'the horizon', 1, 400);
  const history = counts.slice(0, origin + 1);
  checkInteger(period, 'the season length', 1, history.length);
  const rules = baselineForecasts({ history, horizon, period });
  const targetIndices = Array.from({ length: horizon }, (_unused, index) => origin + 1 + index);
  const available = targetIndices.every(index => index < counts.length);
  const actual = targetIndices.map(index => (index < counts.length ? counts[index] : null));
  const donors = rules.seasonalDonors.map(index => ({
    index,
    date: dates[index],
    weekday: weekdayName(dates[index]),
    value: history[index],
  }));
  const summaries = Object.fromEntries(BASELINE_RULES.map(rule => [rule, errorSummary(actual, rules[rule])]));
  return {
    origin,
    originDate: dates[origin],
    originWeekday: weekdayName(dates[origin]),
    period,
    horizon,
    historyLength: history.length,
    targetIndices,
    targetDates: targetIndices.map(index => dates[index] ?? null),
    targetWeekdays: targetIndices.map(index => (dates[index] ? weekdayName(dates[index]) : null)),
    actual,
    outcomesAvailable: available,
    forecasts: Object.fromEntries(BASELINE_RULES.map(rule => [rule, rules[rule]])),
    donors,
    summaries,
    rules,
    /* The request reads counts[0..origin] only. Stated as an audit so that the
       claim is checked rather than asserted in a caption. */
    audit: informationSetAudit({ origin, predictionFeatures: [origin] }),
  };
}

/** Horizon 7 under a seven-day season copies the origin's own count, which is
 *  exactly what the naive rule repeats. The equality in the final table is a
 *  consequence of the rules, not a coincidence -- so it is computed here rather
 *  than asserted in prose. */
export function seasonalNaiveCoincidence({ historyLength, period, horizon }) {
  const donor = seasonalDonorIndex({ historyLength, period, horizon });
  return { donor, lastIndex: historyLength - 1, identical: donor === historyLength - 1 };
}

/* ======================================================= drawing primitives */

/** A linear map from a data domain to a drawing range. Every x and y this
 *  lesson paints goes through one of these, so a figure and its verifier read
 *  the same positions. */
export function linearScale({ domain, range }) {
  const [low, high] = domain;
  const [start, end] = range;
  [low, high, start, end].forEach((value, index) => checkFinite(value, `scale bound ${index}`));
  demand(high !== low, 'a scale needs a domain with width');
  const slope = (end - start) / (high - low);
  const map = value => start + (checkFinite(value, 'a scaled value') - low) * slope;
  map.domain = [low, high];
  map.range = [start, end];
  map.invert = position => low + (checkFinite(position, 'a position') - start) / slope;
  return map;
}

/** Ticks that land on round numbers inside the domain, including both ends.
 *  Computed rather than typed, because a tick label is a claim about the axis
 *  it sits on. */
export function axisTicks({ low, high, count = 5 }) {
  checkFinite(low, 'the axis minimum');
  checkFinite(high, 'the axis maximum');
  demand(high > low, `an axis needs high above low, got ${low} to ${high}`);
  checkInteger(count, 'the tick count', 2, 20);
  const rawStep = (high - low) / (count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 2.5, 5, 10].map(factor => factor * magnitude)
    .find(candidate => candidate >= rawStep) ?? 10 * magnitude;
  const ticks = [];
  for (let value = Math.ceil(low / step) * step; value <= high + step * 1e-9; value += step) {
    ticks.push(Number(value.toPrecision(12)));
  }
  if (!ticks.length || ticks[0] > low) ticks.unshift(Number(low.toPrecision(12)));
  return ticks;
}

/* ======================================== F1 · one target, two issue dates */

/**
 * Two fourteen-day calendar lanes ending on the same Saturday target.
 *
 * The dates are the real ones from the served series' first development week,
 * so the figure and section 5 are talking about the same calendar. Only the
 * shaded region differs between lanes: the target square is identical, which is
 * the entire point.
 */
export const ISSUE_TIME_FIXTURE = {
  firstDay: '2011-12-25',
  days: 14,
  targetDate: '2012-01-07',
  lanes: [
    { id: 'seven', originDate: '2011-12-31', label: 'issued Saturday evening' },
    { id: 'one', originDate: '2012-01-06', label: 'issued Friday evening' },
  ],
  arrival: { eventDate: '2012-01-04', arrivalDate: '2012-01-06', decisionDate: '2012-01-05' },
};

export function issueTimeGeometry({ fixture = ISSUE_TIME_FIXTURE, width = 300, inset = 14, cellGap = 1.4 } = {}) {
  demand(fixture.days >= 2, 'a calendar lane needs at least two days');
  const cellWidth = (width - 2 * inset) / fixture.days;
  const dates = Array.from({ length: fixture.days }, (_unused, index) => shiftDay(fixture.firstDay, index));
  const targetIndex = dates.indexOf(fixture.targetDate);
  demand(targetIndex >= 0, `the target ${fixture.targetDate} is not inside the drawn lane`);
  /* Offsets are measured from each lane's own top, and every row that carries
     text gets its own band. Stacking the weekday letters flush under the cells
     and the horizon label flush under the arrow put three rows of 9px text
     inside 18 units, which reads as one smear at rendered size. The long
     `y[target | origin]` annotation is NOT in here: it is 26 characters, and at
     this scale it would span most of the lane. It reflows in HTML beneath. */
  const rows = {
    top: 16,
    laneGap: 56,
    cellHeight: 16,
    letterBaseline: 25,
    arrowOffset: 33,
    /* Fifteen units below the arrow, not eleven. The horizon-1 arrow spans a
       SINGLE cell while its label is several cells wide, so the label is
       necessarily centred under the arrowhead; the browser curve sampler found
       the arrowhead polygon crossing the label's box at 1366 px. Horizontal
       separation is impossible for a one-cell arrow, so the clearance has to be
       vertical, and the label text was shortened as well. */
    arrowLabelBaseline: 48,
  };
  const lanes = fixture.lanes.map((lane, order) => {
    const originIndex = dates.indexOf(lane.originDate);
    demand(originIndex >= 0, `the origin ${lane.originDate} is not inside the drawn lane`);
    demand(originIndex < targetIndex, `${lane.originDate} is not before the target ${fixture.targetDate}`);
    const horizon = daysBetween(lane.originDate, fixture.targetDate);
    const y = rows.top + order * rows.laneGap;
    return {
      ...lane,
      order,
      y,
      originIndex,
      horizon,
      originWeekday: weekdayName(lane.originDate),
      /* The shaded region is the information available at this lane's issue
         time: every day up to and including the origin. Its right edge IS the
         split boundary, so it is computed, never nudged for looks. */
      known: { fromIndex: 0, toIndex: originIndex, x: inset, width: (originIndex + 1) * cellWidth },
      letterBaseline: y + rows.letterBaseline,
      arrow: {
        x1: inset + (originIndex + 0.5) * cellWidth,
        x2: inset + (targetIndex + 0.5) * cellWidth,
        y: y + rows.arrowOffset,
        labelX: inset + ((originIndex + targetIndex) / 2 + 0.5) * cellWidth,
        labelBaseline: y + rows.arrowLabelBaseline,
      },
      annotation: `y[${fixture.targetDate} | ${lane.originDate}]`,
      notation: `\\hat y_{t+${horizon}\\mid t}`,
    };
  });
  const horizons = lanes.map(lane => lane.horizon);
  demand(new Set(horizons).size === lanes.length, 'the two lanes must differ in horizon, or the figure shows nothing');
  return {
    width,
    inset,
    cellWidth,
    cellGap,
    height: rows.top + (fixture.lanes.length - 1) * rows.laneGap + rows.arrowLabelBaseline + 10,
    rows,
    dates,
    days: dates.map((date, index) => ({
      index,
      date,
      weekday: weekdayName(date),
      short: shortDay(date),
      x: inset + index * cellWidth,
      isTarget: index === targetIndex,
    })),
    targetIndex,
    targetDate: fixture.targetDate,
    targetWeekday: weekdayName(fixture.targetDate),
    lanes,
    arrival: {
      ...fixture.arrival,
      eventIndex: dates.indexOf(fixture.arrival.eventDate),
      arrivalIndex: dates.indexOf(fixture.arrival.arrivalDate),
      decisionIndex: dates.indexOf(fixture.arrival.decisionDate),
      /* The whole inset in one computed boolean: an observation that has not
         arrived by the decision day cannot be a feature of that day's forecast,
         however early the event itself was. */
      availableAtDecision: daysBetween(fixture.arrival.arrivalDate, fixture.arrival.decisionDate) >= 0,
    },
  };
}

/* ==================================== F2/I2 · the label-arrival timeline */

export const ELIGIBILITY_FIXTURE = { origins: [6, 7, 8, 9, 10, 11, 12], cutoff: 12, horizon: 3, delay: 2 };
export const FIGURE_TWO_FIXTURE = { issueDay: 8, horizon: 3, delay: 2, cutoff: 12, featureSpan: 7 };

export function arrivalGeometry({ fixture = FIGURE_TWO_FIXTURE, width = 300, inset = 18, axisDays = 21 } = {}) {
  checkInteger(axisDays, 'the day axis length', 2, 200);
  const row = directRidgeRow({ origin: fixture.issueDay, horizon: fixture.horizon, delay: fixture.delay });
  const features = featureWindow({ origin: fixture.issueDay, delay: fixture.delay, span: fixture.featureSpan });
  const highest = Math.max(axisDays - 1, row.labelArrival, fixture.cutoff);
  /* The axis is EXPANDED to hold a late-arriving label rather than clipping it.
     A clipped arrival would hide exactly the quantity the figure is about. */
  const days = highest + 1;
  const scale = linearScale({ domain: [0, days - 1], range: [inset, width - inset] });
  return {
    width,
    /* 124, not 112. The axis title is end-anchored at the same x as the last
       tick label, so it needs a band of its own below it; at 112 the two boxes
       overlapped by two pixels on the rendered page. */
    height: 124,
    inset,
    days,
    scale,
    axisExpandedBeyondDefault: days > axisDays,
    ticks: Array.from({ length: days }, (_unused, index) => index)
      .filter(index => index % 2 === 0 || index === days - 1)
      .map(index => ({ value: index, x: scale(index) })),
    issueDay: fixture.issueDay,
    cutoff: fixture.cutoff,
    horizon: fixture.horizon,
    delay: fixture.delay,
    features: features.map(index => ({ index, x: scale(index), arrival: index + fixture.delay, arrivalX: scale(index + fixture.delay) })),
    latestFeatureDay: features[features.length - 1],
    targetDay: row.label,
    labelArrival: row.labelArrival,
    marks: {
      issue: { day: fixture.issueDay, x: scale(fixture.issueDay) },
      cutoff: { day: fixture.cutoff, x: scale(fixture.cutoff) },
      target: { day: row.label, x: scale(row.label) },
      arrival: { day: row.labelArrival, x: scale(row.labelArrival) },
    },
    /* Two independent decisions, reported separately so the figure cannot
       shade one as if it settled the other. */
    featuresLegitimate: features[features.length - 1] + fixture.delay <= fixture.cutoff,
    labelAvailable: row.labelArrival <= fixture.cutoff,
    /* Three marked rows plus an axis, each with its own band. The feature row
       and the label row are separated because the figure's whole job is to show
       that they are two independent decisions; drawn on one line they read as
       one shaded interval.
       Row labels sit ABOVE their marks with real clearance. The first draft put
       the word "features" four units above squares of half-height 3.2, and the
       rendered screenshot showed it sitting on top of them. */
    rows: {
      featureLabelBaseline: 22,
      featureY: 34,
      targetLabelBaseline: 50,
      targetY: 62,
      axisY: 88,
      tickY: 92,
      tickLabelBaseline: 103,
    },
  };
}

/* ==================================== F4 · the staircase of real rehearsals */

export const STAIRCASE_FIXTURE = { issueOrigins: [364, 371, 378], firstTrainOrigin: 6, horizons: 7, window: 90 };

/**
 * TWO PANELS, because this figure spans two scales and one axis cannot hold
 * both honestly.
 *
 * An expanding rehearsal at origin 364 trains on 358 days and forecasts 7. On a
 * single linear axis from day 6 to day 371 the entire informative region -- the
 * per-horizon boundary at t − h, the issue day, and the seven targets -- is the
 * last 2% of the width, and the browser screenshots showed exactly that: one
 * long bar with a smear at its right end. The boundary a reader is supposed to
 * read off this figure was indistinguishable from the issue mark.
 *
 *   OVERVIEW  the whole history each rehearsal sees, on absolute day numbers.
 *             Its only job is the LEFT EDGE: fixed in expanding mode, moving in
 *             sliding mode. No targets and no boundary are drawn here, because
 *             at this scale they would be a lie about legibility.
 *   DETAIL    the last three weeks of each rehearsal, on an axis of days
 *             RELATIVE to that rehearsal's own origin, so all three lanes share
 *             one scale. The boundary sits at −h, the issue day at 0 and the
 *             targets at +1…+7, each a clear distance apart. Training that
 *             continues off the left of the window is marked as continuing
 *             rather than redrawn as if it started there.
 */
export function staircaseGeometry({
  /* inset 30, not 16: each lane is named by its own three-digit origin, printed
     end-anchored just left of the drawing area. At 16 those labels began about
     two pixels left of the viewBox on the rendered page at every width -- small,
     but clipped, and the shared layout inspector did not see it. The inset is
     sized for the label, not the other way round. */
  fixture = STAIRCASE_FIXTURE, mode = 'expanding', horizon = 1, width = 300, inset = 30,
  detailDaysBefore = 14, detailDaysAfter = 7,
} = {}) {
  demand(mode === 'expanding' || mode === 'sliding', `mode must be expanding or sliding, got ${mode}`);
  checkInteger(horizon, 'the inspected horizon', 1, fixture.horizons);
  checkInteger(detailDaysBefore, 'the detail window before the origin', fixture.horizons + 1, 120);
  const window = mode === 'sliding' ? fixture.window : null;
  const fits = fixture.issueOrigins.map(issueOrigin =>
    horizonFit({ issueOrigin, horizon, firstTrainOrigin: fixture.firstTrainOrigin, window }));
  const low = Math.min(...fits.map(fit => fit.trainStart ?? fit.firstTrainOrigin));
  const high = Math.max(...fits.map(fit => fit.issueOrigin + fixture.horizons));
  const scale = linearScale({ domain: [low, high], range: [inset, width - inset] });
  const rows = { top: 16, laneGap: 20, barHeight: 9 };
  /* The axis band is sized for THREE rows of its own -- the line, the tick
     labels and the axis title -- because the first draft put the title 30 units
     above the foot of a 126-unit drawing, which landed it on the third lane's
     target marks. Point marks are `rect` elements, which neither the curve
     sampler nor the shared line inspector looks at, so nothing but reading the
     screenshot caught it. */
  const detailRows = {
    top: 20, laneGap: 32, barHeight: 9, targetHeight: 8,
    axisOffset: 34, tickOffset: 30, tickLabelOffset: 20, titleOffset: 4,
  };
  const detailScale = linearScale({
    domain: [-detailDaysBefore, detailDaysAfter], range: [inset, width - inset],
  });

  const overviewLanes = fits.map((fit, order) => ({
    order,
    y: rows.top + order * rows.laneGap,
    issueOrigin: fit.issueOrigin,
    trainStart: fit.trainStart,
    trainEnd: fit.trainEnd,
    trainCount: fit.trainCount,
    eligibleCount: fit.eligibleCount,
    train: { x: scale(fit.trainStart), width: scale(fit.trainEnd) - scale(fit.trainStart) },
    issue: { day: fit.issueOrigin, x: scale(fit.issueOrigin) },
  }));

  const detailLanes = fits.map((fit, order) => {
    const y = detailRows.top + order * detailRows.laneGap;
    const trainStartOffset = fit.trainStart - fit.issueOrigin;
    const clippedStart = Math.max(trainStartOffset, -detailDaysBefore);
    return {
      order,
      y,
      issueOrigin: fit.issueOrigin,
      trainCount: fit.trainCount,
      eligibleCount: fit.eligibleCount,
      /* The right edge is t − h, the per-horizon boundary. The left edge is
         clipped to the window and `continuesLeft` says so, rather than the bar
         being redrawn as if the training data began there. */
      train: {
        x: detailScale(clippedStart),
        width: detailScale(-horizon) - detailScale(clippedStart),
        continuesLeft: trainStartOffset < -detailDaysBefore,
        startOffset: trainStartOffset,
      },
      boundary: { day: fit.lastTrainOrigin, offset: -horizon, x: detailScale(-horizon) },
      issue: { day: fit.issueOrigin, offset: 0, x: detailScale(0) },
      targets: Array.from({ length: fixture.horizons }, (_unused, index) => index + 1).map(offset => ({
        offset,
        index: fit.issueOrigin + offset,
        x: detailScale(offset),
        isInspected: offset === horizon,
      })),
      maxObservationIndex: fit.audit.maxObservationIndex,
      clean: fit.audit.clean,
    };
  });

  return {
    width,
    inset,
    mode,
    horizon,
    window,
    low,
    high,
    fits,
    violations: fits.flatMap(fit => fit.audit.violations),
    overview: {
      width,
      height: rows.top + fits.length * rows.laneGap + 26,
      inset,
      scale,
      rows,
      lanes: overviewLanes,
      /* Asserted rather than assumed: expanding keeps ONE left edge, sliding
         must move it, or the toggle this figure exists for shows nothing. */
      distinctLeftEdges: new Set(overviewLanes.map(lane => lane.trainStart)).size,
      /* The HIGH end is always labelled. `axisTicks` picks a round step, and
         over 6..385 the only round multiple inside the domain is 200 — so the
         axis carried two labels and every bar ended past the last of them, in
         the panel that exists for legibility. Adding the domain's own upper
         bound costs nothing and puts a label beyond the marks rather than
         behind them. */
      ticks: (() => {
        const chosen = axisTicks({ low, high, count: 4 }).filter(value => value >= low && value <= high);
        const top = Math.round(high);
        if (!chosen.length || top - chosen[chosen.length - 1] > (high - low) * 0.08) chosen.push(top);
        return chosen.map(value => ({ value, x: scale(value) }));
      })(),
    },
    detail: {
      width,
      /* The last lane's marks end at `top + (n-1)*laneGap + targetHeight + 2`;
         the axis band adds its own 48 units below that, so the title clears
         every mark by construction rather than by a number chosen by eye. */
      height: detailRows.top + (fits.length - 1) * detailRows.laneGap + detailRows.targetHeight + 2 + 48,
      inset,
      scale: detailScale,
      rows: detailRows,
      daysBefore: detailDaysBefore,
      daysAfter: detailDaysAfter,
      lanes: detailLanes,
      /* The boundary and the issue day must be far enough apart on screen to be
         told apart. At h = 1 they are one day apart, which at this scale is
         about twelve units -- the whole reason the detail panel exists. */
      boundaryToIssueUnits: detailScale(0) - detailScale(-horizon),
      ticks: Array.from({ length: detailDaysBefore + detailDaysAfter + 1 },
        (_unused, index) => index - detailDaysBefore)
        .filter(offset => offset % 7 === 0 || offset === detailDaysAfter)
        .map(offset => ({ value: offset, x: detailScale(offset) })),
    },
  };
}

/** The other panel of figure 4: the same issue origin at every horizon, so the
 *  per-horizon boundary t − h is visible as a staircase in its own right. */
export function horizonBoundaryTable({ issueOrigin, firstTrainOrigin, horizons = 7, delay = 0, window = null }) {
  return Array.from({ length: horizons }, (_unused, index) => {
    const fit = horizonFit({ issueOrigin, horizon: index + 1, firstTrainOrigin, delay, window });
    return {
      horizon: fit.horizon,
      lastTrainOrigin: fit.lastTrainOrigin,
      trainCount: fit.trainCount,
      eligibleCount: fit.eligibleCount,
      targetIndex: fit.targetIndex,
      maxObservationIndex: fit.audit.maxObservationIndex,
      clean: fit.audit.clean,
    };
  });
}

/* ================================== F5 · the measured comparison geometry */

export function horizonCurveGeometry({ series, width = 300, height = 180, padding = { top: 26, right: 12, bottom: 34, left: 44 } }) {
  demand(Array.isArray(series) && series.length > 0, 'a horizon chart needs at least one method');
  const values = series.flatMap(entry => entry.values);
  values.forEach((value, index) => checkFinite(value, `plotted value ${index}`));
  const highest = Math.max(...values);
  const horizons = series[0].values.length;
  demand(series.every(entry => entry.values.length === horizons), 'every method must carry the same horizons');
  /* The axis starts at zero and extends BEYOND the largest plotted value, so no
     point sits on the frame and no reader has to guess whether a curve was
     clipped. */
  const top = highest * 1.08;
  const x = linearScale({ domain: [1, horizons], range: [padding.left, width - padding.right] });
  const y = linearScale({ domain: [0, top], range: [height - padding.bottom, padding.top] });
  return {
    width,
    height,
    padding,
    x,
    y,
    horizons,
    maximum: highest,
    axisTop: top,
    yTicks: axisTicks({ low: 0, high: top, count: 5 }).filter(value => value <= top).map(value => ({ value, y: y(value) })),
    xTicks: Array.from({ length: horizons }, (_unused, index) => index + 1).map(value => ({ value, x: x(value) })),
    series: series.map(entry => ({
      ...entry,
      points: entry.values.map((value, index) => ({ horizon: index + 1, value, x: x(index + 1), y: y(value) })),
    })),
    /* Coincident points are found, not assumed. The h = 7 naive/seasonal
       equality is a required consequence of the rules, and a figure that draws
       one mark over the other without saying so hides its best teaching
       moment. */
    coincidences: Array.from({ length: horizons }, (_unused, index) => index + 1)
      .map(horizon => {
        const here = series.map(entry => ({ key: entry.key, value: entry.values[horizon - 1] }));
        const groups = new Map();
        here.forEach(entry => {
          const bucket = groups.get(entry.value) ?? [];
          bucket.push(entry.key);
          groups.set(entry.value, bucket);
        });
        const shared = [...groups.entries()].filter(([, keys]) => keys.length > 1);
        return shared.length ? { horizon, value: shared[0][0], keys: shared[0][1] } : null;
      })
      .filter(Boolean),
    winners: Array.from({ length: horizons }, (_unused, index) => {
      const best = Math.min(...series.map(entry => entry.values[index]));
      /* EVERY method attaining the minimum, never `winners[0]`. A tie graded as
         a miss because grading took the first winner is a defect this
         repository has shipped. */
      return { horizon: index + 1, value: best, keys: series.filter(entry => entry.values[index] === best).map(entry => entry.key) };
    }),
  };
}

/** Short comparison bars for the six development candidates. Length encodes
 *  MAE from a zero baseline, so a bar twice as long is twice the error. */
export function developmentBarGeometry({ rows, width = 300, barHeight = 13, gap = 7, labelWidth = 128 }) {
  demand(Array.isArray(rows) && rows.length > 0, 'the development comparison needs at least one candidate');
  const highest = Math.max(...rows.map(row => row.mae));
  const scale = linearScale({ domain: [0, highest * 1.02], range: [labelWidth, width - 6] });
  const best = Math.min(...rows.map(row => row.mae));
  return {
    width,
    height: rows.length * (barHeight + gap) + 20,
    labelWidth,
    barHeight,
    scale,
    maximum: highest,
    zeroX: scale(0),
    rows: rows.map((row, index) => ({
      ...row,
      order: index,
      y: 6 + index * (barHeight + gap),
      barWidth: scale(row.mae) - scale(0),
      selected: row.mae === best,
    })),
    bestMae: best,
    selectedKeys: rows.filter(row => row.mae === best).map(row => row.key),
  };
}

/** The history chart of investigation 3: the last days before the origin, the
 *  cutoff, and the block the seasonal rule copies. */
export function historyChartGeometry({
  dates, counts, origin, period, shown = 14, width = 300, height = 150,
  padding = { top: 22, right: 10, bottom: 30, left: 44 },
}) {
  checkInteger(shown, 'the number of shown days', 2, 60);
  checkInteger(origin, 'the origin', shown - 1, counts.length - 1);
  const start = origin - shown + 1;
  const indices = Array.from({ length: shown }, (_unused, index) => start + index);
  const values = indices.map(index => counts[index]);
  const highest = Math.max(...values);
  const x = linearScale({ domain: [start, origin], range: [padding.left, width - padding.right] });
  const y = linearScale({ domain: [0, highest * 1.1], range: [height - padding.bottom, padding.top] });
  const donorIndices = Array.from({ length: Math.min(period, shown) }, (_unused, index) => origin - period + 1 + index)
    .filter(index => index >= start);
  return {
    width,
    height,
    padding,
    x,
    y,
    start,
    origin,
    shown,
    maximum: highest,
    points: indices.map(index => ({
      index,
      date: dates[index],
      weekday: weekdayName(dates[index]),
      count: counts[index],
      x: x(index),
      y: y(counts[index]),
      isDonor: donorIndices.includes(index),
    })),
    donorIndices,
    cutoff: { index: origin, x: x(origin) },
    yTicks: axisTicks({ low: 0, high: highest * 1.1, count: 4 }).map(value => ({ value, y: y(value) })),
  };
}

/* ====================================================== graded investigations
 *
 * Each `...Answer` takes the COMMITTED inputs and returns an outcome category,
 * an optional numeric value and an explanation. Nothing here reads the current
 * on-screen draft: the component hands over the inputs a prediction was
 * recorded with, so a prediction is always graded against what it answered.
 *
 * Every outcome category is a value the grader can actually return for some
 * admissible input, and every degenerate case -- an unscored horizon, an empty
 * eligible set, an exact tie -- has a named category rather than being folded
 * into a neighbouring one.
 */

/** A tie is its own answer. Comparisons on this page return one of three
 *  categories, never a boolean, because "seasonal is not lower" conflates
 *  "higher" with "exactly equal" -- and the lesson's headline example is an
 *  exact equality. */
export function compareScores(left, right, tolerance = 1e-9) {
  if (left === null || right === null) return 'unscored';
  checkFinite(left, 'the left score');
  checkFinite(right, 'the right score');
  const difference = left - right;
  if (Math.abs(difference) <= tolerance) return 'equal';
  return difference < 0 ? 'lower' : 'higher';
}

export const TOY_HISTORY = [10, 20, 10, 20, 12, 22];
export const TOY_FUTURE = [12, 22, 12, 22];

/** Investigation 1: which observation feeds the selected horizon, and what is
 *  the seasonal forecast there? */
export function donorAnswer({ history, period, horizonCount, selectedHorizon, future }) {
  checkCounts(history, 'the history', { minimum: 2 });
  checkInteger(period, 'the season length', 1, history.length);
  checkInteger(horizonCount, 'the horizon count', 1, 8);
  checkInteger(selectedHorizon, 'the selected horizon', 1, horizonCount);
  const rules = baselineForecasts({ history, horizon: horizonCount, period });
  const donorIndex = rules.seasonalDonors[selectedHorizon - 1];
  const outcomes = (future ?? []).slice(0, horizonCount);
  const padded = Array.from({ length: horizonCount }, (_unused, index) =>
    (index < outcomes.length && outcomes[index] !== null && outcomes[index] !== undefined ? outcomes[index] : null));
  const summaries = Object.fromEntries(BASELINE_RULES.map(rule => [rule, errorSummary(padded, rules[rule])]));
  return {
    /* The graded category is the DONOR's one-based position. It is a fact about
       the rule's indexing, which is what the investigation asks about; the
       numeric commitment beside it is the forecast value itself. */
    outcome: `source-${donorIndex + 1}`,
    value: rules.seasonal[selectedHorizon - 1],
    donorIndex,
    donorPosition: donorIndex + 1,
    rules,
    summaries,
    scoredHorizons: padded.filter(value => value !== null).length,
    explain: `Horizon ${selectedHorizon} under season length ${period} copies history position `
      + `${donorIndex + 1} of ${history.length}: T − m + ((h − 1) mod m) = ${history.length} − ${period} + `
      + `${(selectedHorizon - 1) % period} = ${donorIndex} in zero-based terms.`,
  };
}

/** Investigation 2: the exact set of eligible training origins. */
export function eligibilityAnswer({ origins, cutoff, horizon, delay, selected }) {
  const rows = eligibilityRows({ origins, cutoff, horizon, delay });
  const eligible = rows.filter(row => row.eligible).map(row => row.origin);
  const chosen = [...new Set(selected ?? [])].sort((left, right) => left - right);
  const exact = chosen.length === eligible.length && chosen.every((value, index) => value === eligible[index]);
  const missed = eligible.filter(origin => !chosen.includes(origin));
  const extra = chosen.filter(origin => !eligible.includes(origin));
  return {
    /* Three outcomes, because an empty eligible set is a meaningful result the
       lab must be able to report rather than a state it refuses to reach. */
    outcome: eligible.length === 0 ? 'none' : exact ? 'exact' : 'different',
    value: eligible.length,
    rows,
    eligible,
    chosen,
    exact,
    missed,
    extra,
    explain: eligible.length === 0
      ? `No offered origin satisfies s + ${horizon} + ${delay} ≤ ${cutoff}: the earliest offered origin is `
        + `${Math.min(...origins)}, whose label would arrive on day ${Math.min(...origins) + horizon + delay}.`
      : `The rule is s + h + d ≤ cutoff, here s + ${horizon} + ${delay} ≤ ${cutoff}, so s ≤ ${cutoff - horizon - delay}.`,
  };
}

/** Investigation 3: will the seasonal rule's MAE come in below the naive
 *  rule's on this week, and what is the horizon-3 forecast? */
export function requestAnswer({ counts, dates, origin, period, horizon = 7 }) {
  const request = forecastRequest({ counts, dates, origin, period, horizon });
  const seasonal = request.summaries.seasonal.mae;
  const naive = request.summaries.naive.mae;
  return {
    outcome: compareScores(seasonal, naive),
    value: request.forecasts.seasonal[2] ?? null,
    request,
    seasonalMae: seasonal,
    naiveMae: naive,
    /* Six decimals, not the raw double. Interpolating the number straight into
       the sentence printed "1042.5714285714287" on the page -- a reading a
       learner cannot check against anything else on screen, and a precision the
       measurement does not have. Every other readout in this lesson prints at
       six decimals, and the verdict has to match them. */
    explain: seasonal === null || naive === null
      ? 'No outcome was supplied for any horizon, so neither rule has a score to compare.'
      : `Seasonal MAE ${seasonal.toFixed(6)} against naive MAE ${naive.toFixed(6)} over `
        + `${request.summaries.seasonal.denominator} scored horizons.`,
  };
}

/* ========================================================= exported fixtures
 *
 * Every fixture the page uses by name. They live here rather than in the
 * components so that the verifier exercises the same starting states a reader
 * opens, including the degenerate ones.
 */
export const fixtures = {
  toy: { history: TOY_HISTORY, future: TOY_FUTURE, period: 2, horizonCount: 4, selectedHorizon: 3 },
  toyEdited: { history: [10, 20, 10, 20, 18, 22], future: TOY_FUTURE, period: 2, horizonCount: 4, selectedHorizon: 3 },
  toyLongHorizon: { history: TOY_HISTORY, future: TOY_FUTURE, period: 2, horizonCount: 8, selectedHorizon: 7 },
  toyPeriodOne: { history: TOY_HISTORY, future: TOY_FUTURE, period: 1, horizonCount: 4, selectedHorizon: 3 },
  toyFutureShifted: { history: TOY_HISTORY, future: [13, 23, 13, 23], period: 2, horizonCount: 4, selectedHorizon: 3 },
  eligibility: ELIGIBILITY_FIXTURE,
  eligibilityNoDelay: { origins: [6, 7, 8, 9, 10, 11, 12], cutoff: 12, horizon: 3, delay: 0 },
  eligibilityHorizonOne: { origins: [6, 7, 8, 9, 10, 11, 12], cutoff: 12, horizon: 1, delay: 0 },
  eligibilityEmpty: { origins: [6, 7, 8, 9, 10, 11, 12], cutoff: 10, horizon: 3, delay: 2 },
  recursion: { lastObserved: 22, step: 2, horizon: 4, outcomes: TOY_FUTURE },
  recursionChanged: { lastObserved: 22, step: 2, horizon: 4, outcomes: [22, 24, 26, 28] },
  staircase: STAIRCASE_FIXTURE,
  request: { origin: 364, period: 7, horizon: 7 },
  requestSecond: { origin: 371, period: 7, horizon: 7 },
  requestFortnight: { origin: 476, period: 14, horizon: 7 },
};

/** The six-value cycle worked out in full, for the prose and the figure. Built
 *  once so a printed number and a drawn cell cannot disagree. */
export const toyComparison = (() => {
  const rules = baselineForecasts({ history: TOY_HISTORY, horizon: 4, period: 2 });
  const summaries = Object.fromEntries(BASELINE_RULES.map(rule => [rule, errorSummary(TOY_FUTURE, rules[rule])]));
  return { rules, summaries };
})();

export const toyPeriodTable = Array.from({ length: 6 }, (_unused, index) => {
  const period = index + 1;
  return { period, forecasts: baselineForecasts({ history: TOY_HISTORY, horizon: 8, period }).seasonal };
});
