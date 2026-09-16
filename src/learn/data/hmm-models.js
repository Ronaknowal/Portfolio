/** Pure models for the hidden Markov model lesson.
 *
 * Everything a figure or an investigation draws is computed here, so that a
 * drawn trellis edge, a selected predecessor, a path bar, a fractional count
 * flow, a duration bar or a likelihood track is an asserted mathematical claim
 * rather than a shape chosen inside a component. No proportion is eyeballed in
 * a stylesheet and no coordinate is interpolated from a picture.
 *
 * Three categories of quantity live here and are never mixed:
 *
 *   1. Exact declared constructions. The two-state Rainy/Sunny model, its four
 *      activity reports, the three-state constrained graph, the two recordings
 *      and the geometric duration family are exact arithmetic on stated inputs.
 *   2. Recorded observations. The English Web Treebank quantities are read from
 *      hmm-data.js; nothing here refits a tagger or re-runs EM on real text.
 *   3. Numerical-representation facts. Underflow, log-space and retained
 *      scaling factors are computed, not asserted in prose.
 *
 * Two independent numerical routes are deliberately kept, because the lesson
 * teaches the difference between them and because a verifier that calls one
 * helper twice has checked nothing:
 *
 *   * `trellis` computes raw joint masses by direct multiplication. Those are
 *      the numbers the manuscript prints and the figure draws, and for the
 *      bounded sizes this lesson allows they are exactly representable.
 *   * `infer` computes posteriors by RETAINED SCALING: each forward column is
 *      normalised, its scale factor kept, and the backward pass divided by the
 *      same factors. The content packet's author program uses log-space
 *      throughout, so agreement between the two is evidence rather than an echo.
 *
 * Load-bearing conventions, kept everywhere:
 *
 *   * A symbol of `MISSING` (−1) is a RETAINED time step with no observation.
 *     Summing over every symbol gives emission likelihood one, so the state
 *     still transitions. Deleting the step instead is a different model, and
 *     the two are never conflated.
 *   * An impossible observation sequence has zero evidence and NO conditional
 *     posterior. `infer` returns `impossible: true` with null beliefs rather
 *     than normalising zero by zero into an invented distribution.
 *   * A structural zero stays zero. Nothing here adds an epsilon to a forbidden
 *     transition or emission to make an answer exist.
 *   * An unvisited state's expected-count row is unidentified, so `emStep`
 *     retains the previous row rather than dividing by zero.
 *
 * Every entry point refuses input it cannot honour rather than substituting a
 * silent default: a non-finite number, a row that does not sum to one, an
 * out-of-range setting, an empty sequence or an enumeration too large to
 * perform raises a RangeError.
 */

/* ------------------------------------------------------------------ limits */

/** A retained time step whose observation is absent. Not a symbol index. */
export const MISSING = -1;

/** Two declared size envelopes, because they answer different questions.
 *
 * `toy` is the visual contract's bound on free exploration: a learner may build
 * any model up to this size and the browser recalculates it exactly. `real` is
 * the envelope of the recorded English Web Treebank tagger, whose 146-symbol
 * emission table and 15-token sentences are read from saved results and never
 * refitted here. Passing the wrong envelope is a RangeError, not a silent
 * widening, so an investigation cannot quietly grow past what it promises.
 */
export const envelopes = {
  toy: { maximumStates: 4, maximumSymbols: 4, maximumTimeSteps: 12, name: 'toy exploration' },
  real: { maximumStates: 4, maximumSymbols: 200, maximumTimeSteps: 20, name: 'the recorded tagger' },
};

export const limits = {
  /** Sizes live in `envelopes`, once, so a control and a guard cannot drift. */
  minimumTimeSteps: 1,
  /** Enumerating every path is exponential; it is offered only when it is small. */
  maximumEnumeratedPaths: 4096,
  /** Probability entries a learner may type, and the tolerance a row must meet. */
  probability: { minimum: 0, maximum: 1 },
  rowSumTolerance: 1e-9,
  /** Self-transition probability of the duration family. 1 is absorbing. */
  stay: { minimum: 0, maximum: 1 },
  durationBars: 8,
  /** Independent recordings the count investigation may hold. */
  maximumSequences: 4,
  /** Equality tolerance for the lesson's declared nulls. */
  nullTolerance: 1e-10,
  /** Below this relative gap a graded quantity is reported as having NOT moved.
   *
   * It is deliberately tied to the precision the verdict prints. A verdict that
   * says "it rose" while showing two identical numbers is the defect this
   * lesson's investigations exist to avoid, so the tolerance and the displayed
   * digits are chosen together: twelve decimal places beside a 1e-12 threshold.
   */
  moveTolerance: 1e-12,
  moveDigits: 12,
};

/** Which way a graded quantity moved between two committed states.
 *
 * Exported and used by every direction question in the lesson, so that the rule
 * a verdict applies is one function with one definition rather than an
 * open-coded comparison per lab. `null` on either side means the quantity had no
 * value in that state - an impossible sequence has no posterior - and that is
 * reported as its own outcome instead of being compared against a number.
 */
export function direction(before, after, tolerance = limits.moveTolerance) {
  if (before === null || after === null || before === undefined || after === undefined) return 'undefined';
  checkFinite(before, 'the earlier value');
  checkFinite(after, 'the later value');
  const gap = after - before;
  if (Math.abs(gap) <= tolerance * Math.max(1, Math.abs(before))) return 'unchanged';
  return gap > 0 ? 'rises' : 'falls';
}

export function checkFinite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number.`);
  }
  return value;
}
export function checkRange(value, range, name) {
  checkFinite(value, name);
  if (value < range.minimum || value > range.maximum) {
    throw new RangeError(`Keep ${name} between ${range.minimum} and ${range.maximum}.`);
  }
  return value;
}
const sum = values => values.reduce((total, value) => total + value, 0);
const zeros = length => Array.from({ length }, () => 0);
const matrix = (rows, columns) => Array.from({ length: rows }, () => zeros(columns));

/* --------------------------------------------------------- model validation */

/** A probability row: nonnegative, finite, summing to one within tolerance.
 *
 * The tolerance is on the SUM, never on an individual entry, so a structural
 * zero stays exactly zero and is never quietly lifted to a tiny positive value.
 */
export function checkRow(row, name) {
  if (!Array.isArray(row) || row.length === 0) throw new RangeError(`${name} must be a nonempty row.`);
  row.forEach((value, index) => {
    checkFinite(value, `${name}[${index}]`);
    if (value < 0) throw new RangeError(`${name}[${index}] must not be negative.`);
  });
  const total = sum(row);
  if (Math.abs(total - 1) > limits.rowSumTolerance) {
    throw new RangeError(`${name} must sum to 1; it sums to ${total}.`);
  }
  return row;
}

/** A complete categorical HMM. `stateNames` and `symbolNames` travel with it so
 * a diagram never has to guess which index it is drawing. */
export function checkModel(model, envelope = envelopes.toy) {
  const { start, transition, emission } = model;
  checkRow(start, 'the initial-state row');
  const states = start.length;
  if (states < 1 || states > envelope.maximumStates) {
    throw new RangeError(`${envelope.name} supports 1 to ${envelope.maximumStates} states.`);
  }
  if (!Array.isArray(transition) || transition.length !== states) {
    throw new RangeError('The transition matrix needs one row per state.');
  }
  transition.forEach((row, index) => {
    if (row.length !== states) throw new RangeError(`Transition row ${index} needs one entry per state.`);
    checkRow(row, `transition row ${index}`);
  });
  if (!Array.isArray(emission) || emission.length !== states) {
    throw new RangeError('The emission matrix needs one row per state.');
  }
  const symbols = emission[0].length;
  if (symbols < 1 || symbols > envelope.maximumSymbols) {
    throw new RangeError(`${envelope.name} supports 1 to ${envelope.maximumSymbols} observation symbols.`);
  }
  emission.forEach((row, index) => {
    if (row.length !== symbols) throw new RangeError(`Emission row ${index} needs one entry per symbol.`);
    checkRow(row, `emission row ${index}`);
  });
  return { states, symbols };
}

function checkObservations(observations, symbols, envelope = envelopes.toy) {
  if (!Array.isArray(observations) || observations.length < limits.minimumTimeSteps) {
    throw new RangeError('A sequence needs at least one time step.');
  }
  if (observations.length > envelope.maximumTimeSteps) {
    throw new RangeError(`Keep sequences to at most ${envelope.maximumTimeSteps} time steps under ${envelope.name}.`);
  }
  observations.forEach((value, index) => {
    if (!Number.isInteger(value)) throw new RangeError(`Observation ${index} must be a whole number.`);
    if (value !== MISSING && (value < 0 || value >= symbols)) {
      throw new RangeError(`Observation ${index} must be a symbol index or ${MISSING} for a missing report.`);
    }
  });
  return observations;
}

/** The emission likelihood of one time step, for every state.
 *
 * A missing report is marginalised over every symbol, which gives one for each
 * state because the emission rows are distributions. That is the whole reason a
 * retained missing step differs from a deleted step: it keeps its transition.
 */
export function localLikelihood(model, value) {
  return model.emission.map(row => (value === MISSING ? 1 : row[value]));
}

/* ================================================== the trellis, unscaled */

/** Every cell, every incoming edge contribution and every selected predecessor
 * of the trellis the figures draw.
 *
 * `mode` is `'sum'` for the forward recursion and `'max'` for Viterbi. The two
 * differ only in how a destination combines its incoming contributions, which
 * is exactly the contrast section 5 is about, so they share one implementation
 * and the caller states which operator it wants.
 *
 * Each edge carries `carried = previousCell * transition`, BEFORE the shared
 * destination emission. That is the quantity a Viterbi predecessor is chosen
 * on, and choosing on the previous cell alone is the error the lesson exists
 * to prevent, so the figure is given the post-transition number to draw.
 */
export function trellis(model, observations, mode = 'sum', envelope = envelopes.toy) {
  const { symbols } = checkModel(model, envelope);
  checkObservations(observations, symbols, envelope);
  if (mode !== 'sum' && mode !== 'max') throw new RangeError('The trellis operator is "sum" or "max".');
  const states = model.start.length;
  const columns = [];
  observations.forEach((value, time) => {
    const local = localLikelihood(model, value);
    if (time === 0) {
      columns.push({
        time,
        observation: value,
        emission: local,
        cells: model.start.map((prior, state) => ({
          state,
          incoming: [],
          aggregate: prior,
          chosen: null,
          emission: local[state],
          value: prior * local[state],
        })),
      });
      return;
    }
    const previous = columns[time - 1].cells.map(cell => cell.value);
    const cells = [];
    for (let destination = 0; destination < states; destination += 1) {
      const incoming = previous.map((mass, origin) => ({
        from: origin,
        previousValue: mass,
        transition: model.transition[origin][destination],
        carried: mass * model.transition[origin][destination],
      }));
      const carried = incoming.map(edge => edge.carried);
      const aggregate = mode === 'sum'
        ? sum(carried)
        : carried.reduce((best, entry) => (entry > best ? entry : best), -Infinity);
      /** The FIRST index attaining the maximum owns a tie, declared here so a
       * drawn arrow and a graded answer cannot disagree about which one wins. */
      let chosen = null;
      if (mode === 'max') {
        chosen = 0;
        for (let origin = 1; origin < states; origin += 1) {
          if (carried[origin] > carried[chosen]) chosen = origin;
        }
      }
      const tied = mode === 'max'
        ? carried.filter(entry => entry === carried[chosen]).length > 1
        : false;
      cells.push({
        state: destination,
        incoming,
        aggregate,
        chosen,
        tiedPredecessor: tied,
        emission: local[destination],
        value: aggregate * local[destination],
      });
    }
    columns.push({ time, observation: value, emission: local, cells });
  });
  const totals = columns.map(column => sum(column.cells.map(cell => cell.value)));
  return {
    mode,
    columns,
    columnTotals: totals,
    /** For `'sum'` this is P(o); for `'max'` it is the best path's joint mass. */
    final: mode === 'sum'
      ? totals[totals.length - 1]
      : columns[columns.length - 1].cells.reduce((best, cell) => (cell.value > best ? cell.value : best), 0),
    /** Transition contributions evaluated: (T−1)·N² is the cost claim of §11. */
    transitionContributions: (observations.length - 1) * model.start.length ** 2,
  };
}

/** Backtrack the stored predecessors of a max-mode trellis into one path. */
export function backtrack(maxTrellis) {
  const columns = maxTrellis.columns;
  const last = columns[columns.length - 1].cells;
  let best = 0;
  for (let state = 1; state < last.length; state += 1) if (last[state].value > last[best].value) best = state;
  const path = new Array(columns.length);
  path[columns.length - 1] = best;
  for (let time = columns.length - 2; time >= 0; time -= 1) {
    path[time] = columns[time + 1].cells[path[time + 1]].chosen;
  }
  return {
    path,
    joint: last[best].value,
    finalTie: last.filter(cell => cell.value === last[best].value).length > 1,
  };
}

/* ====================================== inference by retained scaling */

/** Forward filtering with the scale factor of every step retained.
 *
 * `factors[0] = P(o_0)` and `factors[t] = P(o_t | o_0..o_{t-1})`, so their
 * product is the evidence and the sum of their logarithms is its logarithm.
 * This is the exact alternative to log-space that section 7 contrasts, and it
 * is what the browser uses, so agreement with the packet's log-space program is
 * agreement between two routes rather than one helper called twice.
 */
export function forwardScaled(model, observations, envelope = envelopes.toy) {
  const { symbols } = checkModel(model, envelope);
  checkObservations(observations, symbols, envelope);
  const states = model.start.length;
  const factors = [];
  const filtered = [];
  let belief = model.start.slice();
  let impossible = false;
  for (let time = 0; time < observations.length && !impossible; time += 1) {
    const local = localLikelihood(model, observations[time]);
    const predicted = time === 0
      ? belief.slice()
      : Array.from({ length: states }, (unused, destination) =>
        sum(belief.map((mass, origin) => mass * model.transition[origin][destination])));
    const mass = predicted.map((entry, state) => entry * local[state]);
    const factor = sum(mass);
    if (factor === 0) {
      // The sequence is impossible from this step onward. Stop rather than
      // dividing zero by zero into a distribution the model never assigned.
      factors.push(0);
      impossible = true;
    } else {
      belief = mass.map(entry => entry / factor);
      factors.push(factor);
      filtered.push(belief.slice());
    }
  }
  return {
    factors,
    filtered,
    impossible,
    /** The step at which evidence vanished, so an interface can point at it. */
    impossibleAt: impossible ? factors.length - 1 : null,
    evidence: impossible ? 0 : factors.reduce((product, factor) => product * factor, 1),
    logEvidence: impossible ? -Infinity : sum(factors.map(factor => Math.log(factor))),
    filteredLast: impossible ? null : filtered[filtered.length - 1],
  };
}

/** One state-space prediction step, with no observation applied.
 *
 * Section 3's forecast: propagate the whole filtered row through A, then read
 * the predicted observation distribution off the emission matrix. Collapsing
 * the belief to its mode first is the mistake the investigation contrasts, so
 * both quantities are returned and neither is derived from the other.
 */
export function predictNext(model, filteredRow) {
  checkModel(model);
  checkRow(filteredRow, 'the filtered row');
  const states = model.start.length;
  const nextState = Array.from({ length: states }, (unused, destination) =>
    sum(filteredRow.map((mass, origin) => mass * model.transition[origin][destination])));
  const symbols = model.emission[0].length;
  const nextObservation = Array.from({ length: symbols }, (unused, symbol) =>
    sum(nextState.map((mass, state) => mass * model.emission[state][symbol])));
  let mode = 0;
  for (let state = 1; state < states; state += 1) if (filteredRow[state] > filteredRow[mode]) mode = state;
  const collapsedState = model.transition[mode].slice();
  const collapsedObservation = Array.from({ length: symbols }, (unused, symbol) =>
    sum(collapsedState.map((mass, state) => mass * model.emission[state][symbol])));
  return { nextState, nextObservation, mode, collapsedState, collapsedObservation };
}

/** Complete inference: evidence, filtered and smoothed beliefs, pair
 * posteriors, the best path and the pointwise modes.
 *
 * Scaling detail that matters: the backward row is divided by the SAME factors
 * the forward pass produced, so `smoothed = filteredHat * backwardHat` needs no
 * further normalisation and `pair` divides by one extra factor rather than by
 * the evidence. Mixing a scaled forward row with an unscaled backward row is a
 * classic way to get a posterior that silently fails to sum to one.
 */
export function infer(model, observations, envelope = envelopes.toy) {
  const { symbols } = checkModel(model, envelope);
  checkObservations(observations, symbols, envelope);
  const states = model.start.length;
  const length = observations.length;
  /* The envelope has to travel with the delegation. Dropping it here made
     `infer(realTagger, sentence, envelopes.real)` validate the 146-symbol
     emission table against the four-symbol toy bound and throw on first paint,
     which no assertion about the returned value could have caught. */
  const scaled = forwardScaled(model, observations, envelope);
  if (scaled.impossible) {
    return {
      impossible: true, evidence: 0, logEvidence: -Infinity,
      factors: scaled.factors, filtered: null, smoothed: null, pair: null,
      path: null, pathJoint: 0, pathPosterior: null, marginalModes: null,
      impossibleAt: scaled.impossibleAt,
      reason: 'Every state assigns zero probability to one of these reports, so the sequence has zero evidence and no conditional posterior.',
    };
  }
  const filtered = scaled.filtered;
  /* The scaled backward pass. With alphaHat(t) = alpha(t) / prod_{s<=t} c(s)
     and betaHat(t) = beta(t) / prod_{s>t} c(s), those two products multiply
     back to P(o). So `smoothed = alphaHat * betaHat` needs no further
     normalisation, and xi(t) = alphaHat(t) A b betaHat(t+1) / c(t+1). Dividing
     the backward step by c(t) instead of c(t+1) is the classic way to get a
     posterior that quietly fails to sum to one, so every index is spelled out
     rather than inferred, and the sums are asserted below. */
  const backward = Array.from({ length }, () => zeros(states));
  backward[length - 1] = zeros(states).map(() => 1);
  for (let time = length - 2; time >= 0; time -= 1) {
    const local = localLikelihood(model, observations[time + 1]);
    for (let origin = 0; origin < states; origin += 1) {
      backward[time][origin] = sum(Array.from({ length: states }, (unused, destination) =>
        model.transition[origin][destination] * local[destination] * backward[time + 1][destination]))
        / scaled.factors[time + 1];
    }
  }
  const smoothed = filtered.map((row, time) => row.map((mass, state) => mass * backward[time][state]));
  const pair = [];
  for (let time = 0; time < length - 1; time += 1) {
    const local = localLikelihood(model, observations[time + 1]);
    const block = matrix(states, states);
    for (let origin = 0; origin < states; origin += 1) {
      for (let destination = 0; destination < states; destination += 1) {
        block[origin][destination] = filtered[time][origin] * model.transition[origin][destination]
          * local[destination] * backward[time + 1][destination] / scaled.factors[time + 1];
      }
    }
    pair.push(block);
  }
  const maxTrellis = trellis(model, observations, 'max', envelope);
  const best = backtrack(maxTrellis);
  const marginalModes = smoothed.map(row => {
    let mode = 0;
    for (let state = 1; state < states; state += 1) if (row[state] > row[mode]) mode = state;
    return mode;
  });
  return {
    impossible: false,
    evidence: scaled.evidence,
    logEvidence: scaled.logEvidence,
    factors: scaled.factors,
    filtered,
    smoothed,
    backward,
    pair,
    path: best.path,
    pathJoint: best.joint,
    pathPosterior: best.joint / scaled.evidence,
    pathTie: best.finalTie,
    marginalModes,
    /** Whether the pointwise modes form a path the transition matrix allows. */
    modesLegal: marginalModes.every((state, time) =>
      time === 0 ? model.start[state] > 0 : model.transition[marginalModes[time - 1]][state] > 0),
    modesJoint: pathJointOf(model, marginalModes, observations, envelope),
    /** Expected correct positions of a decision, under the model's own
     * marginals. Pointwise modes maximise it over ALL paths; Viterbi maximises
     * the joint over legal ones. Both are returned so the lab can show that a
     * better position count and validity as a path are separate properties. */
    modesExpectedCorrect: sum(marginalModes.map((state, time) => smoothed[time][state])),
    pathExpectedCorrect: sum(best.path.map((state, time) => smoothed[time][state])),
  };
}

/** The highest-probability path the model actually permits, by exhaustive
 * search over legal paths. This is a second route to Viterbi's answer, kept
 * separate so a verifier can compare the dynamic program with brute force
 * rather than calling the dynamic program twice. */
export function bestLegalPath(model, observations) {
  const enumerated = enumeratePaths(model, observations);
  const legal = enumerated.paths.filter(entry => pathLegal(model, entry.path));
  if (legal.length === 0) return { path: null, joint: 0, legalPaths: 0 };
  const best = legal.reduce((winner, entry) => (entry.joint > winner.joint ? entry : winner));
  return {
    path: best.path,
    joint: best.joint,
    legalPaths: legal.length,
    tied: legal.filter(entry => entry.joint === best.joint).length > 1,
  };
}

/** The exact joint probability of one named state path with its observations.
 *
 * Read it as the manuscript does: start, emit, transition, emit, transition,
 * emit. A four-report sequence therefore contains three transitions, not four.
 */
export function pathJointOf(model, path, observations, envelope = envelopes.toy) {
  const { symbols } = checkModel(model, envelope);
  checkObservations(observations, symbols, envelope);
  if (!Array.isArray(path) || path.length !== observations.length) {
    throw new RangeError('A path needs exactly one state per time step.');
  }
  path.forEach((state, time) => {
    if (!Number.isInteger(state) || state < 0 || state >= model.start.length) {
      throw new RangeError(`Path position ${time} names no state of this model.`);
    }
  });
  let joint = model.start[path[0]] * localLikelihood(model, observations[0])[path[0]];
  for (let time = 1; time < path.length; time += 1) {
    joint *= model.transition[path[time - 1]][path[time]];
    joint *= localLikelihood(model, observations[time])[path[time]];
  }
  return joint;
}

/** Whether a named path uses only edges the model permits. */
export function pathLegal(model, path) {
  checkModel(model);
  if (model.start[path[0]] === 0) return false;
  for (let time = 1; time < path.length; time += 1) {
    if (model.transition[path[time - 1]][path[time]] === 0) return false;
  }
  return true;
}

/** Every path, with its joint probability. Exponential, so explicitly bounded. */
export function enumeratePaths(model, observations, envelope = envelopes.toy) {
  const { symbols } = checkModel(model, envelope);
  checkObservations(observations, symbols, envelope);
  const states = model.start.length;
  const total = states ** observations.length;
  if (total > limits.maximumEnumeratedPaths) {
    throw new RangeError(`${total} paths is more than the ${limits.maximumEnumeratedPaths} this lesson enumerates; `
      + 'the trellis exists precisely so that this list never has to be built.');
  }
  const paths = [];
  for (let code = 0; code < total; code += 1) {
    const path = [];
    let remainder = code;
    for (let time = observations.length - 1; time >= 0; time -= 1) {
      path[time] = remainder % states;
      remainder = Math.floor(remainder / states);
    }
    paths.push({ path, joint: pathJointOf(model, path, observations) });
  }
  const evidence = sum(paths.map(entry => entry.joint));
  const largest = paths.reduce((best, entry) => (entry.joint > best.joint ? entry : best));
  return {
    paths,
    count: total,
    evidence,
    largest,
    /** The share of posterior mass that does NOT belong to the best path. */
    remainingPosterior: evidence === 0 ? null : 1 - largest.joint / evidence,
  };
}

/* ============================================ expected counts and one update */

/** Fractional start, transition and emission events over independent recordings.
 *
 * The boundary rule is the point of section 6: every recording contributes one
 * start, T emissions and T−1 within-recording transitions. Nothing is added
 * across a boundary, so a length-one recording contributes no transition at all.
 */
export function expectedCounts(model, sequences) {
  const { symbols } = checkModel(model);
  if (!Array.isArray(sequences) || sequences.length === 0) {
    throw new RangeError('Expected counts need at least one recording.');
  }
  if (sequences.length > limits.maximumSequences) {
    throw new RangeError(`Keep to at most ${limits.maximumSequences} independent recordings.`);
  }
  const states = model.start.length;
  const initial = zeros(states);
  const edge = matrix(states, states);
  const symbol = matrix(states, symbols);
  let logLikelihood = 0;
  let observedEmissions = 0;
  let transitions = 0;
  const perSequence = sequences.map(observations => {
    const result = infer(model, observations);
    if (result.impossible) {
      throw new RangeError('One of these recordings is impossible under the model, so it has no expected counts.');
    }
    result.smoothed[0].forEach((mass, state) => { initial[state] += mass; });
    result.pair.forEach(block => {
      block.forEach((row, origin) => row.forEach((mass, destination) => { edge[origin][destination] += mass; }));
    });
    observations.forEach((value, time) => {
      if (value === MISSING) return;
      observedEmissions += 1;
      result.smoothed[time].forEach((mass, state) => { symbol[state][value] += mass; });
    });
    transitions += observations.length - 1;
    logLikelihood += result.logEvidence;
    return { observations, result };
  });
  return {
    initial,
    edge,
    symbol,
    logLikelihood,
    perSequence,
    totals: {
      starts: sequences.length,
      transitions,
      /** Emissions counted only where a report exists, matching the M-step. */
      emissions: observedEmissions,
      startMass: sum(initial),
      edgeMass: sum(edge.map(sum)),
      symbolMass: sum(symbol.map(sum)),
    },
  };
}

/** Rows of expected counts become probability rows; an unvisited row is kept. */
export function normalizeCounts(counts, previous) {
  return counts.map((row, index) => {
    const total = sum(row);
    return total > 0 ? row.map(value => value / total) : previous[index].slice();
  });
}

/** One complete exact EM update over independent recordings. */
export function emStep(model, sequences) {
  const counts = expectedCounts(model, sequences);
  const startTotal = sum(counts.initial);
  if (startTotal === 0) throw new RangeError('No start mass, so there is no initial-state update.');
  const updated = {
    ...model,
    start: counts.initial.map(value => value / startTotal),
    transition: normalizeCounts(counts.edge, model.transition),
    emission: normalizeCounts(counts.symbol, model.emission),
  };
  return {
    counts,
    model: updated,
    logLikelihoodBefore: counts.logLikelihood,
    logLikelihoodAfter: expectedCounts(updated, sequences).logLikelihood,
  };
}

/* ============================================== duration and parameter count */

/** The geometric dwell-time family implied by one self-transition probability.
 *
 * The first eight bars deliberately do NOT sum to one: the tail mass a^8 is the
 * whole reason a duration model is a modelling assumption rather than a
 * histogram, so it is returned as its own quantity instead of being normalised
 * away. An absorbing state has no finite mean, which is said rather than
 * computed as a division by zero.
 */
export function durationModel(stay, bars = limits.durationBars) {
  checkRange(stay, limits.stay, 'the self-transition probability');
  if (!Number.isInteger(bars) || bars < 1) throw new RangeError('The bar count must be a positive whole number.');
  const absorbing = stay === 1;
  const probabilities = Array.from({ length: bars }, (unused, index) => stay ** index * (1 - stay));
  return {
    stay,
    absorbing,
    mean: absorbing ? null : 1 / (1 - stay),
    /** Constant, whatever the elapsed time: the memorylessness the lab grades. */
    exitProbability: 1 - stay,
    probabilities,
    tailMass: stay ** bars,
    conserved: Math.abs(sum(probabilities) + stay ** bars - 1) < 1e-12,
  };
}

/** Exit probability conditional on having already stayed `elapsed` steps. */
export function exitAfter(stay, elapsed) {
  checkRange(stay, limits.stay, 'the self-transition probability');
  if (!Number.isInteger(elapsed) || elapsed < 0) throw new RangeError('Elapsed steps must be a nonnegative whole number.');
  // P(D > elapsed) = stay^elapsed. At zero survival the conditional
  // probability is undefined, even though the transition kernel is specified.
  return stay === 0 && elapsed > 0 ? null : 1 - stay;
}

/** Free parameters of a fully unconstrained categorical HMM. */
export function parameterCount(states, symbols) {
  if (!Number.isInteger(states) || states < 1) throw new RangeError('States must be a positive whole number.');
  if (!Number.isInteger(symbols) || symbols < 1) throw new RangeError('Symbols must be a positive whole number.');
  return (states - 1) + states * (states - 1) + states * (symbols - 1);
}

/* =========================================== numerical scale, three routes */

/** Numerically stable log of a sum of exponentials, with an all-empty row said.
 *
 * An all −∞ row represents a zero sum. Subtracting −∞ from itself would create
 * a NaN, so the case is answered directly rather than being computed.
 */
export function logSumExp(values) {
  if (values.length === 0) throw new RangeError('logSumExp needs at least one value.');
  const largest = values.reduce((best, value) => (value > best ? value : best), -Infinity);
  if (largest === -Infinity) return -Infinity;
  return largest + Math.log(sum(values.map(value => Math.exp(value - largest))));
}

/** The same repeated-product evidence by ordinary float, by logarithm and by
 * retained scaling, so section 7's contrast is a calculation and not a claim. */
export function repeatedProduct(factor, count) {
  checkRange(factor, { minimum: 0, maximum: 1 }, 'the repeated factor');
  if (!Number.isInteger(count) || count < 1) throw new RangeError('The count must be a positive whole number.');
  const ordinary = factor ** count;
  return {
    factor,
    count,
    ordinary,
    /** True when the mathematical value is positive but the float rounds to 0. */
    underflowed: factor > 0 && ordinary === 0,
    logValue: count * Math.log(factor),
    scaledLogValue: sum(Array.from({ length: count }, () => Math.log(factor))),
  };
}

/* =============================================== declared lesson fixtures */

const weather = {
  id: 'weather',
  stateNames: ['Rainy', 'Sunny'],
  symbolNames: ['Walk', 'Shop', 'Clean'],
  start: [0.6, 0.4],
  transition: [[0.7, 0.3], [0.4, 0.6]],
  emission: [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
};

/** The three-state graph of section 5, whose only symbol is certain in every
 * state, so all of its information is in the path masses. */
const constrained = {
  id: 'constrained',
  stateNames: ['A', 'B', 'C'],
  symbolNames: ['same'],
  start: [0.4, 0.35, 0.25],
  transition: [[0, 0.5, 0.5], [1, 0, 0], [1, 0, 0]],
  emission: [[1], [1], [1]],
};

export const fixtures = {
  weather,
  constrained,
  /** Walk, Shop, Walk, Clean. */
  reports: [0, 1, 0, 2],
  /** The same prefix with the final Clean corrected to Walk. */
  correctedFinal: [0, 1, 0, 0],
  /** The middle report retained but absent, against the same step deleted. */
  missingMiddle: [0, MISSING, 0, 2],
  deletedMiddle: [0, 0, 2],
  /** A changed Rainy emission row: the model changes, the best path's joint
   * mass does not, and its posterior does. */
  changedRainyEmission: [0.2, 0.3, 0.5],
  /** Section 5's changed prior, under which both decoders agree. */
  changedConstrainedStart: [0.2, 0.55, 0.25],
  /** Two independent recordings whose lengths sum to four. */
  splitRecordings: [[0, 1], [0, 2]],
  joinedRecording: [[0, 1, 0, 2]],
  /** Practice 5: three recordings of lengths 4, 1 and 3. */
  practiceLengths: [4, 1, 3],
  /** Section 7's rare-event and impossible-event examples. */
  rareFactor: 0.01,
  rareCount: 400,
  representableFactor: 0.3,
  representableCount: 100,
  impossibleModel: {
    id: 'impossible',
    stateNames: ['S0', 'S1'],
    symbolNames: ['0', '1'],
    start: [0.6, 0.4],
    transition: [[0.7, 0.3], [0.4, 0.6]],
    emission: [[1, 0], [1, 0]],
  },
  impossibleObservations: [1],
  /** Declared duration settings of section 9. */
  durationDefault: 0.7,
  durationContrasts: [0, 0.7, 0.95],
  durationPractice: 0.8,
  /** The construction target of investigation A. */
  smoothedTarget: 0.41,
  /** The construction target of investigation B. */
  legalPathTarget: 0.3,
};

/** The model with one emission row replaced, built here so no component edits a
 * frozen fixture in place. */
export function withRainyEmission(row) {
  checkRow(row, 'the replacement Rainy emission row');
  return { ...weather, emission: [row.slice(), weather.emission[1].slice()] };
}
export function withStart(model, start) {
  checkRow(start, 'the replacement initial-state row');
  if (start.length !== model.start.length) throw new RangeError('The initial row needs one entry per state.');
  return { ...model, start: start.slice() };
}

/* ============================================ quantities the figures draw */

/** The filtered and smoothed rows of one state, as a paired bar track.
 *
 * Both series share the 0–1 scale, and each point carries the exact value, so a
 * reader comparing two bars is comparing the numbers rather than an impression.
 */
export function beliefTrack(model, observations, state = 0) {
  const result = infer(model, observations);
  if (result.impossible) return { impossible: true, rows: [], reason: result.reason };
  return {
    impossible: false,
    rows: observations.map((value, time) => ({
      time,
      observation: value,
      label: value === MISSING ? 'missing' : model.symbolNames[value],
      filtered: result.filtered[time][state],
      smoothed: result.smoothed[time][state],
      /** Exactly zero at the last step, where no later report remains. */
      difference: result.smoothed[time][state] - result.filtered[time][state],
    })),
    state,
    stateName: model.stateNames[state],
    evidence: result.evidence,
  };
}

/** The best path against everything else, as two shares of one posterior. */
export function pathShare(model, observations) {
  const result = infer(model, observations);
  if (result.impossible) return { impossible: true };
  return {
    impossible: false,
    path: result.path,
    joint: result.pathJoint,
    evidence: result.evidence,
    posterior: result.pathPosterior,
    remaining: 1 - result.pathPosterior,
    conserved: Math.abs(result.pathPosterior + (1 - result.pathPosterior) - 1) < 1e-12,
  };
}

/** Where every fractional event of section 6 goes, as the flow figure draws it.
 *
 * Start mass, edge mass and symbol mass are returned with their exact totals,
 * because the figure's claim is that fractional parts add to whole event counts:
 * one start per recording, one emission per observed report, one transition per
 * within-recording step.
 */
export function countFlow(model, sequences) {
  const counts = expectedCounts(model, sequences);
  const states = model.start.length;
  const symbols = model.emission[0].length;
  return {
    recordings: sequences.map((observations, index) => ({
      index,
      observations,
      labels: observations.map(value => (value === MISSING ? 'missing' : model.symbolNames[value])),
      startMass: counts.perSequence[index].result.smoothed[0].slice(),
      transitions: observations.length - 1,
    })),
    start: counts.initial.map((mass, state) => ({ state, name: model.stateNames[state], mass })),
    edges: Array.from({ length: states }, (unused, origin) =>
      Array.from({ length: states }, (unusedInner, destination) => ({
        origin, destination, mass: counts.edge[origin][destination],
      }))),
    symbols: Array.from({ length: states }, (unused, state) =>
      Array.from({ length: symbols }, (unusedInner, symbol) => ({
        state, symbol, mass: counts.symbol[state][symbol],
      }))),
    totals: counts.totals,
    logLikelihood: counts.logLikelihood,
    /** The conservation the figure exists to make visible. */
    conserved: Math.abs(counts.totals.startMass - counts.totals.starts) < 1e-9
      && Math.abs(counts.totals.edgeMass - counts.totals.transitions) < 1e-9
      && Math.abs(counts.totals.symbolMass - counts.totals.emissions) < 1e-9,
  };
}

/** The bars of the duration figure, each with its own exact value plus the tail. */
export function durationBars(stay, bars = limits.durationBars) {
  const model = durationModel(stay, bars);
  return {
    ...model,
    bars: model.probabilities.map((probability, index) => ({
      duration: index + 1,
      probability,
      /** Constant hazard: identical for every bar, which is the teaching point. */
      hazard: model.exitProbability,
    })),
  };
}

/* ======================================== quantities a drawing encodes */

/** The stroke width a trellis edge gets, as a function of the share of its
 * destination's incoming total that it carries.
 *
 * This lives here rather than in a component because a drawn width is a
 * quantitative claim: two edges carrying the same share must be drawn the same
 * width, a larger share must never be drawn thinner, and an edge carrying
 * nothing must not be drawn as a thin version of one that carries a little. A
 * zero share returns `null`, which the drawing renders as a distinctly styled
 * forbidden edge instead of a hairline.
 */
export const edgeWidthRange = { zero: 0.5, minimum: 0.9, maximum: 4.4 };
export function edgeWidth(share) {
  checkRange(share, { minimum: 0, maximum: 1 }, 'the edge share');
  /* A zero share is drawn THINNER than the smallest positive one, not thicker.
     Returning null here left the component with no attribute to set, so the
     line painted at the SVG initial 1px while a share of one part in a million
     painted at 0.9px - this lesson's own zero-versus-tiny distinction, inverted
     on screen. The function is now monotone non-decreasing across its whole
     domain and always returns a number. */
  if (share === 0) return edgeWidthRange.zero;
  return edgeWidthRange.minimum + (edgeWidthRange.maximum - edgeWidthRange.minimum) * share;
}

/** Every edge of one trellis with the share it carries and the width that
 * encodes it, so a figure never computes a proportion of its own. */
export function trellisEdges(model, observations, mode = 'sum') {
  const built = trellis(model, observations, mode);
  return built.columns.slice(1).map(column => ({
    time: column.time,
    edges: column.cells.flatMap(cell => cell.incoming.map(edge => {
      const share = cell.aggregate === 0 ? 0 : edge.carried / cell.aggregate;
      return {
        time: column.time,
        from: edge.from,
        to: cell.state,
        carried: edge.carried,
        transition: edge.transition,
        previousValue: edge.previousValue,
        share,
        width: edgeWidth(share),
        chosen: mode === 'max' && cell.chosen === edge.from,
        /** The model has no such transition at all. */
        forbidden: edge.transition === 0,
        /** The transition exists but nothing arrives along it, because the
         * evidence has ruled out its source. That is a different statement
         * from a forbidden edge and it is drawn differently. */
        carriesNothing: edge.carried === 0 && edge.transition !== 0,
      };
    })),
  }));
}

/** A bar's height as a fraction of its axis, so no component invents a scale.
 *
 * The axis maximum is passed in rather than derived per bar, because a set of
 * bars that must be compared has to share one scale, and a bar drawn against
 * its own maximum would make every set look identical.
 */
export function barHeight(value, axisMaximum) {
  checkFinite(value, 'the bar value');
  checkFinite(axisMaximum, 'the axis maximum');
  if (value < 0) throw new RangeError('A bar value must be nonnegative.');
  if (axisMaximum <= 0) throw new RangeError('The axis maximum must be positive.');
  if (value > axisMaximum) throw new RangeError('A bar cannot exceed its own axis.');
  return value / axisMaximum;
}

/** Two declared vertical windows for the EM objective, and the separation each
 * one actually gives the values the prose asks a reader to compare.
 *
 * The overview has to contain an initial score near -566 and therefore cannot
 * also resolve a gap of 0.4 between two final scores: at that range a linear
 * axis turns the whole interesting region into one flat line. So there are two
 * windows, and `separationInPixels` is what lets a verifier check that the
 * second one really does separate them at the size the figure renders, rather
 * than leaving "the curves differ" as a claim about a picture.
 */
export const objectiveWindows = {
  overview: { low: -570, high: -385, pixels: 96 },
  detail: { low: -396, high: -389.5, pixels: 96 },
};

export function separationInPixels(values, window) {
  const sorted = [...values].sort((left, right) => left - right);
  let smallest = Infinity;
  for (let index = 1; index < sorted.length; index += 1) {
    const gap = (sorted[index] - sorted[index - 1]) / (window.high - window.low) * window.pixels;
    if (gap < smallest) smallest = gap;
  }
  return smallest;
}

/** Whether a value lies inside a window, so a figure never silently clips a
 * quantity it claims to be drawing. */
export function insideWindow(value, window) {
  return value >= window.low && value <= window.high;
}

/* =============================================== the rules the labs grade with
 *
 * These live here, not inside a component, for one reason: a rule a learner is
 * graded against has to be exercisable over the whole grid of inputs the
 * control can reach. A comparison open-coded inside a render function can be
 * checked at the two values someone happened to think of; a function here can
 * be checked at every sequence, every prior and every boundary the interface
 * allows, and at the degenerate ones a designer would not think to try.
 *
 * Each returns a plain outcome string, and each is paired below with the
 * property the verifier asserts about it.
 */

/** Where one complete path stands among all of them.
 *
 * The order is load-bearing. An impossible path is reported as impossible even
 * when EVERY path is impossible, so a learner is never told that a
 * probability-zero story is "the most likely one" merely because nothing beats
 * it. Ties at the top all report `largest`, because they all are.
 */
export function pathRankOutcome(model, path, observations, envelope = envelopes.toy) {
  const joint = pathJointOf(model, path, observations, envelope);
  if (joint === 0) return 'zero';
  const enumeratedPaths = enumeratePaths(model, observations, envelope);
  return joint === enumeratedPaths.largest.joint ? 'largest' : 'smaller';
}

/** How the two beliefs about one time move between two committed recordings.
 *
 * Both are reported separately, because "something changed" is not an answer to
 * "which of these two changed": editing a report later than the queried time
 * cannot move the filtered value and must move nothing but the smoothed one.
 */
export function beliefMoveOutcomes(beforeModel, beforeObservations, afterModel, afterObservations, query) {
  const read = (model, observations) => {
    const index = Math.min(query, observations.length - 1);
    const result = infer(model, observations);
    if (result.impossible) return { filtered: null, smoothed: null };
    return { filtered: result.filtered[index][0], smoothed: result.smoothed[index][0] };
  };
  const before = read(beforeModel, beforeObservations);
  const after = read(afterModel, afterObservations);
  return {
    outcomes: {
      filtering: direction(before.filtered, after.filtered),
      smoothing: direction(before.smoothed, after.smoothed),
    },
    before,
    after,
  };
}

/** Whether joining the highest-marginal cell at each time gives the
 * highest-probability path the model permits.
 *
 * Two ways for the answer to be no, and they are different: the pointwise route
 * can be illegal, or it can be legal but beaten. Both return `'no'`, and the
 * accompanying record says which, so the feedback can too.
 */
export function legalPathGuarantee(model, observations, envelope = envelopes.toy) {
  const result = infer(model, observations, envelope);
  const best = bestLegalPath(model, observations, envelope);
  const legal = pathLegal(model, result.marginalModes);
  const modesJoint = pathJointOf(model, result.marginalModes, observations, envelope);
  return {
    outcome: legal && modesJoint === best.joint ? 'yes' : 'no',
    reason: legal ? (modesJoint === best.joint ? 'they coincide' : 'legal but beaten') : 'the modes are not a path',
    modes: result.marginalModes,
    modesJoint,
    best,
  };
}

/** How many events of each kind a set of independent recordings contributes.
 *
 * One start per recording, one transition per step inside a recording, and one
 * emission per OBSERVED report. A length-one recording therefore contributes no
 * transition, and a retained missing report contributes no emission while still
 * contributing its transition.
 */
export function recordingTotals(recordings) {
  if (!Array.isArray(recordings) || recordings.length === 0) {
    throw new RangeError('There has to be at least one recording.');
  }
  recordings.forEach((row, index) => {
    if (!Array.isArray(row) || row.length === 0) throw new RangeError(`Recording ${index} has no time steps.`);
  });
  return {
    starts: recordings.length,
    transitions: recordings.reduce((total, row) => total + row.length - 1, 0),
    emissions: recordings.reduce((total, row) => total + row.filter(value => value !== MISSING).length, 0),
  };
}

/** Whether a state that has already lasted longer leaves sooner.
 *
 * Equal hazards require possible conditioning histories. At a=0, any positive
 * elapsed duration is impossible; at a=1 both defined exit probabilities are 0.
 */
export function hazardOutcome(stay, shortElapsed, longElapsed) {
  const early = exitAfter(stay, shortElapsed);
  const late = exitAfter(stay, longElapsed);
  if (early === null || late === null) return 'undefined';
  const moved = direction(early, late);
  return moved === 'unchanged' ? 'same' : (moved === 'rises' ? 'higher' : 'lower');
}

/* ------------------------------------------------- construction task graders */

/** Investigation A: change only the final report, move the smoothed belief
 * below a threshold, leave the filtered belief where it was. */
export function gradeSmoothedConstruction({ observations, modelEdited }, reference, target, query = 1) {
  const original = infer(weather, reference);
  const prefixKept = observations.length === reference.length
    && observations.slice(0, -1).every((value, index) => value === reference[index]);
  const result = modelEdited ? null : infer(weather, observations);
  const usable = result && !result.impossible && query >= 0 && query < observations.length;
  const smoothed = usable ? result.smoothed[query][0] : null;
  const filtered = usable ? result.filtered[query][0] : null;
  const held = filtered !== null && Math.abs(filtered - original.filtered[query][0]) <= limits.nullTolerance;
  return {
    solved: Boolean(!modelEdited && prefixKept && smoothed !== null && smoothed < target && held),
    modelEdited, prefixKept, filtered, smoothed, held,
  };
}

/** Investigation B: a path the model permits whose joint probability clears a
 * declared floor. */
export function gradeLegalPathConstruction(model, path, observations, target, envelope = envelopes.toy) {
  const legal = pathLegal(model, path);
  const joint = pathJointOf(model, path, observations, envelope);
  return { solved: legal && joint >= target, legal, joint };
}

/** Investigation C: restore the declared two-session split.
 *
 * Both the event totals and the session lengths are checked, because a
 * one-and-three split reaches the same three totals and is a different claim
 * about what was recorded. Checking the totals alone would accept it.
 */
export function gradeBoundaryConstruction(recordings, declaredLengths) {
  const totals = recordingTotals(recordings);
  const lengths = recordings.map(row => row.length);
  const expected = {
    starts: declaredLengths.length,
    transitions: declaredLengths.reduce((total, value) => total + value - 1, 0),
    emissions: declaredLengths.reduce((total, value) => total + value, 0),
  };
  const totalsMatch = totals.starts === expected.starts
    && totals.transitions === expected.transitions
    && totals.emissions === expected.emissions;
  const structureMatches = lengths.length === declaredLengths.length
    && lengths.every((value, index) => value === declaredLengths[index]);
  return { solved: totalsMatch && structureMatches, totals, expected, lengths, totalsMatch, structureMatches };
}
