/** Independent checks for the ICA lesson's model layer.
 *
 * Every oracle here is written from the manuscript's algebra, not from the
 * production helper it checks: eigenpairs are confirmed by applying the matrix,
 * the fixed-point step is recomputed by an explicit loop, and the projection
 * kurtosis uses the identity cos^4 t + sin^4 t = 1 - 2 cos^2 t sin^2 t rather
 * than the curve the page plots. A figure's geometry is a mathematical claim,
 * so the whitener and the eigen-decomposition that place its points are checked
 * on degenerate inputs too.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  MIXING, SOURCE_STATES, STATE_PROBABILITY, CONTRASTS, SOURCE_FAMILIES, KEEP_LABELS, DEPENDENCE_POINTS,
  applyMatrix, multiplyMatrices, invert2, covarianceOfRows, symmetricEigen2, whitenerFromCovariance,
  whitenedFixture, fastIcaStep, deflate, projectionKurtosis, rotationModel, gradeRotation,
  contributionModel, gradeContribution, rescaleComponent, dependenceSummary, absoluteCorrelation,
  checkEnteredAmplitude, ENTERED_AMPLITUDE_LIMIT, MODEL_AMPLITUDE_LIMIT,
} from '../src/learn/data/ica-models.js';
import { ICA_RECORDING } from '../src/learn/data/ica-data.js';

let assertions = 0;
const groups = [];
const close = (actual, expected, tolerance = 1e-12) => {
  assertions += 1;
  assert.ok(Number.isFinite(actual), `${actual} is not finite`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected} by ${Math.abs(actual - expected)}`);
};
const closeVector = (actual, expected, tolerance = 1e-12) => expected.forEach((value, index) => close(actual[index], value, tolerance));
const equal = (actual, expected) => { assertions += 1; assert.deepEqual(actual, expected); };
const rejects = (fn, message) => { assertions += 1; assert.throws(fn, undefined, message); };
const checked = (name, fn) => { fn(); groups.push(name); };

const SQRT2 = Math.SQRT2;

checked('Mixing algebra: one sample, inverse rows, transpose safety and singular refusal', () => {
  closeVector(applyMatrix(MIXING, [1, -1]), [1, -1]);
  closeVector(applyMatrix(MIXING, [-1, -1]), [-3, -3]);
  closeVector(applyMatrix(MIXING, [-1, 1]), [-1, 1]);
  closeVector(applyMatrix(MIXING, [1, 1]), [3, 3]);
  const inverse = invert2(MIXING);
  closeVector(inverse[0], [2 / 3, -1 / 3]);
  closeVector(inverse[1], [-1 / 3, 2 / 3]);
  closeVector(applyMatrix(inverse, [1, -1]), [1, -1]);
  multiplyMatrices(inverse, MIXING).forEach((row, i) => row.forEach((value, j) => close(value, i === j ? 1 : 0)));
  // Practice 1: a different sensor recipe with the same algebra.
  closeVector(applyMatrix(invert2([[3, 1], [1, 1]]), [5, -1]), [3, -4]);
  closeVector(applyMatrix([[1.5, 1], [0.5, 1]], [6, -4]), [5, -1]);
  // A mixing column is not an inverse row: (2,1) against (2,-1)/3.
  closeVector([MIXING[0][0], MIXING[1][0]], [2, 1]);
  closeVector(inverse[0], [2 / 3, -1 / 3]);
  rejects(() => invert2([[1, 2], [2, 4]]), 'singular mixture');
  rejects(() => invert2([[1, 2], [2, Number.NaN]]));
  rejects(() => applyMatrix(MIXING, [1]));
  rejects(() => applyMatrix([[1, 2, 3], [1, 2, 3]], [1, 1]));
});

checked('Symmetric eigen-decomposition: applied-matrix oracle, ordering, diagonal and isotropic branches', () => {
  const cases = [[[5, 4], [4, 5]], [[9, 0], [0, 1]], [[1, 0], [0, 9]], [[2, 0], [0, 2]], [[3, -1], [-1, 3]],
    [[10, 3], [3, 2]], [[0.25, 0.1], [0.1, 4]], [[1e-6, 0], [0, 5]]];
  for (const matrix of cases) {
    const { values, vectors } = symmetricEigen2(matrix);
    assertions += 1;
    assert.ok(values[0] >= values[1], 'eigenvalues descend');
    close(values[0] + values[1], matrix[0][0] + matrix[1][1]);
    close(values[0] * values[1], matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
    // The definition, applied: A v must equal lambda v for each returned pair.
    vectors.forEach((vector, index) => {
      const image = applyMatrix(matrix, vector);
      closeVector(image, vector.map(value => value * values[index]), 1e-9);
      close(Math.hypot(vector[0], vector[1]), 1);
    });
    close(vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1], 0, 1e-12);
    // The diagonal branch must not hand back one direction twice.
    assertions += 1;
    assert.ok(Math.abs(vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]) < 1e-12, 'two distinct axes');
  }
  closeVector(symmetricEigen2([[5, 4], [4, 5]]).values, [9, 1]);
  closeVector(symmetricEigen2([[5, 4], [4, 5]]).vectors[0], [Math.SQRT1_2, Math.SQRT1_2]);
  closeVector(symmetricEigen2([[5, 4], [4, 5]]).vectors[1], [Math.SQRT1_2, -Math.SQRT1_2]);
  equal(symmetricEigen2([[9, 0], [0, 1]]).vectors, [[1, 0], [0, 1]]);
  equal(symmetricEigen2([[1, 0], [0, 9]]).vectors, [[0, 1], [1, 0]]);
  equal(symmetricEigen2([[2, 0], [0, 2]]).vectors, [[1, 0], [0, 1]]);
  rejects(() => symmetricEigen2([[1, 2], [3, 4]]), 'asymmetric input');
  rejects(() => whitenerFromCovariance([[1, 1], [1, 1]]), 'a zero eigenvalue cannot be whitened');
  rejects(() => whitenerFromCovariance([[0, 0], [0, 0]]));
  for (const covariance of cases.filter(matrix => matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2 > 1e-9)) {
    const { whitener } = whitenerFromCovariance(covariance);
    const product = multiplyMatrices(multiplyMatrices(whitener, covariance), [[whitener[0][0], whitener[1][0]], [whitener[0][1], whitener[1][1]]]);
    product.forEach((row, i) => row.forEach((value, j) => close(value, i === j ? 1 : 0, 1e-9)));
  }
});

checked('Four-state fixture: exact covariance, whitened diamond, absent joint zero and recovery', () => {
  const fixture = whitenedFixture();
  equal(fixture.states.map(state => state.id), ['A', 'B', 'C', 'D']);
  close(STATE_PROBABILITY * 4, 1);
  closeVector(fixture.mean, [0, 0]);
  closeVector(fixture.covariance[0], [5, 4]);
  closeVector(fixture.covariance[1], [4, 5]);
  closeVector(fixture.eigenvalues, [9, 1]);
  closeVector(fixture.whitener[0], [1 / (3 * SQRT2), 1 / (3 * SQRT2)]);
  closeVector(fixture.whitener[1], [1 / SQRT2, -1 / SQRT2]);
  const expectedWhitened = [[-SQRT2, 0], [0, -SQRT2], [0, SQRT2], [SQRT2, 0]];
  fixture.states.forEach((state, index) => {
    closeVector(state.observed, [[-3, -3], [-1, 1], [1, -1], [3, 3]][index]);
    closeVector(state.whitened, expectedWhitened[index]);
    closeVector(state.recovered, state.source);
  });
  fixture.whitenedCovariance.forEach((row, i) => row.forEach((value, j) => close(value, i === j ? 1 : 0)));
  // Sensor 1's average squared value and the average sensor product, by hand.
  close((9 + 1 + 1 + 9) / 4, fixture.covariance[0][0]);
  close((9 - 1 - 1 + 9) / 4, fixture.covariance[0][1]);
  // Each whitened marginal is zero half the time, yet never both at once.
  [0, 1].forEach(axis => close(fixture.states.filter(state => Math.abs(state.whitened[axis]) < 1e-12).length / 4, 0.5));
  assertions += 1;
  assert.equal(fixture.states.filter(state => state.whitened.every(value => Math.abs(value) < 1e-12)).length, 0, 'no probability mass at the joint origin');
  // The orthogonal recovery includes a reflection: determinant -1.
  close(fixture.recovery[0][0] * fixture.recovery[1][1] - fixture.recovery[0][1] * fixture.recovery[1][0], -1);
  multiplyMatrices(fixture.recovery, [[fixture.recovery[0][0], fixture.recovery[1][0]], [fixture.recovery[0][1], fixture.recovery[1][1]]])
    .forEach((row, i) => row.forEach((value, j) => close(value, i === j ? 1 : 0)));
  // Q = K A is orthogonal, which is why an orthogonal search after whitening suffices.
  const Q = multiplyMatrices(fixture.whitener, MIXING);
  multiplyMatrices(Q, [[Q[0][0], Q[1][0]], [Q[0][1], Q[1][1]]]).forEach((row, i) => row.forEach((value, j) => close(value, i === j ? 1 : 0)));
  // B = W K recovers the sources directly from centred observations.
  fixture.states.forEach(state => closeVector(applyMatrix(fixture.unmixing, state.observed), state.source));
  const moved = covarianceOfRows([[1, 1], [3, 3], [-1, -1], [-3, -3]]);
  closeVector(moved.mean, [0, 0]);
  rejects(() => covarianceOfRows([[1, 1]]), 'a single row has no covariance here');
  rejects(() => covarianceOfRows([[1, 1], [2, Number.POSITIVE_INFINITY]]));
  rejects(() => whitenedFixture([[2, 4], [1, 2]]), 'a singular mixture cannot be whitened into a full-rank fixture');
});

checked('Fixed-point update: exact section-5 trace, independent recomputation, contrasts and zero-update refusal', () => {
  const fixture = whitenedFixture();
  const points = fixture.states.map(state => state.whitened);
  const step = fastIcaStep(points, [0.8, 0.6]);
  closeVector(step.rows.map(row => row.projection), [-0.8 * SQRT2, -0.6 * SQRT2, 0.6 * SQRT2, 0.8 * SQRT2]);
  closeVector(step.weightedMean, [1.024, 0.432], 1e-12);
  close(step.derivativeMean, 3);
  closeVector(step.correction, [2.4, 1.8]);
  closeVector(step.raw, [-1.376, -1.368], 1e-12);
  closeVector(step.next, [-0.7091653, -0.70504225], 5e-8);
  closeVector(step.signAligned, [0.7091653, 0.70504225], 5e-8);
  close(Math.hypot(step.next[0], step.next[1]), 1);
  // Independent recomputation with an explicit loop, no shared helper.
  let sumX = 0; let sumY = 0; let sumDerivative = 0;
  for (const point of points) {
    const y = point[0] * 0.8 + point[1] * 0.6;
    sumX += point[0] * y ** 3;
    sumY += point[1] * y ** 3;
    sumDerivative += 3 * y ** 2;
  }
  closeVector(step.weightedMean, [sumX / 4, sumY / 4]);
  close(step.derivativeMean, sumDerivative / 4);
  // The update moved toward (1,1)/sqrt2, the direction that extracts s1.
  assertions += 1;
  assert.ok(step.signAligned[0] * Math.SQRT1_2 + step.signAligned[1] * Math.SQRT1_2 > 0.999, 'moves toward the source axis');
  const settled = fastIcaStep(points, [Math.SQRT1_2, Math.SQRT1_2]);
  close(settled.convergence, 0, 1e-12);
  closeVector(settled.signAligned, [Math.SQRT1_2, Math.SQRT1_2], 1e-12);
  // Practice 3, program A: a linear contrast gives exactly the zero update.
  rejects(() => fastIcaStep(points, [0.8, 0.6], 'linear'), 'linear contrast is the documented failure');
  close(fastIcaStep(points, [0.8, 0.6], 'tanh').convergence >= 0 ? 1 : 0, 1);
  equal(Object.keys(CONTRASTS), ['cube', 'tanh', 'linear']);
  rejects(() => fastIcaStep(points, [0.8, 0.6], 'quartic'));
  rejects(() => fastIcaStep(points, [0.8, 0.8]), 'the direction must be a unit vector');
  rejects(() => fastIcaStep([[1, 1]], [1, 0]));
  // Deflation inside the iteration, not after convergence.
  const first = [Math.SQRT1_2, Math.SQRT1_2];
  const deflated = deflate([0.9, 0.1], [first]);
  close(deflated.unit[0] * first[0] + deflated.unit[1] * first[1], 0, 1e-12);
  close(Math.hypot(deflated.unit[0], deflated.unit[1]), 1);
  rejects(() => deflate([Math.SQRT1_2, Math.SQRT1_2], [first]), 'a zero residual needs diagnosis, not normalization');
  rejects(() => deflate([1, 0], [[2, 0]]));
});

checked('Projection kurtosis: independent identity, every family, symmetry, endpoints and refusals', () => {
  const identity = (kappa, degrees) => {
    const radians = (degrees * Math.PI) / 180;
    const a = Math.cos(radians); const b = Math.sin(radians);
    return kappa * (1 - 2 * a * a * b * b);
  };
  const table = { binary: [-2, -1.25, -1, -2], laplace: [3, 1.875, 1.5, 3], gaussian: [0, 0, 0, 0] };
  for (const [family, expected] of Object.entries(table)) {
    [0, 30, 45, 90].forEach((angle, index) => {
      close(projectionKurtosis(family, angle), expected[index], 1e-12);
      close(projectionKurtosis(family, angle), identity(SOURCE_FAMILIES[family].kurtosis, angle), 1e-12);
    });
    for (let angle = 0; angle <= 180; angle += 0.5) {
      close(projectionKurtosis(family, angle), identity(SOURCE_FAMILIES[family].kurtosis, angle), 1e-12);
      close(projectionKurtosis(family, angle), projectionKurtosis(family, 180 - angle), 1e-12);
      if (angle <= 90) close(projectionKurtosis(family, angle), projectionKurtosis(family, angle + 90), 1e-12);
    }
    // Arbitrary decimals are permitted, and the extremes sit where they should.
    close(projectionKurtosis(family, 12.3456), identity(SOURCE_FAMILIES[family].kurtosis, 12.3456), 1e-12);
    assertions += 1;
    const magnitudes = Array.from({ length: 361 }, (unused, index) => Math.abs(projectionKurtosis(family, index / 2)));
    assert.ok(Math.abs(Math.max(...magnitudes) - Math.abs(SOURCE_FAMILIES[family].kurtosis)) < 1e-12, 'source axes are the maxima');
  }
  // Practice 2: weights sqrt3/2 and 1/2 on two Laplace sources.
  close(Math.hypot(Math.sqrt(3) / 2, 0.5), 1);
  close(3 * ((Math.sqrt(3) / 2) ** 4 + 0.5 ** 4), 1.875, 1e-12);
  close(projectionKurtosis('laplace', 30), 1.875, 1e-12);
  close(projectionKurtosis('laplace', 60), 1.875, 1e-12);
  // Equal source kurtoses leave a directional contrast, not a flat ring.
  assertions += 1;
  assert.ok(Math.abs(projectionKurtosis('laplace', 45)) < Math.abs(projectionKurtosis('laplace', 0)), 'no ring of equal maxima');
  rejects(() => projectionKurtosis('laplace', 181));
  rejects(() => projectionKurtosis('laplace', -1));
  rejects(() => projectionKurtosis('laplace', Number.NaN));
  rejects(() => projectionKurtosis('cauchy', 30));
});

checked('Rotation investigation: unit covariance at every angle, exact support, contours and grading', () => {
  for (const family of ['binary', 'laplace', 'gaussian']) {
    for (const angle of [0, 17.5, 30, 45, 90, 123.75, 180]) {
      const model = rotationModel(family, angle);
      close(model.covariance[0][0], 1);
      close(model.covariance[1][1], 1);
      close(model.covariance[0][1], 0);
      close(model.kurtosis, projectionKurtosis(family, angle), 1e-12);
      close(Math.hypot(model.direction[0], model.direction[1]), 1);
      close(model.direction[0] * model.companionDirection[0] + model.direction[1] * model.companionDirection[1], 0, 1e-12);
      close(model.companionKurtosis, model.kurtosis, 1e-12);
      assertions += 1;
      assert.equal(model.curve.length, 361);
      close(model.curve[0].angle, 0);
      close(model.curve.at(-1).angle, 180);
      model.curve.forEach(point => close(point.kurtosis, projectionKurtosis(family, point.angle), 1e-12));
      if (family === 'binary') {
        assertions += 1;
        assert.equal(model.support.length, 4);
        model.support.forEach(state => {
          close(state.probability, 0.25);
          close(Math.hypot(state.projected[0], state.projected[1]), Math.hypot(state.source[0], state.source[1]), 1e-12);
        });
        close(model.support.reduce((total, state) => total + state.probability, 0), 1);
        // The rotated support really is the projected pair, by hand.
        const radians = (angle * Math.PI) / 180;
        model.support.forEach(state => closeVector(state.projected, [
          Math.cos(radians) * state.source[0] + Math.sin(radians) * state.source[1],
          -Math.sin(radians) * state.source[0] + Math.cos(radians) * state.source[1],
        ], 1e-12));
        equal(model.contours, []);
      } else {
        assertions += 1;
        assert.equal(model.contours.length, 3);
        equal(model.contours.map(contour => contour.level), [1, 2, 3]);
        assert.equal(model.support.length, 0);
        model.contours.forEach(contour => contour.points.forEach(point => {
          if (family === 'gaussian') close(Math.hypot(point[0], point[1]), contour.level, 1e-9);
        }));
        if (family === 'laplace') {
          model.contours.forEach(contour => {
            assertions += 1;
            assert.equal(contour.points.length, 4);
            // A rotated level set keeps its distance from the origin.
            contour.points.forEach(point => close(Math.hypot(point[0], point[1]), contour.level, 1e-12));
          });
        }
      }
    }
  }
  // Graded against the committed snapshot's angles, never the rendered state.
  equal(gradeRotation({ family: 'binary', oldAngle: 45, newAngle: 30, prediction: 'larger' }).correct, true);
  close(gradeRotation({ family: 'binary', oldAngle: 45, newAngle: 30, prediction: 'larger' }).after, 1.25, 1e-12);
  equal(gradeRotation({ family: 'binary', oldAngle: 0, newAngle: 45, prediction: 'smaller' }).correct, true);
  equal(gradeRotation({ family: 'binary', oldAngle: 0, newAngle: 90, prediction: 'same' }).correct, true);
  equal(gradeRotation({ family: 'laplace', oldAngle: 45, newAngle: 30, prediction: 'larger' }).correct, true);
  equal(gradeRotation({ family: 'laplace', oldAngle: 30, newAngle: 60, prediction: 'same' }).correct, true);
  equal(gradeRotation({ family: 'binary', oldAngle: 45, newAngle: 30, prediction: 'smaller' }).correct, false);
  for (const angle of [0, 22.5, 45, 90, 180]) {
    equal(gradeRotation({ family: 'gaussian', oldAngle: angle, newAngle: (angle + 37.5) % 180, prediction: 'same' }).actual, 'same');
  }
  equal(gradeRotation({ family: 'binary', oldAngle: 30, newAngle: 30, prediction: 'same' }).correct, true);
  rejects(() => gradeRotation({ family: 'binary', oldAngle: 30, newAngle: 30, prediction: 'bigger' }));
  rejects(() => gradeRotation(null));
  rejects(() => rotationModel('binary', 45, 400));
  rejects(() => rotationModel('binary', 45, 2));
});

checked('Component contributions: every keep-set, exact nulls, compensated scaling and invalid zero', () => {
  equal(Object.keys(KEEP_LABELS), ['both', 'first', 'second', 'none']);
  const manuscript = contributionModel({ sources: [1, -1], keep: 'first' });
  closeVector(manuscript.observed, [1, -1]);
  closeVector(manuscript.retained, [2, 1]);
  closeVector(manuscript.removed, [-1, -2]);
  closeVector(manuscript.contributions[0].contribution, [2, 1]);
  closeVector(manuscript.contributions[1].contribution, [-1, -2]);
  close(manuscript.contributions[0].columnNorm ** 2, 5);
  close(manuscript.contributions[1].columnNorm ** 2, 5);
  // Independent changed sources from the specification's fixture list.
  const expected = { none: [0, 0], first: [4, 2], second: [-3, -6], both: [1, -4] };
  for (const [keep, retained] of Object.entries(expected)) {
    const model = contributionModel({ sources: [2, -3], keep });
    closeVector(model.observed, [1, -4]);
    closeVector(model.retained, retained);
    closeVector(model.removed, [1 - retained[0], -4 - retained[1]]);
  }
  // Removing source 2 is a null operation when its amplitude is zero.
  closeVector(contributionModel({ sources: [2, 0], keep: 'first' }).retained, contributionModel({ sources: [2, 0], keep: 'both' }).retained);
  closeVector(contributionModel({ sources: [2, 0], keep: 'second' }).retained, [0, 0]);
  for (const keep of Object.keys(KEEP_LABELS)) closeVector(contributionModel({ sources: [0, 0], keep }).retained, [0, 0]);
  // Dropping a component can make sensor 1 larger when source 2 is negative.
  assertions += 1;
  assert.ok(contributionModel({ sources: [1, -1], keep: 'first' }).retained[0] > contributionModel({ sources: [1, -1], keep: 'both' }).retained[0]);
  // Scale ambiguity: compensated rescaling leaves every product unchanged.
  for (const sources of [[2, -3], [2, 0], [0, 0], [1, -1]]) {
    for (const index of [0, 1]) {
      for (const scale of [2, -2, 0.25, -0.5, 4]) {
        const rescaled = rescaleComponent({ sources, index, scale });
        for (const keep of Object.keys(KEEP_LABELS)) {
          closeVector(contributionModel({ ...rescaled, keep }).retained, contributionModel({ sources, keep }).retained, 1e-12);
        }
        close(rescaled.sources[index], sources[index] * scale, 1e-12);
        close(rescaled.mixing[0][index], MIXING[0][index] / scale, 1e-12);
      }
    }
  }
  // Scaling the source alone changes the observation; that is the point.
  const sourceOnly = rescaleComponent({ sources: [2, -3], index: 0, scale: 2, compensate: false });
  closeVector(contributionModel({ ...sourceOnly, keep: 'both' }).observed, [5, -2]);
  equal(sourceOnly.mixing, MIXING);
  // Practice 5: excluding a candidate removes the wanted activity it carries.
  closeVector(contributionModel({ mixing: [[2, 0], [-1, 0]], sources: [0.2, 0], keep: 'none' }).removed, [0.4, -0.2]);
  rejects(() => rescaleComponent({ sources: [1, -1], index: 0, scale: 0 }), 'a zero scale cannot be compensated');
  rejects(() => rescaleComponent({ sources: [1, -1], index: 0, scale: 0.1 }));
  rejects(() => rescaleComponent({ sources: [1, -1], index: 2, scale: 2 }));
  rejects(() => checkEnteredAmplitude(5, 'Source 1 amplitude'), 'entered amplitudes stay inside the declared range');
  rejects(() => contributionModel({ sources: [17, 0], keep: 'both' }), 'the model refuses an amplitude no control can produce');
  equal(checkEnteredAmplitude(-4), -4);
  equal([ENTERED_AMPLITUDE_LIMIT, MODEL_AMPLITUDE_LIMIT], [4, 16]);
  rejects(() => contributionModel({ sources: [1, -1], keep: 'all' }));
  rejects(() => contributionModel({ sources: [1, Number.NaN], keep: 'both' }));
  // Grading uses the committed amplitudes, not whatever is on screen.
  const graded = gradeContribution({ sources: [1, -1], keep: 'first', prediction: 2, sensor: 0 });
  equal(graded.correct, true);
  close(graded.actual, 2);
  equal(gradeContribution({ sources: [2, -3], keep: 'second', prediction: -3, sensor: 0 }).correct, true);
  equal(gradeContribution({ sources: [2, -3], keep: 'second', prediction: -6, sensor: 1 }).correct, true);
  equal(gradeContribution({ sources: [2, -3], keep: 'second', prediction: -5.9999, sensor: 1 }).correct, false);
  equal(gradeContribution({ sources: [2, -3], keep: 'second', prediction: -6 + 5e-10, sensor: 1 }).correct, true);
  rejects(() => gradeContribution({ sources: [1, -1], keep: 'first', prediction: Number.NaN, sensor: 0 }));
  rejects(() => gradeContribution({ sources: [1, -1], keep: 'first', prediction: 2, sensor: 3 }));
});

checked('Zero covariance with complete dependence, and the correlation diagnostic', () => {
  const summary = dependenceSummary();
  equal(summary.points.map(point => point.product), [-1, 0, 1]);
  close(summary.meanU, 0);
  close(summary.meanV, 2 / 3);
  close(summary.covariance, 0);
  close(summary.jointZero, 1 / 3);
  close(summary.marginalProduct, 1 / 9);
  equal(summary.independent, false);
  equal(DEPENDENCE_POINTS.length, 3);
  // An actually independent pair passes the same test.
  const independent = dependenceSummary([
    { u: -1, v: 0, probability: 0.25 }, { u: -1, v: 1, probability: 0.25 },
    { u: 0, v: 0, probability: 0.25 }, { u: 0, v: 1, probability: 0.25 },
  ]);
  close(independent.covariance, 0);
  equal(independent.independent, true);
  rejects(() => dependenceSummary([{ u: 0, v: 0, probability: 0.5 }]));
  rejects(() => dependenceSummary([{ u: 0, v: 0, probability: 0.5 }, { u: 1, v: 1, probability: 0.4 }]));
  rejects(() => dependenceSummary([{ u: 0, v: 0, probability: -1 }, { u: 1, v: 1, probability: 2 }]));
  close(absoluteCorrelation([1, 2, 3, 4], [2, 4, 6, 8]), 1, 1e-12);
  close(absoluteCorrelation([1, 2, 3, 4], [-2, -4, -6, -8]), 1, 1e-12);
  close(absoluteCorrelation([1, 2, 3, 4], [1, 3, 2, 4]), 0.8, 1e-12);
  // A positive affine calibration cannot change a correlation.
  close(absoluteCorrelation([1, 2, 3, 4].map(v => (v + 32768) * (6553.6 / 65535) - 3276.8), [1, 3, 2, 4]), 0.8, 1e-12);
  rejects(() => absoluteCorrelation([1, 2], [1]));
  rejects(() => absoluteCorrelation([1, 1, 1], [1, 2, 3]), 'a constant series has no correlation');
});

checked('Published recording module: selection rule, frozen signs, envelope bounds and stated limits', () => {
  equal(ICA_RECORDING.samples, 20000);
  equal(ICA_RECORDING.split, { train: [0, 12000], development: [12000, 16000], test: [16000, 20000] });
  equal(ICA_RECORDING.iterations, 14);
  equal(ICA_RECORDING.sha256, '7c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc');
  equal(ICA_RECORDING.license, 'ODC-By 1.0');
  assertions += 1;
  assert.ok(ICA_RECORDING.limits.length >= 4 && ICA_RECORDING.attribution.includes('Jezewski'));
  equal(ICA_RECORDING.results.map(item => item.method), ['channel', 'PCA', 'ICA']);
  equal(ICA_RECORDING.results.map(item => item.chosen), [3, 4, 2]);
  equal(ICA_RECORDING.results.map(item => item.testAbs), [0.119806, 0.450178, 0.343966]);
  equal(ICA_RECORDING.results.map(item => item.developmentAbs), [0.201203, 0.18458, 0.169804]);
  for (const item of ICA_RECORDING.results) {
    assertions += 1;
    assert.equal(item.development.length, 4);
    const best = item.development.reduce((bestIndex, value, index, all) => Math.abs(value) > Math.abs(all[bestIndex]) ? index : bestIndex, 0);
    assert.equal(best + 1, item.chosen, 'the published choice is the development argmax');
    close(Math.abs(item.development[best]), item.developmentAbs, 5e-7);
    close(Math.abs(item.testSigned), item.testAbs, 1e-12);
    assert.equal(item.displaySign, item.development[best] >= 0 ? 1 : -1);
    assertions += 1;
  }
  // The honest outcome is published as it came out: PCA leads on held-out data.
  assertions += 1;
  assert.ok(ICA_RECORDING.results[1].testAbs > ICA_RECORDING.results[2].testAbs
    && ICA_RECORDING.results[2].testAbs > ICA_RECORDING.results[0].testAbs, 'PCA leads, ICA improves on the raw channel');
  const trace = ICA_RECORDING.trace;
  equal(trace.columns * trace.samplesPerColumn, 2000);
  equal(Object.keys(trace.series).sort(), ['ICA', 'PCA', 'channel', 'reference']);
  for (const [name, series] of Object.entries(trace.series)) {
    assertions += 1;
    assert.equal(series.length, trace.columns, `${name} envelope column count`);
    series.forEach(([low, high]) => {
      assertions += 1;
      assert.ok(Number.isFinite(low) && Number.isFinite(high) && low <= high, `${name} envelope column is ordered and finite`);
    });
    const mean = series.reduce((total, [low, high]) => total + low + high, 0) / (series.length * 2);
    assert.ok(Math.abs(mean) < 0.35, `${name} display z-score is centred`);
    assertions += 1;
  }
  equal(ICA_RECORDING.window.length, 20);
  close(ICA_RECORDING.window[0].second, 16);
  close(ICA_RECORDING.window.at(-1).second, 16.019);
  ICA_RECORDING.window.forEach(row => { assertions += 1; assert.ok(Number.isFinite(row.reference) && Number.isFinite(row.ICA)); });
  assertions += 1;
  assert.ok(ICA_RECORDING.reconstructionMse < 1e-20, 'exact reconstruction says nothing about usefulness');
});

const files = ['src/learn/data/ica-models.js', 'src/learn/data/ica-data.js', 'src/learn/data/ica-examples.js', 'scripts/verify-ica-models.mjs'];
const evidence = {
  status: 'passed',
  generatedAt: new Date().toISOString(),
  command: 'node scripts/verify-ica-models.mjs',
  groups,
  assertions,
  independentOracles: [
    'eigenpairs confirmed by applying the matrix, plus trace and determinant identities',
    'whitener confirmed by K Sigma K^T = I on every non-degenerate fixture',
    'fixed-point averages recomputed by an explicit loop over the four whitened points',
    'projection kurtosis checked against cos^4 t + sin^4 t = 1 - 2 cos^2 t sin^2 t at 361 angles per family',
    'contribution fixtures and compensated scale nulls taken from the specification, not from production output',
    'published recording selection re-derived from the twelve signed development correlations',
  ],
  sourceHashes: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])),
};
fs.writeFileSync('docs/teaching/evidence/ica-models.json', `${JSON.stringify(evidence, null, 2)}\n`);
console.log(`PASS: ${groups.length} model groups, ${assertions} assertions.`);
