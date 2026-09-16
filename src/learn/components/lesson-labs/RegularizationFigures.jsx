import { useState } from 'react';
import {
  criteria, duplicateAnalysis, earlyStoppingFilter, encodeSixteenBits, factorOptimum, fixtures,
  penaltyBoundary, ridgeFilter, scalarObjective, smoothnessComparison, twoCoordinateSolution,
} from '../../data/regularization-models';
import {
  baselineMeanMse, baselines, candidates, coefficientPaths, featureNames, olsMeanMse, provenance, strengths,
} from '../../data/regularization-data';
import { Table, coefficientText, curve, fixed, round, signed } from './RegularizationShared.jsx';
import './regularization-labs.css';

const familyLabels = { ridge: 'ridge', lasso: 'lasso', elastic_net: 'elastic net' };
const familyStroke = { ridge: '#e7b94a', lasso: '#8eb9a5', elastic_net: '#91aecf' };
const familyDash = { ridge: '', lasso: '5 3', elastic_net: '2 3' };
const familyRatio = { ridge: 0, lasso: 1, elastic_net: 0.5 };

/* ------------------------------------------------------- §1 · F1 */

/** Two quantities make one objective. */
export function ObjectiveSumFigure() {
  const span = [-1, 5];
  const height = 220;
  const place = value => 46 + 262 * (value - span[0]) / (span[1] - span[0]);
  const lift = value => height - 34 - (height - 50) * value / 15;
  const data = w => scalarObjective(w, 3, 1, 0).data;
  const penalty = w => scalarObjective(w, 3, 1, 0).penalty;
  const total = w => scalarObjective(w, 3, 1, 0).total;
  const marked = [3, 1.5].map(w => ({ w, ...scalarObjective(w, 3, 1, 0) }));
  return <figure className="rg-figure">
    <figcaption><strong>Figure 1 — Two quantities make one objective.</strong> The fitted value moves because the sum has a different minimum from the data term. These are exact analytic curves at λ = 1, ρ = 0 and scalar curvature 1, not a measured generalization curve.</figcaption>
    <svg viewBox="0 0 340 220" role="img" aria-label={`Three curves against the coefficient w from minus 1 to 5: the data cost (w minus 3) squared over 2, the ridge penalty w squared over 2, and their sum. At w equals 3 the data cost is 0, the penalty 4.5 and the total 4.5. At w equals 1.5 the data cost is 1.125, the penalty 1.125 and the total 2.25, which is the smallest total.`}>
      <line className="rg-axis" x1="46" x2="320" y1={lift(0)} y2={lift(0)} />
      <line className="rg-axis" x1="46" x2="46" y1="16" y2={lift(0)} />
      {[0, 5, 10, 15].map(value => <g key={value}>
        <line className="rg-grid" x1="46" x2="320" y1={lift(value)} y2={lift(value)} />
        <text x="40" y={lift(value) + 4} textAnchor="end">{value}</text>
      </g>)}
      {[-1, 0, 1, 2, 3, 4, 5].map(value => <text key={value} x={place(value)} y={lift(0) + 16} textAnchor="middle">{value}</text>)}
      <polyline className="rg-curve is-data" points={curve(place, lift, span, data)} />
      <polyline className="rg-curve is-penalty" points={curve(place, lift, span, penalty)} />
      <polyline className="rg-curve is-total" points={curve(place, lift, span, total)} />
      {marked.map(point => <g key={point.w}>
        <line className="rg-grid" x1={place(point.w)} x2={place(point.w)} y1={lift(0)} y2={lift(point.total)} />
        <circle className="rg-mark" cx={place(point.w)} cy={lift(point.total)} r="4" />
      </g>)}
      <text x="60" y="30">total 4.5 at w = 3</text>
      <text x="60" y="46">total 2.25 at w = 1.5</text>
      <text x={place(5)} y={lift(0) + 30} textAnchor="end">coefficient w</text>
    </svg>
    <p className="rg-caption">
      Solid gold is the total, dashed green the data cost (w − 3)²/2 and dotted blue the ridge penalty w²/2. The vertical drops mark
      w = 3 and the minimum at w = 1.5.
    </p>
    <Table caption="Both decompositions, exactly"
      headings={['coefficient w', 'data cost (w − 3)²/2', 'penalty w²/2', 'total']}
      rows={marked.map(point => [round(point.w, 2), fixed(point.data, 3), fixed(point.penalty, 3), fixed(point.total, 3)])} />
    <p>
      At w = 3 the data term is satisfied and the total is 4.5. At w = 1.5 the data term is worse and the total is 2.25, so the new objective
      prefers it. That is a statement about the objective being minimised on these observations. It is not by itself evidence that w = 1.5
      predicts future cases better.
    </p>
    <div className="rg-panel">
      <h4>The same prediction, a different penalty: metres against centimetres</h4>
      <Table caption="Replacing a feature by the same measurement in centimetres, with the coefficient divided by 100"
        headings={['quantity', 'metres', 'centimetres', 'ratio']}
        rows={[
          ['feature value x', '2', '200', '× 100'],
          ['coefficient w', '3', '0.03', '÷ 100'],
          ['contribution xw', '6', '6', 'unchanged'],
          ['L1 cost |w|', '3', '0.03', '÷ 100'],
          ['squared L2 cost w²', '9', '0.0009', '÷ 10,000'],
        ]} />
      <p>
        The prediction is identical and the parameter cost is not. A raw coefficient penalty therefore prefers the larger numerical feature
        scale for the same predictive contribution. The rescaled input did not become less scientifically important.
      </p>
    </div>
  </figure>;
}

/* ------------------------------------------------------- §2 · F2 */

const geometryCases = [1, 0.1].flatMap(strength => ['ridge', 'lasso', 'elastic_net'].map(family => ({
  family, strength, solution: twoCoordinateSolution(fixtures.preferences, strength, familyRatio[family]),
})));

function GeometryPanel({ entry }) {
  const low = -3.7;
  const high = 3.7;
  const size = 260;
  const place = value => 34 + 212 * (value - low) / (high - low);
  const lift = value => 218 - 212 * (value - low) / (high - low);
  const { solution } = entry;
  const [z1, z2] = fixtures.preferences;
  const boundary = penaltyBoundary(entry.solution.ratio, solution.budget)
    .map(([x, y]) => `${place(x).toFixed(2)},${lift(y).toFixed(2)}`).join(' ');
  const contour = (radius, className, key) => <circle key={key} className={className}
    cx={place(z1)} cy={lift(z2)} r={(212 * radius / (high - low)).toFixed(2)} />;
  return <div className="rg-panel">
    <h4>{familyLabels[entry.family]}, λ = {entry.strength}</h4>
    <svg viewBox={`0 0 ${size} ${size}`} role="img"
      aria-label={`${familyLabels[entry.family]} at lambda ${entry.strength}. The data preference z is at 3 and 0.4. The penalised solution is ${solution.weights.map(value => round(value, 6)).join(' and ')}. Its attained penalty measure, used as the matching constraint budget, is ${round(solution.budget, 6)}. The data-loss contour through that solution has radius ${round(solution.contactRadius, 6)}. Both axes use one equal scale from minus 3.7 to 3.7.`}>
      <line className="rg-grid" x1={place(low)} x2={place(high)} y1={lift(0)} y2={lift(0)} />
      <line className="rg-grid" x1={place(0)} x2={place(0)} y1={lift(low)} y2={lift(high)} />
      {[-2, 2].map(value => <g key={value}>
        <text x={place(value)} y={lift(0) + 15} textAnchor="middle">{value}</text>
        <text x={place(0) - 6} y={lift(value) + 4} textAnchor="end">{value}</text>
      </g>)}
      {contour(solution.contactRadius * 1.6, 'rg-curve is-faint', 'outer')}
      {contour(solution.contactRadius * 0.5, 'rg-curve is-faint', 'inner')}
      {contour(solution.contactRadius, 'rg-curve is-data', 'contact')}
      <polyline className="rg-region" points={boundary} />
      <circle className="rg-observation" cx={place(z1)} cy={lift(z2)} r="3.5" />
      <text x={place(z1) - 8} y={lift(z2) - 8} textAnchor="end">z</text>
      <rect className="rg-mark" x={place(solution.weights[0]) - 4} y={lift(solution.weights[1]) - 4} width="8" height="8" />
    </svg>
    <p>
      Pale dot z = (3, 0.4); gold square the solution ({round(solution.weights[0], 4)}, {round(solution.weights[1], 4)}).
      Budget {round(solution.budget, 6)} · objective {round(solution.objective, 6)} ·
      {solution.zeroCoordinates[1] ? ' the second coordinate is exactly 0.' : ' both coordinates survive.'}
    </p>
  </div>;
}

/** Actual contact, including a non-sparse case. */
export function ConstraintGeometryFigure() {
  return <figure className="rg-figure">
    <figcaption><strong>Figure 2 — Actual contact, including a non-sparse case.</strong> Each region is the set where the family's own penalty measure equals the value its penalised solution actually attained, and the solid contour is the data loss through that same solution. The touching point is the arithmetic's answer, not a placed dot.</figcaption>
    <div className="rg-panels">{geometryCases.map(entry => <GeometryPanel key={`${entry.family}-${entry.strength}`} entry={entry} />)}</div>
    <Table caption="The six stated two-coordinate problems with z = (3, 0.4), on one equal scale"
      headings={['family', 'λ', 'coefficients', 'attained penalty measure', 'data loss', 'objective']}
      rows={geometryCases.map(entry => [
        familyLabels[entry.family], entry.strength,
        `(${coefficientText(entry.solution.weights[0], 6)}, ${coefficientText(entry.solution.weights[1], 6)})`,
        fixed(entry.solution.budget, 6), fixed(entry.solution.data, 6), fixed(entry.solution.objective, 6),
      ])} />
    <p>
      At λ = 0.1 the lasso answer (2.9, 0.3) sits on a diamond edge, not on an axis: corners and faces make an exact zero coordinate
      possible over a range of data preferences, and do not force every optimum to a corner. At λ = 1 the same problem gives (2, 0).
      The mixed boundary keeps its nonsmooth corners on the axes; a rounded rectangle would be a different set.
      A constraint budget is the penalty value that solution attained, so the same numerical λ is not the same radius or budget across
      families. The penalty measure here is ρ(|w₁| + |w₂|) + (1 − ρ)(w₁² + w₂²)/2, so ridge reads it as half the squared norm.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §4 · F3 */

/** One prediction, many coefficient allocations. */
export function DuplicateFigure() {
  const lasso = duplicateAnalysis(1, 1);
  const ridge = duplicateAnalysis(1, 0);
  const elastic = duplicateAnalysis(1, 0.5);
  const low = -0.35;
  const high = 1.45;
  const place = value => 40 + 200 * (value - low) / (high - low);
  const lift = value => 220 - 200 * (value - low) / (high - low);
  const sumLine = s => {
    const a = Math.max(low, s - high);
    const b = Math.min(high, s - low);
    return `${place(a).toFixed(2)},${lift(s - a).toFixed(2)} ${place(b).toFixed(2)},${lift(s - b).toFixed(2)}`;
  };
  const solutions = [
    { name: 'lasso (1, 0)', point: [1, 0], objective: lasso.objective([1, 0]).total },
    { name: 'lasso (0.5, 0.5)', point: [0.5, 0.5], objective: lasso.objective([0.5, 0.5]).total },
    { name: 'lasso (0, 1)', point: [0, 1], objective: lasso.objective([0, 1]).total },
    { name: 'ridge (2/3, 2/3)', point: ridge.balanced, objective: ridge.objective(ridge.balanced).total },
    { name: 'elastic net (0.6, 0.6)', point: elastic.balanced, objective: elastic.objective(elastic.balanced).total },
  ];
  return <figure className="rg-figure">
    <figcaption><strong>Figure 3 — One prediction, many coefficient allocations.</strong> Two identical centred columns, two rows, and λ = 1. The prediction depends only on the coefficient sum, so the lasso minimizers form a whole segment while ridge and elastic net each have one answer. These three optima solve <em>different</em> objectives.</figcaption>
    <div className="rg-panels is-pair">
      <div className="rg-panel">
        <h4>The coefficient plane, on one equal scale</h4>
        <svg viewBox="0 0 260 260" role="img" aria-label={`The plane of the two duplicate coefficients. Thin lines are constant sums. The lasso minimizers form the segment from 1, 0 to 0, 1 with objective ${round(lasso.objective([1, 0]).total, 6)}, including the midpoint 0.5, 0.5. Ridge gives two thirds and two thirds with objective ${round(ridge.objective(ridge.balanced).total, 6)}. Elastic net with rho 0.5 gives 0.6 and 0.6 with objective ${round(elastic.objective(elastic.balanced).total, 6)}.`}>
          <line className="rg-axis" x1={place(low)} x2={place(high)} y1={lift(0)} y2={lift(0)} />
          <line className="rg-axis" x1={place(0)} x2={place(0)} y1={lift(low)} y2={lift(high)} />
          {[0, 0.5, 1].map(value => <g key={value}>
            <text x={place(value)} y={lift(0) + 16} textAnchor="middle">{value}</text>
            <text x={place(0) - 6} y={lift(value) + 4} textAnchor="end">{value}</text>
          </g>)}
          {[0.4, 0.8, 1.2].map(s => <polyline key={s} className="rg-curve is-faint" points={sumLine(s)} />)}
          <polyline className="rg-region" style={{ strokeWidth: 4 }} points={`${place(1)},${lift(0)} ${place(0)},${lift(1)}`} />
          {[[1, 0], [0.5, 0.5], [0, 1]].map(point => (
            <circle key={point.join()} className="rg-mark" cx={place(point[0])} cy={lift(point[1])} r="4" />
          ))}
          <rect className="rg-mark is-hollow" x={place(ridge.balanced[0]) - 5} y={lift(ridge.balanced[1]) - 5} width="10" height="10" />
          <polygon className="rg-observation"
            points={`${place(elastic.balanced[0])},${lift(elastic.balanced[1]) - 5} ${place(elastic.balanced[0]) + 5},${lift(elastic.balanced[1])} ${place(elastic.balanced[0])},${lift(elastic.balanced[1]) + 5} ${place(elastic.balanced[0]) - 5},${lift(elastic.balanced[1])}`} />
          <text x={place(ridge.balanced[0]) + 10} y={lift(ridge.balanced[1]) - 8}>ridge</text>
          <text x={place(elastic.balanced[0]) + 10} y={lift(elastic.balanced[1]) + 16}>elastic net</text>
          <text x={place(high)} y={lift(low) + 6} textAnchor="end">w₁ →</text>
          <text x={place(0) + 8} y={lift(high) + 10}>↑ w₂</text>
        </svg>
        <p>
          The thick gold segment holds every lasso minimizer, with filled gold dots at (1, 0), (0.5, 0.5) and (0, 1). The open square is
          ridge and the pale diamond elastic net, and both of those sit off the segment. Thin dashed lines are constant coefficient sums;
          every point on one of them makes the same prediction at every observed row.
        </p>
      </div>
      <div className="rg-panel">
        <h4>What the two sensor readings add up to</h4>
        <svg viewBox="0 0 260 190" role="img" aria-label={`Prediction strips at x equals 1: lasso gives sum ${round(lasso.sum, 6)}, ridge ${round(ridge.sum, 6)}, elastic net ${round(elastic.sum, 6)}. At x equals minus 1 each sign reverses.`}>
          {[{ label: 'lasso', value: lasso.sum }, { label: 'ridge', value: ridge.sum }, { label: 'elastic net', value: elastic.sum }].map((row, index) => {
            const y = 40 + index * 46;
            const width = 150 * row.value / 1.5;
            return <g key={row.label}>
              <text x="8" y={y + 4}>{row.label}</text>
              <rect className="rg-lane" x="86" y={y - 12} width="160" height="24" rx="3" />
              <rect x="88" y={y - 10} width={width.toFixed(1)} height="20" fill={index === 0 ? '#8eb9a5' : index === 1 ? '#e7b94a' : '#91aecf'} />
              <text x="92" y={y + 5}>{round(row.value, 6)}</text>
            </g>;
          })}
          <text x="8" y="16">contribution at x = 1</text>
        </svg>
        <p>
          Bars are the prediction each family makes at x = 1, in target units. At x = −1 every bar has the same length with the opposite
          sign. Different penalty families change the best <em>sum</em> as well as how it is allocated; they do not merely redistribute
          one answer.
        </p>
      </div>
    </div>
    <Table caption="Five coefficient vectors, their shared data cost and their own objectives at λ = 1"
      headings={['coefficients', 'w₁ + w₂', 'data cost', 'penalty', 'objective']}
      rows={solutions.map(entry => {
        const analysis = entry.name.startsWith('lasso') ? lasso : entry.name.startsWith('ridge') ? ridge : elastic;
        const parts = analysis.objective(entry.point);
        return [entry.name, round(entry.point[0] + entry.point[1], 6), fixed(parts.data, 6), fixed(parts.penalty, 6), fixed(parts.total, 6)];
      })} />
    <p>
      The three lasso rows share one objective value, so a solver that returns (1, 0) did so because it visited the first column first.
      That ordering is not evidence about which sensor matters. Ridge and elastic net are strictly convex here and each has a single
      answer, and those answers are not comparable with lasso's because the objectives differ. For columns that are merely correlated
      rather than identical, this becomes a tendency with conditions, not an equality.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §5 · F4 */

const logPlace = (value, low, high, left, width) =>
  left + width * (Math.log10(value) - Math.log10(low)) / (Math.log10(high) - Math.log10(low));

/** The observed path, including its unhelpful end. */
export function PathFigure() {
  const [family, setFamily] = useState('lasso');
  const [fold, setFold] = useState(0);
  const [selected, setSelected] = useState(0.1);
  const [term, setTerm] = useState('all');
  const rows = candidates.filter(row => row[0] === family);
  const row = rows.find(entry => entry[1] === selected);
  const path = coefficientPaths.find(entry => entry.family === family && entry.strength === selected);
  const coefficients = path.folds[fold];
  const place = value => logPlace(value, 0.001, 100, 52, 216);
  const liftFull = value => 128 - 104 * (value - 15) / 32;
  const liftZoom = value => 128 - 104 * (value - 17.25) / 0.55;
  const coefficientExtent = Math.max(...coefficientPaths.filter(entry => entry.family === family)
    .flatMap(entry => entry.folds[fold].map(Math.abs)));
  const coefficientY = value => 72 - 60 * value / (coefficientExtent * 1.05);
  const termIndex = featureNames.indexOf(term);
  const pathOfTerm = termIndex < 0 ? [] : coefficientPaths
    .filter(entry => entry.family === family).map(entry => entry.folds[fold][termIndex]);
  return <figure className="rg-figure">
    <figcaption><strong>Figure 4 — The observed path, including its unhelpful end.</strong> These are the recorded development results: six fitted strengths, three folds, no curve drawn between the fitted points and no confidence band. Mean squared error is in squared decibels.</figcaption>
    <div className="rg-controls">
      <label className="rg-field"><span>Family</span>
        <select value={family} onChange={event => { setFamily(event.target.value); }}>
          {Object.entries(familyLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
        </select>
      </label>
      <label className="rg-field"><span>Fold, for the per-fold lines and the coefficients</span>
        <select value={fold} onChange={event => setFold(Number(event.target.value))}>
          {[0, 1, 2].map(index => <option key={index} value={index}>fold {index + 1}</option>)}
        </select>
      </label>
      <label className="rg-field"><span>Inspect λ</span>
        <select value={selected} onChange={event => setSelected(Number(event.target.value))}>
          {strengths.map(value => <option key={value} value={value}>λ = {value}</option>)}
        </select>
      </label>
      <label className="rg-field"><span>Follow one term</span>
        <select value={term} onChange={event => setTerm(event.target.value)}>
          <option value="all">all twenty together</option>
          {featureNames.map(name => <option key={name} value={name}>{name}</option>)}
        </select>
      </label>
    </div>
    <div className="rg-panels is-pair">
      <div className="rg-panel">
        <h4>Mean validation MSE, full range</h4>
        <svg viewBox="0 0 300 160" role="img" aria-label={`Mean validation MSE against lambda on a logarithmic axis for all three families, from 15 to 47 squared decibels. ${Object.keys(familyLabels).map(key => `${familyLabels[key]}: ${candidates.filter(entry => entry[0] === key).map(entry => `lambda ${entry[1]} gives ${entry[2]}`).join(', ')}`).join('. ')}. The mean baseline is ${baselineMeanMse} and unpenalized OLS is ${olsMeanMse}.`}>
          <line className="rg-axis" x1="52" x2="284" y1="128" y2="128" />
          <line className="rg-axis" x1="52" x2="52" y1="16" y2="128" />
          {[20, 30, 40].map(value => <text key={value} x="46" y={liftFull(value) + 4} textAnchor="end">{value}</text>)}
          {strengths.map(value => <text key={value} x={place(value)} y="150" textAnchor="middle">{value}</text>)}
          <line className="rg-grid" x1="52" x2="284" y1={liftFull(baselineMeanMse)} y2={liftFull(baselineMeanMse)} strokeDasharray="4 3" />
          <line className="rg-grid" x1="52" x2="284" y1={liftFull(olsMeanMse)} y2={liftFull(olsMeanMse)} strokeDasharray="4 3" />
          {rows.map(entry => entry[3].map((value, index) => (
            <circle key={`${entry[1]}-${index}`} cx={place(entry[1])} cy={liftFull(value)} r="2" fill="#5c6b64" />
          )))}
          {[0, 1, 2].map(index => (
            <polyline key={index} className="rg-curve is-faint" points={rows.map(entry => `${place(entry[1])},${liftFull(entry[3][index])}`).join(' ')} />
          ))}
          {Object.keys(familyLabels).map(key => (
            <polyline key={key} className="rg-curve" stroke={familyStroke[key]} strokeDasharray={familyDash[key]}
              points={candidates.filter(entry => entry[0] === key).map(entry => `${place(entry[1])},${liftFull(entry[2])}`).join(' ')} />
          ))}
          {candidates.map(entry => (
            <circle key={`${entry[0]}-${entry[1]}`} cx={place(entry[1])} cy={liftFull(entry[2])} r="2.6" fill={familyStroke[entry[0]]} />
          ))}
          <circle className="rg-mark is-hollow" cx={place(selected)} cy={liftFull(row[2])} r="7" />
        </svg>
        <p>
          Horizontal axis λ, logarithmic; vertical axis mean squared error in squared decibels, 15 to 47. The two dashed horizontals are
          the comparators: the upper is the training-mean baseline at {fixed(baselineMeanMse, 6)} and the lower unpenalized OLS at
          {' '}{fixed(olsMeanMse, 6)}. Faint grey: the three folds of the selected family. Solid gold ridge, dashed green lasso, dotted
          blue elastic net are the means, and the open circle is the strength selected in the control above.
        </p>
      </div>
      <div className="rg-panel">
        <h4>The same means, axis changed to 17.25–17.80</h4>
        <svg viewBox="0 0 300 160" role="img" aria-label={`The same three mean curves on a changed vertical axis from 17.25 to 17.80 squared decibels, covering only the three smallest strengths. Ridge reads ${candidates.filter(e => e[0] === 'ridge' && e[1] <= 0.1).map(e => e[2]).join(', ')}; lasso ${candidates.filter(e => e[0] === 'lasso' && e[1] <= 0.1).map(e => e[2]).join(', ')}; elastic net ${candidates.filter(e => e[0] === 'elastic_net' && e[1] <= 0.1).map(e => e[2]).join(', ')}. Differences of this size carry no significance claim.`}>
          <line className="rg-axis" x1="52" x2="284" y1="128" y2="128" />
          <line className="rg-axis" x1="52" x2="52" y1="16" y2="128" />
          {[17.3, 17.5, 17.7].map(value => <text key={value} x="46" y={liftZoom(value) + 4} textAnchor="end">{value}</text>)}
          {[0.001, 0.01, 0.1].map(value => <text key={value} x={logPlace(value, 0.001, 0.1, 62, 200)} y="150" textAnchor="middle">{value}</text>)}
          {Object.keys(familyLabels).map(key => (
            <polyline key={key} className="rg-curve" stroke={familyStroke[key]} strokeDasharray={familyDash[key]}
              points={candidates.filter(entry => entry[0] === key && entry[1] <= 0.1)
                .map(entry => `${logPlace(entry[1], 0.001, 0.1, 62, 200)},${liftZoom(entry[2])}`).join(' ')} />
          ))}
          {candidates.filter(entry => entry[1] <= 0.1).map(entry => (
            <circle key={`${entry[0]}-${entry[1]}`} cx={logPlace(entry[1], 0.001, 0.1, 62, 200)} cy={liftZoom(entry[2])} r="3" fill={familyStroke[entry[0]]} />
          ))}
          <line className="rg-grid" x1="52" x2="284" y1={liftZoom(olsMeanMse)} y2={liftZoom(olsMeanMse)} strokeDasharray="4 3" />
          <text x="284" y={liftZoom(olsMeanMse) - 5} textAnchor="end">OLS {olsMeanMse.toFixed(3)}</text>
        </svg>
        <p>
          Changed bounds: this vertical axis runs 17.25 to 17.80 and only the three smallest λ fit inside it, so it is not comparable with
          the panel beside it. All three families select λ = 0.001, and their separations are small.
        </p>
      </div>
    </div>
    <div className="rg-panel">
      <h4>Signed coefficients for {familyLabels[family]}, fold {fold + 1}, across the six fitted strengths</h4>
      <svg viewBox="0 0 340 160" role="img" aria-label={`Twenty signed standardized coefficients for ${familyLabels[family]} on fold ${fold + 1}, plotted against lambda on a logarithmic axis. At the selected lambda ${selected} the nonzero count is ${row[4][fold]} of twenty; the terms that are exactly zero are marked with an open cross on the zero line.`}>
        <line className="rg-axis" x1="52" x2="316" y1={coefficientY(0)} y2={coefficientY(0)} />
        <line className="rg-axis" x1="52" x2="52" y1="14" y2="132" />
        {[coefficientExtent, coefficientExtent / 2, 0, -coefficientExtent / 2, -coefficientExtent].map(value => (
          <text key={value} x="46" y={coefficientY(value) + 4} textAnchor="end">{round(value, 2)}</text>
        ))}
        {strengths.map(value => <text key={value} x={logPlace(value, 0.001, 100, 62, 244)} y="150" textAnchor="middle">{value}</text>)}
        {featureNames.map((name, index) => {
          const followed = term === name;
          return <polyline key={name} className="rg-curve" fill="none"
            style={{ strokeWidth: followed ? 2.4 : 1.1 }} stroke={followed ? '#e7b94a' : '#6f8b7e'}
            strokeOpacity={term === 'all' || followed ? 1 : 0.28}
            points={coefficientPaths.filter(entry => entry.family === family)
              .map(entry => `${logPlace(entry.strength, 0.001, 100, 62, 244)},${coefficientY(entry.folds[fold][index])}`).join(' ')} />;
        })}
        {coefficientPaths.filter(entry => entry.family === family).flatMap(entry => entry.folds[fold]
          .map((value, term) => (value === 0
            ? <g key={`${entry.strength}-${term}`}>
              <line className="rg-zero-mark" x1={logPlace(entry.strength, 0.001, 100, 62, 244) - 3.5} x2={logPlace(entry.strength, 0.001, 100, 62, 244) + 3.5} y1={coefficientY(0) - 3.5} y2={coefficientY(0) + 3.5} />
              <line className="rg-zero-mark" x1={logPlace(entry.strength, 0.001, 100, 62, 244) - 3.5} x2={logPlace(entry.strength, 0.001, 100, 62, 244) + 3.5} y1={coefficientY(0) + 3.5} y2={coefficientY(0) - 3.5} />
            </g>
            : null)))}
        {coefficients.map((value, term) => (
          <circle key={term} cx={logPlace(selected, 0.001, 100, 62, 244)} cy={coefficientY(value)} r="2.6"
            fill={value === 0 ? 'none' : '#e7b94a'} stroke={value === 0 ? '#da9c86' : 'none'} strokeWidth="1.4" />
        ))}
      </svg>
      <p>
        Vertical axis: coefficient value in this fold's own standardized coordinates, labelled at ±{round(coefficientExtent, 2)} and
        ±{round(coefficientExtent / 2, 2)}, with the horizontal line at zero. Horizontal axis λ, logarithmic.{' '}
        {term === 'all'
          ? 'All twenty paths are drawn alike; use the selector above to follow one of them in gold.'
          : `${term} is drawn in gold at ${coefficientText(pathOfTerm[0], 6)} when λ = 0.001 and ${coefficientText(pathOfTerm.at(-1), 6)} when λ = 100; the other nineteen are dimmed.`}{' '}
        A red cross on the zero line is an exact zero, not a small value rounded for display. One fold's scaler and coefficients belong to
        that fold alone: these are not averaged across folds and not combined with the final all-development refit.
      </p>
    </div>
    <Table caption={`Every term at ${familyLabels[family]}, λ = ${selected}, fold ${fold + 1}`}
      headings={['term', 'standardized coefficient']}
      rows={featureNames.map((name, index) => [name, coefficientText(coefficients[index], 6)])}
      rowClass={index => (featureNames[index] === term ? 'is-selected' : coefficients[index] === 0 ? 'is-zero' : undefined)} scroll />
    <p className="rg-readout" aria-live="polite">
      {familyLabels[family]} at λ = {selected}: fold MSEs {row[3].map(value => fixed(value, 6)).join(', ')},
      mean {fixed(row[2], 6)}, nonzero counts {row[4].join(', ')} of twenty. Fold {fold + 1} keeps {row[4][fold]}.
    </p>
    <details>
      <summary>Inspect all eighteen fitted candidates and the two baselines</summary>
      <Table caption="Recorded development results, rounded to six decimals"
        headings={['family', 'λ', 'fold 1', 'fold 2', 'fold 3', 'mean MSE', 'nonzero per fold']}
        rows={candidates.map(entry => [familyLabels[entry[0]], entry[1], fixed(entry[3][0], 6), fixed(entry[3][1], 6),
          fixed(entry[3][2], 6), fixed(entry[2], 6), entry[4].join(', ')])} scroll />
      <Table caption="The two comparators, on the same folds"
        headings={['fold', 'training mean prediction (dB)', 'mean-baseline MSE', 'unpenalized OLS MSE']}
        rows={baselines.map((entry, index) => [`fold ${index + 1}`, fixed(entry.meanPrediction, 6), fixed(entry.meanMse, 6), fixed(entry.olsMse, 6)])} />
    </details>
    <p>
      Source: the observed {provenance.name} collection, {provenance.rows} rows, licensed {provenance.license}.
      {' '}{provenance.developmentRows} development rows and {provenance.reservedRows} reserved rows split with seed {provenance.splitSeed};
      three shuffled folds with seed {provenance.foldSeed}, {provenance.foldTrainRows} fitting and {provenance.foldValidationRows} validating
      in each. The reserved rows receive no prediction and no score. These are development comparison and selection records, not
      independent performance claims, and they cannot substantiate performance on a new airfoil or experimental run. The later
      learning-curves lesson reuses the same input under a different protocol; its numbers are not a contest against these.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §7 · F5 */

/** Strong and weak directions. */
export function DirectionsFigure() {
  const [first, second] = fixtures.singularValues;
  const cases = [1, 4].flatMap(a => fixtures.singularValues.map(sigma => ({ a, sigma, ...ridgeFilter(sigma, a) })));
  const low = 0.1;
  const high = 100;
  const place = value => logPlace(value, low, high, 52, 216);
  const lift = value => 128 - 104 * value;
  const early = [1, 4].map(eigenvalue => earlyStoppingFilter(eigenvalue, 0.1, 1));
  return <figure className="rg-figure">
    <figcaption><strong>Figure 5 — Strong and weak directions.</strong> Analytic multipliers for the chosen singular values 4 and 0.5. The horizontal variable is nλ, matching the displayed equation; it is not a library <code>alpha</code> axis and these are not fitted airfoil singular values.</figcaption>
    <div className="rg-panels is-pair">
      <div className="rg-panel">
        <h4>Two coefficient directions with different data sensitivity</h4>
        <svg viewBox="0 0 300 190" role="img" aria-label={`Two orthogonal unit directions in coefficient space, drawn at the same length because both are unit vectors. Beside them, the data sensitivity of each: moving one unit along v one changes the fitted observations by ${first}, and along v two by ${second}. The bars are on one shared scale.`}>
          <line className="rg-grid" x1="18" x2="130" y1="112" y2="112" />
          <line className="rg-grid" x1="74" x2="74" y1="52" y2="150" />
          <line className="rg-stem is-data" x1="74" y1="112" x2="120" y2="112" />
          <line className="rg-stem is-penalty" x1="74" y1="112" x2="74" y2="66" />
          <circle className="rg-mark" cx="74" cy="112" r="3" />
          <text x="80" y="60">v₂</text>
          <text x="104" y="128">v₁</text>
          <text x="18" y="168">two unit directions,</text>
          <text x="18" y="182">equal length</text>
          <line className="rg-axis" x1="160" x2="292" y1="150" y2="150" />
          <rect x="160" y="60" width={(28 * first).toFixed(1)} height="18" fill="#8eb9a5" />
          <text x="160" y="54">‖Zv₁‖ = σ = {first}</text>
          <rect x="160" y="108" width={(28 * second).toFixed(1)} height="18" fill="#91aecf" />
          <text x="160" y="102">‖Zv₂‖ = σ = {second}</text>
          <text x="160" y="168">data sensitivity,</text>
          <text x="160" y="182">one shared scale</text>
        </svg>
        <p>
          Coefficient space, equal scale on both axes. The two direction stems are drawn at the <strong>same</strong> length, because both
          are unit vectors; what differs is what moving along them does to the fitted observations. The two bars beside them carry that:
          each is drawn to its own direction's singular value on one shared scale, so moving one unit along v₁ changes the fitted
          observations by {first} and one unit along v₂ changes them by {second}.
        </p>
      </div>
      <div className="rg-panel">
        <h4>Fitted-data multiplier σ²/(σ² + nλ)</h4>
        <svg viewBox="0 0 300 160" role="img" aria-label={`Fitted-data multipliers against n lambda on a logarithmic axis. At n lambda equals 1 the multipliers are ${round(ridgeFilter(first, 1).dataMultiplier, 10)} and ${round(ridgeFilter(second, 1).dataMultiplier, 1)}; at n lambda equals 4 they are ${round(ridgeFilter(first, 4).dataMultiplier, 1)} and ${round(ridgeFilter(second, 4).dataMultiplier, 10)}.`}>
          <line className="rg-axis" x1="52" x2="284" y1="128" y2="128" />
          <line className="rg-axis" x1="52" x2="52" y1="16" y2="128" />
          {[0, 0.5, 1].map(value => <text key={value} x="46" y={lift(value) + 4} textAnchor="end">{value}</text>)}
          {[0.1, 1, 10, 100].map(value => <text key={value} x={place(value)} y="150" textAnchor="middle">{value}</text>)}
          <polyline className="rg-curve is-total" points={Array.from({ length: 121 }, (_, index) => {
            const a = low * (high / low) ** (index / 120);
            return `${place(a).toFixed(2)},${lift(ridgeFilter(first, a).dataMultiplier).toFixed(2)}`;
          }).join(' ')} />
          <polyline className="rg-curve is-penalty" points={Array.from({ length: 121 }, (_, index) => {
            const a = low * (high / low) ** (index / 120);
            return `${place(a).toFixed(2)},${lift(ridgeFilter(second, a).dataMultiplier).toFixed(2)}`;
          }).join(' ')} />
          {cases.map(entry => (
            <circle key={`${entry.a}-${entry.sigma}`} className="rg-mark" cx={place(entry.a)} cy={lift(entry.dataMultiplier)} r="3.5" />
          ))}
        </svg>
        <p>
          Solid gold is σ = 4 and dotted blue σ = 0.5; the horizontal axis is nλ on a logarithmic scale and the vertical axis the
          multiplier from 0 to 1. The weakly identified direction loses most of its unpenalized fit while the strong direction keeps
          nearly all of it.
        </p>
      </div>
    </div>
    <Table caption="The four stated multipliers, computed from σ²/(σ² + nλ)"
      headings={['nλ', 'σ', 'fitted-data multiplier', 'coefficient multiplier σ/(σ² + nλ)']}
      rows={cases.map(entry => [entry.a, entry.sigma, fixed(entry.dataMultiplier, 9), fixed(entry.coefficientMultiplier, 9)])} />
    <div className="rg-panel">
      <h4>Early stopping is a different filter</h4>
      <Table caption="Gradient descent from zero after one step with η = 0.1, and the ridge strength that would match each factor separately"
        headings={['Gram eigenvalue a', 'fit factor 1 − (1 − ηa)ᵗ', 'matching ridge λ = a(1 − f)/f']}
        rows={early.map(entry => [entry.eigenvalue, fixed(entry.factor, 6), fixed(entry.matchingStrength, 6)])} />
      <p>
        A Gram eigenvalue is not a singular value, and this normalized λ is not the nλ on the axis above. One common λ cannot reproduce
        both factors: matching them separately needs 9 and 6. Both filters suppress weakly learned directions; they are not the same filter,
        and the iteration also needs a step size that keeps it stable.
      </p>
    </div>
  </figure>;
}

/* ------------------------------------------------------- §8 · F6 */

/** The same-prediction curve and the actual optimum. */
export function FactorFigure() {
  const strength = 0.25;
  const optimum = factorOptimum(strength);
  const low = -2.6;
  const high = 2.6;
  const place = value => 130 + 92 * value / high;
  const lift = value => 130 - 92 * value / high;
  const branch = sign => Array.from({ length: 81 }, (_, index) => {
    const a = sign * (0.4 + 2.2 * index / 80);
    return `${place(a).toFixed(2)},${lift(1 / a).toFixed(2)}`;
  }).join(' ');
  const pSpan = [-0.6, 2];
  const pPlace = value => 46 + 214 * (value - pSpan[0]) / (pSpan[1] - pSpan[0]);
  const pLift = value => 124 - 96 * value / 1.4;
  const scalarCost = p => 0.5 * (p - 1) ** 2 + 2 * strength * Math.abs(p);
  return <figure className="rg-figure">
    <figcaption><strong>Figure 6 — The same-prediction curve and the actual optimum.</strong> The predictor is abx and the data cost is ½(ab − 1)². Every pair on ab = 1 makes the same prediction, and the penalty λ(a² + b²) is different along that curve. At λ = 0.25 the full optimum is <em>not</em> on it.</figcaption>
    <div className="rg-panels is-pair">
      <div className="rg-panel">
        <h4>The factor plane at λ = {strength}, one equal scale</h4>
        <svg viewBox="0 0 260 260" role="img" aria-label={`The two branches of a b equals 1. The balanced zero-loss point 1, 1 costs ${round(optimum.balancedZeroLoss.total, 6)} in total. The same-prediction alternative 2, 0.5 has penalty ${round(strength * (4 + 0.25), 6)}. The full optimum has product ${round(optimum.product, 6)} with both factors of magnitude ${round(optimum.magnitude, 9)} and total cost ${round(optimum.total, 6)}, including the equal negative pair.`}>
          <line className="rg-grid" x1={place(low)} x2={place(high)} y1={lift(0)} y2={lift(0)} />
          <line className="rg-grid" x1={place(0)} x2={place(0)} y1={lift(low)} y2={lift(high)} />
          {[-2, 2].map(value => <g key={value}>
            <text x={place(value)} y={lift(0) + 15} textAnchor="middle">{value}</text>
            <text x={place(0) - 6} y={lift(value) + 4} textAnchor="end">{value}</text>
          </g>)}
          <polyline className="rg-curve is-faint" strokeDasharray="" points={branch(1)} />
          <polyline className="rg-curve is-faint" strokeDasharray="" points={branch(-1)} />
          <circle className="rg-observation" cx={place(1)} cy={lift(1)} r="4" />
          <text x={place(1) + 8} y={lift(1) - 6}>(1, 1)</text>
          <circle className="rg-observation" cx={place(2)} cy={lift(0.5)} r="4" />
          <text x={place(2) - 6} y={lift(0.5) + 18} textAnchor="end">(2, 0.5)</text>
          <circle className="rg-mark" cx={place(optimum.magnitude)} cy={lift(optimum.magnitude)} r="5" />
          <circle className="rg-mark" cx={place(-optimum.magnitude)} cy={lift(-optimum.magnitude)} r="5" />
          <text x={place(optimum.magnitude) - 10} y={lift(optimum.magnitude) - 12} textAnchor="end">optimum</text>
          <text x={place(low) + 6} y={lift(high) + 4}>dashed: ab = 1</text>
        </svg>
        <p>
          Both marked points on the dashed curve make the same prediction: (1, 1) costs {round(2 * strength, 6)} in penalty and (2, 0.5)
          costs {round(strength * 4.25, 6)}. Balancing the factors minimises the penalty <em>among</em> zero-data-loss pairs. The full
          problem prefers nonzero data loss, and its optimum is the gold pair off the curve.
        </p>
      </div>
      <div className="rg-panel">
        <h4>The reduced scalar problem in p = ab</h4>
        <svg viewBox="0 0 300 150" role="img" aria-label={`The scalar objective one half of p minus 1 squared plus 2 lambda absolute p, at lambda ${strength}. Its minimum is at p equals ${round(optimum.product, 6)} with value ${round(optimum.total, 6)}. The absolute value gives a corner at p equals 0.`}>
          <line className="rg-axis" x1="46" x2="284" y1={pLift(0)} y2={pLift(0)} />
          <line className="rg-axis" x1="46" x2="46" y1="16" y2={pLift(0)} />
          {[0, 0.5, 1].map(value => <text key={value} x="40" y={pLift(value) + 4} textAnchor="end">{value}</text>)}
          {[-0.5, 0, 0.5, 1, 1.5, 2].map(value => <text key={value} x={pPlace(value)} y={pLift(0) + 16} textAnchor="middle">{value}</text>)}
          <polyline className="rg-curve is-total" points={curve(pPlace, pLift, pSpan, scalarCost)} />
          <circle className="rg-mark" cx={pPlace(optimum.product)} cy={pLift(optimum.total)} r="4" />
          <text x={pPlace(optimum.product)} y={pLift(optimum.total) - 10} textAnchor="middle">p* = {round(optimum.product, 6)}</text>
          <text x="46" y="12">total cost</text>
        </svg>
        <p>
          Horizontal axis: the product p = ab. Because a² + b² ≥ 2|ab| with equality at equal magnitudes, the whole problem reduces to
          this one-dimensional soft-threshold.
        </p>
      </div>
    </div>
    <Table caption="Four strengths under the same reduction, with p* = max(1 − 2λ, 0)"
      headings={['λ', 'optimal product p*', 'balanced factor magnitude', 'data cost', 'penalty', 'total']}
      rows={[0, 0.1, 0.25, 0.5].map(value => {
        const entry = factorOptimum(value);
        return [value, fixed(entry.product, 6), fixed(entry.magnitude, 6), fixed(entry.data, 6), fixed(entry.penalty, 6), fixed(entry.total, 6)];
      })} />
    <p>
      At λ = 0.25 the optimum has product 0.5 with both factors ±√0.5, data cost 0.125, penalty 0.25 and total 0.375, below the 0.5 of the
      balanced zero-loss pair (1, 1). At λ ≥ 0.5 both factors are zero. The λ = 0 row is different in kind: with no penalty every ab = 1 pair
      minimises, so the single listed pair is one representative. None of this is evidence that balancing arbitrary neural layers improves
      prediction; the point is to inspect the whole objective rather than a symmetry of its data term.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §8 · F7 */

/** Penalize magnitude or neighbouring change? */
export function SmoothnessFigure() {
  const base = smoothnessComparison(fixtures.denoising, 1);
  const shifted = smoothnessComparison(fixtures.shiftedDenoising, 1);
  const place = index => 70 + 90 * index;
  const lift = value => 96 - 30 * value;
  const series = [
    { name: 'observed y', values: base.signal, className: 'rg-stem is-data', marker: 'rg-observation' },
    { name: 'identity penalty', values: base.identity, className: 'rg-stem is-penalty', marker: 'rg-mark is-hollow' },
    { name: 'difference penalty', values: base.difference, className: 'rg-stem', marker: 'rg-mark' },
  ];
  return <figure className="rg-figure">
    <figcaption><strong>Figure 7 — Penalize magnitude or neighbouring change?</strong> Three positions, λ = 1, under the <strong>unaveraged</strong> objective ½‖y − w‖² + (λ/2)‖Lw‖². This local convention is declared because the regression sections elsewhere average by n. This is an exact constructed calculation, not a claim about noise-removal quality.</figcaption>
    <svg viewBox="0 0 340 146" role="img" aria-label={`Three discrete positions. The observed signal is 0, 2, 0. The identity-penalty solution is 0, 1, 0. The difference-penalty solution is 0.5, 1, 0.5. Neighbouring differences are ${base.signalEdges.join(' and ')} for the observation, ${base.identityEdges.join(' and ')} under the identity penalty and ${base.differenceEdges.join(' and ')} under the difference penalty.`}>
      <line className="rg-axis" x1="40" x2="300" y1={lift(0)} y2={lift(0)} />
      {[0, 1, 2].map(index => <text key={index} x={place(index)} y={lift(0) + 18} textAnchor="middle">position {index + 1}</text>)}
      {[0, 1, 2].map(value => <g key={`v${value}`}>
        <line className="rg-grid" x1="40" x2="300" y1={lift(value)} y2={lift(value)} />
        <text x="34" y={lift(value) + 4} textAnchor="end">{value}</text>
      </g>)}
      {series.map((entry, order) => <g key={entry.name}>
        <polyline className="rg-curve" style={{ strokeWidth: 1.1 }} strokeDasharray="3 3" fill="none"
          stroke={['#d8dcd9', '#91aecf', '#e7b94a'][order]}
          points={entry.values.map((value, index) => `${place(index) + (order - 1) * 7},${lift(value)}`).join(' ')} />
        {entry.values.map((value, index) => <g key={index}>
          <line className={entry.className} x1={place(index) + (order - 1) * 7} x2={place(index) + (order - 1) * 7} y1={lift(0)} y2={lift(value)} />
          {entry.marker === 'rg-mark is-hollow'
            ? <rect className="rg-mark is-hollow" x={place(index) + (order - 1) * 7 - 4} y={lift(value) - 4} width="8" height="8" />
            : <circle className={entry.marker} cx={place(index) + (order - 1) * 7} cy={lift(value)} r="4" />}
        </g>)}
      </g>)}
    </svg>
    <p className="rg-caption">
      Filled pale dot: the observed value. Open square: the identity-penalty solution. Gold dot: the difference-penalty solution. The three
      positions are joined only to show the neighbour relation; nothing is measured between them.
    </p>
    <div className="rg-panels is-pair">
      <div className="rg-panel">
        <h4>The system that was solved</h4>
        <p className="rg-matrices">
          <span>L = [−1, 1, 0; 0, −1, 1]</span>
          <span>(I + LᵀL) w = y</span>
        </p>
        <Table caption="I + LᵀL at λ = 1, with the observed right-hand side"
          headings={['row', 'w₁', 'w₂', 'w₃', '= y']}
          rows={base.system.map((row, index) => [index + 1, ...row.map(value => round(value, 6)), round(base.signal[index], 6)])} />
      </div>
      <div className="rg-panel">
        <h4>Neighbouring differences under each preference</h4>
        <Table caption="Both reduce the middle spike; only one spreads it"
          headings={['quantity', 'values', 'edges']}
          rows={[
            ['observed y', base.signal.map(value => round(value, 6)).join(', '), base.signalEdges.map(value => signed(value, 6)).join(', ')],
            ['identity penalty', base.identity.map(value => round(value, 6)).join(', '), base.identityEdges.map(value => signed(value, 6)).join(', ')],
            ['difference penalty', base.difference.map(value => round(value, 6)).join(', '), base.differenceEdges.map(value => signed(value, 6)).join(', ')],
          ]} />
      </div>
    </div>
    <Table caption="Adding a constant to every input: L annihilates a constant vector, so the difference solution shifts by the same constant"
      headings={['input', 'difference-penalty solution', 'identity-penalty solution']}
      rows={[
        [base.signal.join(', '), base.difference.map(value => round(value, 6)).join(', '), base.identity.map(value => round(value, 6)).join(', ')],
        [shifted.signal.join(', '), shifted.difference.map(value => round(value, 6)).join(', '), shifted.identity.map(value => round(value, 6)).join(', ')],
      ]} />
    <p>
      Both preferences lower the middle value, and only the difference penalty spreads it into its neighbours. Adding three to every input
      shifts the difference solution by exactly three, while the identity solution's shift is only 1.5, because a constant vector costs
      nothing under L and a great deal under the identity.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §9 · F8 */

/** Three complexity accounts. */
export function ComplexityFigure() {
  const records = fixtures.criteriaRecords.map(record => ({ ...record, ...criteria(record.logLikelihood, record.parameters, 100) }));
  const codes = fixtures.messages.slice(0, 2).map(message => encodeSixteenBits(message));
  const scale = 0.34;
  const bar = (record, kind, x) => {
    const penalty = kind === 'aic' ? record.aicPenalty : record.bicPenalty;
    const fitHeight = record.fit * scale;
    const penaltyHeight = penalty * scale;
    return <g key={`${record.model}-${kind}`}>
      <rect x={x} y={128 - fitHeight} width="46" height={fitHeight} fill="#3d4f47" />
      <rect x={x} y={128 - fitHeight - penaltyHeight} width="46" height={penaltyHeight} fill="#e7b94a" />
      <text x={x + 23} y="142" textAnchor="middle">{record.model}</text>
      <text x={x + 23} y={128 - fitHeight - penaltyHeight - 6} textAnchor="middle">{round(kind === 'aic' ? record.aic : record.bic, 4)}</text>
    </g>;
  };
  return <figure className="rg-figure">
    <figcaption><strong>Figure 8 — Three complexity accounts.</strong> Two constructed fitted-model records on n = 100. Natural logarithms, and lower is better for both criteria. The bit code below sits on its own scale; bits and log-likelihood units never share an axis here.</figcaption>
    <div className="rg-panels is-pair">
      {['aic', 'bic'].map(kind => (
        <div className="rg-panel" key={kind}>
          <h4>{kind.toUpperCase()} = −2ℓ + {kind === 'aic' ? '2k' : 'k log n'}, lower is preferred</h4>
          <svg viewBox="0 0 260 160" role="img" aria-label={`${kind.toUpperCase()} components. ${records.map(record => `${record.model}: minus two log-likelihood ${record.fit}, complexity charge ${round(kind === 'aic' ? record.aicPenalty : record.bicPenalty, 6)}, total ${round(kind === 'aic' ? record.aic : record.bic, 6)}`).join('. ')}.`}>
            <line className="rg-axis" x1="46" x2="240" y1="128" y2="128" />
            <line className="rg-axis" x1="46" x2="46" y1="14" y2="128" />
            {[0, 100, 200, 300].map(value => <g key={value}>
              <line className="rg-grid" x1="46" x2="240" y1={128 - value * scale} y2={128 - value * scale} />
              <text x="40" y={128 - value * scale + 4} textAnchor="end">{value}</text>
            </g>)}
            {bar(records[0], kind, 86)}
            {bar(records[1], kind, 166)}
          </svg>
          <p>{kind === 'aic'
            ? 'Dark: −2ℓ. Gold: the complexity charge. AIC charges 4 for the two added parameters, so the eight-unit likelihood improvement pays for them.'
            : `Dark: −2ℓ. Gold: the complexity charge. BIC charges about ${round(records[1].bicPenalty - records[0].bicPenalty, 4)} for the same two parameters, which the same eight units do not cover.`}</p>
        </div>
      ))}
    </div>
    <Table caption="The two constructed records, computed from the same maximized log-likelihoods"
      headings={['model', 'maximized log likelihood', 'k', '−2ℓ', 'AIC', 'BIC']}
      rows={records.map(record => [record.model, round(record.logLikelihood, 0), record.parameters, round(record.fit, 0),
        fixed(record.aic, 4), fixed(record.bic, 4)])} />
    <div className="rg-panel">
      <h4>The declared sixteen-bit code, in bits</h4>
      {codes.map(code => <div key={code.message}>
        <p className="rg-bits">
          {code.message} → <span className="rg-flag">{code.mode}</span>|<span className="rg-payload">{code.payload}</span>
          {' '}= {code.totalBits} bits
        </p>
        <p>
          {code.repeats
            ? `Mode 1: the flag plus a four-bit pattern the receiver repeats four times. The pattern was learned from the data, so its four bits are paid for. Decoding gives back ${code.decoded}.`
            : 'Mode 0: no four-bit pattern repeated four times reproduces this message, so the scheme falls back to the flag plus sixteen literal bits.'}
        </p>
      </div>)}
      <p>
        The first flag makes the remaining length unambiguous; both parties already know the message is sixteen bits and know the repeat
        rule. Never charge one bit for a “perfect” explanation: identifying which of sixteen patterns was chosen costs four. This code is
        deliberately limited, and refined universal codes are a formula explanation in the section, not an operational compressor here.
      </p>
    </div>
  </figure>;
}
