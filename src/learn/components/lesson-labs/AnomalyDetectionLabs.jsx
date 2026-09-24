import { useState } from 'react';
import {
  correction, firstCutIntervals, isolationExpectations, isolationLimits, isolationPath,
  kernelBoundary, kernelLimits, lofLimits, lofModeComparison, lofState, alarmCounts,
} from '../../data/anomaly-detection-models';
import { Field, Investigation, NumberField, NumberLine, Table, integer, percent, round } from './AnomalyDetectionShared.jsx';
import './anomaly-detection-labs.css';

const name = id => `P${id}`;

/** §3. Isolation: which position do random cuts separate first? */
const isolationPresets = {
  gap: { label: 'One distant position', values: [0, 1, 2, 3, 12] },
  regular: { label: 'Regular spacing', values: [0, 1, 2, 3, 4] },
  identical: { label: 'Identical records', values: [2, 2, 2, 2, 2] },
};
export function IsolationLab() {
  const [values, setValues] = useState(isolationPresets.gap.values);
  const [depthCap, setDepthCap] = useState(3);
  const [query, setQuery] = useState(4);
  const [cuts, setCuts] = useState([6, 1.5, 0.5, 2.5, 3.5]);

  const expectations = isolationExpectations(values, depthCap);
  const cutIntervals = firstCutIntervals(values);
  const walk = isolationPath(values, cuts, query, depthCap);
  
  const reset = () => {
    setValues(isolationPresets.gap.values); setDepthCap(3); setQuery(4); setCuts([6, 1.5, 0.5, 2.5, 3.5]); 
  };
  const setValue = (index, next) => setValues(values.map((value, position) => (position === index ? next : value)));
  return <Investigation
    title="Which position do random cuts separate first?"
    question="A cut is drawn uniformly between the smallest and largest position, then again inside whichever side holds the query. An empty gap gives the position beyond it many chances to be separated early. Edit the positions and watch the exact expectation over every cut sequence update."
    onReset={reset}>
    <div className="ad-controls">
      {values.map((value, index) => (
        <NumberField key={index} label={`${name(index)} position`} value={value} min={isolationLimits.coordinate.minimum}
          max={isolationLimits.coordinate.maximum} step="0.5" onChange={next => setValue(index, next)} />
      ))}
      <Field label="Depth cap">
        <select value={depthCap} onChange={event => setDepthCap(Number(event.target.value))}>
          {[1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value}</option>)}
        </select>
      </Field>
    </div>
    <div className="ad-buttons">
      {Object.entries(isolationPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => setValues(preset.values)}>{preset.label}</button>
      ))}
    </div>
    <p className="lesson-live-note">{expectations.constant ? 'With every position identical no cut can separate anything, so all five corrected paths equal c(5) and the scores are all 0.5. Equal scores give no ranking.' : `The normalizer is c(${values.length}) = ${expectations.normalizer.toString()}.`}</p>
    <NumberLine from={Math.min(...values) - 0.5} to={Math.max(...values) + 0.5} height={104}
      caption="The positions, with each gap a first cut can land in"
      describe={`Positions ${values.map((value, index) => `${name(index)} at ${round(value)}`).join(', ')}. ${cutIntervals.constant ? 'Every position is identical, so there is no gap to cut.' : cutIntervals.intervals.map(interval => `A cut between ${round(interval.from.toNumber())} and ${round(interval.to.toNumber())} has probability ${interval.probability.toString()}.`).join(' ')}`}>
      {(project, baseline) => <>
        {cutIntervals.intervals.map((interval, index) => {
          const left = project(interval.from.toNumber());
          const width = project(interval.to.toNumber()) - left;
          const singleton = interval.right.length === 1 || interval.left.length === 1;
          return <g key={index}>
            <rect className={`ad-span${singleton ? ' is-selected' : ''}`} x={left} y="16" width={width} height={baseline - 16} />
            {width > 26 && <text x={left + width / 2} y="12" textAnchor="middle">{interval.probability.toString()}</text>}
          </g>;
        })}
        {values.map((value, index) => <g key={index}>
          <circle className={`ad-point${index === query ? ' is-query' : ''}`} cx={project(value)} cy={baseline} r="5" />
          <text x={project(value)} y={baseline - 8} textAnchor="middle">{name(index)}</text>
        </g>)}
      </>}
    </NumberLine>
    {<>
      <Table caption="Every gap a first cut can fall in, and what it would separate" headings={['gap', 'width', 'probability', 'goes left', 'goes right']} rows={cutIntervals.constant ? [['no gap', '0', '—', 'every position', 'nothing']] : cutIntervals.intervals.map(interval => [`${round(interval.from.toNumber())} to ${round(interval.to.toNumber())}`, interval.width.toString(), interval.probability.toString(), interval.left.map(name).join(', '), interval.right.map(name).join(', ')])} />
      <Table caption={`Expected corrected path and score, integrated exactly over every cut sequence to depth ${depthCap}`} headings={['position', 'value', 'expected corrected path', 'score']} rows={expectations.rows.map(row => [name(row.id), round(row.value.toNumber()), `${row.meanPath.toString()} ≈ ${round(row.meanPath.toNumber())}`, row.score === null ? 'undefined' : round(row.score, 6)])} rowClass={index => !expectations.constant && expectations.rows[index].id === expectations.shortest ? 'is-selected' : undefined} />
      <p className="ad-caption">These are expectations over the cut construction, not the output of any finite random forest. The score rescales a path: 0.71 is not a 71% chance of a fault. A corrected path equal to the normalizer gives exactly 0.5.</p>
    </>}
    <details>
      <summary>Follow one chosen sequence of cuts instead of the average</summary>
      <div className="ad-controls">
        <Field label="Query position">
          <select value={query} onChange={event => setQuery(Number(event.target.value))}>
            {values.map((value, index) => <option key={index} value={index}>{name(index)} at {round(value)}</option>)}
          </select>
        </Field>
        {cuts.slice(0, depthCap).map((cut, index) => (
          <NumberField key={index} label={`Cut ${index + 1}`} value={cut} min={isolationLimits.coordinate.minimum}
            max={isolationLimits.coordinate.maximum} step="0.5"
            onChange={next => setCuts(cuts.map((value, position) => (position === index ? next : value)))} />
        ))}
      </div>
      <div className="ad-tree">
        {walk.steps.map(step => <div className="ad-tree-step" key={step.depth}>
          <span>depth {step.depth}</span>
          <span className="ad-tree-node is-path">
            {step.members.map(name).join(', ')}
            {step.usable
              ? ` — cut at ${step.cut.toString()} sends the query ${step.side}: {${(step.side === 'left' ? step.leftIds : step.rightIds).map(name).join(', ')}}`
              : ' — no usable cut here, so this node is terminal'}
          </span>
        </div>)}
      </div>
      <p className="ad-readout" aria-live="polite">
        <span className="ad-leaf-badge">{walk.leaf.size} row{walk.leaf.size === 1 ? '' : 's'} remain; depth {walk.leaf.depth}</span>{' '}
        Corrected path {walk.leaf.depth} + c({walk.leaf.size}) = {walk.pathLength.toString()}; normalized score
        2^(-{walk.pathLength.toString()} / {walk.normalizer.toString()}) = {walk.score === null ? 'undefined' : round(walk.score, 10)}.
        Stopping at a leaf of {walk.leaf.size} rows does not mean those rows were separated one by one, which is what c({walk.leaf.size}) = {correction(walk.leaf.size).toString()} corrects for.
      </p>
    </details>
  </Investigation>;
}

/** §4. LOF: a neighbourhood made of neighbours that have neighbourhoods. */
const defaultReferences = [0, 1, 2, 20, 24, 28];
function lofRows(state, target) {
  return target.neighbours.map(entry => [
    name(entry.id), round(entry.value.toNumber()), entry.distance.toString(), entry.radius.toString(),
    entry.reach.toString(), state.rows[entry.id].density.toString(),
  ]);
}
export function LofNeighbourhoodLab() {
  const [references, setReferences] = useState(defaultReferences);
  const [k, setK] = useState(2);
  const [query, setQuery] = useState(4);
  const [error, setError] = useState(null);

  let state = null;
  let four = null;
  let seventeen = null;
  let current = null;
  try {
    state = lofState(references, k);
    four = lofState(references, k, 4).query;
    seventeen = lofState(references, k, 17).query;
    current = lofState(references, k, query).query;
  } catch (problem) {
    return <Investigation title="Compare a neighbourhood with its neighbours' neighbourhoods"
      question="This exact calculation needs distinct reference coordinates." onReset={() => { setReferences(defaultReferences); setK(2); setQuery(4); setError(null); }}>
      <p className="ad-note">{problem.message}</p>
    </Investigation>;
  }
  const pairAnswer = four.factor.compare(seventeen.factor) === 0 ? 'equal' : four.factor.compare(seventeen.factor) > 0 ? 'four' : 'seventeen';
  const isDefaultReference = references.every((value, index) => value === defaultReferences[index]);

  const reset = () => { setReferences(defaultReferences); setK(2); setQuery(4); setError(null);   };
  return <Investigation
    title="Compare a neighbourhood with its neighbours' neighbourhoods"
    question={`Six reference rows sit in two groups with different spacing. Each neighbour contributes its own radius as a floor, so a query pressed against a tightly packed group is judged differently from one beside a loose group. Neighbours are exactly ${k} other rows, with ties broken by row order.`}
    onReset={reset}>
    <p className="lesson-live-note">{pairAnswer === 'equal' ? `Both give ${four.factor.toString()} with these references and k = ${k}. Their mean-neighbour-density to query-density ratios agree; equality does not require identical neighbours or identical distances.` : `LOF(4) = ${four.factor.toString()} and LOF(17) = ${seventeen.factor.toString()}. Each ratio compares the selected neighbours' mean density with that query's own density. The tables below show whose radius sets each floor for these edited references.`}</p>
    {<>
      <Table caption={`Query 4: its ${k} chosen neighbour${k === 1 ? '' : 's'}, each with its own radius as the floor`} headings={['neighbour', 'at', 'distance d(p, o)', 'radius r(o)', 'reach = max', 'lrd(o)']} rows={lofRows(state, four)} />
      <Table caption={`Query 17: nearest-reference distance ${seventeen.neighbours[0].distance.toString()}, and a factor ${pairAnswer === 'equal' ? 'equal to' : pairAnswer === 'four' ? 'lower than' : 'higher than'} query 4`} headings={['neighbour', 'at', 'distance d(p, o)', 'radius r(o)', 'reach = max', 'lrd(o)']} rows={lofRows(state, seventeen)} />
      <p className="ad-readout">
        Query 4: mean reach {four.neighbours.map(entry => entry.reach.toString()).join(' and ')} gives lrd {four.density.toString()}, and dividing the mean neighbour density by it gives {four.factor.toString()} ≈ {round(four.factor.toNumber())}.
        Query 17: mean reach {seventeen.neighbours.map(entry => entry.reach.toString()).join(' and ')} gives lrd {seventeen.density.toString()} and factor {seventeen.factor.toString()} ≈ {round(seventeen.factor.toNumber())}.
      </p>
    </>}
    <div className="ad-controls">
      <NumberField label="Query position" value={query} min={lofLimits.coordinate.minimum} max={lofLimits.coordinate.maximum} step="0.5" onChange={setQuery} />
      <Field label="Neighbours k (other rows)">
        <select value={k} onChange={event => setK(Number(event.target.value))}>
          {[1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value}</option>)}
        </select>
      </Field>
      {references.map((value, index) => (
        <NumberField key={index} label={`Reference ${name(index)}`} value={value} min={lofLimits.coordinate.minimum}
          max={lofLimits.coordinate.maximum} step="0.5"
          onChange={next => {
            const updated = references.map((current, position) => (position === index ? next : current));
            if (updated.some((value, position) => updated.some((other, other_position) => other_position !== position && other === value))) {
              setError('Two references would share a coordinate. This exact calculation requires distinct coordinates: duplicate neighbourhoods can have zero mean reach and no finite reciprocal.');
              return;
            }
            setError(null);
            setReferences(updated);
          }} />
      ))}
    </div>
    {error && <p className="ad-note">{error}</p>}
    <p className="lesson-live-note">{current.factor === null ? 'Every reach is zero here, so the density is not finite.' : `LOF = ${current.factor.toString()} ≈ ${round(current.factor.toNumber())}. A factor near 1 means the query is supported about as well as its neighbours are; values below 1 are ordinary, not impossible.`}</p>
    <NumberLine from={Math.min(...references, query) - 1.5} to={Math.max(...references, query) + 1.5} height={174} bottomPadding={42} tickSide="above"
      caption="References, the query, and the reach used for each chosen neighbour"
      describe={`References at ${references.map(round).join(', ')}; query at ${round(query)}. Chosen neighbours: ${current.neighbours.map(entry => `${name(entry.id)} at distance ${entry.distance.toString()} with radius ${entry.radius.toString()}, giving reach ${entry.reach.toString()}`).join('; ')}.`}>
      {(project, baseline) => <>
        {current.neighbours.map((entry, index) => {
          const y = 34 + index * 15;
          const from = project(Math.min(query, entry.value.toNumber()));
          const to = project(Math.max(query, entry.value.toNumber()));
          const radiusWidth = Math.abs(project(entry.radius.toNumber()) - project(0));
          const radiusEnd = project(entry.value.toNumber()) + radiusWidth <= 306
            ? project(entry.value.toNumber()) + radiusWidth
            : project(entry.value.toNumber()) - radiusWidth;
          const label = `${name(entry.id)}: max(${entry.distance.toString()}, ${entry.radius.toString()}) = ${entry.reach.toString()}`;
          // Put the annotation on whichever side of the span still has room.
          const flip = to + 8 + label.length * 6.3 > 316;
          return <g key={entry.id}>
            <line className="ad-reach" x1={from} x2={to} y1={y} y2={y} />
            <line className="ad-radius" x1={project(entry.value.toNumber())} x2={radiusEnd} y1={y - 5} y2={y - 5} />
            <text x={flip ? from - 8 : to + 8} y={y + 4} textAnchor={flip ? 'end' : 'start'}>{label}</text>
          </g>;
        })}
        {references.map((value, index) => <g key={index}>
          <circle className={`ad-point${current.neighbours.some(entry => entry.id === index) ? ' is-neighbour' : ''}`} cx={project(value)} cy={baseline} r="5" />
          <text x={project(value)} y={baseline + (index % 2 ? 26 : 14)} textAnchor="middle">{name(index)}</text>
        </g>)}
        <circle className="ad-point is-query" cx={project(query)} cy={baseline} r="6" />
        <text className="ad-point is-query" x={project(query)} y={baseline - 10} textAnchor="middle">query</text>
      </>}
    </NumberLine>
    {<>
      <Table caption={`The query's ${k} chosen neighbour${k === 1 ? '' : 's'}, and the floor each one contributes`} headings={['neighbour', 'at', 'distance d(p, o)', 'radius r(o)', 'reach = max', 'lrd(o)']} rows={lofRows(state, current)} />
      <p className="ad-readout" aria-live="polite">
        Mean reach {current.neighbours.map(entry => entry.reach.toString()).join(' + ')} over {k} gives lrd(query) = {current.density === null ? 'undefined' : current.density.toString()};
        the mean of the neighbours' densities divided by it gives LOF = {current.factor === null ? 'undefined' : current.factor.toString()}.
      </p>
      <Table caption="Every reference row's own factor, for comparison" headings={['row', 'at', 'radius', 'lrd', 'LOF']} rows={state.rows.map(row => [name(row.id), round(row.value.toNumber()), row.radius.toString(), row.density.toString(), row.factor.toString()])} />
      <p className="ad-caption">{isDefaultReference ? 'For the original six references, setting k = 3 makes both query 4 and query 17 score 1. That is a fact about these coordinates, not a rule that larger k always lowers a factor.' : 'You changed the reference geometry. Compare both queries again after changing k: the original two-group result need not survive these edits. Each radius and density is recomputed from the current references.'}</p>
    </>}
  </Investigation>;
}

/** §5. The same coordinate, scored two ways. */
export function LofModeLab() {
  const [coordinate, setCoordinate] = useState(1);
  const [k, setK] = useState(2);
  
  const comparison = lofModeComparison(defaultReferences, k, coordinate);
  
  return <Investigation
    title="A reference row and a new query at the same place"
    question="A reference row leaves its own identity out of its neighbourhood. A new query has no identity here, so a reference sitting at the same coordinate is an ordinary neighbour at distance zero. That is a difference in the mathematics, not a library defect."
    onReset={() => { setCoordinate(1); setK(2);  }}>
    <div className="ad-controls">
      <Field label="Coordinate to compare">
        <select value={coordinate} onChange={event => setCoordinate(Number(event.target.value))}>
          {defaultReferences.map(value => <option key={value} value={value}>{value} (reference {name(defaultReferences.indexOf(value))})</option>)}
        </select>
      </Field>
      <Field label="Neighbours k">
        <select value={k} onChange={event => setK(Number(event.target.value))}>
          {[1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value}</option>)}
        </select>
      </Field>
    </div>
    <p className="lesson-live-note">{comparison.trainingRow === null ? '' : comparison.agree ? `Both give ${comparison.trainingRow.factor.toString()} here, from different neighbour sets: the reference row uses {${comparison.trainingRow.neighbours.map(entry => name(entry.id)).join(', ')}} and the query uses {${comparison.queryRow.neighbours.map(entry => name(entry.id)).join(', ')}}. Two contracts landing on one number is not the same as one contract.` : `The reference row gives ${comparison.trainingRow.factor.toString()} and the new query gives ${comparison.queryRow.factor.toString()}, because the query can choose the reference standing at its own coordinate.`}</p>
    {<>
      <div className="ad-fitting-strips">
      <style>{'.ad-fitting-strips svg text { font-size: 17px; }'}</style>
      {[['Reference row: exclude its own identity', comparison.trainingRow, true], ['New query: the coincident reference remains available', comparison.queryRow, false]].map(([caption, target, training]) => <div key={caption}><NumberLine from={-2} to={30} height={180} tickSide="above" caption={caption} describe={`${caption}. Selected neighbours ${target.neighbours.map(entry => name(entry.id)).join(', ')}. ${training ? `${name(comparison.trainingRow.id)} is excluded.` : `Query Q is separate from reference ${name(comparison.trainingRow.id)}, although both sit at ${coordinate}.`}`}>
        {(project, baseline) => <>
          {target.neighbours.map(entry => <line key={entry.id} className="ad-reach" x1={project(coordinate)} x2={project(entry.value.toNumber())} y1={baseline - 100} y2={baseline} />)}
          {defaultReferences.map((value, id) => {
            const selected = target.neighbours.some(entry => entry.id === id);
            const excluded = training && comparison.trainingRow.id === id;
            // Labels may move; their leaders return to the unchanged coordinates.
            // Fan out the close 0/1/2 labels instead of shrinking them on phones.
            const labelX = project(value) + (id < 3 ? (id - 1) * 24 : 0);
            const labelY = baseline - (id % 2 ? 61 : 38);
            return <g key={id}>
              <line className="ad-grid" x1={project(value)} x2={labelX} y1={baseline - 7} y2={labelY + 5} />
              <circle className={`ad-point${selected ? ' is-neighbour' : ''}`} cx={project(value)} cy={baseline} r="5" />
              {excluded && <path d={`M${project(value) - 7},${baseline - 7} l14,14 m0,-14 l-14,14`} stroke="#da9c86" strokeWidth="2" />}
              <rect x={labelX - 13} y={labelY - 15} width="26" height="20" fill="#0b0f10" />
              <text x={labelX} y={labelY} textAnchor="middle">{name(id)}</text>
            </g>;
          })}
          <circle className="ad-point is-query" cx={project(coordinate)} cy={baseline - 100} r="6" />
          <text x={Math.max(60, Math.min(280, project(coordinate)))} y={baseline - 112} textAnchor="middle">{training ? 'row' : 'Q'} at {coordinate}</text>
        </>}
      </NumberLine>
      <p className="ad-caption">Selected neighbours: {target.neighbours.map(entry => `${name(entry.id)} at ${round(entry.value.toNumber())}`).join('; ')}.
        {' '}{training ? `${name(comparison.trainingRow.id)} at ${coordinate} is crossed out because its own identity is excluded.` : `Q at ${coordinate} is a new observation; reference ${name(comparison.trainingRow.id)} at the same coordinate remains available.`}
        {' '}The lines connect identities; horizontal position alone represents the coordinate.</p>
      </div>)}
      </div>
      <p className="ad-caption">Reference row {name(comparison.trainingRow.id)}: its own identity is excluded.</p>
      <Table caption="Training-row neighbours" headings={['neighbour', 'distance', 'radius', 'reach', 'lrd(o)']} rows={comparison.trainingRow.neighbours.map(entry => [name(entry.id), entry.distance.toString(), entry.radius.toString(), entry.reach.toString(), comparison.references[entry.id].density.toString()])} />
      <p className="ad-caption">For a new query at the same coordinate, every reference is available.</p>
      <Table caption="New-query neighbours" headings={['neighbour', 'distance', 'radius', 'reach', 'lrd(o)']} rows={comparison.queryRow.neighbours.map(entry => [name(entry.id), entry.distance.toString(), entry.radius.toString(), entry.reach.toString(), comparison.references[entry.id].density.toString()])} />
      <p className="ad-readout" aria-live="polite">
        Reference factor {comparison.trainingRow.factor.toString()}; new-query score {comparison.queryRow.factor === null ? 'undefined' : comparison.queryRow.factor.toString()}.
        Use the fitted rows' own factors to review the reference collection, and query scores for later observations. Pooling them as if they were the same held-out measurement compares two different neighbourhood contracts.
      </p>
    </>}
  </Investigation>;
}

/** §6. The kernel support boundary from two reference points. */
export function KernelBoundaryLab() {
  const [anchor, setAnchor] = useState(1);
  const [gamma, setGamma] = useState(1);
  const [query, setQuery] = useState(0);
  
  const boundary = kernelBoundary(anchor, gamma, query);
  const scaleX = value => 34 + 272 * (value - kernelLimits.query.minimum) / (kernelLimits.query.maximum - kernelLimits.query.minimum);
  const top = Math.max(1, boundary.rho * 1.15, ...boundary.curve.map(point => point.sum));
  const scaleSum = value => 96 - 76 * value / top;
  const extreme = Math.max(0.08, ...boundary.curve.map(point => Math.abs(point.decision)));
  // The lower panel is drawn in its own band, so the curve cannot reach the
  // prose underneath however deep the decision goes.
  const scaleDecision = value => 52 - 40 * value / extreme;
  return <Investigation
    title="Two reference points, a similarity sum and an offset"
    question="With two reference observations and nu = 1/2, symmetry fixes both weights at 1/2 and puts both references exactly on the boundary. That closed solution lets you watch gamma reshape the accepted region without a solver in the way."
    onReset={() => { setAnchor(1); setGamma(1); setQuery(0);  }}>
    <div className="ad-controls">
      <NumberField label="Reference anchors at ±a" value={anchor} min={kernelLimits.anchor.minimum} max={kernelLimits.anchor.maximum} step="0.1" decimals={2} onChange={setAnchor} />
      <NumberField label="Kernel width gamma" value={gamma} min={kernelLimits.gamma.minimum} max={kernelLimits.gamma.maximum} step="0.05" decimals={2} onChange={setGamma} />
      <NumberField label="Query x" value={query} min={kernelLimits.query.minimum} max={kernelLimits.query.maximum} step="0.1" decimals={2} onChange={setQuery} />
    </div>
    <div className="ad-buttons">
      <button type="button" onClick={() => setQuery(anchor)}>Put the query on a reference point</button>
      <button type="button" onClick={() => setQuery(0)}>Put the query at the midpoint</button>
    </div>
    <p className="lesson-live-note">{`g(${round(query)}) = ${round(boundary.decision, 6)}, with rho = ${round(boundary.rho, 6)}. A reference point scores exactly zero at every gamma in this symmetric solution.`}</p>
    {<><figure className="ad-plot">
      <figcaption>Each reference contributes a bump; their weighted sum is compared with rho</figcaption>
      <svg viewBox="0 0 340 232" role="img" aria-label={`Two similarity bumps centred at minus ${round(anchor)} and ${round(anchor)}, their weighted sum, and the horizontal offset rho at ${round(boundary.rho, 4)}. Below, the signed decision with its zero line. ${boundary.positiveIntervals.length === 0 ? 'No strictly positive interval is resolved by this numerical scan.' : `The resolved accepted intervals run ${boundary.positiveIntervals.map(([from, to]) => `from ${round(from, 3)} to ${round(to, 3)}`).join(' and ')}.`}`}>
        <line className="ad-axis" x1="34" x2="306" y1="96" y2="96" />
        {[-4, -2, 0, 2, 4].map(value => <g key={value}>
          <line className="ad-grid" x1={scaleX(value)} x2={scaleX(value)} y1="14" y2="96" />
          <text x={scaleX(value)} y="108" textAnchor="middle">{value}</text>
        </g>)}
        <polyline className="ad-curve" stroke="#8eb9a5" points={boundary.curve.map(point => `${scaleX(point.x)},${scaleSum(point.left)}`).join(' ')} />
        <polyline className="ad-curve" stroke="#91aecf" points={boundary.curve.map(point => `${scaleX(point.x)},${scaleSum(point.right)}`).join(' ')} />
        <polyline className="ad-curve" stroke="#e7b94a" strokeWidth="2.2" points={boundary.curve.map(point => `${scaleX(point.x)},${scaleSum(point.sum)}`).join(' ')} />
        <line className="ad-threshold" x1="34" x2="306" y1={scaleSum(boundary.rho)} y2={scaleSum(boundary.rho)} />
        <text x="308" y={scaleSum(boundary.rho) + 4}>rho</text>
        <text x="34" y="12">kernel compatibility sum</text>
        <line className="ad-axis" x1="34" x2="306" y1={scaleDecision(0)} y2={scaleDecision(0)} transform="translate(0 128)" />
        <g transform="translate(0 128)">
          {boundary.positiveIntervals.map(([from, to], index) => <rect key={index} className="ad-span is-selected" x={scaleX(from)} y="10" width={scaleX(to) - scaleX(from)} height="84" />)}
          <polyline className="ad-curve" stroke="#d9d3bf" strokeWidth="2" points={boundary.curve.map(point => `${scaleX(point.x)},${scaleDecision(point.decision)}`).join(' ')} />
          <circle className="ad-point is-query" cx={scaleX(query)} cy={scaleDecision(boundary.decision)} r="5" />
          <text x="34" y="4">signed decision g(x), shaded where positive</text>
          <text x={scaleX(query)} y={scaleDecision(boundary.decision) - 9} textAnchor="middle">{round(boundary.decision, 4)}</text>
        </g>
      </svg>
    </figure>
    <p className="ad-readout" aria-live="polite">
      Contributions {round(boundary.contributions[0], 6)} and {round(boundary.contributions[1], 6)} sum to {round(boundary.sum, 6)}; subtracting rho = {round(boundary.rho, 6)} gives {round(boundary.decision, 6)}.
      {boundary.positiveIntervals.length === 0 ? ' No strictly positive interval is resolved by this numerical scan. Very narrow positive intervals near the anchors can be smaller than its resolution; this is not an exact proof that every other point falls outside.' : boundary.positiveIntervals.length === 1 ? ` The accepted region is the single interval from ${round(boundary.positiveIntervals[0][0], 4)} to ${round(boundary.positiveIntervals[0][1], 4)}.` : ` The accepted region has ${boundary.positiveIntervals.length} separated pieces: ${boundary.positiveIntervals.map(([from, to]) => `${round(from, 4)} to ${round(to, 4)}`).join(', ')}. A hyperplane in the kernel's feature space need not be one blob back in the original coordinates.`}
    </p></>}
    <p className="ad-caption">The vertical axes are a kernel compatibility sum and a signed decision. Neither is a probability, and neither is a distance in the original coordinate. Query classification treats |g| ≤ 10⁻¹² as boundary to allow for floating-point arithmetic. Doubling the anchors while dividing gamma by four preserves corresponding decisions when the query coordinates are doubled too.</p>
  </Investigation>;
}

/** §8. A good rate still has to survive the population it runs on. */
export function AlertPopulationLab() {
  const [population, setPopulation] = useState(100000);
  const [prevalence, setPrevalence] = useState(0.1);
  const [sensitivity, setSensitivity] = useState(80);
  const [falseRate, setFalseRate] = useState(1);
  const [budget, setBudget] = useState(200);

  const counts = alarmCounts(population, prevalence / 100, sensitivity / 100, falseRate / 100, budget);

  const widest = Math.max(counts.trueAlerts, counts.falseAlerts, 1);
  return <Investigation
    title="Turn two rates and a prevalence into a review queue"
    question="A detector that catches most faults and flags a small fraction of everything else can still bury a team, because the population of ordinary observations is so much larger. Enter the rates and see the queue."
    onReset={() => { setPopulation(100000); setPrevalence(0.1); setSensitivity(80); setFalseRate(1); setBudget(200);   }}>
    <div className="ad-controls">
      <NumberField label="Observations N" value={population} min={100} max={1000000} step="100" decimals={0} onChange={setPopulation} />
      <NumberField label="Fault prevalence (percent)" value={prevalence} min={0} max={100} step="0.05" decimals={3} onChange={setPrevalence} suffix="%" />
      <NumberField label="Sensitivity: faults flagged (percent)" value={sensitivity} min={0} max={100} step="1" decimals={2} onChange={setSensitivity} suffix="%" />
      <NumberField label="False-positive rate (percent)" value={falseRate} min={0} max={100} step="0.1" decimals={3} onChange={setFalseRate} suffix="%" />
      <NumberField label="Review budget (alerts)" value={budget} min={0} max={1000000} step="10" decimals={0} onChange={setBudget} />
    </div>
    <p className="lesson-live-note">{counts.precision === null ? 'With these rates nothing is flagged, so precision is undefined rather than zero or one.' : `Expected alerts ${round(counts.total, 2)} of which ${round(counts.trueAlerts, 2)} are faults: ${percent(counts.precision, 2)}.`}</p>
    <p className="lesson-live-note">{counts.withinBudget ? `${round(counts.total, 2)} expected alerts against ${integer(budget)} slots. Spare capacity is not a reason to lower the threshold without deciding what the extra reviews are for.` : `${round(counts.total, 2)} expected alerts against ${integer(budget)} slots, short by ${round(counts.budgetShortfall, 2)}. Reviewing only the top ${integer(budget)} is a different operating point, so the sensitivity above no longer applies to it.`}</p>
    {false}
    {<>
      <div className="ad-bars" role="img" aria-label={`Population ${integer(population)}: ${round(counts.faults, 2)} faults and ${round(counts.nonFaults, 2)} others. Alerts ${round(counts.total, 2)}: ${round(counts.trueAlerts, 2)} faults flagged and ${round(counts.falseAlerts, 2)} others flagged.`}>
        <div className="ad-bar-row"><span>faults in N</span><span className="ad-bar-track"><span className="ad-bar-fill" style={{
          width: `${100 * counts.faults / population}%`
        }} /></span><span className="ad-bar-value">{round(counts.faults, 2)}</span></div>
        <div className="ad-bar-row"><span>everything else</span><span className="ad-bar-track"><span className="ad-bar-fill is-quiet" style={{
          width: `${100 * counts.nonFaults / population}%`
        }} /></span><span className="ad-bar-value">{round(counts.nonFaults, 2)}</span></div>
      </div>
      <p className="ad-caption">The two bars above share the population scale, so the faults are almost invisible. The two below share the alert scale instead, which is the denominator that decides precision.</p>
      <div className="ad-bars" role="img" aria-label={`Alert composition: ${round(counts.trueAlerts, 2)} faults and ${round(counts.falseAlerts, 2)} non-faults.`}>
        <div className="ad-bar-row"><span>faults flagged</span><span className="ad-bar-track"><span className="ad-bar-fill" style={{
          width: `${100 * counts.trueAlerts / widest}%`
        }} /></span><span className="ad-bar-value">{round(counts.trueAlerts, 2)}</span></div>
        <div className="ad-bar-row"><span>others flagged</span><span className="ad-bar-track"><span className="ad-bar-fill is-false" style={{
          width: `${100 * counts.falseAlerts / widest}%`
        }} /></span><span className="ad-bar-value">{round(counts.falseAlerts, 2)}</span></div>
      </div>
      <Table caption="Expected counts. These are averages over the stated rates, so a fractional count is an average and not a fraction of a record." headings={['quantity', 'value']} rows={[['faults present', round(counts.faults, 2)], ['faults flagged', round(counts.trueAlerts, 2)], ['faults missed', round(counts.missedFaults, 2)], ['others flagged', round(counts.falseAlerts, 2)], ['total alerts', round(counts.total, 2)], ['fraction of alerts that are faults', counts.precision === null ? 'undefined: no alerts' : percent(counts.precision, 4)], ['review budget', integer(budget)], ['over budget by', counts.withinBudget ? 'within budget' : round(counts.budgetShortfall, 2)]]} />
      <p className="ad-caption">Raising prevalence while holding both conditional rates fixed raises precision without changing the detector at all. Dropping the false-positive rate to zero makes every alert a fault, and dropping sensitivity to zero as well leaves no alerts and no defined precision.</p>
    </>}
  </Investigation>;
}
