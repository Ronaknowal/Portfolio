import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { BOOSTING_FIXTURE, fitBoosting, predictBoosting, boostingPredictionSegments, correctionLeaves, parseBoostingTargets, newtonInvestigation, histogramInvestigation, gossInvestigation, orderedStatistics, bundleExclusive, growthPolicies, validationInvestigation } from '../../data/gradient-boosted-trees-models.js';
import './gradient-boosted-trees-labs.css';
const gold = '#f0bd58';
const green = '#80d6b2';
const blue = '#8bbce8';

const format = value => Math.abs(value) < 1e-12 ? '0' : Number(value.toFixed(5)).toString();
function Lab({
  title,
  name,
  children
}) {
  const id = useId();
  return <section className="gbt-lab" data-gbt-lab={name} aria-labelledby={id} data-live-exploration>
    <h3 id={id}>{title}</h3>{children}
  </section>;
}
function Range({
  label,
  value,
  min,
  max,
  step = 1,
  onChange
}) {
  return <label>{label}: <strong>{format(value)}</strong><input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Select({
  label,
  value,
  onChange,
  options
}) {
  return <label>{label}<select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, title]) => <option key={key} value={key}>{title}</option>)}</select></label>;
}
function Readouts({
  rows
}) {
  return <dl className="gbt-readouts" aria-live="polite">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function DataTable({
  caption,
  headings,
  rows
}) {
  return <details className="gbt-data"><summary>{caption}</summary><div role="region" aria-label={caption} tabIndex={0}><table><thead><tr>{headings.map(heading => <th scope="col" key={heading}>{heading}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((value, column) => column === 0 ? <th scope="row" key={column}>{value}</th> : <td key={column}>{value}</td>)}</tr>)}</tbody></table></div></details>;
}

/** Match the SVG coordinate width to its actual container: labels do not shrink
 * into unreadable miniature text on narrow screens. Extents use every series.
 */
function Chart({
  title,
  series,
  xLabel,
  yLabel,
  selected,
  cursor,
  zero = false,
  nonnegative = false
}) {
  const ref = useRef(null);
  const [width, setWidth] = useState(320);
  const id = useId();
  useEffect(() => {
    const element = ref.current;
    const observer = new ResizeObserver(entries => setWidth(Math.max(220, entries[0].contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const points = series.flatMap(item => item.points);
  const xMin = Math.min(...points.map(point => point[0]));
  const xMax = Math.max(...points.map(point => point[0]));
  const rawMin = Math.min(...points.map(point => point[1]), ...(zero ? [0] : []));
  const rawMax = Math.max(...points.map(point => point[1]), ...(zero ? [0] : []));
  const padding = Math.max(0.15, (rawMax - rawMin) * 0.13);
  const signedExtent = Math.max(Math.abs(rawMin), Math.abs(rawMax)) + padding;
  const yMin = nonnegative ? Math.max(0, rawMin - padding) : zero ? -signedExtent : rawMin - padding;
  const yMax = zero && !nonnegative ? signedExtent : rawMax + padding;
  const left = 44;
  const right = width - 14;
  const top = 20;
  const bottom = 192;
  const x = value => left + (value - xMin) / (xMax - xMin || 1) * (right - left);
  const y = value => bottom - (value - yMin) / (yMax - yMin) * (bottom - top);
  const tick = value => Math.abs(value) >= 100 ? value.toFixed(0) : Number(value.toFixed(2)).toString();
  return <figure className="gbt-chart" ref={ref}>
    <figcaption>{title}</figcaption>
    <p className="gbt-axis-note">{yLabel}</p>
    <svg viewBox={`0 0 ${width} 225`} role="img" aria-labelledby={id}>
      <title id={id}>{title}. {series.map(item => item.label).join('; ')}. Full values are available in the table.</title>
      {[0, .5, 1].map(fraction => {
        const value = yMin + fraction * (yMax - yMin);
        return <g key={fraction}><line x1={left} x2={right} y1={y(value)} y2={y(value)} className="gbt-grid" /><text x={left - 6} y={y(value) + 4} textAnchor="end">{tick(value)}</text></g>;
      })}
      {zero && <line x1={left} x2={right} y1={y(0)} y2={y(0)} className="gbt-zero" />}
      {cursor !== undefined && <line x1={x(cursor)} x2={x(cursor)} y1={top} y2={bottom} stroke={gold} strokeDasharray="4 4" />}
      {series.map(item => <g key={item.label}>
        {item.line && <polyline points={item.points.flatMap((point, index) => {
          if (!item.step || index === 0) return [`${x(point[0])},${y(point[1])}`];
          const previous = item.points[index - 1];
          const boundary = (previous[0] + point[0]) / 2;
          return [`${x(boundary)},${y(previous[1])}`, `${x(boundary)},${y(point[1])}`, `${x(point[0])},${y(point[1])}`];
        }).join(' ')} fill="none" stroke={item.color} strokeWidth="2" strokeDasharray={item.dashed ? '4 3' : undefined} />}
        {item.stems && item.points.map((point, index) => <line key={index} x1={x(point[0])} x2={x(point[0])} y1={y(0)} y2={y(point[1])} stroke={item.color} strokeWidth="3" opacity=".65" />)}
        {!item.hidePoints && item.points.map((point, index) => <circle key={index} cx={x(point[0])} cy={y(point[1])} r={index === selected ? 5 : 3} fill={item.open ? '#111719' : item.color} stroke={item.color} strokeWidth="2" />)}
      </g>)}
      <text x={left} y={212}>{tick(xMin)}</text><text x={right} y={212} textAnchor="end">{tick(xMax)}</text>
    </svg>
    <p className="gbt-axis-note">{xLabel}</p>
    <div className="gbt-legend">{series.map(item => <span key={item.label} style={{
        color: item.color
      }}>{item.open ? '○' : item.dashed ? '┄' : '━'} {item.label}</span>)}</div>
  </figure>;
}
export function AdditivePredictionFigure() {
  const model = fitBoosting({
    rounds: 2
  });
  const feature = 4.5;
  return <figure className="gbt-inline gbt-additive">
    <figcaption>A new row supplies x = 4.5. It does not supply y.</figcaption>
    <div className="gbt-additive-route"><div><span>Start</span><strong>5</strong><small>mean baseline</small></div>{model.trees.map((tree, index) => <div key={index}><span>Tree {index + 1}: x ≤ {tree.threshold}?</span><strong>No → {format(tree.right.value)}</strong><small>× rate 0.5 → +{format(.5 * tree.right.value)}</small></div>)}<div><span>Add saved scores</span><strong>{format(predictBoosting(model, feature))}</strong><small>final numeric prediction</small></div></div>
    <p>The two trees were fitted in sequence. Prediction visits both saved trees and adds their contributions.</p>
  </figure>;
}
export function BoostingCorrectionLab() {
  const [targets, setTargets] = useState([...BOOSTING_FIXTURE.y]);
  const [draft, setDraft] = useState('2, 2, 3, 7, 8, 8');
  const [error, setError] = useState('');
  const [rate, setRate] = useState(.5);
  const [round, setRound] = useState(1);
  const [row, setRow] = useState(3);
  const model = useMemo(() => fitBoosting({
    y: targets,
    rate,
    rounds: 8
  }), [targets, rate]);
  const stage = model.stages[round];
  const leaves = correctionLeaves(stage.tree);
  function reset() {
    setTargets([...BOOSTING_FIXTURE.y]);
    setDraft('2, 2, 3, 7, 8, 8');
    setRate(.5);
    setRound(1);
    setRow(3);
    setError('');
  }
  return <Lab title="Fit a correction; then add it" name="correction">
    <p>The hollow circles show the current predictor. Residual stems say how far each observed target lies above or below it. A leaf gives all its rows the same correction.</p>
    <form onSubmit={event => {
      event.preventDefault();
      try {
        setTargets(parseBoostingTargets(draft));
        setRound(1);
        setError('');
      } catch (failure) {
        setError(failure.message);
      }
    }}><label>Six target values<input aria-label="Six target values" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Apply targets and restart</button></form>
    {error && <p role="alert" className="gbt-error">{error} The previous valid data remain active.</p>}
    <div className="gbt-controls"><Range label="Correction learning rate" value={rate} min={0} max={1.5} step={.1} onChange={value => {
        setRate(value);
        setRound(1);
      }} /><Select label="Inspect row" value={row} onChange={value => setRow(Number(value))} options={model.x.map((value, index) => [index, `row ${index + 1}, x=${value}`])} /></div>
    <div className="gbt-plots"><Chart title={`Before tree ${round}`} xLabel="Feature x" yLabel="Target / current prediction" selected={row} series={[{
        label: 'Observed target',
        color: green,
        points: model.x.map((x, i) => [x, targets[i]])
      }, {
        label: 'Current prediction',
        color: blue,
        open: true,
        points: model.x.map((x, i) => [x, stage.before[i]])
      }]} /><Chart title="The tree fits these residuals" xLabel="Feature x" yLabel="Signed target − prediction" zero selected={row} series={[{
        label: 'Residual',
        color: green,
        stems: true,
        points: model.x.map((x, i) => [x, stage.residual[i]])
      }, {
        label: 'Leaf correction',
        color: gold,
        line: true,
        step: true,
        points: model.x.map((x, i) => [x, stage.correction[i]])
      }]} /></div>
    <div className="gbt-leaf-bands">{leaves.map(leaf => <div key={leaf.id}><strong>Rows {leaf.indices.map(index => index + 1).join(', ')}</strong><span>mean residual = {format(leaf.value)}</span><span>add {format(rate * leaf.value)} to each prediction</span></div>)}</div>
    <Readouts rows={[[`Row ${row + 1}: before + scaled leaf`, `${format(stage.before[row])} + ${format(rate * stage.correction[row])} = ${format(stage.prediction[row])}`], ['Training MSE before → after', `${format(model.stages[round - 1].mse)} → ${format(stage.mse)}`]]} />
    <div className="gbt-actions"><button onClick={() => setRound(round - 1)} disabled={round === 1}>Previous tree</button><span>Tree {round} of 8</span><button onClick={() => setRound(round + 1)} disabled={round === 8}>Next tree</button><button onClick={reset}>Reset correction</button></div>
    <DataTable caption="Exact linked row values" headings={['Row', 'Target', 'Before', 'Residual', 'Leaf', 'After']} rows={model.x.map((_, i) => [i + 1, targets[i], format(stage.before[i]), format(stage.residual[i]), format(stage.correction[i]), format(stage.prediction[i])])} />
    <p className="gbt-scope">One-feature greedy stumps, half-squared training loss, at most eight rounds. The leaf correction is constant within each region, with a jump at the chosen boundary. Floating-point display is rounded. A lower training MSE is not held-out evidence.</p>
  </Lab>;
}
export function NewtonSplitLab() {
  const [kind, setKind] = useState('square');
  const [split, setSplit] = useState(2);
  const [lambda, setLambda] = useState(1);
  const [alpha, setAlpha] = useState(0);
  const [gamma, setGamma] = useState(0);
  const [rate, setRate] = useState(.3);
  const state = newtonInvestigation({
    kind,
    split,
    lambda,
    alpha,
    gamma,
    rate
  });
  const {
    statistics
  } = state;
  return <Lab title="Price a candidate split" name="newton">
    <p>Rows are ordered by x. Their gradients and curvatures flow into child leaves. The comparison subtracts the improvement already available to the parent and the cost of one additional leaf.</p>
    <div className="gbt-controls"><Select label="Loss fixture" value={kind} onChange={setKind} options={[["square", "Half-squared: current score 2.5"], ["logistic", "Binary: current score 0"], ["confident", "Binary: current score −4"]]} /><Range label="Rows sent left" value={split} min={1} max={4} onChange={setSplit} /><Range label="Leaf L2 lambda" value={lambda} min={0} max={5} step={.25} onChange={setLambda} /><Range label="Leaf L1 alpha" value={alpha} min={0} max={5} step={.25} onChange={setAlpha} /><Range label="New-leaf gamma" value={gamma} min={0} max={5} step={.25} onChange={setGamma} /><Range label="Newton learning rate" value={rate} min={0} max={1} step={.05} onChange={setRate} /></div>
    <div className="gbt-row-strip">{state.x.map((value, index) => <div key={value} className={index < split ? 'gbt-left' : 'gbt-right'}><strong>x={value} → {index < split ? 'L' : 'R'}</strong><span>y={state.y[index]}</span><span>g={format(state.gradients[index])}</span><span>h={format(state.hessians[index])}</span></div>)}</div>
    <div className="gbt-leaf-bands">{[['Left', statistics.left], ['Right', statistics.right]].map(([label, leaf]) => <div key={label}><strong>{label} leaf</strong><span>G={format(leaf.G)}; H={format(leaf.H)}</span><span>optimal w={format(leaf.weight)}</span><span>surrogate improvement {format(leaf.improvement)}</span></div>)}</div>
    <div className="gbt-balance"><span>Children <strong>{format(statistics.left.improvement + statistics.right.improvement)}</strong></span><span>− parent <strong>{format(statistics.parent.improvement)}</strong></span><span>− extra leaf <strong>{format(gamma)}</strong></span><span>= net gain <strong>{format(statistics.netGain)}</strong></span></div>
    <Readouts rows={[["Split decision from this surrogate", statistics.netGain > 0 ? 'Positive net gain: eligible' : 'No positive net gain: do not split'], ['Unregularized actual data loss, before → proposed children', `${format(state.beforeLoss)} → ${format(state.afterLoss)}`]]} />
    <p>The proposed child update is shown even when the split is rejected so you can inspect the tradeoff. For binary loss, try score −4, three rows left, lambda 0 and rate 1: a large Newton correction worsens actual loss. Gamma/alpha/lambda price the surrogate; the separate data-loss number excludes their penalties.</p>
    <DataTable caption="Scores, probabilities and proposed updates" headings={['Row', 'g', 'h', 'Before F', 'After F', 'After p (binary)']} rows={state.x.map((_, i) => [i + 1, format(state.gradients[i]), format(state.hessians[i]), format(state.before[i]), format(state.after[i]), kind === 'square' ? 'not used' : format(state.afterProbabilities[i])])} />
    <button onClick={() => {
      setKind('square');
      setSplit(2);
      setLambda(1);
      setAlpha(0);
      setGamma(0);
      setRate(.3);
    }}>Reset split</button>
  </Lab>;
}
export function HistogramMissingLab() {
  const [coarse, setCoarse] = useState(true);
  const [missingTarget, setMissingTarget] = useState(8);
  const state = histogramInvestigation({
    coarse,
    missingTarget
  });
  return <Lab title="What a histogram can no longer split" name="histogram">
    <p>Adjacent x values share bins. A threshold can go between bins, not through one. The missing row is kept separate while both default routes are scored.</p>
    <div className="gbt-controls"><Select label="Candidate boundaries" value={coarse ? 'coarse' : 'fine'} onChange={value => setCoarse(value === 'coarse')} options={[["coarse", "Coarse: cuts 2.5 and 4.5"], ["fine", "Fine: every distinct boundary"]]} /><Range label="Missing-row target" value={missingTarget} min={0} max={10} onChange={setMissingTarget} /></div>
    <div className="gbt-histogram">{state.bins.map(bin => <div key={bin.index}><strong>Bin {bin.index + 1}</strong><div>{bin.rows.map(index => <span key={index}>x={state.x[index]}<small>y={state.y[index]}</small></span>)}</div><small>G={format(bin.rows.reduce((total, index) => total + state.gradients[index], 0))}<br />H={bin.rows.length}</small></div>)}</div>
    <div className="gbt-missing-route"><strong>x missing; y={missingTarget}</strong><span>best training route → {state.best.missing}</span></div>
    <Readouts rows={[["Best allowed threshold", `x ≤ ${state.best.threshold}`], ['Two-route candidates inspected', state.candidates.length], ['Net gain (lambda=alpha=gamma=0)', format(state.best.netGain)]]} />
    <DataTable caption="All allowed split and missing-route scores" headings={['Threshold', 'Missing goes', 'Gain']} rows={state.candidates.map(candidate => [candidate.threshold, candidate.missing, format(candidate.netGain)])} />
    <button onClick={() => {
      setCoarse(true);
      setMissingTarget(8);
    }}>Reset histogram</button>
    <p className="gbt-scope">These fixed bins isolate candidate restriction. Production libraries build their own bins and apply additional constraints. A missing training value is not automatically equivalent to a measured zero.</p>
  </Lab>;
}
export function GrowthPolicyFigure() {
  const state = growthPolicies();
  function diagram(policy) {
    const nodes = ['root', ...new Set(policy.splits.flatMap(node => node === 'root' ? ['L', 'R'] : [`${node}L`, `${node}R`]))];
    const coordinates = node => node === 'root' ? [150, 25] : [150 + [...node].reduce((offset, side, depth) => offset + (side === 'L' ? -1 : 1) * 70 / 2 ** depth, 0), 25 + node.length * 53];
    return <svg viewBox="0 0 300 224" role="img" aria-label={`${policy.policy === 'best' ? 'Best-first' : 'Level-wise'}: split ${policy.splits.join(', ')}, four leaves, depth ${policy.depth}`}>
      {nodes.filter(node => node !== 'root').map(node => {
        const [x1, y1] = coordinates(node.length === 1 ? 'root' : node.slice(0, -1));
        const [x2, y2] = coordinates(node);
        return <line key={node} x1={x1} y1={y1 + 11} x2={x2} y2={y2 - 11} stroke="#647174" />;
      })}
      {nodes.map(node => {
        const [x, y] = coordinates(node);
        return <g key={node}><circle cx={x} cy={y} r="13" fill={policy.splits.includes(node) ? '#403421' : '#17382d'} stroke={policy.splits.includes(node) ? gold : green} /><text x={x} y={y + 4} textAnchor="middle">{node === 'root' ? '•' : node}</text></g>;
      })}
      <text x="150" y="217" textAnchor="middle">4 leaves; depth {policy.depth}</text>
    </svg>;
  }
  return <figure className="gbt-inline"><figcaption>Leaf budget and depth are different constraints</figcaption><p>Policy illustration with specified candidate gains: root 10, L 6, R 2, LL 5. Both policies spend three splits. These are hypothetical frontier scores, not library measurements.</p><div className="gbt-plots"><div><strong>Level-wise: root → L → R</strong>{diagram(state.level)}</div><div><strong>Best-first: root → L → LL</strong>{diagram(state.best)}</div></div><p>A symmetric tree adds another constraint: the same predicate is used across all nodes at a given depth. Level-wise alone does not impose that shared predicate, nor force every possible child to split.</p></figure>;
}
export function GossSamplingLab() {
  const [sample, setSample] = useState(0);
  const state = gossInvestigation({
    sample
  });
  return <Lab title="An unbiased sum can have a biased square" name="goss">
    <p>Keep gradients 8 and −6. Sample exactly two of the other four rows. Each remaining row has inclusion probability 1/2, so a sampled contribution is doubled.</p>
    <Chart title="Each row's contribution to this estimated sum" xLabel="Row number" yLabel="Signed gradient contribution" zero series={[{
      label: 'Full data contribution',
      color: blue,
      open: true,
      points: state.gradients.map((value, index) => [index + 1, value])
    }, {
      label: 'Kept/reweighted contribution',
      color: gold,
      stems: true,
      points: state.contribution.map((value, index) => [index + 1, value])
    }]} />
    <div className="gbt-actions"><button disabled={sample === 0} onClick={() => setSample(sample - 1)}>Previous sample</button><span>Sample {sample + 1} of {state.subsets.length}</span><button disabled={sample === state.subsets.length - 1} onClick={() => setSample(sample + 1)}>Next sample</button><button onClick={() => setSample(0)}>Reset sample</button></div>
    <div className="gbt-sample-strip">{state.totals.map((total, index) => <div key={index} className={index === sample ? 'is-selected' : ''}><small>draw {state.subsets[index].map(row => row + 1).join(',')}</small><strong>{format(total)}</strong></div>)}</div>
    <Readouts rows={[["Current estimated sum / full sum", `${format(state.estimate)} / ${state.full}`], ['Mean over all six equally likely samples', format(state.average)], ['Mean squared estimate / squared full sum', `${format(state.averageSquare)} / ${state.full ** 2}`]]} />
    <DataTable caption="Weighted rows and all sample totals" headings={['Row', 'Gradient', 'Role now', 'Contribution']} rows={state.gradients.map((value, index) => [index + 1, value, state.retained.includes(index) ? 'always kept' : state.selected.includes(index) ? 'sampled ×2' : 'not sampled', state.contribution[index]])} />
    <p>Expectation is over the sampling rule with gradients held fixed. Split scores include squares and possibly random denominators; choosing their maximum is another nonlinear operation. This finite example proves why unbiased sums alone cannot guarantee unbiased split selection.</p>
  </Lab>;
}
export function ExclusiveBundleFigure() {
  const state = bundleExclusive([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 2]]);
  return <figure className="gbt-inline"><figcaption>Exact exclusive bundling preserves the feature identity in its bin range</figcaption><div className="gbt-bundle-ranges"><span>A bins 1–2</span><span>B bins 3–4</span><span>C bins 5–6</span></div><div className="gbt-bundle-rows">{state.rows.map((row, index) => <div key={index}><code>[{row.join(', ')}]</code><span>→</span><strong>{state.encoded[index]}</strong><span>→</span><code>[{state.decoded[index].join(', ')}]</code></div>)}</div><p>All-zero rows use code 0. A row such as [1,1,0] has two active features and cannot use this exact single-code scheme. Allowing collisions introduces an approximation; it is no longer this reversible example.</p></figure>;
}
export function OrderedCategoricalLab() {
  const [row, setRow] = useState(2);
  const [reverse, setReverse] = useState(false);
  const [flipped, setFlipped] = useState(false);
  const [smoothing, setSmoothing] = useState(1);
  const state = orderedStatistics({
    row,
    order: reverse ? [5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5],
    flipped,
    smoothing
  });
  return <Lab title="Only earlier targets may enter this row's statistic" name="ordered">
    <p>The external prior is fixed at 0.5. A selected row's target is revealed only after its prefix encoding has been calculated. Green rows contribute to that encoding; later rows and other categories do not.</p>
    <div className="gbt-controls"><Select label="Categorical row" value={row} onChange={value => {
        setRow(Number(value));
        setFlipped(false);
      }} options={state.rows.map((item, index) => [index, `row ${index + 1}: category ${item.category}`])} /><Range label="Prior strength" value={smoothing} min={.25} max={4} step={.25} onChange={setSmoothing} /><label className="gbt-check"><input type="checkbox" checked={reverse} onChange={event => setReverse(event.target.checked)} />Reverse the permutation</label><label className="gbt-check"><input type="checkbox" checked={flipped} onChange={event => setFlipped(event.target.checked)} />Flip the selected row's target</label></div>
    <div className="gbt-prefix-strip">{state.order.map(index => <div key={index} className={index === row ? 'is-selected' : state.preceding.includes(index) ? 'is-eligible' : 'is-excluded'}><small>row {index + 1}</small><strong>{state.rows[index].category}: y={state.rows[index].target}</strong><span>{index === row ? 'encode before target' : state.preceding.includes(index) ? 'earlier + same category' : state.order.indexOf(index) > state.position ? 'later: excluded' : 'other category'}</span></div>)}</div>
    <div className="gbt-fraction"><span>Earlier matching target sum + strength × 0.5<strong>{format(state.numerator)}</strong></span><span>Earlier matching count + strength<strong>{format(state.denominator)}</strong></span></div>
    <Readouts rows={[["Selected row's prefix statistic", format(state.prefix)], ['Naive full-training statistic', format(state.naive)], ['Unseen category inference fallback', '0.5 (the external prior)']]} />
    <DataTable caption="Every row's actual prefix encoding" headings={['Row in order', 'Earlier matching rows', 'Prefix value']} rows={state.prefixes.map(item => [item.index + 1, item.eligible.length ? item.eligible.map(index => index + 1).join(', ') : 'none', format(item.value)])} />
    <p>Flipping a row can change later encodings that legitimately include it. A prior estimated using that same row's target would break the strict own-label invariance demonstrated here. CatBoost's internal statistics/modes are more elaborate; this is the isolated information-boundary mechanism.</p>
    <button onClick={() => {
      setRow(2);
      setReverse(false);
      setFlipped(false);
      setSmoothing(1);
    }}>Reset categories</button>
  </Lab>;
}
export function BoostingValidationLab() {
  const [rate, setRate] = useState(.3);
  const [depth, setDepth] = useState(2);
  const [round, setRound] = useState(5);
  const state = useMemo(() => validationInvestigation({
    rate,
    depth
  }), [rate, depth]);
  const predictionSegments = boostingPredictionSegments(state.model, round);
  const predictionPoints = predictionSegments.flatMap(segment => [[segment.left, segment.value], [segment.right, segment.value]]);
  return <Lab title="Training loss is not a stopping rule" name="validation">
    <p>Training rows have fixed noise. Validation rows are interleaved and use the constructed noiseless signal. The model can improve its fit to noise while making worse predictions between rows.</p>
    <div className="gbt-controls"><Range label="Validation learning rate" value={rate} min={.05} max={1} step={.05} onChange={value => {
        setRate(value);
        setRound(1);
      }} /><Range label="Correction tree depth" value={depth} min={1} max={3} onChange={value => {
        setDepth(value);
        setRound(1);
      }} /><Range label="Inspect boosting round" value={round} min={0} max={60} onChange={setRound} /></div>
    <div className="gbt-plots"><Chart title="Actual loss at each fitted prefix" xLabel="Number of correction trees" yLabel="Mean squared error" cursor={round} zero nonnegative series={[{
        label: 'Training',
        color: blue,
        line: true,
        hidePoints: true,
        points: state.curves.map(point => [point.round, point.train])
      }, {
        label: 'Validation',
        color: green,
        line: true,
        hidePoints: true,
        points: state.curves.map(point => [point.round, point.validation])
      }]} /><Chart title={`Predictor using ${round} trees`} xLabel="Feature x (including outside training range)" yLabel="Target / predicted response" series={[{
        label: 'Training observations',
        color: blue,
        points: state.model.x.map((x, index) => [x, state.model.y[index]])
      }, {
        label: 'Validation observations',
        color: green,
        open: true,
        points: state.validationX.map((x, index) => [x, state.validationY[index]])
      }, {
        label: 'Saved predictor',
        color: gold,
        line: true,
        hidePoints: true,
        points: predictionPoints
      }]} /></div>
    <Readouts rows={[["Round shown: train / validation MSE", `${format(state.curves[round].train)} / ${format(state.curves[round].validation)}`], ['Best round before patience stops', state.best], ['Stop after six nonimproving rounds', state.stopped], ['Mean-baseline validation MSE', format(state.baselineValidation)]]} />
    <p>The full 60-round trajectory is available for explanation. The simulated selection procedure considers rounds in order and stops at {state.stopped}; later points do not revise its chosen round {state.best}. Round 0 is the separately reported mean baseline. No test set is used in this illustration.</p>
    <div className="gbt-actions"><button onClick={() => setRound(state.best)}>Show validation choice</button><button onClick={() => setRound(state.stopped)}>Show stopping round</button><button onClick={() => {
        setRate(.3);
        setDepth(2);
        setRound(5);
      }}>Reset validation</button></div>
    <DataTable caption="Training and validation curve values" headings={['Round', 'Training MSE', 'Validation MSE']} rows={state.curves.map(point => [point.round, format(point.train), format(point.validation)])} />
    <DataTable caption="Saved predictor: constant regions" headings={['Feature interval (rounded boundaries)', 'Prediction']} rows={predictionSegments.map((segment, index) => [`${index === 0 ? '[' : '('}${format(segment.left)}, ${format(segment.right)}]`, format(segment.value)])} />
    <DataTable caption="Observed rows and selected predictions" headings={['Role', 'Feature x', 'Observed target', 'Prediction']} rows={[...state.model.x.map((x, index) => ['Training', format(x), format(state.model.y[index]), format(predictBoosting(state.model, x, round))]), ...state.validationX.map((x, index) => ['Validation', format(x), format(state.validationY[index]), format(predictBoosting(state.model, x, round))])]} />
    <p className="gbt-scope">{state.provenance} Constant-leaf predictions stay constant beyond the outermost split. Each horizontal piece uses actual saved thresholds; vertical joins mark jumps, not interpolated predictions. At a split, the ≤ rule uses the left-side value.</p>
  </Lab>;
}
