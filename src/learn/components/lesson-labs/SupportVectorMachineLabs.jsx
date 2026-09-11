import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { marginGeometry, supportMotion, softPairState, xorKernelState, xorScore, svmPairFixture, svmPairStep, svrTubeState, spectrumCounts, SVM_XOR_POINTS, SVM_XOR_LABELS } from '../../data/support-vector-machines-models.js';
import validation from '../../data/svm-validation-fixtures.js';
import './support-vector-machines-labs.css';
const gold = '#efc36e';
const blue = '#8cbce8';
const rose = '#ef9dad';
const green = '#87d9b1';
const fmt = value => Math.abs(value) < 1e-10 ? '0' : Number(value.toFixed(4)).toString();
function Lab({
  name,
  title,
  children
}) {
  const id = useId();
  return <section className="svm-lab" data-svm-lab={name} aria-labelledby={id}><h3 id={id}>{title}</h3>{children}</section>;
}
function Range({
  label,
  value,
  min,
  max,
  step = .05,
  onChange,
  disabled = false
}) {
  return <label>{label}: <strong>{fmt(value)}</strong><input type="range" aria-label={label} value={value} min={min} max={max} step={step} disabled={disabled} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Select({
  label,
  value,
  onChange,
  options
}) {
  return <label>{label}<select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, text]) => <option value={key} key={key}>{text}</option>)}</select></label>;
}
function Readouts({
  rows
}) {
  return <dl className="svm-readouts" aria-live="polite">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Table({
  caption,
  headings,
  rows
}) {
  return <details className="svm-data"><summary>{caption}</summary><div role="region" aria-label={caption} tabIndex={0}><table><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((value, column) => column === 0 ? <th scope="row" key={column}>{value}</th> : <td key={column}>{value}</td>)}</tr>)}</tbody></table></div></details>;
}
function Plot({
  title,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  children,
  square = false
}) {
  const ref = useRef(null);
  const [width, setWidth] = useState(300);
  const id = useId();
  useEffect(() => {
    const observer = new ResizeObserver(entries => setWidth(Math.max(210, entries[0].contentRect.width)));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const plotWidth = Math.min(width - 57, 440);
  const left = 43 + Math.max(0, (width - 57 - plotWidth) / 2);
  const right = left + plotWidth;
  const top = 18;
  const plotHeight = square ? plotWidth : Math.min(230, plotWidth * .75);
  const bottom = top + plotHeight;
  const x = value => left + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * plotWidth;
  const y = value => bottom - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * plotHeight;
  const area = {
    left,
    right,
    top,
    bottom,
    plotWidth,
    plotHeight
  };
  const path = points => points.map(point => `${x(point[0])},${y(point[1])}`).join(' ');
  return <figure className="svm-plot" ref={ref}><figcaption>{title}</figcaption><p className="svm-axis">{yLabel}</p>
    <svg viewBox={`0 0 ${width} ${bottom + 35}`} role="img" aria-labelledby={id}>
      <title id={id}>{title}. Horizontal: {xLabel}. Vertical: {yLabel}. Exact values and interpretation accompany the figure.</title>
      {[0, .5, 1].map(fraction => {
        const xv = xDomain[0] + fraction * (xDomain[1] - xDomain[0]);
        const yv = yDomain[0] + fraction * (yDomain[1] - yDomain[0]);
        return <g key={fraction}><line x1={x(xv)} x2={x(xv)} y1={top} y2={bottom} className="svm-grid" /><line x1={left} x2={right} y1={y(yv)} y2={y(yv)} className="svm-grid" /><text x={x(xv)} y={bottom + 21} textAnchor="middle">{Number(xv.toPrecision(2))}</text><text x={left - 7} y={y(yv) + 4} textAnchor="end">{Number(yv.toPrecision(2))}</text></g>;
      })}
      {yDomain[0] < 0 && yDomain[1] > 0 && <line x1={left} x2={right} y1={y(0)} y2={y(0)} className="svm-zero" />}
      {children({
        x,
        y,
        path,
        area,
        width
      })}
    </svg><p className="svm-axis">{xLabel}</p>
  </figure>;
}
function Point({
  x,
  y,
  label,
  name,
  open = false,
  radius = 5,
  nameBelow = false
}) {
  const color = label > 0 ? blue : rose;
  return <g>{label > 0 ? <circle cx={x} cy={y} r={radius} fill={open ? '#101719' : color} stroke={color} strokeWidth="2" /> : <rect x={x - radius} y={y - radius} width={2 * radius} height={2 * radius} fill={open ? '#101719' : color} stroke={color} strokeWidth="2" />}{name && <text x={x + 9} y={y + (nameBelow ? 20 : -9)}>{name}</text>}</g>;
}
function Legend() {
  return <div className="svm-legend"><span style={{
      color: blue
    }}>● positive class</span><span style={{
      color: rose
    }}>■ negative class</span></div>;
}
function boundarySegment(normal, offset, bounds) {
  const [lo, hi] = bounds;
  const candidates = [];
  if (Math.abs(normal[0]) > 1e-12) {
    for (const y of [lo, hi]) {
      const x = (offset - normal[1] * y) / normal[0];
      if (x >= lo - 1e-10 && x <= hi + 1e-10) candidates.push([x, y]);
    }
  }
  if (Math.abs(normal[1]) > 1e-12) {
    for (const x of [lo, hi]) {
      const y = (offset - normal[0] * x) / normal[1];
      if (y >= lo - 1e-10 && y <= hi + 1e-10) candidates.push([x, y]);
    }
  }
  return candidates.slice(0, 2);
}
export function GeometricMarginLab() {
  const [angle, setAngle] = useState(0);
  const [offset, setOffset] = useState(0);
  const [scale, setScale] = useState(1);
  const state = marginGeometry(angle, offset, scale);
  return <Lab name="margin" title="Move the boundary; then change only its score scale">
    <p>Solid gold is the decision boundary f=0. Dashed lines are the score levels f=±1. The green segment is the perpendicular projection of the least favorable labeled observation.</p>
    <div className="svm-controls"><Range label="Normal angle" value={angle} min={-45} max={45} step={5} onChange={setAngle} /><Range label="Boundary offset" value={offset} min={-.75} max={.75} onChange={setOffset} /><Range label="Score scale" value={scale} min={.5} max={3} step={.25} onChange={setScale} /></div>
    <Plot title="Coordinates and perpendicular separation" xDomain={[-2.6, 2.6]} yDomain={[-2.6, 2.6]} xLabel="feature x₁" yLabel="feature x₂" square>{({
        x,
        y,
        path
      }) => <>
      {[-1, 0, 1].map(level => <polyline key={level} points={path(boundarySegment(state.normal, offset + level / scale, [-2.6, 2.6]))} fill="none" stroke={gold} strokeWidth={level === 0 ? 2.5 : 1.5} strokeDasharray={level === 0 ? undefined : '5 4'} />)}
      <line x1={x(state.nearest.x[0])} x2={x(state.nearest.projection[0])} y1={y(state.nearest.x[1])} y2={y(state.nearest.projection[1])} stroke={green} strokeWidth="4" />
      {state.rows.map(row => <Point key={row.id} x={x(row.x[0])} y={y(row.x[1])} label={row.y} name={row.id} />)}
      <line x1={x(state.normal[0] * offset)} y1={y(state.normal[1] * offset)} x2={x(state.normal[0] * (offset + .65))} y2={y(state.normal[1] * (offset + .65))} stroke={green} strokeWidth="2" />
      <circle cx={x(state.normal[0] * (offset + .65))} cy={y(state.normal[1] * (offset + .65))} r="3" fill={green} />
    </>}</Plot><Legend />
    <Readouts rows={[["Least label-signed distance", fmt(state.geometricMargin)], ['Separates these labels?', state.separating ? 'Yes' : 'No'], ['All canonical constraints y f ≥ 1?', state.canonicalFeasible ? 'Yes' : 'No'], ['Weight norm', fmt(state.norm)], ['Score-level corridor width', fmt(state.scoreCorridorWidth)]]} />
    <p>Scaling changes the dashed score levels, not the solid boundary or any perpendicular distance. A negative label-signed distance identifies a misclassification; it is not a negative physical length. The green normal segment points toward increasing scores.</p>
    <Table caption="Inspect scores, distances and projections" headings={['Point', 'Label', 'Raw f', 'y f / norm', 'Projection']} rows={state.rows.map(row => [row.id, row.y, fmt(row.score), fmt(row.labelDistance), row.projection.map(fmt).join(', ')])} />
    <button onClick={() => {
      setAngle(0);
      setOffset(0);
      setScale(1);
    }}>Reset margin</button>
  </Lab>;
}
export function SupportCertificateLab() {
  const [moved, setMoved] = useState(2);
  const state = supportMotion(moved);
  return <Lab name="support" title="A zero coefficient is not permission to move a point anywhere">
    <p>A and B are negative; C and D are positive. Move D while comparing the original boundary at zero with the new hard-margin optimum. The ring identifies a point with a positive coefficient in the displayed certificate.</p>
    <Range label="Point D coordinate" value={moved} min={.25} max={3} step={.05} onChange={setMoved} />
    <Plot title="Same data, two candidate certificates" xDomain={[-2.5, 3.5]} yDomain={[-.5, 1.5]} xLabel="one feature x" yLabel="original row below; refitted row above">{({
        x,
        y
      }) => <>
      {[0, 1].map(row => <g key={row}><line x1={x(-2.5)} x2={x(3.5)} y1={y(row)} y2={y(row)} className="svm-zero" /><line x1={x(row ? state.boundary : 0)} x2={x(row ? state.boundary : 0)} y1={y(row + .35)} y2={y(row - .35)} stroke={gold} strokeWidth="3" />{state.rows.map(point => <g key={point.id}><Point x={x(point.x)} y={y(row)} label={point.y} name={point.id} nameBelow={point.id === 'D'} />{row === 1 && point.alpha > 0 && <circle cx={x(point.x)} cy={y(row)} r="10" fill="none" stroke={gold} />}</g>)}</g>)}
    </>}</Plot><Legend />
    <Readouts rows={[["Old certificate still feasible?", state.oldFeasible ? 'Yes' : 'No: D has y f < 1'], ['New boundary', fmt(state.boundary)], ['New w and b', `${fmt(state.w)}, ${fmt(state.b)}`], ['Primal = dual', fmt(state.primal)]]} />
    <p>{state.oldFeasible ? 'The original feasible solution and lower-bound certificate still meet. There is no reason to move the optimal boundary.' : 'The original lower bound no longer meets a feasible original solution. D becomes the nearest positive observation, and the new boundary moves halfway between B and D.'} At D=1, C and D coincide: this certificate assigns the positive weight to C; other optimal distributions are possible.</p>
    <Table caption="Inspect both margin constraints and new coefficients" headings={['Point', 'x', 'Old y f', 'New y f', 'alpha']} rows={state.rows.map(row => [row.id, fmt(row.x), fmt(row.oldMargin), fmt(row.margin), fmt(row.alpha)])} />
    <button onClick={() => setMoved(2)}>Reset support point</button>
  </Lab>;
}
export function SoftMarginKktLab() {
  const [c, setC] = useState(.25);
  const [conflict, setConflict] = useState(false);
  const [bias, setBias] = useState(0);
  const state = softPairState(c, conflict, bias);
  return <Lab name="soft" title="The same optimum can allow more than one intercept">
    <p>With separated inputs x=−1,+1, reducing C accepts margin shortfalls. The conflicting fixture puts both labels at x=0. The intercept slider moves only within the analytically optimal interval.</p>
    <div className="svm-controls"><Select label="Penalty C" value={c} onChange={value => setC(Number(value))} options={[.1, .25, .5, 1, 2].map(value => [value, String(value)])} /><Select label="Two-point fixture" value={conflict ? 'conflict' : 'separate'} onChange={value => setConflict(value === 'conflict')} options={[["separate", 'Separated inputs'], ['conflict', 'Conflicting labels at x=0']]} /><Range label="Position in optimal bias interval" value={state.biasLimit ? bias : 0} min={-1} max={1} step={.25} onChange={setBias} disabled={state.biasLimit === 0} /></div>
    <Plot title="A margin of one is where hinge loss turns off" xDomain={[-1.5, 2.5]} yDomain={[-.2, 2.8]} xLabel="label-signed margin m = y f" yLabel="hinge loss max(0, 1−m)">{({
        x,
        y,
        path
      }) => <><polyline points={path([[-1.5, 2.5], [1, 0], [2.5, 0]])} fill="none" stroke={gold} strokeWidth="3" /><line x1={x(1)} x2={x(1)} y1={y(-.2)} y2={y(2.8)} stroke={green} strokeDasharray="4 4" />{[...state.rows].reverse().map(row => <Point key={row.y} x={x(row.margin)} y={y(row.hinge)} label={row.y} radius={row.y > 0 ? 7 : 4} />)}</>}</Plot><Legend />
    <Readouts rows={[["w, b", `${fmt(state.w)}, ${fmt(state.b)}`], ['Optimal b interval', `[${fmt(-state.biasLimit)}, ${fmt(state.biasLimit)}]`], ['Each alpha', fmt(state.alpha)], ['Norm cost + loss cost', `${fmt(state.normCost)} + ${fmt(state.lossCost)}`], ['Primal / dual', `${fmt(state.primal)} / ${fmt(state.dual)}`], ['Geometric margin', state.geometricMargin === null ? 'Undefined: w=0' : fmt(state.geometricMargin)]]} />
    <p>{state.biasLimit === 0 ? 'The optimal bias interval is a single value here. Both points are still support vectors even when every hinge loss is zero.' : 'The displayed losses can move between the two observations while their sum and the optimum stay unchanged. The coefficients are bounded at C; a free support vector is not available to choose a unique b.'}</p>
    <Table caption="Inspect each KKT margin and loss" headings={['Label', 'Input', 'f', 'y f', 'Hinge', 'alpha']} rows={state.rows.map(row => [row.y, fmt(row.x), fmt(row.score), fmt(row.margin), fmt(row.hinge), fmt(row.alpha)])} />
    <button onClick={() => {
      setC(.25);
      setConflict(false);
      setBias(0);
    }}>Reset soft margin</button>
  </Lab>;
}
export function KernelValidityFigure() {
  return <figure className="svm-inline"><figcaption>Zero curvature and negative curvature say different things</figcaption><div className="svm-comparison"><div><h4>Duplicate feature vectors</h4><div className="svm-matrix"><span>1</span><span>1</span><span>1</span><span>1</span></div><p>For v=(1,−1), vᵀKv=0. Moving opposite coefficient amounts changes their representation but not the summed feature vector.</p><strong>PSD can have a null direction.</strong></div><div><h4>An invalid two-point Gram matrix</h4><div className="svm-matrix"><span>1</span><span>2</span><span>2</span><span>1</span></div><p>The same v gives vᵀKv=−2. A squared feature-vector length cannot be negative.</p><strong>One negative witness refutes validity.</strong></div></div><p>The finite-C dual still has a bounded box. Losing concavity does not turn that compact feasible set into an unbounded one.</p></figure>;
}
function sampledField(score, xDomain, yDomain, count = 32) {
  return Array.from({
    length: count * count
  }, (_, index) => {
    const column = index % count;
    const row = Math.floor(index / count);
    const point = [xDomain[0] + (column + .5) / count * (xDomain[1] - xDomain[0]), yDomain[0] + (row + .5) / count * (yDomain[1] - yDomain[0])];
    return {
      column,
      row,
      score: score(point)
    };
  });
}
function Field({
  cells,
  area,
  count = 32
}) {
  return <g>{cells.map((cell, index) => <rect key={index} x={area.left + cell.column / count * area.plotWidth} y={area.bottom - (cell.row + 1) / count * area.plotHeight} width={area.plotWidth / count + .15} height={area.plotHeight / count + .15} fill={cell.score >= 0 ? blue : rose} opacity={.08 + .37 * Math.tanh(Math.abs(cell.score))} />)}</g>;
}
export function KernelGeometryLab() {
  const [kernel, setKernel] = useState('rbf');
  const [c, setC] = useState(1);
  const [gamma, setGamma] = useState(.5);
  const [qx, setQx] = useState(.5);
  const [qy, setQy] = useState(.5);
  const state = xorKernelState(kernel, c, gamma, [qx, qy]);
  const contributionExtent = Math.max(1, ...state.contributions.map(row => Math.abs(row.value)));
  const field = useMemo(() => sampledField(point => xorScore(point, kernel, c, gamma), [-2, 2], [-2, 2]), [kernel, c, gamma]);
  return <Lab name="kernel" title="A nonlinear boundary is a sum of support-point similarities">
    <p>The four XOR labels depend on the sign of x₁x₂. Background color evaluates the fitted score at each of 32×32 cell centers. The green query has four explicit contributions; changing the query does not refit the model.</p>
    <div className="svm-controls"><Select label="Kernel" value={kernel} onChange={setKernel} options={[["linear", 'Linear dot product'], ['poly', 'Homogeneous degree two'], ['rbf', 'Gaussian RBF']]} /><Select label="Kernel penalty C" value={c} onChange={value => setC(Number(value))} options={[.05, .25, 1, 4].map(value => [value, String(value)])} /><Range label="RBF gamma" value={gamma} min={.05} max={4} step={.05} onChange={setGamma} disabled={kernel !== 'rbf'} /><Range label="Query x1" value={qx} min={-2} max={2} step={.1} onChange={setQx} /><Range label="Query x2" value={qy} min={-2} max={2} step={.1} onChange={setQy} /></div>
    <div className="svm-plots"><Plot title="Input coordinates: sampled fitted score" xDomain={[-2, 2]} yDomain={[-2, 2]} xLabel="x₁" yLabel="x₂" square>{({
          x,
          y,
          area
        }) => <><Field cells={field} area={area} />{SVM_XOR_POINTS.map((point, index) => <Point key={index} x={x(point[0])} y={y(point[1])} label={SVM_XOR_LABELS[index]} />)}<circle cx={x(qx)} cy={y(qy)} r="8" fill="none" stroke={green} strokeWidth="3" /></>}</Plot>
      <Plot title="Degree-two feature view" xDomain={[-.5, 1.5]} yDomain={[-2, 2]} xLabel="first lifted coordinate x₁²" yLabel="second lifted coordinate √2 x₁x₂" square>{({
          x,
          y,
          area
        }) => <><line x1={area.left} x2={area.right} y1={y(0)} y2={y(0)} stroke={gold} strokeWidth="2" />{[1, -1].map(label => <g key={label}><Point x={x(1)} y={y(label * Math.SQRT2)} label={label} /><text x={x(1) - 10} y={y(label * Math.SQRT2) - 12} textAnchor="end">two rows</text></g>)}</>}</Plot></div><Legend />
    <p>The feature view shows the fixed polynomial construction, whichever kernel is selected on the left. Its third coordinate x₂² also equals one for these four training points. The relevant middle coordinate separates the labels; it is not a claim that every transformed query has the same other coordinates.</p>
    <Readouts rows={[["Each dual alpha", fmt(state.alpha)], ['Query score / tie rule', `${fmt(state.score)} → ${state.prediction > 0 ? '+1' : '−1'} (zero → +1)`], ['Primal / dual', `${fmt(state.primal)} / ${fmt(state.dual)}`]]} />
    <div className="svm-contributions">{state.contributions.map((row, index) => <div key={index}><span>{SVM_XOR_POINTS[index].join(', ')}</span><div className="svm-contribution-track"><i style={{
            width: `${50 * Math.abs(row.value) / contributionExtent}%`,
            left: row.value >= 0 ? '50%' : undefined,
            right: row.value < 0 ? '50%' : undefined,
            background: row.value >= 0 ? blue : rose
          }} /></div><strong>{fmt(row.value)}</strong></div>)}</div>
    <p className="svm-scope">Contribution bars share a symmetric ±{fmt(contributionExtent)} scale, expanded to include every current contribution. Dark cells have scores near zero; brightness is not a probability or a confidence guarantee. The linear optimum has w=0 here and cannot separate XOR.</p>
    <Table caption="Inspect the query sum and the four training margins" headings={['Point', 'Label', 'k(point,query)', 'alpha × label × k', 'Training margin']} rows={state.contributions.map((row, index) => [index + 1, row.label, fmt(row.kernel), fmt(row.value), fmt(state.margins[index])])} />
    <Table caption="Inspect the actual Gram matrix" headings={['Row', '1', '2', '3', '4']} rows={state.gram.map((row, index) => [index + 1, ...row.map(fmt)])} />
    <button onClick={() => {
      setKernel('rbf');
      setC(1);
      setGamma(.5);
      setQx(.5);
      setQy(.5);
    }}>Reset kernel</button>
  </Lab>;
}
export function SmoPairLab() {
  const [duplicate, setDuplicate] = useState(false);
  const [alpha, setAlpha] = useState(svmPairFixture().alpha);
  const [pair, setPair] = useState('0,1');
  const [steps, setSteps] = useState(0);
  const fixture = svmPairFixture(duplicate);
  const [i, j] = pair.split(',').map(Number);
  const state = svmPairStep(fixture.points, fixture.labels, alpha, fixture.c, i, j);
  const maxGain = Math.max(.1, ...state.curve.map(point => Math.abs(point.gain)));
  function reset(nextDuplicate = duplicate) {
    setDuplicate(nextDuplicate);
    setAlpha(svmPairFixture(nextDuplicate).alpha);
    setPair('0,1');
    setSteps(0);
  }
  return <Lab name="pair" title="Two coefficients must move on one legal segment">
    <p>The box requires 0≤alpha≤C. The gold segment also preserves the label-weighted sum. Green marks the best point on that segment with every other coefficient held fixed; it is not necessarily the optimum of the whole training problem.</p>
    <div className="svm-controls"><Select label="Pair fixture" value={duplicate ? 'duplicate' : 'three'} onChange={value => reset(value === 'duplicate')} options={[["three", 'Three distinct inputs'], ['duplicate', 'Opposite labels, identical input']]} /><Select label="Selected coefficient pair" value={pair} onChange={setPair} options={duplicate ? [['0,1', '0 and 1: opposite labels']] : [['0,1', '0 and 1: opposite labels'], ['1,2', '1 and 2: same label'], ['0,2', '0 and 2: opposite labels']]} /></div>
    <div className="svm-plots"><Plot title="Feasible coordinate geometry" xDomain={[-.1, 1.3]} yDomain={[-.1, 1.3]} xLabel={`alpha ${j}`} yLabel={`alpha ${i}`} square>{({
          x,
          y,
          path
        }) => <><polygon points={path([[0, 0], [1.2, 0], [1.2, 1.2], [0, 1.2]])} fill="none" stroke="#87928a" /><polyline points={path([state.lower, state.upper].map(delta => [alpha[j] + delta, alpha[i] - state.sign * delta]))} fill="none" stroke={gold} strokeWidth="4" /><circle cx={x(alpha[j])} cy={y(alpha[i])} r="6" fill={blue} /><circle cx={x(state.after[j])} cy={y(state.after[i])} r="9" fill="none" stroke={green} strokeWidth="3" /></>}</Plot>
      <Plot title="Dual improvement along that segment" xDomain={state.lower === state.upper ? [state.lower - .1, state.upper + .1] : [state.lower, state.upper]} yDomain={[-maxGain * 1.1, maxGain * 1.1]} xLabel="delta added to alpha j" yLabel="D(after) − D(current)">{({
          x,
          y,
          path
        }) => <><polyline points={path(state.curve.map(point => [point.delta, point.gain]))} fill="none" stroke={gold} strokeWidth="2.5" /><circle cx={x(0)} cy={y(0)} r="5" fill={blue} /><circle cx={x(state.bestDelta)} cy={y(state.improvement)} r="7" fill="none" stroke={green} strokeWidth="3" /></>}</Plot></div>
    <div className="svm-legend"><span style={{
        color: blue
      }}>● current</span><span style={{
        color: green
      }}>○ segment optimum</span></div>
    <Readouts rows={[["Curvature q", fmt(state.q)], ['Best delta', fmt(state.bestDelta)], ['Current → candidate dual', `${fmt(state.beforeDual)} → ${fmt(state.afterDual)}`], ['Label balance after move', fmt(state.balance)], ['Committed moves', steps]]} />
    <p>{state.q === 0 ? 'The objective is linear along this segment. Compare the two endpoints; skipping the move would leave useful progress behind.' : state.sign === 1 ? 'Both labels have the same sign, so increasing one coefficient requires decreasing the other.' : 'The labels have opposite signs, so these two coefficients can increase together without changing label balance.'}</p>
    <Table caption="Inspect the proposed atomic update" headings={['Index', 'Label', 'Current alpha', 'Candidate alpha']} rows={alpha.map((value, index) => [index, fixture.labels[index], fmt(value), fmt(state.after[index])])} />
    <div className="svm-actions"><button disabled={steps >= 16 || state.improvement <= 1e-12} onClick={() => {
        setAlpha(state.after);
        setSteps(steps + 1);
      }}>Commit segment optimum</button><button onClick={() => reset()}>Reset pair moves</button></div><p className="svm-scope">At most sixteen manual moves. Disabled progress is not a whole-problem convergence certificate; the complete program checks a primal–dual criterion separately.</p>
  </Lab>;
}
function savedScore(point, model) {
  const standardized = point.map((value, index) => (value - validation.scaler.mean[index]) / validation.scaler.scale[index]);
  return model.bias + model.support.reduce((sum, support, index) => sum + model.coefficients[index] * Math.exp(-model.gamma * support.reduce((distance, value, dimension) => distance + (value - standardized[dimension]) ** 2, 0)), 0);
}
export function SvmValidationLab() {
  const [selected, setSelected] = useState(4);
  const [showTraining, setShowTraining] = useState(false);
  const model = validation.models[selected];
  const field = useMemo(() => sampledField(point => savedScore(point, model), [-1.7, 2.7], [-1.3, 1.7]), [model]);
  const rows = showTraining ? validation.train : validation.validation;
  return <Lab name="validation" title="Choose a fitted model with validation evidence">
    <p>Each button reports accuracy on the same forty validation rows. No test labels are used by these controls. The background uses the selected model's saved coefficients and training-fitted scaler.</p>
    <div className="svm-selection" role="group" aria-label="Validation model selection"><span>gamma →<br />C ↓</span>{[.1, 1, 10].map(value => <strong key={value}>{value}</strong>)}{[.1, 1, 10].map((c, row) => <div className="svm-selection-row" key={c}><strong>{c}</strong>{validation.models.slice(row * 3, row * 3 + 3).map((item, column) => <button key={item.gamma} aria-pressed={selected === row * 3 + column} aria-label={`C ${c} gamma ${item.gamma} validation ${(100 * item.validation).toFixed(1)} percent`} onClick={() => setSelected(row * 3 + column)}>{(100 * item.validation).toFixed(1)}%</button>)}</div>)}</div>
    <label className="svm-check"><input type="checkbox" checked={showTraining} onChange={event => setShowTraining(event.target.checked)} />Show training observations instead of validation observations</label>
    <Plot title={`${showTraining ? 'Training' : 'Validation'} observations on the selected score field`} xDomain={[-1.7, 2.7]} yDomain={[-1.3, 1.7]} xLabel="raw feature x₁" yLabel="raw feature x₂">{({
        x,
        y,
        area
      }) => <><Field cells={field} area={area} />{rows.map(row => <Point key={row.id} x={x(row.x[0])} y={y(row.x[1])} label={row.y === 1 ? 1 : -1} open={!showTraining} radius={3} />)}</>}</Plot><Legend />
    <Readouts rows={[["Selected C, gamma", `${model.c}, ${model.gamma}`], ['Training accuracy', `${(100 * model.training).toFixed(1)}%`], ['Validation accuracy', `${(100 * model.validation).toFixed(1)}%`], ['Support observations', `${model.supportCount} / 120`]]} />
    <p>C=10, gamma=10 fits 99.2% of training labels here but scores 92.5% on validation. C=1, gamma=1 scores 97.5% on both. These are results for one fixed split; neither a universal optimum nor uncertainty-free estimates.</p>
    <Table caption="Inspect all fitted selection results" headings={['C', 'gamma', 'Train accuracy', 'Validation accuracy', 'Support count']} rows={validation.models.map(item => [item.c, item.gamma, fmt(item.training), fmt(item.validation), item.supportCount])} />
    <Table caption="Inspect the displayed observations and saved-model scores" headings={['Original row', 'x1', 'x2', 'Class', 'Score']} rows={rows.map(row => [row.id, ...row.x.map(fmt), row.y, fmt(savedScore(row.x, model))])} />
    <p className="svm-scope">Measured with scikit-learn {validation.provenance.sklearn}; make_moons(200, noise=.2, seed42), stratified split seeds17/19. The 32×32 score rendering is sampled; the values in the table are evaluated at the actual observations. The complete program reserves a separate forty-row test set.</p><button onClick={() => {
      setSelected(4);
      setShowTraining(false);
    }}>Reset validation</button>
  </Lab>;
}
export function SvrTubeLab() {
  const [amplitude, setAmplitude] = useState(2);
  const [epsilon, setEpsilon] = useState(.5);
  const [c, setC] = useState(.5);
  const state = svrTubeState(amplitude, epsilon, c);
  const extent = Math.max(amplitude, state.slope * 1.2 + epsilon) + .5;
  return <Lab name="tube" title="A regression tube measures vertical target error">
    <p>Targets are −a,0,+a at x=−1,0,+1. This symmetric fixture permits b=0. Solid gold is the optimal line; dashed gold encloses predictions ±epsilon. Rose stems show only error beyond the tube.</p>
    <div className="svm-controls"><Range label="Target amplitude a" value={amplitude} min={1} max={3} step={.25} onChange={setAmplitude} /><Range label="Tube epsilon" value={epsilon} min={0} max={2} step={.25} onChange={setEpsilon} /><Select label="Regression penalty C" value={c} onChange={value => setC(Number(value))} options={[.1, .25, .5, 1, 2].map(value => [value, String(value)])} /></div>
    <Plot title="Prediction, tolerance and paid excess" xDomain={[-1.3, 1.3]} yDomain={[-extent, extent]} xLabel="feature x" yLabel="target units">{({
        x,
        y,
        path
      }) => <><polygon points={path([[-1.2, -1.2 * state.slope - epsilon], [1.2, 1.2 * state.slope - epsilon], [1.2, 1.2 * state.slope + epsilon], [-1.2, -1.2 * state.slope + epsilon]])} fill={gold} opacity=".09" />{[-1, 0, 1].map(level => <polyline key={level} points={path([[-1.2, -1.2 * state.slope + level * epsilon], [1.2, 1.2 * state.slope + level * epsilon]])} fill="none" stroke={gold} strokeWidth={level ? 1.5 : 2.5} strokeDasharray={level ? '4 4' : undefined} />)}{state.rows.map(row => <g key={row.x}><line x1={x(row.x)} x2={x(row.x)} y1={y(row.target)} y2={y(row.target - Math.sign(row.residual) * row.excess)} stroke={rose} strokeWidth="4" /><circle cx={x(row.x)} cy={y(row.target)} r="5" fill={blue} /></g>)}</>}</Plot>
    <Readouts rows={[["Optimal slope", fmt(state.slope)], ['Norm + excess cost', `${fmt(state.normCost)} + ${fmt(state.lossCost)}`], ['Objective', fmt(state.objective)], ['Tolerance units', `${fmt(epsilon)} target units`]]} />
    <p>Increasing epsilon can stop paying for an error without moving an observation. C controls the price of the remaining excess. This is an analytically solved finite fixture, not a general-purpose fitting widget.</p>
    <Table caption="Inspect vertical errors and the epsilon loss" headings={['x', 'Target', 'Prediction', 'Residual', 'Paid excess']} rows={state.rows.map(row => [row.x, fmt(row.target), fmt(row.prediction), fmt(row.residual), fmt(row.excess)])} />
    <button onClick={() => {
      setAmplitude(2);
      setEpsilon(.5);
      setC(.5);
    }}>Reset regression tube</button>
  </Lab>;
}
export function SpectrumKernelFigure() {
  const sequences = ['ACACA', 'CACAC', 'AACCA'].map(sequence => spectrumCounts(sequence));
  return <figure className="svm-inline"><figcaption>Similarity can count sequence pieces rather than measure coordinate distance</figcaption><div className="svm-sequence-rows">{sequences.map(row => <div key={row.sequence}><strong>{row.sequence}</strong><div className="svm-windows">{row.windows.map(window => <span key={window.index}><small>start {window.index}</small>{window.word}</span>)}</div><p>{['AA', 'AC', 'CA', 'CC'].map(word => `${word}: ${row.counts[word] ?? 0}`).join(' · ')}</p></div>)}</div><p>The first two sequences both map to (0,2,2,0), so their dot product is 8 and this representation cannot tell them apart. The third maps to (1,1,1,1), giving similarity 4 with either. The windows overlap; counting only disjoint pairs would be a different kernel.</p></figure>;
}
