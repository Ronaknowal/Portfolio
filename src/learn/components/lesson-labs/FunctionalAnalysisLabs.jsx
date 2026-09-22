import { useState } from 'react';
import { spikeState, integralSpaceState, kernelValidityState, representerState, kernelRidgeState, distributionEmbeddingState, kernelQuadratureState, functionalNumber as fmt } from '../../data/functional-analysis-models.js';
import { LessonTable } from './LessonElements';
import './functional-analysis-labs.css';
function Slider({
  label,
  value,
  set,
  min,
  max,
  step
}) {
  return <label className="functional-control">{label}: <strong>{fmt(value)}</strong><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => set(Number(event.target.value))} /></label>;
}
function Metrics({
  rows
}) {
  return <dl className="functional-metrics">{rows.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>;
}
function Actions({
  children
}) {
  return <div className="functional-actions">{children}</div>;
}
function CurvePlot({
  curves,
  xRange = [0, 1],
  yRange = [-1, 2],
  markers = [],
  query = null,
  title,
  fill = null,
  knots = []
}) {
  const [xmin, xmax] = xRange;
  const [ymin, ymax] = yRange;
  const sx = x => 42 + (x - xmin) * 250 / (xmax - xmin);
  const sy = y => 190 - (y - ymin) * 162 / (ymax - ymin);
  const coordinates = [...new Set([...Array.from({
    length: 241
  }, (_, i) => xmin + (xmax - xmin) * i / 240), ...knots])].sort((a, b) => a - b);
  const path = fn => coordinates.map((x, i) => {
    return `${i === 0 ? 'M' : 'L'}${sx(x)},${sy(fn(x))}`;
  }).join(" ");
  return <figure className="functional-plot"><svg viewBox="-24 0 364 240" role="img" aria-label={title}>
    <title>{title}</title>
    {[ymin, (ymin + ymax) / 2, ymax].map(y => <g key={y}><line className="functional-grid" x1={42} x2={292} y1={sy(y)} y2={sy(y)} /><text x={35} y={sy(y) + 6} textAnchor="end">{fmt(y)}</text></g>)}
    {[xmin, (xmin + xmax) / 2, xmax].map((x, index) => <g key={x}><line className="functional-grid" x1={sx(x)} x2={sx(x)} y1={28} y2={190} /><text x={sx(x)} y={220} textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{fmt(x)}</text></g>)}
    {ymin <= 0 && ymax >= 0 && <line className="functional-zero" x1={42} x2={292} y1={sy(0)} y2={sy(0)} />}
    {fill && <path d={`${path(fill)} L${sx(xmax)},${sy(0)} L${sx(xmin)},${sy(0)} Z`} className="functional-fill" />}
    {curves.map((curve, i) => <path key={curve.label} d={path(curve.fn)} className={`functional-curve functional-tone-${i}`} style={{
        strokeDasharray: curve.dash || undefined
      }} />)}
    {query !== null && <line className="functional-query" x1={sx(query)} x2={sx(query)} y1={28} y2={190} />}
    {markers.map((point, i) => point.square ? <rect key={i} x={sx(point.x) - 4} y={sy(point.y) - 4} width={8} height={8} className="functional-marker-alt" /> : <circle key={i} cx={sx(point.x)} cy={sy(point.y)} r={4} className="functional-marker" />)}
  </svg><figcaption>{title}. Horizontal axis: input coordinate; vertical axis: the named function value. {curves.map((curve, i) => <span className={`functional-legend functional-legend-${i}`} key={curve.label}>{curve.dash ? '┄' : '━'} {curve.label}</span>)}</figcaption></figure>;
}
export function CompletionFigure() {
  return <figure className="functional-inline"><h3>A limit can leave the chosen space</h3><div className="functional-sequence" tabIndex={0} role="region" aria-label="Sequence truncations; scroll horizontally to inspect all coordinates">
    {[[1, 0, 0, 0, 0], [1, .5, 0, 0, 0], [1, .5, .25, .125, 0]].map((row, index) => <div className="functional-sequence-row" key={index}><strong>{['n=1', 'n=2', 'n=4'][index]}</strong>{row.map((value, col) => <span key={col} className={value ? 'functional-kept' : ''}>{value === 0 ? '0' : fmt(value)}</span>)}<span>…</span></div>)}
    <div className="functional-sequence-row"><strong>limit</strong>{[1, .5, .25, .125, .0625].map(value => <span className="functional-kept" key={value}>{fmt(value)}</span>)}<span>…</span></div>
  </div><figcaption>Each displayed row is a sequence, not a curve sampled at five locations. After retaining n terms, the squared tail is (4/3)·4⁻ⁿ. The infinitely many nonzero coordinates are missing from the space of finitely supported sequences.</figcaption></figure>;
}
export function FunctionSpikeLab() {
  const [width, setWidth] = useState(.125);
  const state = spikeState(width);
  return <section className="functional-lab" aria-label="Spike and evaluation investigation"><h3>Small area, unchanged reading</h3><p>Inspect what happens to f(.5) when the spike narrows. The shaded graph is f², whose area is the squared L² norm; the peak remains at the same location.</p>
    <Slider label="Spike half-width" value={width} set={setWidth} min={.005} max={.45} step={.005} />
    <CurvePlot curves={[{
      label: 'f',
      fn: state.value
    }, {
      label: 'f²',
      fn: t => state.value(t) ** 2,
      dash: "5 4"
    }]} yRange={[0, 1.2]} query={.5} knots={[.5 - width, .5, .5 + width]} fill={t => state.value(t) ** 2} title="A triangular function and its squared-area measurement" />
    <Metrics rows={[["Point value f(.5)", '1'], ["Squared L² norm", fmt(state.squaredSize)], ["Squared-slope integral", fmt(state.slopeEnergy)]]} />
    <p>The integrated error shrinks, but the point measurement does not. Meanwhile the slope must grow to cross the same height in less horizontal distance. These are exact formulas for this triangle; the plot samples its shape.</p>
    <Actions><button onClick={() => setWidth(.125)}>Reset spike</button><button onClick={() => setWidth(.025)}>Narrow spike</button></Actions>
  </section>;
}
export function EvaluationKernelLab() {
  const [slopes, setSlopes] = useState([2, -1, 1, 0]);
  const [draft, setDraft] = useState("2, -1, 1, 0");
  const [query, setQuery] = useState(.6);
  const [error, setError] = useState('');
  const state = integralSpaceState(slopes, query);
  const functionValues = [0, .25, .5, .75, 1].map(state.valueAt);
  const plotRange = [Math.min(-.25, Math.floor(Math.min(...functionValues) * 4) / 4), Math.max(.75, Math.ceil(Math.max(...functionValues, query) * 4) / 4)];
  const apply = event => {
    event.preventDefault();
    try {
      const parts = draft.split(',').map(value => value.trim());
      if (parts.some(value => value === '')) throw new Error("Enter four nonempty comma-separated numbers.");
      const next = parts.map(Number);
      integralSpaceState(next, query);
      setSlopes(next);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  };
  const preset = (next, x) => {
    setSlopes(next);
    setDraft(next.join(", "));
    setQuery(x);
    setError('');
  };
  return <section className="functional-lab" aria-label="Evaluation kernel investigation"><h3>An inner product that reads one point</h3><p>Each slope acts over a quarter of the interval. Move the query: only the part before x contributes to f(x). The kernel's slope is 1 before x and 0 afterward.</p>
    <form onSubmit={apply}><label className="functional-control">Four slopes in [−4,4]<input aria-label="Four slope values" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Apply slopes</button></form>
    {error && <p className="functional-error" role="alert">{error} The last valid slopes remain active.</p>}
    <Slider label="Evaluation query" value={query} set={setQuery} min={0} max={1} step={.025} />
    <div className="functional-slope-strip" aria-label="Four slope intervals; highlighted widths are the parts before the query">{state.intervals.map(item => <div key={item.left}><strong>{fmt(item.slope)}</strong><div className="functional-overlap"><span style={{
            width: `${item.overlap * 400}%`
          }} /></div><small>{fmt(item.left)}–{fmt(item.right)}</small></div>)}</div>
    <CurvePlot curves={[{
      label: "integrated f",
      fn: state.valueAt
    }, {
      label: 'kₓ(t)=min(x,t)',
      fn: t => Math.min(query, t),
      dash: "6 4"
    }]} yRange={plotRange} query={query} markers={[{
      x: query,
      y: state.evaluation
    }]} title="Accumulated function and the query's kernel section; vertical scale adjusts to the active values" />
    <LessonTable caption="Signed integral contributions from the active slopes" headers={['Interval', 'Slope', 'Overlap', 'Product']} rows={state.intervals.map(item => [`${fmt(item.left)}–${fmt(item.right)}`, fmt(item.slope), fmt(item.overlap), fmt(item.contribution)])} />
    <Metrics rows={[["f(x)=〈f,kₓ〉", fmt(state.evaluation)], ["||f||² = slope energy", fmt(state.squaredNorm)], ["Point-value bound", fmt(state.bound)]]} />
    <p>Absolute value {fmt(Math.abs(state.evaluation))} ≤ {fmt(state.bound)}. Equality occurs when the active derivative is proportional to the kernel derivative; the attainment preset shows this. All values shown are rounded.</p>
    <Actions><button onClick={() => preset([2, -1, 1, 0], .6)}>Reset evaluation</button><button onClick={() => preset([2, 2, 0, 0], .5)}>Attain the bound</button><button onClick={() => setQuery(0)}>Read the anchor</button></Actions>
  </section>;
}
export function KernelValidityLab() {
  const [kind, setKind] = useState('polynomial');
  const state = kernelValidityState(kind);
  return <section className="functional-lab" aria-label="Kernel validity investigation"><h3>Check the whole quadratic form</h3><p>The coefficients are fixed at (1,−1,1). Follow one product cᵢKᵢⱼcⱼ from the Gram matrix into the signed sum. For the invalid preset, every two-point principal matrix passes, while three points fail.</p>
    <label className="functional-control">Kernel construction<select aria-label="Kernel construction" value={kind} onChange={event => setKind(event.target.value)}><option value="polynomial">Polynomial features</option><option value="bigrams">String bigram counts</option><option value="invalid">Invalid three-object similarity</option></select></label>
    {state.features && <LessonTable caption={kind === 'bigrams' ? "Feature columns count AB and BA, including overlaps" : "Feature columns are 1, √2 x and x²"} headers={['Input', 'Features']} rows={state.labels.map((label, i) => [label, state.features[i].map(fmt).join(", ")])} />}
    <div className="functional-matrix-pair"><LessonTable caption="K: pair inner products" headers={['Input', ...state.labels]} rows={state.gram.map((row, i) => [state.labels[i], ...row.map(fmt)])} /><LessonTable caption="cᵢ Kᵢⱼ cⱼ: nine terms to add" headers={['Input', ...state.labels]} rows={state.contributions.map((row, i) => [state.labels[i], ...row.map(fmt)])} /></div>
    <Metrics rows={[["cᵀKc", fmt(state.quadratic)], ["This construction", state.valid ? "PSD by feature factorization" : "Invalid: negative squared norm"]]} />
    <p>{state.valid ? "For every coefficient vector, cᵀΦΦᵀc=||Φᵀc||²≥0. That algebraic construction proves validity on the entire declared feature domain; this one displayed sum alone would not." : "The three-point sum is −0.6. A squared norm cannot be negative. Nonnegative entries, symmetry and the three positive pair determinants 0.19,1,0.19 do not rescue this matrix."}</p>
    <Actions><button onClick={() => setKind('polynomial')}>Reset kernel</button></Actions>
  </section>;
}
export function RepresenterGeometryLab() {
  const [amplitude, setAmplitude] = useState(.5);
  const [extra, setExtra] = useState(false);
  const state = representerState(amplitude, extra);
  return <section className="functional-lab" aria-label="Representer geometry investigation"><h3>Move a curve where the observations cannot see</h3><p>The measured targets lie at (.5,1) and (1,0), with the fixed anchor (0,0). Add a wiggle ending before .5. Inspect which sample values and which energy change.</p>
    <Slider label="Wiggle amplitude" value={amplitude} set={setAmplitude} min={-1.5} max={1.5} step={.05} />
    <CurvePlot curves={[{
      label: "minimal-energy curve",
      fn: state.baseline
    }, {
      label: "curve + wiggle",
      fn: state.total,
      dash: "6 3"
    }, {
      label: "wiggle only",
      fn: state.wiggle,
      dash: "2 3"
    }]} yRange={[-1.5, 2.5]} markers={state.observations.map(x => ({
      x,
      y: state.baseline(x)
    }))} title={extra ? "The new reading exposes the former invisible wiggle" : "Same observed values, a different function between them"} />
    <Metrics rows={[["Baseline energy", '4'], ["Extra energy", fmt(state.addedEnergy)], ["Total energy", fmt(state.totalEnergy)], ["Residual at new .25 reading", extra ? fmt(state.extraResidual) : "No reading there"]]} />
    <label className="functional-checkbox"><input type="checkbox" checked={extra} onChange={event => setExtra(event.target.checked)} />Add a target (.25,.5)</label>
    <p>{extra ? "Now the former invisible wiggle changes a measured value by its amplitude. It is no longer orthogonal to the enlarged span of evaluation sections. We are diagnosing the old decomposition, not claiming these curves have been refitted." : "The slope cross term integrates to zero on each observed interval. The two sample losses remain identical, while a strictly increasing norm penalty prefers amplitude 0."}</p>
    <Actions><button onClick={() => {
        setAmplitude(.5);
        setExtra(false);
      }}>Reset wiggle</button><button onClick={() => setAmplitude(0)}>Remove unseen component</button></Actions>
  </section>;
}
export function KernelRidgeLab() {
  const [gamma, setGamma] = useState(1);
  const [lambda, setLambda] = useState(.05);
  const [query, setQuery] = useState(.5);
  const [fixture, setFixture] = useState('original');
  const state = kernelRidgeState(gamma, lambda, query, fixture);
  const curveValues = Array.from({
    length: 241
  }, (_, i) => state.predict(-.25 + 1.5 * i / 240));
  const plotRange = [Math.min(-.5, Math.floor(Math.min(...curveValues) * 4) / 4), Math.max(1.75, Math.ceil(Math.max(...curveValues) * 4) / 4)];
  return <section className="functional-lab" aria-label="Kernel ridge investigation"><h3>Build a prediction from signed kernel sections</h3><p>λ belongs to the average-loss objective. Gamma and λ refit the same active data; moving the query only reads the resulting function. Circles are training targets; squares are declared synthetic validation targets.</p>
    <label className="functional-control">Observed data<select aria-label="Observed data" value={fixture} onChange={event => setFixture(event.target.value)}><option value="original">Original two observations</option><option value="curve">Five noisy synthetic observations</option><option value="duplicates">Conflicting duplicate location</option></select></label>
    <Slider label="RBF gamma" value={gamma} set={setGamma} min={.05} max={32} step={.05} />
    <Slider label="Average-loss lambda" value={lambda} set={setLambda} min={.0001} max={.5} step={.0001} />
    <Slider label="Prediction query" value={query} set={setQuery} min={-.25} max={1.25} step={.025} />
    <CurvePlot curves={[{
      label: "fitted function",
      fn: state.predict
    }]} xRange={[-.25, 1.25]} yRange={plotRange} query={query} markers={[...state.xs.map((x, i) => ({
      x,
      y: state.ys[i]
    })), ...state.validation.map(([x, y]) => ({
      x,
      y,
      square: true
    }))]} title="Fitted predictions and separately identified targets; vertical scale expands when needed" />
    <LessonTable caption="Contributions at the query: alpha times similarity" headers={['Center', 'α', 'k(center,query)', 'Contribution']} rows={state.xs.map((x, i) => [fmt(x), fmt(state.alpha[i]), fmt(Math.exp(-gamma * (x - query) ** 2)), fmt(state.contributions[i])])} />
    <Metrics rows={[["Sum = prediction", fmt(state.prediction)], ["Diagonal shift nλ", fmt(state.shift)], ["Squared RKHS norm ≈", fmt(state.squaredNorm)], ["Training MSE", fmt(state.trainMse)], ["Validation MSE", fmt(state.validationMse)]]} />
    <p>{fixture === 'duplicates' ? "Both targets at .5 receive the same function value, even though their labels differ. Their separate coefficients can be large with opposite signs." : "The norm measures this function relative to this kernel. The displayed synthetic targets illustrate the mechanism; they are not evidence of performance on an external dataset."} Norm arithmetic uses a nonnegative Gaussian feature sum through degree 192, independently checked against the Gram form on the declared ranges. It is a numerical approximation, not an exact finite feature representation.</p>
    <Actions><button onClick={() => {
        setGamma(1);
        setLambda(.05);
        setQuery(.5);
        setFixture('original');
      }}>Reset ridge</button></Actions>
  </section>;
}
export function KernelMeanLab() {
  const [gamma, setGamma] = useState(1);
  const [same, setSame] = useState(false);
  const state = distributionEmbeddingState(gamma, same);
  return <section className="functional-lab" aria-label="Kernel distribution witness investigation"><h3>Equal moments can hide a different distribution</h3><p>Compare the atom probabilities first. The polynomial features 1,√2x,x² have the same expectation. A Gaussian kernel can still provide a function with different expectations.</p>
    <div className="functional-atoms" role="img" aria-label="Atom probability columns for P and Q; exact heights appear in the adjacent table">{state.support.map((x, i) => <div key={x}><div className="functional-atom-bars"><span style={{
            height: `${state.p[i] * 120}px`
          }} /><span style={{
            height: `${state.q[i] * 120}px`
          }} /></div><strong>{x}</strong></div>)}</div>
    <p>Each paired column: P on the left, Q on the right. Height is probability, not density.</p>
    <LessonTable caption="Exact atom probabilities" headers={['Location', 'P', 'Q']} rows={state.support.map((x, i) => [x, fmt(state.p[i]), fmt(state.q[i])])} />
    <Slider label="Witness gamma" value={gamma} set={setGamma} min={.05} max={8} step={.05} />
    <CurvePlot curves={[{
      label: 'μP(t)−μQ(t)',
      fn: state.witness
    }]} xRange={[-3, 3]} yRange={[-1, 1]} title="A signed expectation witness, not a probability density" />
    <Metrics rows={[["P / Q mean", `${state.pMoments[1]} / ${state.qMoments[1]}`], ["P / Q second moment", `${state.pMoments[2]} / ${state.qMoments[2]}`], ["P / Q fourth moment", `${state.pMoments[3]} / ${state.qMoments[3]}`], ["Gaussian MMD²", fmt(state.squaredMmd)]]} />
    <p>{same ? "The laws coincide, so the difference function and its norm are exactly zero in this fixture. There is no nonzero direction to normalize." : "Normalize this difference function by its RKHS norm to obtain a unit-norm witness. Its expectation gap is the MMD. A finite positive result verifies this pair, not the general characteristic-kernel theorem."}</p>
    <Actions><button onClick={() => setSame(!same)}>{same ? "Restore different laws" : "Set Q equal to P"}</button><button onClick={() => {
        setGamma(1);
        setSame(false);
      }}>Reset witness</button></Actions>
  </section>;
}
export function KernelQuadratureLab() {
  const [node, setNode] = useState(.5);
  const [manual, setManual] = useState(false);
  const [weight, setWeight] = useState(.75);
  const state = kernelQuadratureState(node, manual ? weight : null);
  return <section className="functional-lab" aria-label="Kernel quadrature investigation"><h3>Approximate an integral with one controlled reading</h3><p>We estimate ∫₀¹f(t)dt by w f(x). Project the integration representer m onto the evaluation section kₓ. Its residual is the error measurement left over.</p>
    <Slider label="Quadrature node" value={node} set={setNode} min={0} max={1} step="any" />
    <label className="functional-checkbox"><input type="checkbox" checked={manual} onChange={event => setManual(event.target.checked)} />Choose the weight manually</label>
    {manual && <Slider label="Quadrature weight" value={weight} set={setWeight} min={-1} max={2} step={.025} />}
    <CurvePlot curves={[{
      label: 'm(t)=t−t²/2',
      fn: state.mean
    }, {
      label: "w kₓ(t)",
      fn: t => state.weight * Math.min(node, t),
      dash: "6 4"
    }, {
      label: 'residual',
      fn: state.residual,
      dash: "2 3"
    }]} yRange={manual ? [-2, 2] : [-.2, .8]} query={node} title="Integration representer, one-section approximation and residual" />
    <Metrics rows={[["Weight", fmt(state.weight)], ["Worst-case error, ||f||≤1", fmt(state.error)], ["Squared bound", fmt(state.squaredError)], ["Estimate for f(t)=t", fmt(state.estimateLinear)], ["Actual integral of t", '.5'], ["Actual error for t", fmt(state.linearError)]]} />
    <p>{state.endpointNote || "The actual error for f(t)=t lies below the unit-ball bound because this function has RKHS norm 1. The normalized residual itself attains the bound. This is a deterministic guarantee for the declared function class; there is no sampling confidence level."}</p>
    <Actions><button onClick={() => {
        setNode(.5);
        setManual(false);
        setWeight(.75);
      }}>Reset quadrature</button><button onClick={() => {
        setNode(2 / 3);
        setManual(false);
      }}>Best one-node location</button><button onClick={() => setNode(0)}>Observe only the anchor</button></Actions>
  </section>;
}
export function OperatorModesFigure() {
  return <figure className="functional-inline"><h3>Same function, different geometric cost</h3><div className="functional-mode-row"><span>Constant mode 1</span><span>L² eigenvalue 1</span><span>RKHS cost a²</span></div><div className="functional-mode-row"><span>Linear mode √3x</span><span>L² eigenvalue 1/3</span><span>RKHS cost 3c²</span></div><figcaption>For uniform measure on [−1,1] and k(x,z)=1+xz, write f=a+c√3x. Its L² squared norm is a²+c²; its RKHS squared norm is a²+3c². The coefficient c is measured in the displayed normalized mode, not in the unnormalized coordinate x.</figcaption></figure>;
}
