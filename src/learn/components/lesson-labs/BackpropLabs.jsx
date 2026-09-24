import { useState } from 'react';
import { finiteDifference, fitDefaults, lineFit, sharedSquare } from '../../data/backprop-models.js';
import { BackpropFigure, BackpropNumber, BackpropSteps, BackpropTable, numberText as f } from './BackpropShared.jsx';

export function BackpropFitLab() {
  const [inputs, setInputs] = useState(fitDefaults);
  const [resetKey, setResetKey] = useState(0);
  const model = lineFit(inputs);
  const domainValues = [0, ...model.rows.flatMap(row => [row.target, row.prediction, row.nextPrediction])];
  const low = Math.min(...domainValues) - 1;
  const high = Math.max(...domainValues) + 1;
  const px = x => 48 + (x - 0.75) * 170;
  const py = y => 228 - (y - low) / (high - low) * 200;
  return <section className="backprop-lab" data-backprop-lab="fit" aria-labelledby="backprop-fit-title">
    <h3 id="backprop-fit-title">Investigation A · A correct direction can be an unsafe step</h3>
    <p>Two examples, x = (1, 2). Change a model parameter, target or learning rate. Both new parameters use the same old gradient; the proposed line and its loss update immediately.</p>
    <p className="backprop-baseline"><strong>Starting case, kept for comparison:</strong> w = 1, b = 0, targets (1, 3), η = 0.1. MSE 0.5 → 0.17; gradients (−2, −1). These reference values stay fixed while you edit the current case.</p>
    <div className="backprop-controls">{[
      ['weight', 'Weight w', -3, 3], ['bias', 'Bias b', -3, 3], ['rate', 'Learning rate η', 0, 1],
      ['target1', 'Target at x = 1', -3, 5], ['target2', 'Target at x = 2', -3, 5],
    ].map(([key, name, min, max]) => <BackpropNumber key={key} name={name} value={inputs[key]} min={min} max={max} resetKey={resetKey}
      onChange={value => setInputs(previous => ({ ...previous, [key]: value }))} />)}</div>
    <div className="backprop-actions"><button type="button" onClick={() => { setInputs({ ...fitDefaults }); setResetKey(key => key + 1); }}>Reset line fit</button></div>
    <BackpropFigure id="fit-line" title="Observed targets and simultaneous update" caption="Circles: observed targets. Solid amber: current fit. Dashed light line: proposed fit. Vertical red segments: current residuals. The horizontal scale stays fixed; the vertical range includes every current value.">
      <svg className="backprop-svg backprop-fit-svg" viewBox="0 0 330 278" role="img" aria-label="Current and proposed fitted line at x equals 1 and 2">
        <line x1="60" x2="300" y1="228" y2="228" className="bp-axis" />
        <line x1="60" x2="60" y1="28" y2="228" className="bp-axis" />
        {[low, (low + high) / 2, high].map(value => <g key={value}><line x1="60" x2="300" y1={py(value)} y2={py(value)} className="bp-grid" /><text x="55" y={py(value) + 4} textAnchor="end">{Number(value.toPrecision(3))}</text></g>)}
        <text x="60" y="16">target / output</text>
        <line x1={px(1)} y1={py(model.rows[0].prediction)} x2={px(2)} y2={py(model.rows[1].prediction)} className="bp-current" />
        <line x1={px(1)} y1={py(model.rows[0].nextPrediction)} x2={px(2)} y2={py(model.rows[1].nextPrediction)} className="bp-proposed" />
        {model.rows.map(row => <g key={row.x}><line x1={px(row.x)} x2={px(row.x)} y1={py(row.target)} y2={py(row.prediction)} className="bp-residual" /><circle cx={px(row.x)} cy={py(row.target)} r="5" className="bp-target" /><text x={px(row.x)} y="250" textAnchor="middle">x = {row.x}</text></g>)}
      </svg>
      <BackpropTable caption="Current residuals, gradient contributions and proposed outputs" headers={['x', 'Target', 'Current q', 'r = q − y', 'r × x', 'New q']} rows={model.rows.map(row => [row.x, f(row.target), f(row.prediction), f(row.residual), f(row.weightContribution), f(row.nextPrediction)])} />
    </BackpropFigure>
    <div className="backprop-result" data-fit-result data-weight={inputs.weight} data-bias={inputs.bias} data-rate={inputs.rate} data-old-loss={model.loss} data-new-loss={model.nextLoss} data-weight-gradient={model.weightGradient} data-bias-gradient={model.biasGradient}>
      <p><strong>MSE: {f(model.loss)} → {f(model.nextLoss)} ({model.change}).</strong> Change = {f(model.delta)}.</p>
      <p>g<sub>w</sub> = {model.rows.map(row => `(${f(row.weightContribution)})`).join(' + ')} = {f(model.weightGradient)}; g<sub>b</sub> = {model.rows.map(row => `(${f(row.biasContribution)})`).join(' + ')} = {f(model.biasGradient)}. The mean over two entries cancels the factor 2 from squaring.</p>
      <p>w′ = {f(inputs.weight)} − {f(inputs.rate)} × ({f(model.weightGradient)}) = {f(model.nextWeight)}; b′ = {f(inputs.bias)} − {f(inputs.rate)} × ({f(model.biasGradient)}) = {f(model.nextBias)}.</p>
      <p>{inputs.rate === 0 ? 'A zero learning rate leaves this model unchanged.' : model.weightGradient === 0 && model.biasGradient === 0 ? 'Both gradients vanish, so every proposed parameter change is zero.' : model.change === 'increased' ? 'This step overshoots: the derivative is local, and this learning rate makes the evaluated loss larger.' : model.change === 'decreased' ? 'This evaluated step lowers the loss. The gradient supplied the direction; the new forward pass established the improvement.' : 'The before/after losses agree within absolute 1e−12 plus relative 1e−10 tolerance; equal loss need not imply identical parameters.'}</p>
    </div>
  </section>;
}

export function BackpropSharedLab() {
  const [point, setPoint] = useState(3);
  const [coefficient, setCoefficient] = useState(2);
  const [stage, setStage] = useState(0);
  const [resetKey, setResetKey] = useState(0);
  const model = sharedSquare(point, coefficient);
  const descriptions = [
    'Seed the final loss sensitivity with 1. The complete derivative remains visible throughout this trace.',
    `The direct route contributes 1 to u, and the scaled route contributes c = ${f(coefficient)}. Their total is ${f(model.uGradient)}.`,
    `Each operand slot of x × x receives (${f(model.uGradient)}) × (${f(point)}) = ${f(model.slotContribution)}. Add both slots to get ${f(model.gradient)}.`,
  ];
  return <section className="backprop-lab" data-backprop-lab="shared" aria-labelledby="backprop-shared-title">
    <h3 id="backprop-shared-title">Investigation B · One value, several paths</h3>
    <p>Edit x and the reused-branch coefficient c in u = x × x, L = u + c × u. One x node feeds both multiply slots; one u node feeds both loss paths.</p>
    <p className="backprop-baseline"><strong>Starting case, kept for comparison:</strong> x = 3, c = 2, u = 9, L = 27 and dL/dx = 18. Current edits below do not change this reference.</p>
    <div className="backprop-controls"><BackpropNumber name="Shared input x" value={point} min={-3} max={3} step={0.25} onChange={setPoint} resetKey={resetKey} /><BackpropNumber name="Branch coefficient c" value={coefficient} min={-3} max={3} step={0.25} onChange={setCoefficient} resetKey={resetKey} /></div>
    <BackpropFigure id="shared-graph" title="A shared node is not a single operand slot" caption={`${stage === 0 ? 'Arrows show forward use.' : 'Highlighted arrows now point backward as sensitivity returns; other arrows retain forward direction.'} The two curved edges between x and u are distinct operand slots, not two independent x values. The table follows sensitivity backward through both kinds of reuse.`}>
      <svg className="backprop-svg" viewBox="0 0 360 265" role="img" aria-label="Shared x twice into square u; u reused directly and through coefficient c into loss L">
        <defs><marker id="backprop-shared-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" /></marker></defs>
        <g className={`bp-edges ${stage === 2 ? 'bp-active' : ''}`} markerStart={stage === 2 ? 'url(#backprop-shared-arrow)' : undefined} markerEnd={stage === 2 ? undefined : 'url(#backprop-shared-arrow)'}><path d="M 58 117 Q 95 56 137 117" /><path d="M 58 147 Q 95 208 137 147" /></g>
        <g className={`bp-edges ${stage === 1 ? 'bp-active' : ''}`} markerStart={stage === 1 ? 'url(#backprop-shared-arrow)' : undefined} markerEnd={stage === 1 ? undefined : 'url(#backprop-shared-arrow)'}><path d="M 177 117 L 238 54" /><path d="M 178 145 L 237 207" /><path d="M 276 54 L 319 116" /><path d="M 276 207 L 319 147" /></g>
        {[[45, 132, 'x'], [155, 132, 'u'], [256, 40, '×1'], [256, 220, '×c'], [331, 132, 'L']].map(([cx, cy, label]) => <g key={label}><circle cx={cx} cy={cy} r="23" className="bp-node" /><text x={cx} y={cy + 6} textAnchor="middle" className="bp-node-label">{label}</text></g>)}
        <text x="97" y="75" textAnchor="middle">slot 1</text><text x="97" y="210" textAnchor="middle">slot 2</text>
      </svg>
      <BackpropTable caption="Forward values and reverse contributions" headers={['Node', 'Forward value', 'Incoming reverse contributions', 'Total sensitivity']} rows={[
        ['L', f(model.loss), 'seed', '1'], ['u', f(model.square), `1 + (${f(coefficient)})`, f(model.uGradient)], ['x (one shared value)', f(point), `${f(model.slotContribution)} + ${f(model.slotContribution)}`, f(model.gradient)],
      ]} />
    </BackpropFigure>
    <BackpropSteps index={stage} setIndex={setStage} count={3} name="Backward" /><p className="backprop-stage">{descriptions[stage]}</p>
    <div className="backprop-result" data-shared-result data-x={point} data-coefficient={coefficient} data-loss={model.loss} data-gradient={model.gradient} data-stage={stage}>
      <p><strong>L = {f(model.loss)}; dL/dx = {f(model.gradient)}.</strong> Direct loss path: 2x = {f(model.firstPath)}. Scaled loss path: 2cx = {f(model.secondPath)}. Add paths to obtain 2x(1 + c).</p>
      <p>{coefficient === -1 ? 'The branches cancel exactly for every x: L = 0 and dL/dx = 0. Further x edits preserve both zeros.' : point === 0 ? 'At x = 0, both multiplication-slot contributions vanish for every c. This is a second null case, not a failure to update.' : 'Multiply local sensitivities along a path; add contributions across reused paths and repeated operand slots. Try a negative x and c below −1 to inspect the sign reversal.'}</p>
    </div>
    <button type="button" onClick={() => { setPoint(3); setCoefficient(2); setStage(0); setResetKey(key => key + 1); }}>Reset shared graph</button>
  </section>;
}

function LocalFunction({ result }) {
  const { point, step, kind, offset } = result;
  const fn = kind === 'linear' ? x => offset + x : kind === 'square' ? x => x * x : Math.sin;
  const samples = Array.from({ length: 41 }, (_, i) => {
    const x = point + (i / 40 * 2 - 1) * step;
    return { x, y: fn(x) };
  });
  const minX = samples[0].x;
  const maxX = samples.at(-1).x;
  const minY = Math.min(...samples.map(p => p.y));
  const maxY = Math.max(...samples.map(p => p.y));
  const px = x => maxX === minX ? 160 : 28 + 264 * ((x - minX) / (maxX - minX));
  const py = y => maxY === minY ? 77 : 127 - 100 * ((y - minY) / (maxY - minY));
  return <div>
    <p className="backprop-small">Magnified computed function over [x − h, x + h]. Both axes rescale to this local interval; the table supplies their exact coordinates.</p>
    <svg className="backprop-svg" viewBox="0 0 320 165" role="img" aria-label="Magnified local function sampled with binary64 arithmetic">
      <polyline points={samples.map(p => `${px(p.x)},${py(p.y)}`).join(' ')} className="bp-current" />
      <line x1={px(result.left)} x2={px(result.right)} y1={py(result.lower)} y2={py(result.upper)} className="bp-proposed" />
      <circle cx={px(result.left)} cy={py(result.lower)} r="5" className="bp-target" /><rect x={px(result.right) - 4} y={py(result.upper) - 4} width="8" height="8" className="bp-target" />
      <text x="28" y="155">● x − h</text><text x="292" y="155" textAnchor="end">■ x + h</text>
    </svg>
  </div>;
}

export function BackpropDifferenceLab() {
  const [kind, setKind] = useState('sine');
  const [point, setPoint] = useState(1);
  const [offset, setOffset] = useState(0);
  const [firstExponent, setFirstExponent] = useState(-3);
  const [secondExponent, setSecondExponent] = useState(-5);
  const [resetKey, setResetKey] = useState(0);
  const candidates = [firstExponent, secondExponent].map(exponent => finiteDifference({ kind, point, offset, exponent }));
  const difference = candidates[1].absoluteError - candidates[0].absoluteError;
  const equalityTolerance = 1e-18 + 1e-12 * Math.max(candidates[0].absoluteError, candidates[1].absoluteError);
  const comparison = Math.abs(difference) <= equalityTolerance ? 'equal to Check A’s within the stated comparison tolerance' : difference < 0 ? 'smaller than Check A’s' : 'larger than Check A’s';
  return <section className="backprop-lab" data-backprop-lab="difference" aria-labelledby="backprop-difference-title">
    <h3 id="backprop-difference-title">Investigation F · Inspect the numbers being subtracted</h3>
    <p>Two simultaneous checks of the same function. Compare perturbations while holding the point and function fixed. These are current JavaScript binary64 evaluations; last bits may differ from the saved Python observations.</p>
    <p className="backprop-baseline"><strong>Starting case, kept for comparison:</strong> sin(x) at x = 1; h<sub>A</sub> = 10⁻³ and h<sub>B</sub> = 10⁻⁵. Recorded absolute errors are approximately 9.005 × 10⁻⁸ and 1.114 × 10⁻¹¹, respectively. These reference inputs and errors stay fixed.</p>
    <label htmlFor="backprop-function">Function to differentiate</label><select id="backprop-function" value={kind} onChange={event => setKind(event.target.value)}><option value="sine">sin(x)</option><option value="linear">C + x</option><option value="square">x²</option></select>
    <div className="backprop-controls"><BackpropNumber name="Evaluation point x" value={point} min={-2} max={2} onChange={setPoint} resetKey={resetKey} />
      {kind === 'linear' && <BackpropNumber name="Constant offset C" value={offset} min={0} max={1e12} step={1e6} onChange={setOffset} resetKey={resetKey} />}
      <BackpropNumber name="Check A exponent (h = 10ᵉ)" value={firstExponent} min={-15} max={0} step={1} onChange={setFirstExponent} resetKey={resetKey} />
      <BackpropNumber name="Check B exponent (h = 10ᵉ)" value={secondExponent} min={-15} max={0} step={1} onChange={setSecondExponent} resetKey={resetKey} /></div>
    {kind === 'linear' && <div className="backprop-actions" aria-label="Offset starting values">{[0, 1e6, 1e12].map(value => <button type="button" key={value} onClick={() => { setOffset(value); setResetKey(key => key + 1); }}>C = {value.toExponential(0)}</button>)}</div>}
    <div className="backprop-pair">{candidates.map((result, index) => <div key={index} className="backprop-difference-case" data-difference-case={index === 0 ? 'A' : 'B'} data-error={result.absoluteError} data-estimate={result.estimate} data-analytic={result.analytic} data-relative={result.relativeError ?? 'undefined'}>
      <h4>Check {index === 0 ? 'A' : 'B'} · h = {f(result.step)}</h4><LocalFunction result={result} />
      <dl className="backprop-exact">{[
        ['x − h', result.left], ['x + h', result.right], ['f(x − h)', result.lower], ['f(x + h)', result.upper], ['Central difference', result.estimate], ['Analytic derivative', result.analytic], ['Absolute error', result.absoluteError], ['Relative error', result.relativeError ?? 'Undefined: analytic derivative is zero'],
      ].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{String(value)}</dd></div>)}</dl>
      {result.coordinatesCoalesce && <p>The two input coordinates round to the same binary64 number; they coincide in the graph.</p>}
      {result.evaluationsCoalesce && <p>The two evaluated outputs are identical. Their difference is exactly zero{result.analytic === 0 ? ', matching the analytic derivative in this symmetric quadratic case.' : '; the nonzero analytic derivative is not recovered.'}</p>}
    </div>)}</div>
    <div className="backprop-result"><p><strong>Check B’s absolute error is {comparison}.</strong> Comparison tolerance: 10⁻¹⁸ + 10⁻¹² × the larger error; raw errors above remain unrounded.</p><p>{kind === 'linear' ? 'Changing C leaves the mathematical derivative at 1. Large offsets can erase the difference between nearby evaluations; inspect both numbers before blaming the backward rule.' : kind === 'square' && point === 0 ? 'At x = 0, symmetric quadratic samples agree and the analytic derivative is zero. Absolute error is meaningful; relative error has no denominator.' : 'A large h introduces truncation error; an extremely small h amplifies rounding and cancellation. Sweep h instead of assuming that a smaller perturbation is always better.'}</p>
      <p>Separate nonsmooth case: at ReLU(0), central difference = 0.5 and the chosen AD rule = 0. There is no ordinary derivative at that corner.</p></div>
    <button type="button" onClick={() => { setKind('sine'); setPoint(1); setOffset(0); setFirstExponent(-3); setSecondExponent(-5); setResetKey(key => key + 1); }}>Reset numerical checks</button>
  </section>;
}
