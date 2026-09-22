import { useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, neuralColors, formatNeural as fmt } from './NeuralLessonElements.jsx';
import { normalizationMean as mean, normalizeVector, tensorGroup, normalizeTensor, batchNormalizationStep, normalizationGradient } from '../../data/normalization-models.js';
import measurements from '../../data/normalization-measurements.js';
import './normalization-labs.css';
const methods = [['batch', 'Training BatchNorm'], ['layer', 'LayerNorm over C,H,W'], ['group', 'GroupNorm'], ['instance', 'InstanceNorm']];
export function NormalizationRulerFigure() {
  const raw = [1, 3, 5, 7],
    normalized = normalizeVector(raw);
  return <figure className="normalization-ruler"><figcaption>The same four observations, expressed relative to a shared ruler</figcaption><NeuralTable caption="Exact construction, ε = 0.00001" headers={['Identity', 'Raw value', 'Subtract mean 4', 'Square deviation', 'Divide by √(5+ε)']} rows={raw.map((value, i) => [String.fromCharCode(65 + i), value, value - 4, (value - 4) ** 2, fmt(normalized.output[i], 6)])} /><NeuralPlot title="Centering preserves order and the pattern of gaps" xLabel="raw value" yLabel="normalized value" xDomain={[0, 8]} yDomain={[-1.6, 1.6]} series={[{
      label: '(x − 4) / √(5 + ε)',
      values: raw.map((value, i) => [value, normalized.output[i]])
    }]} points={raw.map((value, i) => ({
      x: value,
      y: normalized.output[i],
      label: `${String.fromCharCode(65 + i)}: ${value} becomes ${fmt(normalized.output[i])}`
    }))} /><p>Mean = 0; output variance = 5 / (5 + ε) = {fmt(5 / (5 + 1e-5), 8)}. The transform does not make an arbitrary distribution Gaussian.</p></figure>;
}
export function NormalizationMembershipLab() {
  const original = Array.from({
    length: 16
  }, (_, i) => i + 1);
  const [values, setValues] = useState(original),
    [method, setMethod] = useState('batch'),
    [groups, setGroups] = useState(2),
    [selected, setSelected] = useState(0),
    [epsilon, setEpsilon] = useState(1e-5);
  const group = tensorGroup(selected, method, groups),
    statistics = normalizeVector(group.map(i => values[i]), epsilon),
    output = normalizeTensor(values, method, groups, epsilon);
  const baseline = normalizeTensor(original, method, groups, epsilon),
    delta = Math.max(...output.slice(0, 8).map((value, i) => Math.abs(value - baseline[i])));
  const edit = (i, value) => setValues(values.map((v, j) => j === i ? value : v));
  return <NeuralLab title="Which cells share this value’s ruler?" id="normalization-membership"><p>Shape 2 examples × 4 channels × 1 row × 2 positions. Select a cell to inspect its statistics group. Edit any cell directly: gold borders mark the selected group, while each output shows its actual change from the original 1–16 fixture.</p>
    <div className="neural-controls"><NeuralSelect label="Statistics rule" value={method} onChange={setMethod} options={methods} /><NeuralSelect label="Group count for GroupNorm" value={groups} onChange={v => setGroups(Number(v))} options={[[1, '1 group'], [2, '2 groups'], [4, '4 groups']]} /><NeuralNumber label="Membership epsilon" value={epsilon} onChange={setEpsilon} min={.00000001} max={1} range={false} /></div>
    <div className="normalization-examples">{[0, 1].map(n => <div key={n}><h4>Example {n}</h4>{[0, 1, 2, 3].map(c => <div className="normalization-channel" key={c}><strong>Channel {c}</strong><div>{[0, 1].map(position => {
              const i = n * 8 + c * 2 + position;
              return <div className={`normalization-cell ${group.includes(i) ? 'normalization-member' : ''}`} key={position}><button aria-pressed={selected === i} onClick={() => setSelected(i)}>Inspect n{n}, c{c}, p{position}</button><NeuralNumber label={`Value n${n} c${c} p${position}`} value={values[i]} onChange={value => edit(i, value)} min={-50} max={50} range={false} /><p>Output {fmt(output[i], 4)}<br />Δ {fmt(output[i] - baseline[i], 4)}</p></div>;
            })}</div></div>)}</div>)}</div>
    <p className="neural-result" data-result="membership">Selected group: {group.length} values [{group.map(i => fmt(values[i])).join(', ')}]. Mean {fmt(statistics.center)}; variance {fmt(statistics.variance)}. Maximum change in example 0: {fmt(delta, 9)}.</p>
    <div className="neural-buttons"><button onClick={() => edit(8, 19)}>Change example 1’s first value to 19</button><button onClick={() => {
        setValues(original);
        setMethod('batch');
        setGroups(2);
        setSelected(0);
        setEpsilon(1e-5);
      }}>Reset tensor</button></div>
    <p>Change only example 1 while watching example 0: training BatchNorm crosses that boundary. LayerNorm, GroupNorm and InstanceNorm do not. Group count has no effect when a different rule is selected. These comparisons use identity affine parameters; equal statistics do not imply equal parameter-sharing contracts.</p>
  </NeuralLab>;
}
export function NormalizationGeometryLab() {
  const [vector, setVector] = useState([1, 3]),
    [offset, setOffset] = useState(0),
    [scale, setScale] = useState(1),
    [epsilon, setEpsilon] = useState(1e-5);
  const x = vector.map(value => scale * value + offset),
    centered = normalizeVector(x, epsilon),
    rms = normalizeVector(x, epsilon, true),
    originalCentered = normalizeVector(vector, epsilon),
    originalRms = normalizeVector(vector, epsilon, true);
  const radius = Math.sqrt(mean(x.map(v => v ** 2))),
    bound = Math.max(2, ...x.map(Math.abs)) * 1.25,
    coordinate = value => 160 + value / bound * 115;
  return <NeuralLab title="Removing an offset is different from rescaling" id="normalization-geometry"><p>Compare a two-feature vector with its transformed version, x′ = scale × x + offset. LayerNorm subtracts a shared mean; RMSNorm measures distance from zero. Both views use identity affine parameters.</p>
    <div className="neural-controls">{vector.map((value, i) => <NeuralNumber key={i} label={`Original feature ${i}`} value={value} onChange={v => setVector(vector.map((a, j) => i === j ? v : a))} min={-20} max={20} />)}<NeuralNumber label="Common offset" value={offset} onChange={setOffset} min={-20} max={20} /><NeuralNumber label="Common scale" value={scale} onChange={setScale} min={-3} max={3} /><NeuralNumber label="Geometry epsilon" value={epsilon} onChange={setEpsilon} min={.00000001} max={1} range={false} /></div>
    <figure><figcaption>Center and direction in two-feature space</figcaption><svg className="neural-chart normalization-vector" viewBox="0 0 320 320" role="img" aria-label="Equal-scale two-feature plane. Amber is the current vector; green is its mean point; blue joins the mean point to the vector."><line x1="30" y1="160" x2="290" y2="160" stroke="#83948a" /><line x1="160" y1="30" x2="160" y2="290" stroke="#83948a" /><line x1="45" y1="275" x2="275" y2="45" stroke="#596b61" strokeDasharray="4 5" /><line x1="160" y1="160" x2={coordinate(x[0])} y2={320 - coordinate(x[1])} stroke={neuralColors[0]} strokeWidth="3" /><line x1={coordinate(centered.center)} y1={320 - coordinate(centered.center)} x2={coordinate(x[0])} y2={320 - coordinate(x[1])} stroke={neuralColors[2]} strokeWidth="3" /><circle cx={coordinate(centered.center)} cy={320 - coordinate(centered.center)} r="5" fill={neuralColors[1]} /><circle cx={coordinate(x[0])} cy={320 - coordinate(x[1])} r="5" fill={neuralColors[0]} /><text x="160" y="308" textAnchor="middle">feature 0: ±{fmt(bound, 2)}</text><text x="12" y="18">feature 1, same scale</text></svg><p>Amber: vector [{x.map(v => fmt(v)).join(', ')}]. Green: mean point [{fmt(centered.center)}, {fmt(centered.center)}]. Blue: centered vector. The diagonal contains constant-feature vectors. RMS = {fmt(radius)}; Euclidean radius = √2 × RMS = {fmt(radius * Math.sqrt(2))}.</p></figure>
    <NeuralTable caption="Same input, two normalization operations" headers={['Operation', 'Original output', 'Transformed output', 'Maximum change']} rows={[['LayerNorm', originalCentered.output.map(v => fmt(v)).join(', '), centered.output.map(v => fmt(v)).join(', '), fmt(Math.max(...centered.output.map((v, i) => Math.abs(v - originalCentered.output[i]))), 9)], ['RMSNorm', originalRms.output.map(v => fmt(v)).join(', '), rms.output.map(v => fmt(v)).join(', '), fmt(Math.max(...rms.output.map((v, i) => Math.abs(v - originalRms.output[i]))), 9)]]} />
    <p className="neural-result" data-result="geometry">Current LayerNorm [{centered.output.map(v => fmt(v, 7)).join(', ')}]; RMSNorm [{rms.output.map(v => fmt(v, 7)).join(', ')}]. {Math.abs(centered.center) < 1e-10 ? 'Zero mean: the denominators and outputs agree.' : 'The nonzero mean enters RMS but is removed before LayerNorm’s scale calculation.'}</p>
    <div className="neural-buttons"><button onClick={() => {
        setVector([-1, 1]);
        setOffset(0);
        setScale(1);
      }}>Zero-mean comparison</button><button onClick={() => {
        setVector([5, 5]);
        setOffset(0);
        setScale(1);
      }}>Equal features</button><button onClick={() => {
        setVector([1, 3]);
        setOffset(0);
        setScale(1);
        setEpsilon(1e-5);
      }}>Reset geometry</button></div>
  </NeuralLab>;
}
export function BatchNormalizationStateLab() {
  const [values, setValues] = useState([1, 3, 5, 7]),
    [buffers, setBuffers] = useState({
      mean: 0,
      variance: 1
    }),
    [momentum, setMomentum] = useState(.1),
    [training, setTraining] = useState(true),
    [gamma, setGamma] = useState(1),
    [beta, setBeta] = useState(0),
    [passes, setPasses] = useState(0);
  const step = batchNormalizationStep(values, buffers, momentum, training, 1e-5, gamma, beta);
  return <NeuralLab title="Batch statistics, parameters and memory are separate" id="normalization-state"><p>Editing inputs immediately previews one forward pass from the stored buffers. “Advance one forward pass” commits that genuine state transition. It does not unlock a result or run an optimizer.</p>
    <div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={`Batch value ${i}`} value={value} onChange={v => setValues(values.map((a, j) => i === j ? v : a))} min={-20} max={20} />)}<NeuralNumber label="BatchNorm momentum" value={momentum} onChange={setMomentum} min={0} max={1} /><NeuralSelect label="Forward mode" value={training ? 'training' : 'evaluation'} onChange={v => setTraining(v === 'training')} options={[['training', 'Training: current batch'], ['evaluation', 'Evaluation: stored buffers']]} /><NeuralNumber label="Shared affine gamma" value={gamma} onChange={setGamma} min={-3} max={3} /><NeuralNumber label="Shared affine beta" value={beta} onChange={setBeta} min={-5} max={5} /></div>
    <div className="neural-flow"><div><strong>Current batch</strong><br />Mean {fmt(step.batchMean)}; population variance {fmt(step.populationVariance)}; corrected variance {fmt(step.correctedVariance)}.</div><div><strong>Stored memory → proposed memory</strong><br />Mean {fmt(buffers.mean)} → {fmt(step.next.mean)}; variance {fmt(buffers.variance)} → {fmt(step.next.variance)}.</div><div><strong>Parameters stay fixed during this forward pass</strong><br />γ = {fmt(gamma)}, β = {fmt(beta)}. No optimizer step has occurred.</div></div>
    <NeuralTable caption="Current forward output and subsequent evaluation" headers={['Input', 'Output of previewed pass', 'Evaluation using proposed buffers']} rows={values.map((value, i) => [fmt(value), fmt(step.output[i], 8), fmt(step.evaluationAfter[i], 8)])} />
    <p className="neural-result" data-result="state">{passes} committed forward passes. Proposed running mean {fmt(step.next.mean, 8)}, variance {fmt(step.next.variance, 8)}. {training ? momentum === 0 ? 'Momentum zero leaves memory unchanged, while training still normalizes by the batch.' : 'Training uses population variance for its output and corrected variance for the memory update.' : 'Evaluation reads the stored memory and leaves it unchanged.'}</p>
    <div className="neural-buttons"><button onClick={() => {
        setBuffers(step.next);
        setPasses(passes + 1);
      }}>Advance one forward pass</button><button onClick={() => {
        setValues([1, 3, 5, 7]);
        setBuffers({
          mean: 0,
          variance: 1
        });
        setMomentum(.1);
        setTraining(true);
        setGamma(1);
        setBeta(0);
        setPasses(0);
      }}>Reset state ledger</button></div>
    <NeuralTable caption="One image is not always one value per channel" headers={['Shape for one channel', 'Values contributing', 'Training behavior']} rows={[['(1, 1, 1, 2)', '[1, 3]', 'Valid; outputs approximately −1, 1'], ['(1, 1, 1, 1)', '[1]', 'PyTorch rejects a single value per channel']]} />
  </NeuralLab>;
}
export function NormalizationGradientLab() {
  const [x, setX] = useState([1, 3]),
    [rate, setRate] = useState(.1);
  const initial = normalizationGradient(x, [1, 2], [0, 0], [0, 1]);
  const gamma = [1, 2].map((value, i) => value - rate * initial.scaleGradient[i]),
    beta = initial.shiftGradient.map(value => -rate * value);
  const after = normalizationGradient(x, gamma, beta, [0, 1]);
  const upstream = initial.shiftGradient.map((value, i) => value * [1, 2][i]);
  const average = (upstream[0] + upstream[1]) / 2;
  const centered = normalizeVector(x);
  const coupling = (upstream[0] * centered.output[0] + upstream[1] * centered.output[1]) / 2;
  return <NeuralLab title="A gradient follows every statistics path" id="normalization-gradients"><p>Input branches into the mean, variance and direct numerator. Their gradient contributions must combine. This experiment updates γ and β only, holding x fixed; changing the rate previews the resulting output immediately.</p><div className="neural-controls">{x.map((v, i) => <NeuralNumber key={i} label={`Gradient input ${i}`} value={v} onChange={value => setX(x.map((a, j) => j === i ? value : a))} min={-5} max={5} />)}<NeuralNumber label="Affine update rate" value={rate} onChange={setRate} min={0} max={.5} /></div>
    <figure><figcaption>One input, branching dependencies, summed backward contributions</figcaption><div className="normalization-graph-scroll" role="region" tabIndex={0} aria-label="Normalization dependency graph; scroll horizontally on narrow screens"><svg className="normalization-dependency-graph" viewBox="0 0 380 455" role="img" aria-label="Input feeds the shared mean and the numerator. Centered differences feed both the numerator and variance. Variance sets the denominator. Numerator and denominator join at normalized output, then affine output and loss. Reverse contributions add at each shared input.">
      <g fill="none" stroke="#8fc9b5" strokeWidth="2"><path d="M190 45 V65 H65 V95 M190 45 V165 M65 125 V180 H130 M250 195 H300 V245 M190 215 V300 M300 275 V325 H250 M190 345 V385" /><path d="M60 87 L65 95 L70 87 M185 157 L190 165 L195 157 M122 175 L130 180 L122 185 M295 237 L300 245 L305 237 M185 292 L190 300 L195 292 M258 320 L250 325 L258 330 M185 377 L190 385 L195 377" /></g>
      <g fill="#16231d" stroke="#607f6e"><rect x="130" y="10" width="120" height="35" rx="3" /><rect x="10" y="95" width="110" height="30" rx="3" /><rect x="130" y="165" width="120" height="50" rx="3" /><rect x="245" y="245" width="125" height="30" rx="3" /><rect x="130" y="300" width="120" height="45" rx="3" /><rect x="75" y="385" width="230" height="35" rx="3" /></g>
      <g fill="#e7ebe8" fontSize="14" textAnchor="middle"><text x="190" y="33">Input x</text><text x="65" y="115">Mean μ</text><text x="190" y="187">Differences</text><text x="190" y="206">d = x − μ</text><text x="307" y="265">v = mean(d²)</text><text x="190" y="319">x̂ = d / s</text><text x="190" y="337">s = √(v + ε)</text><text x="190" y="408">γx̂ + β → mean squared loss</text><text x="251" y="82">direct input</text><text x="340" y="212">variance</text><text x="340" y="230">path</text><text x="30" y="148">mean</text><text x="30" y="164">path</text><text x="185" y="449">Backward: add all three contributions</text></g>
    </svg></div><NeuralTable caption="Backward paths add at each input (before any parameter update)" headers={['Input', 'Direct uᵢ / s', 'Mean −mean(u) / s', 'Variance −x̂ᵢ mean(u x̂) / s', 'Sum = input derivative']} rows={x.map((_, i) => [i, fmt(upstream[i] / centered.denominator, 9), fmt(-average / centered.denominator, 9), fmt(-centered.output[i] * coupling / centered.denominator, 9), fmt(initial.inputGradient[i], 9)])} /><p>Here uᵢ = γᵢ ∂L/∂yᵢ, and s = √(v + ε). The direct contribution alone is incomplete. The mean and variance terms distribute the shared statistics’ effects across the group; their sum can nearly cancel the direct term. This uses population variance and the exact ε term.</p></figure>
    <NeuralTable caption="The shared-statistics derivative and affine update" headers={['Feature', 'Input derivative', 'γ derivative', 'β derivative', 'Before → after output']} rows={x.map((_, i) => [i, fmt(initial.inputGradient[i], 9), fmt(initial.scaleGradient[i], 9), fmt(initial.shiftGradient[i], 9), `${fmt(initial.output[i], 7)} → ${fmt(after.output[i], 7)}`])} />
    <p className="neural-result" data-result="gradients">Loss {fmt(initial.loss, 10)} → {fmt(after.loss, 10)}. {rate === 0 ? 'Zero learning rate leaves the affine parameters and outputs unchanged.' : 'The derivative includes changing the group’s mean and variance; it is not just division by a fixed denominator.'}</p><button onClick={() => {
      setX([1, 3]);
      setRate(.1);
    }}>Reset gradient trace</button>
  </NeuralLab>;
}
export function NormalizationMeasuredLab() {
  const [metric, setMetric] = useState('validation_loss'),
    [seed, setSeed] = useState(1);
  const runs = measurements.records.filter(run => run.seed === seed),
    isCount = metric === 'validation_correct';
  const maximum = isCount ? 120 : Math.max(...runs.flatMap(run => run.history.map(row => row[metric]))) * 1.08;
  return <NeuralLab title="Inspect what the matched runs actually measured" id="normalization-measured"><p>Recorded CPU experiment: 280 training specimens, 120 validation specimens, matched initial weights and batch order for each seed. No fitting occurs in this browser. These connected segments join recorded epochs only; intermediate epochs were not retained.</p><div className="neural-controls"><NeuralSelect label="Measured quantity" value={metric} onChange={setMetric} options={[['validation_loss', 'Validation cross-entropy'], ['train_evaluation_loss', 'Training-set CE in evaluation mode'], ['validation_correct', 'Correct validation specimens']]} /><NeuralSelect label="Recorded seed" value={seed} onChange={v => setSeed(Number(v))} options={[[1, 'Seed 1'], [2, 'Seed 2'], [3, 'Seed 3']]} /></div>
    <NeuralPlot title={`Four normalization choices, seed ${seed}`} xLabel="recorded epoch" yLabel={isCount ? 'correct / 120' : 'cross-entropy, nats'} xDomain={[0, 50]} yDomain={[0, maximum]} series={runs.map((run, i) => ({
      label: run.normalization,
      color: neuralColors[i],
      values: run.history.map(row => [row.epoch, row[metric]])
    }))} points={runs.flatMap((run, i) => run.history.map(row => ({
      x: row.epoch,
      y: row[metric],
      color: neuralColors[i],
      label: `${run.normalization}; epoch ${row.epoch}: ${fmt(row[metric])}`
    })))} />
    <NeuralTable caption="Exact observed values, no smoothed curve" headers={['Normalization', 'Epoch 0', 'Epoch 1', 'Epoch 5', 'Epoch 20', 'Epoch 50']} rows={runs.map(run => [run.normalization, ...run.history.map(row => fmt(row[metric], 6))])} />
    <p className="neural-result">Training-set CE was measured in evaluation mode, including BatchNorm. All four variants learned. These three seeds do not establish a universal ranking or a population confidence interval.</p><button onClick={() => {
      setSeed(1);
      setMetric('validation_loss');
    }}>Reset recorded comparison</button>
  </NeuralLab>;
}
function ResidualPathFigure({
  pre,
  multiplier,
  output
}) {
  const normPosition = pre ? 105 : 260;
  return <figure><figcaption>{pre ? 'Pre-norm: x + F(LN(x))' : 'Post-norm: LN(x + F(x))'}</figcaption><svg className="normalization-residual-graph" viewBox="0 0 300 310" role="img" aria-label={pre ? 'Input forks: one path runs through normalization and the linear branch, the identity path skips both. They add at the output.' : 'Input forks: the learned branch and identity path add, then normalization transforms their sum.'}>
    <g stroke="#8fc9b5" strokeWidth="2" fill="none"><path d="M150 38 V55 H60 V205 H150 V220 M150 55 H245 V230 H170 M150 242 V280" /><path d="M55 85 L60 93 L65 85 M240 210 L245 218 L250 210 M178 225 L170 230 L178 235 M145 272 L150 280 L155 272" /></g>
    <g fill="#14251d" stroke="#607f6e"><rect x="100" y="10" width="100" height="28" rx="3" /><rect x="12" y="145" width="100" height="32" rx="3" /><rect x={pre ? 12 : 100} y={normPosition - 12} width="100" height="28" rx="3" /><circle cx="150" cy="230" r="15" /></g>
    <g fill="#e7ebe8" fontSize="14" textAnchor="middle"><text x="150" y="30">Input [1, 3]</text><text x="62" y="166">F(u) = {fmt(multiplier)}u</text><text x={pre ? 62 : 150} y={normPosition + 7}>LayerNorm</text><text x="150" y="235">+</text><text x="190" y="110">identity</text><text x="190" y="129">x unchanged</text><text x="150" y="302">Output</text></g>
  </svg><p>Output [{output.map(v => fmt(v, 7)).join(', ')}]. {pre ? 'The identity path bypasses normalization.' : 'The final normalization also transforms the identity contribution.'}</p></figure>;
}
export function NormalizationPlacementLab() {
  const [branch, setBranch] = useState(0),
    x = [1, 3],
    normalized = normalizeVector(x).output;
  const pre = x.map((value, i) => value + branch * normalized[i]),
    post = normalizeVector(x.map(value => value + branch * value)).output;
  return <NeuralLab title="Move the normalization; change the function" id="normalization-placement"><p>Here F(u) = a × u. The input stays [1, 3]. At a = 0, the learned branch contributes nothing, but the two placements still return different outputs.</p><NeuralNumber label="Linear branch multiplier a" value={branch} onChange={setBranch} min={-1} max={1} /><div className="neural-two"><ResidualPathFigure pre multiplier={branch} output={pre} /><ResidualPathFigure pre={false} multiplier={branch} output={post} /></div><p className="neural-result">Pre-norm retains a direct identity derivative path. The total derivative also includes its branch; the identity path is not a guarantee against cancellation or amplification.</p><button onClick={() => setBranch(0)}>Reset zero branch</button></NeuralLab>;
}
