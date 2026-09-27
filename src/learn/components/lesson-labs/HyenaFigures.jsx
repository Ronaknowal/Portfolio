import { useState } from 'react';
import { NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { Figure, Node, Arrow, Matrix, Stems, Plot, DnaStrip, Probabilities, Values, f, vec } from './HyenaPrimitives.jsx';
import { directConvolution, fullConvolution, circularConvolution, toeplitz, gatedConvolution, gatePaths, modalFilter, modalScan, finiteTruncation } from '../../data/hyena-convolution-models.js';
import { hyenaExamples as examples } from '../../data/hyena-example-inputs.js';
const gold = '#e6bb60',
  blue = '#87bdf1',
  rose = '#e9979f';
export function DelayAddressFigure() {
  return <Figure title="H01 · Distance and content ask different questions" height={290} description="Illustrated tasks, not measured accuracy. A fixed delay retains one offset; a key request must locate the matching content after reordering."><text x="20" y="25">Fixed delay of three positions</text>{[4, 7, 2].map((v, i) => <g key={i}><Node x={20 + i * 85} y={42} width={60} lines={[String(v)]} /><Arrow x1={50 + i * 85} y1={86} x2={345 + i * 85} y2={130} /><Node x={315 + i * 85} y={130} width={60} lines={[String(v)]} /></g>)}<text x="20" y="211">A→7</text><text x="115" y="211">B→2</text><Node x={225} y={185} lines={['query B']} /><Arrow x1={145} y1={215} x2={225} y2={207} color={blue} /><Arrow x1={375} y1={207} x2={470} y2={207} color={blue} /><Node x={470} y={185} width={110} lines={['read 2']} /><text x="20" y="269">Swap the pair order: the matching key stays B, its distance changes.</text></Figure>;
}
export function EchoFigure() {
  const u = [1, 2, 3, 4],
    h = [1, .5, .25, .125];
  return <><Figure title="H02 · Each pulse contributes a delayed copy" height={320} description="Columns are output positions; each row is one input pulse’s contribution. Gold entries sum to y₂=4.25.">{Array.from({
        length: 7
      }, (_, t) => <text key={t} x={150 + t * 66} y="24" textAnchor="middle">t={t}</text>)}{u.map((v, j) => <g key={j}><text x="10" y={64 + j * 50}>u{j}={v}</text><Arrow x1={62} y1={60 + j * 50} x2={132 + j * 66} y2={60 + j * 50} />{h.map((a, r) => <g key={r}><rect x={128 + (j + r) * 66} y={40 + j * 50} width="45" height="32" fill={j + r === 2 ? '#59471f' : '#252525'} stroke="#555" /><text x={150 + (j + r) * 66} y={62 + j * 50} textAnchor="middle">{f(v * a)}</text></g>)}</g>)}<text x="10" y="289">At t=2: 1×.25 + 2×.5 + 3×1 = .25 + 1 + 3 = 4.25</text></Figure><Stems title="The complete causal output" values={directConvolution(u, h)} selected={2} /></>;
}
export function ToeplitzFigure() {
  return <><Matrix title="H03 · One lag occupies one lower diagonal" values={toeplitz([1, .5, .25, .125], 4)} selected={[3, 1]} causal /><p>Selected entry (receiver 3, sender 1) uses lag 3−1=2: h₂=.25. The same .25 occurs at (2,0). Crossed cells would read a future input and are zero.</p></>;
}
export function PaddingFigure() {
  const full = fullConvolution([1, 2, 3, 4], [1, .5, .25, .125]);
  return <><Figure title="H04 · A tail needs its own positions" height={315} description="An FFT length of four wraps output positions4–6 onto0–2. Padding to eight keeps the seven linear-convolution positions separate.">{Array.from({
        length: 8
      }, (_, t) => <g key={t}><rect x={10 + t * 76} y="45" width="67" height="40" fill={t < 4 ? '#303030' : t < 7 ? '#533137' : '#151515'} stroke="#666" /><text x={43 + t * 76} y="32" textAnchor="middle">{t}</text><text x={43 + t * 76} y="70" textAnchor="middle">{f(full[t] ?? 0, 4)}</text>{t >= 4 && t < 7 && <><path d={`M${43 + t * 76},87 Q${43 + (t - 2) * 76},170 ${43 + (t - 4) * 76},205`} stroke={rose} fill="none" /><Arrow x1={43 + (t - 4) * 76} y1={197} x2={43 + (t - 4) * 76} y2={210} color={rose} /></>}</g>)}{circularConvolution([1, 2, 3, 4], [1, .5, .25, .125]).map((v, t) => <Node key={t} x={10 + t * 76} y={213} width={67} lines={[f(v, 4)]} color={rose} />)}<text x="345" y="241">P=4: wrapped sums</text><text x="10" y="296">P=8: return positions0–3; keep tail4–6 separate; slot7 is zero.</text></Figure><p>Changing the last input from4 to8 changes the circular first output from4 to6. The correctly padded first three outputs remain [1,2.5,4.25].</p></>;
}
export function GateRailsFigure() {
  return <Figure title="H05 · One long filter between two gates" height={400} description="Three projected streams receive separate causal width-three preprocessing. The sending gate multiplies before the long convolution; the receiving gate multiplies after it."><Node x={230} y={10} width={180} lines={['input → projection']} />{['q: receiver', 'k: sender', 'v: value'].map((label, i) => <g key={label}><Arrow x1={320} y1={54} x2={100 + i * 215} y2={99} /><Node x={20 + i * 215} y={100} width={160} lines={[label, 'short causal conv']} height={60} /></g>)}<Arrow x1={315} y1={160} x2={390} y2={210} /><Arrow x1={530} y1={160} x2={435} y2={210} /><Node x={365} y={210} width={90} lines={['k × v']} /><Arrow x1={410} y1={254} x2={410} y2={281} /><Node x={315} y={282} width={190} lines={['long causal filter h']} /><path d="M100,160 L100,350 L315,350" stroke={blue} fill="none" /><Arrow x1={285} y1={350} x2={315} y2={350} color={blue} /><Arrow x1={410} y1={326} x2={410} y2={340} /><Node x={315} y={340} width={190} lines={['q × filtered values']} /></Figure>;
}
export function CoefficientFigure() {
  const u = [1, 2, 3, 4],
    h = [1, .5, .25, .125],
    q = [1, 2, -1, .5],
    k = [1, 0, -1, 2],
    r = gatedConvolution(u, h, q, k);
  return <><Matrix title="H06 · Conditional mixing coefficients qₜ hₜ₋ⱼ kⱼ" values={r.matrix} selected={[3, 2]} causal /><Figure title="A selected coefficient and its signed contribution" height={115}><Node x={10} y={20} width={130} lines={['q₃=.5']} /><Arrow x1={140} y1={42} x2={165} y2={42} /><Node x={165} y={20} width={135} lines={['h₁=.5']} /><Arrow x1={300} y1={42} x2={325} y2={42} /><Node x={325} y={20} width={130} lines={['k₂=−1']} /><text x="480" y="47">= −.25</text><text x="20" y="101">Coefficient × v₂ = −.25 × 3 = −.75; the row sums need not equal1.</text></Figure><Values caption="Worked gate calculation" rows={[["Transmitted k×v", vec(r.transmitted)], ["Filtered", vec(r.filtered)], ["Final q×filtered", vec(r.output)]]} /></>;
}
export function PathGeometry({
  t = 3,
  j = 0,
  paths
}) {
  return <Figure title={`Routes from source ${j} through m to receiver ${t}`} height={100 + paths.length * 64} description="Each complete route contributes the product of its edge/gate factors. Summing routes gives one conditional matrix entry.">{paths.map(({
      m,
      factors,
      coefficient
    }, i) => <g key={m}><text x="8" y={40 + i * 64}>j={j}</text><Arrow x1={48} y1={35 + i * 64} x2={215} y2={35 + i * 64} /><text x="128" y={23 + i * 64} textAnchor="middle">φ={f(factors[3])}</text><Node x={215} y={15 + i * 64} width={120} lines={[`m=${m}; k=${f(factors[2])}`]} /><Arrow x1={335} y1={35 + i * 64} x2={478} y2={35 + i * 64} /><text x="406" y={23 + i * 64} textAnchor="middle">h={f(factors[1])}</text><text x="485" y={40 + i * 64}>q={f(factors[0])}</text><text x="565" y={40 + i * 64}>{f(coefficient, 4)}</text></g>)}<text x="15" y={80 + paths.length * 64}>Sum = {f(paths.reduce((s, p) => s + p.coefficient, 0), 8)}. Future sender: no legal routes.</text></Figure>;
}
export function HierarchyFigure() {
  const paths = gatePaths([1, -.25, .5, 0], [1, -.5, 2, 1], [.5, 1, 0, -1], [1, -.5, 0, 0], 3, 1);
  return <><PathGeometry t={3} j={1} paths={paths} /><p>Two filters allow several intermediate gated paths. A one-filter sandwich has only the sender’s gate on each sender→receiver coefficient; these are different operators.</p></>;
}
export function CoordinateFigure() {
  const h = (r, denom) => Math.exp(-r / denom) * Math.cos(2 * Math.PI * r / denom),
    stable = Array.from({
      length: 8
    }, (_, r) => [r, h(r, 7)]),
    short = Array.from({
      length: 4
    }, (_, r) => [r, h(r, 3)]);
  return <><Plot title="H09 · Keep the lag ruler fixed when the prefix grows" xLabel="Lag r" yLabel="Illustrative coefficient" series={[{
      label: 'Fixed reference length8: r/7',
      values: stable
    }, {
      label: 'Rescaled length4: r/3',
      values: short
    }]} points={short.map(([x, y]) => ({
      x,
      y,
      label: 'Short-prefix coefficient'
    }))} /><Figure title="Training support and extension are separate contracts" height={100}><line x1="30" x2="420" y1="45" y2="45" stroke={gold} strokeWidth="3" /><line x1="420" x2="610" y1="45" y2="45" stroke={gold} strokeWidth="3" strokeDasharray="5 5" /><text x="30" y="27">0</text><text x="400" y="27">59</text><text x="50" y="85">Fitted support: fixed r/59</text><text x="435" y="85">Unmeasured extension</text></Figure></>;
}
export function GradientFigure() {
  return <><Figure title="H10 · One loss produces two coefficient updates" height={285} description="This scalar squared-error example is an exact calculation; the real sequence model differentiates through all filter and gate parameters."><Node x={20} y={10} width={170} lines={['u=[1,2]', 'h=[.5,.25]']} height={65} /><Arrow x1={190} y1={42} x2={240} y2={42} /><Node x={240} y={10} width={150} lines={['y=.5×2+.25×1', 'y=1.25']} height={65} /><Arrow x1={390} y1={42} x2={440} y2={42} /><Node x={440} y={10} width={175} lines={['target2; error−.75', 'loss=.28125']} height={65} /><path d="M525,75 L525,115 L105,115 L105,145" fill="none" stroke={rose} /><Arrow x1={105} y1={135} x2={105} y2={145} color={rose} /><Node x={20} y={145} width={220} lines={['∂L/∂h = error × [2,1]', '[−1.5,−.75]']} height={65} /><Arrow x1={240} y1={177} x2={295} y2={177} /><Node x={295} y={145} width={305} lines={['h ← h − .1 gradient = [.65,.325]', 'new y=1.625; new loss=.0703125']} height={65} /><text x="20" y="263">The .1 is a learning rate. It is neither a gate nor a filter coefficient.</text></Figure></>;
}
export function SequenceBlockFigure() {
  return <><Figure title="The actual two-block classifier’s residual paths" height={370} description="Tokens embed into width 16. Each of two pre-normalized blocks has a gated long-filter branch plus a pointwise GELU feed-forward branch. The head reads position 59 after both blocks."><Node x={15} y={15} width={145} lines={['60 symbols', 'embedding:16']} height={60} /><Arrow x1={160} y1={45} x2={210} y2={45} /><Node x={210} y={15} width={220} lines={['LayerNorm → gate/filter', '→ output projection']} height={60} /><path d="M180,45 L180,105 L470,105 L470,57" fill="none" stroke={blue} /><Arrow x1={430} y1={45} x2={460} y2={45} /><circle cx="470" cy="45" r="12" fill="#222" stroke={blue} /><text x="470" y="50" textAnchor="middle">+</text><path d="M560,45 L560,130 L470,130" fill="none" stroke="#e6bb60" /><circle cx="560" cy="130" r="3" fill="#e6bb60" /><Arrow x1={470} y1={130} x2={470} y2={165} /><Node x={290} y={165} width={235} lines={['LayerNorm → Linear16→32', 'exact GELU → Linear32→16']} height={65} /><path d="M560,45 L560,260 L482,260" fill="none" stroke={blue} /><path d="M482,45 L560,45" fill="none" stroke={blue} /><Arrow x1={470} y1={230} x2={470} y2={248} /><circle cx="470" cy="260" r="12" fill="#222" stroke={blue} /><text x="470" y="265" textAnchor="middle">+</text><Arrow x1={470} y1={272} x2={470} y2={300} /><Node x={300} y={301} width={290} lines={['repeat block → final position → 3 logits']} /><text x="15" y="172">The long-filter branch adds</text><text x="15" y="195">a learned current-position</text><text x="15" y="218">skip before multiplying q.</text></Figure><Figure title="A causal model’s training and generation targets" height={190}><text x="15" y="25">Teacher forcing: known prefix</text>{['A', 'C', 'G', 'T'].map((v, i) => <g key={i}><Node x={20 + i * 110} y={45} width={75} lines={[v]} />{i < 3 && <><Arrow x1={57 + i * 110} y1={89} x2={167 + i * 110} y2={130} /><text x={167 + i * 110} y="154" textAnchor="middle">target {['C', 'G', 'T'][i]}</text></>}</g>)}<text x="15" y="182">Generation feeds the sampled next symbol back into the next prefix.</text></Figure></>;
}
export function DnaWindowFigure() {
  return <><DnaStrip title="H11 · An observed window, source row3" sequence={examples.workedSequence.sequence} observedLabel="EI" /><Figure title="Boundary direction and ambiguity codes" height={225}><Node x={20} y={15} lines={['exon']} /><Arrow x1={170} y1={37} x2={210} y2={37} /><Node x={210} y={15} lines={['intron']} /><text x="400" y="42">EI: exon → intron</text><Node x={20} y={82} lines={['intron']} /><Arrow x1={170} y1={104} x2={210} y2={104} /><Node x={210} y={82} lines={['exon']} /><text x="400" y="109">IE: intron → exon</text><text x="20" y="163">N label: neither boundary. N input symbol: A/C/G/T ambiguity.</text><text x="20" y="200">D: A/G/T    R: A/G    S: C/G. These are symbols, not extra bases.</text></Figure></>;
}
export function DataRolesFigure() {
  return <Figure title="H12 · Split connected groups before fitting" height={380} description="The diagram illustrates the declared grouping rule. Duplicate edges join source-prefix groups; each resulting component stays entirely in one role. Counts are the actual retained split."><Node x={20} y={25} width={145} lines={['source prefix A']} /><Node x={235} y={25} width={145} lines={['source prefix B']} /><Node x={20} y={115} width={145} lines={['window a']} /><Node x={235} y={115} width={145} lines={['same sequence a']} /><Arrow x1={92} y1={69} x2={92} y2={115} /><Arrow x1={307} y1={69} x2={307} y2={115} /><line x1="165" y1="137" x2="235" y2="137" stroke={blue} strokeWidth="3" /><text x="180" y="113">join</text><rect x="5" y="10" width="390" height="170" fill="none" stroke={blue} strokeDasharray="5 4" /><Arrow x1={200} y1={180} x2={110} y2={235} color={blue} />{[['fit', '2128 rows / 1008 groups'], ['validation', '460 rows / 216 groups'], ['assessment', '416 rows / 216 groups']].map(([a, b], i) => <Node key={a} x={10 + i * 210} y={235} width={200} height={65} lines={[a, b]} />)}<Node x={430} y={25} width={195} height={90} lines={['conflicting labels', 'source 1022 +1969', 'excluded together']} color={rose} /><text x="15" y="337">3190 rows →3005 distinct sequences →3004 retained after conflict exclusion.</text><text x="15" y="363">Observed groups do not establish patient or homology independence.</text></Figure>;
}
export function LearningCurvesFigure() {
  const fits = examples.fits;
  return <><Plot title="H13 · All80 measured validation epochs, one common axis" xLabel="Epoch" yLabel="Validation cross-entropy" series={fits.map(row => ({
      label: row.kind + ' seed' + row.seed,
      values: row.history.map(h => [h[0], h[1]])
    }))} points={fits.map((row, i) => ({
      x: row.selected_epoch,
      y: row.history[row.selected_epoch - 1][1],
      label: 'Selected ' + row.kind + row.seed,
      color: [gold, blue, rose, '#bda0df'][i]
    }))} /><p>Dots mark each fit’s minimum validation cross-entropy. Assessment is used only after selection; its exact final scores are in the lesson’s separate table.</p></>;
}
export function CounterfactualFigure() {
  const r = examples.workedSequence;
  return <><DnaStrip title="H14 · Original source 3, observed EI" sequence={r.sequence} observedLabel="EI" /><DnaStrip title="Synthetic edit: array 30–31 GT→AA" sequence={r.edited_sequence} reference={r.sequence} /><Probabilities title="Fixed gated seed29: current edit versus original" current={r.after_probabilities} reference={r.before_probabilities} /><NeuralTable caption="Saved native counterfactual values" headers={['Class', 'Original logit', 'Edited logit', 'Original probability', 'Edited probability']} rows={['EI', 'IE', 'N'].map((c, i) => [c, f(r.before_logits[i], 8), f(r.after_logits[i], 8), f(r.before_probabilities[i], 8), f(r.after_probabilities[i], 8)])} /></>;
}
export function OverlapFigure() {
  const h = [1, .5, .25, .125],
    a = fullConvolution([1, 2], h),
    b = fullConvolution([3, 4], h),
    sum = fullConvolution([1, 2, 3, 4], h);
  return <Figure title="H15 · Keep and add each block’s tail" height={300} description="Blocks begin at their original global positions. Gold and blue contributions overlap at positions2–4; their sum is full linear convolution, including its tail.">{Array.from({
      length: 7
    }, (_, i) => <text key={i} x={140 + i * 70} y="25" textAnchor="middle">{i}</text>)}{[[a, 0, 'block 0', gold], [b, 2, 'block 1', blue], [sum, 0, 'sum', '#ddd']].map(([row, start, label, color], r) => <g key={label}><text x="10" y={76 + r * 65}>{label}</text>{row.map((v, j) => <g key={j}><rect x={110 + (start + j) * 70} y={48 + r * 65} width="61" height="43" fill={r < 2 && start + j >= 2 && start + j <= 4 ? '#423824' : '#222'} stroke={color} /><text x={140 + (start + j) * 70} y={76 + r * 65} textAnchor="middle">{f(v, 4)}</text></g>)}</g>)}<text x="10" y="277">Reset/crop mistake at t=2:3 instead of 4.25; the earlier tail was discarded.</text></Figure>;
}
export function ModalRegisters({
  row,
  residues,
  poles,
  title = 'The current registers'
}) {
  return <Figure title={title} height={80 + residues.length * 90} description="Each register multiplies its previous state by its pole, adds the same current input, then contributes residue×state to the read.">{residues.map((R, i) => <g key={i}><Node x={10} y={15 + i * 90} width={130} lines={[`s${i}=${f(row.before[i], 4)}`, `× λ=${f(poles[i])}`]} height={60} /><Arrow x1={140} y1={45 + i * 90} x2={185} y2={45 + i * 90} /><Node x={185} y={15 + i * 90} width={170} lines={[`+ input ${f(row.input)}`, `new s=${f(row.state[i], 5)}`]} height={60} /><Arrow x1={355} y1={45 + i * 90} x2={405} y2={45 + i * 90} /><Node x={405} y={15 + i * 90} width={220} lines={[`× R=${f(R)}`, `contribution=${f(row.contributions[i], 6)}`]} height={60} /></g>)}<text x="15" y={60 + residues.length * 90}>Sum of register contributions: y={f(row.output, 9)}</text></Figure>;
}
export function ModesFigure() {
  const R = [.6, .4],
    p = [.5, -.25],
    u = [1, -2, .5, 3, -1, 2];
  return <><Plot title="H16 · Positive and alternating modes add into one filter" xLabel="Lag" yLabel="Coefficient" series={[...R.map((v, i) => ({
      label: `mode${i}: ${v} × ${p[i]}ʳ`,
      values: Array.from({
        length: 6
      }, (_, r) => [r, v * p[i] ** r])
    })), {
      label: 'sum: h',
      values: modalFilter(R, p, 6).map((v, r) => [r, v])
    }]} /><ModalRegisters title="Worked t=2: the state to carry into the second chunk" row={modalScan(u, R, p)[2]} residues={R} poles={p} /></>;
}
export function ApproximationFigure() {
  const u = [1, -2, .5, 3, -1, 2],
    h = modalFilter([.6, .4], [.5, -.25], 6),
    r = finiteTruncation(u, h, 2),
    h11 = modalFilter([.6, .4], [.5, -.25], 11);
  return <><Stems title="H17 · Retain lags0–1; omit the remaining finite mass" values={r.approximation} reference={h} referenceLabel="Original six-coefficient filter" /><Plot title="Observed error and the finite six-position bound" xLabel="Output position" yLabel="Absolute output difference" yDomain={[0, 1]} series={[{
      label: 'Actual error',
      values: r.errors.map((v, t) => [t, v])
    }, {
      label: '‖u‖∞ × omitted mass',
      values: r.errors.map((_, t) => [t, r.bound]),
      dashed: true
    }]} /><Values caption="The finite calculation" rows={[["Omitted absolute mass", f(r.omitted, 9)], ["Input maximum magnitude", 3], ["Bound", f(r.bound, 9)], ["Observed maximum", f(Math.max(...r.errors), 9)]]} /><Matrix title="A different structure: Hankel entries hᵢ₊ⱼ" values={Array.from({
      length: 6
    }, (_, i) => Array.from({
      length: 6
    }, (_, j) => h11[i + j]))} rowLabel="i" columnLabel="j" /><Figure title="Two outer products account for the exact analytic Hankel matrix" height={190} description="For each mode, column [1,λ,…,λ⁵] times its row transpose is rank one. The residues scale and add these matrices.">{[.5, -.25].map((p, i) => <g key={p}><Node x={15 + i * 315} y={15} width={280} lines={[`[1, ${p}, ${p * p}, …]ᵀ`, `× [1, ${p}, ${p * p}, …]`, `× residue ${[.6, .4][i]}`]} height={95} /></g>)}<text x="15" y="142">Sum: rank≤2. Measured singular values:1.08698, .13949, then&lt;10⁻¹⁶.</text><text x="15" y="176">This known analytic filter does not certify every learned neural filter.</text></Figure></>;
}
export function MechanismMapFigure() {
  return <Figure title="H18 · Different operators retain different objects" height={435} description="The paths compare mechanisms, not measured quality or speed. A fixed-gate conditional matrix is distinct from the entire input Jacobian.">{[['Lag-only convolution', 'uⱼ → hₜ₋ⱼ → yₜ', 'history / overlap tail'], ['Gate–filter–gate', 'vⱼ → kⱼ → hₜ₋ⱼ → qₜ', 'short history + filter state'], ['Selective state update', 'uₜ + input-dependent state transition', 'current selective state'], ['Normalized attention', 'qₜ · kⱼ → softmax over j → vⱼ', 'retained keys and values']].map(([name, op, state], i) => <g key={name}><text x="15" y={24 + i * 100}>{name}</text><Node x={15} y={38 + i * 100} width={350} lines={[op]} /><Arrow x1={365} y1={60 + i * 100} x2={395} y2={60 + i * 100} /><Node x={395} y={38 + i * 100} width={230} lines={[state]} /></g>)}</Figure>;
}
export function StripedFigure() {
  return <><Figure title="H19 · Different layer ranges own different decode memory" height={315} description="Conceptual operator strip, not an exact official layer schedule. StripedHyena2 separates short explicit, medium regularized and long implicit filters, with attention in a hybrid.">{[['SE: short', 'explicit few taps', 'short buffer'], ['MR: medium', 'windowed filter', 'overlap/history'], ['LI: long', 'exponential modes', 'modal registers'], ['Attention', 'q/k/v + softmax', 'KV records']].map(([name, filter, state], i) => <g key={name}><Node x={10 + i * 157} y={20} width={145} lines={[name]} />{i < 3 && <Arrow x1={155 + i * 157} y1={42} x2={167 + i * 157} y2={42} />}<Arrow x1={82 + i * 157} y1={64} x2={82 + i * 157} y2={102} /><Node x={10 + i * 157} y={102} width={145} height={65} lines={filter.split(' ').length > 2 ? [filter.split(' ').slice(0, 1).join(' '), filter.split(' ').slice(1).join(' ')] : [filter]} /><Arrow x1={82 + i * 157} y1={167} x2={82 + i * 157} y2={210} /><Node x={10 + i * 157} y={210} width={145} lines={[state]} color={blue} /></g>)}<text x="15" y="294">Attention’s KV cache does not carry the convolution layer’s state.</text></Figure><Figure title="Genomic pretraining and downstream adaptation have different inputs" height={260}><Node x={15} y={15} width={215} lines={['single DNA symbols', 'causal next-symbol targets']} height={65} /><Arrow x1={230} y1={47} x2={275} y2={47} /><Node x={275} y={15} width={345} lines={['HyenaDNA pretrained representation', 'downstream task head / adaptation']} height={65} /><Node x={15} y={130} width={215} lines={['learned prompt vectors', 'not A/C/G/T symbols']} height={65} /><Arrow x1={230} y1={162} x2={275} y2={162} /><Node x={275} y={130} width={345} lines={['prompt prefix + embedded DNA', 'separate from the scratch fit here']} height={65} /><text x="15" y="237">A bidirectional task variant changes the available context; it is not a causal prefix.</text></Figure></>;
}
export function BenchmarkFigure() {
  const [length, setLength] = useState(4096),
    [width, setWidth] = useState(256),
    [batch, setBatch] = useState(1),
    [bytes, setBytes] = useState(2),
    [notes, setNotes] = useState({
      device: '',
      version: '',
      scope: 'forward',
      warmup: ''
    }),
    L = BigInt(length),
    D = BigInt(width),
    B = BigInt(batch);
  return <><Figure title="H20 · Turn an asymptotic claim into a specified experiment" height={185}><Node x={10} y={20} width={180} height={95} lines={['mixing work', 'DL log₂L', 'projection LD²']} /><Arrow x1={190} y1={67} x2={220} y2={67} /><Node x={220} y={20} width={190} height={95} lines={['tensor inventory', 'activations / spectra', 'weights / state']} /><Arrow x1={410} y1={67} x2={440} y2={67} /><Node x={440} y={20} width={190} height={95} lines={['workload + measurement', 'device / precision', 'scope / warm-up']} /><text x="10" y="160">Exact attention need not materialize an L×L score tensor.</text></Figure><div className="neural-controls"><NeuralNumber label="Workload sequence length" value={length} min={1} max={1048576} integer range={false} onChange={setLength} /><NeuralNumber label="Model width" value={width} min={1} max={8192} integer range={false} onChange={setWidth} /><NeuralNumber label="Batch size" value={batch} min={1} max={256} integer range={false} onChange={setBatch} /><NeuralNumber label="Bytes per stored scalar" value={bytes} min={1} max={8} integer range={false} onChange={setBytes} /></div><Values caption="Exact simple tensor counts; these are not full peak memory" rows={[["One B×L×D activation, bytes", (B * L * D * BigInt(bytes)).toString()], ["If explicitly allocated: one B×L×L array, bytes", (B * L * L * BigInt(bytes)).toString()], ["Mixing work scale BDL log₂L", f(batch * width * length * Math.log2(length), 0)], ["Projection work scale BLD²", (B * L * D * D).toString()], ["Attention work scale BL²D", (B * L * L * D).toString()]]} /><div className="neural-controls">{Object.entries(notes).map(([key, value]) => <label className="hy-select" key={key}>{key}<input value={value} onChange={e => setNotes({
          ...notes,
          [key]: e.target.value
        })} /></label>)}</div><p>Timing: unmeasured. Record implementation/version, precision, device, forward/backward or prefill/decode scope and warm-up before measuring. The counts above do not predict a winner.</p></>;
}
