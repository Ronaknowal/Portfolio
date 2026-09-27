import { useId } from 'react';
import { NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { bowlGeometry } from '../../data/advanced-optimizer-models.js';
import { optimizerDigits } from '../../data/advanced-optimizer-study.js';
const colors = ['#e4b65a', '#9ebde4', '#b7a1d4', '#ddd'];
export function OptimizerFigure({ title, description, width = 640, height, children }) {
  const id = useId().replaceAll(':', '');
  return <figure className="optimizer-figure" tabIndex={0}><figcaption>{title}</figcaption><svg className="optimizer-diagram" style={{ minWidth: width }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={description}><defs><marker id={id} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#dcb764" /></marker></defs>{children(`url(#${id})`)}</svg><p>{description}</p></figure>;
}
const Box = ({ x, y, w = 125, text }) => <g><rect x={x} y={y} width={w} height="36" rx="3" fill="#242424" stroke="#aaa" /><text x={x + w / 2} y={y + 23} textAnchor="middle">{text}</text></g>;
const Arrow = ({ d, marker, color = '#dcb764', dashed = false }) => <path d={d} fill="none" stroke={color} strokeWidth="1.8" markerEnd={marker} strokeDasharray={dashed ? '5 4' : undefined} />;
export function OptimizerAnatomy() {
  return <OptimizerFigure title="A gradient changes parameters; parameters change the next prediction" height={300} description="One real digit's fixed pixels and chosen training label define a loss. The optimizer consumes the gradient and its own history, then writes parameters. The feedback loop changes the scores for the same image; it does not change the image's recorded label.">{marker => <>
    {optimizerDigits[277].pixels.map((p, i) => <rect key={i} x={25 + i % 8 * 8} y={10 + Math.floor(i / 8) * 8} width="8" height="8" fill={`rgb(${p / 16 * 255},${p / 16 * 255},${p / 16 * 255})`} />)}<text x="20" y="91">source 277</text><Arrow d="M94 42 H180" marker={marker} /><Box x={185} y={24} text="XΘ: ten scores" /><Arrow d="M310 42 H350" marker={marker} /><Box x={355} y={24} text="softmax P" /><Arrow d="M480 42 H510" marker={marker} /><Box x={515} y={24} w={110} text="loss −log P[y]" />
    <Box x={495} y={106} text="fixed target y" /><Arrow d="M558 106 V63" marker={marker} />
    <Arrow d="M625 42 H636 V181 H437" marker={marker} /><Box x={310} y={163} text="gradient Xᵀ(P−Y)" /><text x="536" y="164" textAnchor="middle">differentiate loss</text>
    <Arrow d="M310 181 H260" marker={marker} /><Box x={130} y={163} text="history → update" /><Arrow d="M130 181 H94 V239" marker={marker} /><Box x={15} y={241} w={180} text="new parameters Θ′" /><Arrow d="M195 259 H249 V64" marker={marker} />
    <text x="400" y="258" textAnchor="middle">next prediction uses the same image</text>
  </>}</OptimizerFigure>;
}
export function AdamHistoryFigure() {
  return <OptimizerFigure title="Two histories answer different questions" height={260} description="At the first step, signed gradients feed m while squared gradients feed v. The zero-initialization correction recovers g and g². Both coordinates have unit-magnitude first normalized direction when epsilon is ignored. That first-step property is not a general bound.">{marker => <>
    <Box x={20} y={95} text="g = [2, −4]" />
    <Arrow d="M145 112 H178 V47 H203 M178 112 V178 H203" marker={marker} />
    <Box x={208} y={29} w={175} text="m = .1g = [.2, −.4]" /><Box x={208} y={160} w={175} text="v = .001g² = [.004, .016]" />
    <Arrow d="M383 47 H425 M383 178 H425" marker={marker} /><Box x={430} y={29} w={185} text="m̂ = m/(1−.9) = [2,−4]" /><Box x={430} y={160} w={185} text="v̂ = v/(1−.999) = [4,16]" />
    <path d="M455 88 V139 M455 199 V218" stroke="#aaa" /><rect x="455" y="88" width="32" height="13" fill="#e4b65a" /><rect x="391" y="114" width="64" height="13" fill="#9ebde4" /><text x="494" y="99">+2</text><text x="355" y="125">−4</text><rect x="455" y="199" width="32" height="7" fill="#e4b65a" /><rect x="455" y="211" width="128" height="7" fill="#9ebde4" /><text x="590" y="219">16</text>
    <text x="235" y="100" textAnchor="middle">signed inputs may cancel</text><text x="235" y="126" textAnchor="middle">squared inputs cannot</text>
    <text x="320" y="238" textAnchor="middle">After 999 zero gradients, an impulse gives |m̂|/√v̂ ≈ 2.514567.</text>
  </>}</OptimizerFigure>;
}
export function OptimizerNumberLine({ title, values, arrows = [], caption, width = 500 }) {
  const numbers = values.map(v => v.value).concat(arrows.flatMap(a => [a.from, a.to]));
  let low = Math.min(0, ...numbers), high = Math.max(0, ...numbers);
  const padding = Math.max(.05, (high - low) * .15); low -= padding; high += padding;
  const x = v => 35 + (width - 70) * (v - low) / (high - low);
  return <OptimizerFigure title={title} width={width} height={110 + 42 * values.length} description={caption}>{marker => <>
    <path d={`M35 35 H${width - 35}`} stroke="#aaa" />
    {[low, 0, high].map(v => <g key={v}><path d={`M${x(v)} 30 V40`} stroke="#aaa" /><text x={x(v)} y="21" textAnchor="middle">{f(v, 3)}</text></g>)}
    {values.map((v, i) => <g key={v.label}><path d={`M${x(v.value)} 40 V${66 + i * 42}`} stroke={colors[i % 4]} strokeDasharray="3 3" /><circle cx={x(v.value)} cy={66 + i * 42} r="5" fill={colors[i % 4]} /><text x="35" y={85 + i * 42}>{v.label}: {f(v.value, 7)}</text></g>)}
    {arrows.map((a, i) => <Arrow key={i} d={`M${x(a.from)} ${57 + 42 * i} H${x(a.to)}`} marker={marker} color={colors[i % 4]} />)}
  </>}</OptimizerFigure>;
}
export function LionWorkedFigure() {
  return <OptimizerNumberLine title="History wins before the sign is taken" values={[{ label: 'β₁m', value: .18 }, { label: '(1−β₁)g', value: -.1 }, { label: 'blend; sign +1', value: .08 }, { label: 'new stored momentum: .198−.01', value: .188 }]} caption="The signed blend adds the .18 history contribution to the −.10 current contribution. Its positive sign selects a negative parameter step. The separate β₂ recurrence stores .188 for the next update." />;
}
export function CurvatureBowlFigure() {
  return <div className="optimizer-two">{[false, true].map(rotated => {
    const g = bowlGeometry(rotated), x = v => 175 + 42 * v, y = v => 165 - 42 * v;
    return <OptimizerFigure key={String(rotated)} title={rotated ? 'Rotated bowl: coordinates are coupled' : 'Axis-aligned bowl: diagonal is complete'} width={350} height={360} description="Calculated contours of ½θᵀHθ, curvatures 1 and 20; both panels use identical axes. Start (2,−1). Amber: gradient step η=.08; blue: diagonal Newton; white: full Newton to the optimum.">{marker => <>
      <path d="M20 165 H330 M175 20 V315" stroke="#777" />{[-3, -2, -1, 0, 1, 2, 3].map(v => <g key={v}><text x={x(v)} y="333" textAnchor="middle">{v}</text><text x="13" y={y(v) + 4}>{v}</text></g>)}
      {[.5, 2, 4, 6].map(level => <ellipse key={level} cx="175" cy="165" rx={42 * Math.sqrt(2 * level)} ry={42 * Math.sqrt(2 * level / 20)} transform={`rotate(${-g.angle * 180 / Math.PI} 175 165)`} fill="none" stroke="#555" />)}
      {[g.descent, g.diagonal, g.newton].map((point, i) => <g key={i}><Arrow d={`M${x(2)} ${y(-1)} L${x(point[0])} ${y(point[1])}`} marker={marker} color={colors[i === 2 ? 3 : i]} /><circle cx={x(point[0])} cy={y(point[1])} r="4" fill={colors[i === 2 ? 3 : i]}><title>{`${['Gradient', 'Diagonal Newton', 'Full Newton'][i]} [${point.map(v => f(v, 5)).join(', ')}]`}</title></circle></g>)}
      <circle cx={x(2)} cy={y(-1)} r="5" fill="#b7a1d4" /><text x="180" y="352">parameter θ₁ → ; θ₂ ↑</text>
    </>}</OptimizerFigure>;
  })}</div>;
}
export function CurvatureLanes() {
  return <OptimizerFigure title="The training gradient and curvature instrument use different labels" height={240} description="For x=2 and probability p=.8, the real label 1 gives gradient −.4, squared .16. An independently model-sampled label gives a probability-weighted squared gradient of .64. Its probabilities come from the same current model; they are not replacement training labels.">{marker => <>
    <Box x={15} y={80} w={130} text="current p = .8" /><Arrow d="M145 98 H177 V38 H208 M177 98 V158 H208" marker={marker} />
    <Box x={213} y={20} w={170} text="real label y=1" /><Arrow d="M383 38 H430" marker={marker} /><Box x={435} y={20} w={185} text="training g = 2(.8−1)" />
    <Box x={213} y={140} w={170} text="sample y ~ Bernoulli(.8)" /><Arrow d="M383 158 H420 V116 H450 M420 158 V197 H450" marker={marker} />
    <text x="454" y="121">y=1: .8 × (−.4)² = .128</text><text x="454" y="202">y=0: .2 × (1.6)² = .512</text><text x="330" y="230" textAnchor="middle">expected estimate .128 + .512 = .64 = x²p(1−p)</text>
  </>}</OptimizerFigure>;
}
export function ScheduleWeightsFigure() {
  const rates = [.1, .2, .3], weights = rates.map(r => r * r / .14);
  return <OptimizerFigure title="Rate, insertion coefficient and final weight are different" height={300} description="During this three-step monotone warmup the rates are .1, .2 and .3. Squared-rate weights yield final contributions 1/14, 4/14 and 9/14 to the averaged model. The insertion coefficients at their own steps are 1, 4/5 and 9/14; they are not each old point's final influence.">{() => <>
    <text x="20" y="27">rate η</text>{rates.map((r, i) => <g key={i}><rect x={165 + i * 145} y={105 - r * 220} width="60" height={r * 220} fill={colors[i]} /><text x={195 + i * 145} y="126" textAnchor="middle">step {i + 1}: {r}</text></g>)}
    <text x="20" y="173">final weights</text>{weights.map((w, i) => <g key={i}><rect x={145 + 470 * weights.slice(0, i).reduce((a, b) => a + b, 0)} y="150" width={470 * w} height="36" fill={colors[i]} /><text x={145 + 470 * (weights.slice(0, i).reduce((a, b) => a + b, 0) + w / 2)} y="208" textAnchor="middle">{[1, 4, 9][i]}/14</text></g>)}
    <text x="330" y="240" textAnchor="middle">the weights partition one complete average</text><text x="20" y="277">insertion c at its own step:</text>{[1, .8, 9 / 14].map((v, i) => <g key={i}><rect x={200 + 145 * i} y="260" width={90 * v} height="9" fill={colors[i]} /><text x={200 + 145 * i} y="291">{['1', '4/5', '9/14'][i]}</text></g>)}
  </>}</OptimizerFigure>;
}
export function OptimizerSelectionFigure() {
  return <OptimizerFigure title="Twenty-four fits, then one validation decision per method" height={265} description="Each method has two declared scales and two seeds, producing four candidates. Fitting data update weights. Validation chooses the scale by mean final loss across seeds. Assessment reports both selected seeds and does not feed the choice.">{marker => <>
    <Box x={15} y={25} w={165} text="240 fitting images" /><Arrow d="M180 43 H220" marker={marker} /><Box x={225} y={25} w={190} text="2 scales × 2 seeds" /><Arrow d="M415 43 H450" marker={marker} /><Box x={455} y={25} w={170} text="6 methods: 24 fits" />
    {[0, 1].flatMap(row => [0, 1].map(col => <g key={`${row}-${col}`}><rect x={245 + col * 82} y={89 + row * 55} width="73" height="45" fill="#222" stroke={colors[row]} /><text x={281 + col * 82} y={116 + row * 55} textAnchor="middle">scale{row + 1}/s{col + 1}</text></g>))}
    <Box x={15} y={110} w={165} text="80 validation images" /><Arrow d="M180 128 H237" marker={marker} /><Arrow d="M405 140 H450 V210" marker={marker} /><Box x={330} y={214} w={260} text="selected scale; report both seeds" />
    <Box x={15} y={214} w={220} text="80 assessment images" /><Arrow d="M235 232 H326" marker={marker} />
  </>}</OptimizerFigure>;
}
export function OptimizerPolarFigure() {
  const matrix = [[2, 1], [1, 2]], sign = [[1, 1], [1, 1]], polar = [[1, 0], [0, 1]];
  return <figure className="optimizer-figure"><figcaption>Entrywise signs and singular directions preserve different structure</figcaption><div className="optimizer-two"><NeuralTable caption="The three exact matrices" headers={['Matrix', 'Row 1', 'Row 2']} rows={[["M", ...matrix], ['sign(M)', ...sign], ['ideal polar UVᵀ', ...polar]].map(([name, ...rows]) => [name, ...rows.map(row => `[${row}]`)])} /><OptimizerFigure title="Action on two independent input directions" width={340} height={290} description="M stretches the diagonal directions by 3 and 1. The sign matrix doubles [1,1] and collapses [1,−1] to zero. The ideal polar factor is identity, preserving both vectors. This is not finite-iteration Muon output.">{marker => <>
    <path d="M20 150 H320 M170 15 V275" stroke="#777" />
    {[[[1, 1], '#e4b65a', false], [[1, -1], '#e4b65a', false], [[2, 2], '#9ebde4', true]].map(([v, color, dashed], i) => <Arrow key={i} d={`M170 150 L${170 + v[0] * 45} ${150 - v[1] * 45}`} marker={marker} color={color} dashed={dashed} />)}
    <circle cx="170" cy="150" r="5" fill="#9ebde4" /><text x="178" y="171">sign collapses [1,−1]</text><text x="35" y="267">solid: input = ideal polar output</text><text x="35" y="285">dashed / dot: entrywise sign output</text>
  </>}</OptimizerFigure></div></figure>;
}

export function CurvatureJacobianFigure() {
  return <OptimizerFigure title="The parameter Jacobian carries logit curvature back to weights" width={680} height={290} description="A perturbation in parameter space travels through J to logits, through softmax curvature C=diag(p)−ppᵀ, then back through Jᵀ. This yields JᵀCJ. The full Hessian also contains a logit-second-derivative term, which vanishes for a linear classifier.">{marker => <>
    <Box x={10} y={35} w={135} text="parameter δθ" /><Arrow d="M145 53 H185" marker={marker} /><Box x={190} y={35} w={120} text="Jδθ: logits" /><Arrow d="M310 53 H350" marker={marker} /><Box x={355} y={35} w={120} text="C(Jδθ)" /><Arrow d="M475 53 H515" marker={marker} /><Box x={520} y={35} w={145} text="JᵀCJδθ" />
    <text x="173" y="94" textAnchor="middle">J: classes × parameters</text><text x="480" y="94" textAnchor="middle">Jᵀ: parameters × classes</text>
    <text x="25" y="143">Binary example: logit f(θ)=θ², θ=1, y=1; p=σ(1)≈.731059</text><text x="25" y="177">J=2θ=2 → G=4p(1−p)≈.786448</text><path d="M345 174 H400 V216 H430" fill="none" stroke="#dcb764" markerEnd={marker} />
    <text x="25" y="215">extra term (p−y)f″=2(p−1)≈−.537883</text><path d="M405 211 H430" fill="none" stroke="#aaa" markerEnd={marker} /><Box x={435} y={198} w={225} text="full H ≈ .248565" />
    <text x="25" y="267">For f(θ)=xθ, f″=0: the extra path disappears and H=G.</text>
  </>}</OptimizerFigure>;
}
