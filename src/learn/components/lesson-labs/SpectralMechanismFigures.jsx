import { useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable } from './NeuralLessonElements.jsx';
import { SpectralFigure, SpectralPlot, f } from './SpectralPrimitives.jsx';
import { probePenalty } from '../../data/spectral-regularization-models.js';
const text = {
  fill: '#ddd',
  fontSize: 13
};
function Box({
  x,
  y,
  width = 120,
  label,
  color = '#e6b854'
}) {
  return <g><rect x={x} y={y} width={width} height="42" fill="#222" stroke={color} /><text x={x + width / 2} y={y + 26} textAnchor="middle" {...text}>{label}</text></g>;
}
export function CriticDerivativeFigure() {
  return <>
 <SpectralFigure title="Values flow forward in both phases; the derivative target changes" height={300} description="Amber arrows carry values. Purple arrows show the backward derivative. The critic phase stops its derivative at the generated input. The generator phase keeps the input derivative through a critic whose parameters are fixed.">
  <text x="15" y="22" {...text}>Critic update: optimize φ</text><Box x={15} y={44} label="Gθ(z)" /><Box x={243} y={44} label="fφ(fake)" /><Box x={486} y={44} label="LD=fake−real+R" width={140} />
  <path d="M135 65 H243 M363 65 H486" fill="none" stroke="#e6b854" /><polygon points="243,65 234,61 234,69" fill="#e6b854" /><polygon points="486,65 477,61 477,69" fill="#e6b854" />
  <path d="M486 95 H193 M363 95 V83" stroke="#ba9fd1" fill="none" /><polygon points="193,95 202,91 202,99" fill="#ba9fd1" /><path d="M186 84 V106 M196 84 V106" stroke="#eee" strokeWidth="3" /><text x="144" y="124" {...text}>detach here</text><text x="373" y="118" {...text}>∂LD/∂φ retained</text>
  <text x="15" y="163" {...text}>Generator update: optimize θ, hold φ fixed</text><Box x={15} y={185} label="z → Gθ(z)" /><Box x={243} y={185} label="fixed fφ" /><Box x={486} y={185} label="LG=−fφ(Gθ(z))" width={140} />
  <path d="M135 206 H243 M363 206 H486" stroke="#e6b854" /><polygon points="243,206 234,202 234,210" fill="#e6b854" /><polygon points="486,206 477,202 477,210" fill="#e6b854" />
  <path d="M556 237 H74 V227" stroke="#ba9fd1" fill="none" /><polygon points="74,227 70,236 78,236" fill="#ba9fd1" /><text x="310" y="268" textAnchor="middle" {...text}>−∂f/∂x · ∂G/∂θ reaches generator parameters</text>
 </SpectralFigure>
 <SpectralFigure title="A sign you can check on one number" height={125} description="Recorded point 0, generator θ=2, fixed f(x)=−x. LG=θ has derivative +1. Subtracting 0.1 times that derivative moves the output toward 0; the wrong loss sign moves it away."><line x1="70" y1="65" x2="580" y2="65" stroke="#777" />{[[0, 70, 'real 0'], [1.9, 465, 'correct 1.9'], [2, 486, 'start 2'], [2.1, 507, 'wrong 2.1']].map(([value, x, label], i) => <g key={value}><circle cx={x} cy="65" r="5" fill={i === 3 ? '#ba9fd1' : '#e6b854'} /><text x={x} y={i === 1 ? 101 : i === 3 ? 119 : 42} textAnchor="middle" {...text}>{label}</text></g>)}<path d="M486 73 L465 73 M486 56 L507 56" stroke="#ddd" /><text x="160" y="100" {...text}>score 0 → −2; loss 2 → 1.9</text></SpectralFigure>
 </>;
}
export function SpectralCompositionFigure() {
  return <>
 <SpectralFigure title="The direction of stretch matters when maps compose" height={220} description="Two independent input basis vectors follow separate lanes. The second matrix contracts exactly the coordinate the first stretched. Multiplying the individual worst-case factors ignores that directional mismatch.">
 {[['e₁', [1, 0], '[3, 0]', '[1, 0]'], ['e₂', [0, 1], '[0, 1/3]', '[0, 1]']].map(([label,, middle, last], i) => <g key={label}><Box x={18} y={42 + 83 * i} width={85} label={label} /><Box x={240} y={42 + 83 * i} width={130} label={middle} /><Box x={501} y={42 + 83 * i} width={120} label={last} /><path d={'M103 ' + (63 + 83 * i) + ' H240 M370 ' + (63 + 83 * i) + ' H501'} stroke="#ddd" /><text x="172" y={33 + 83 * i} textAnchor="middle" {...text}>A=diag(3,⅓)</text><text x="435" y={33 + 83 * i} textAnchor="middle" {...text}>B=diag(⅓,3)</text></g>)}<text x="320" y="215" textAnchor="middle" {...text}>Individual norms: 3 × 3 = 9 bound; composed BA = I has norm 1.</text>
 </SpectralFigure>
 <SpectralFigure title="A residual bypass contributes its own sensitivity" height={165} description="For g(x)=0.5x, the identity and branch derivatives add at the sum. The bound 1+0.5=1.5 is attained here."><Box x={15} y={58} width={80} label="x" /><Box x={260} y={100} label="g(x)=0.5x" /><path d="M95 79 H180 V25 H489 V62 M180 79 V121 H260 M380 121 H489 V96" stroke="#ddd" fill="none" /><text x="310" y="19" textAnchor="middle" {...text}>identity derivative 1</text><circle cx="489" cy="79" r="17" fill="#222" stroke="#e6b854" /><text x="489" y="85" textAnchor="middle" {...text}>+</text><path d="M506 79 H542" stroke="#ddd" /><text x="584" y="83" textAnchor="middle" {...text}>1.5x</text></SpectralFigure>
 </>;
}
export function PenaltyLocationsFigure() {
  const [epsilon, setEpsilon] = useState(.5);
  const location = 3 * (1 - epsilon),
    probe = probePenalty(2, 4, [location], 'target-one', 2);
  return <><NeuralNumber label="Interpolation coefficient ε" value={epsilon} min={0} max={1} onChange={setEpsilon} /><SpectralFigure title="The target and the sampling location are separate choices" height={245} description="Declared scalar example: real input 0, generated input 3. R1 evaluates zero-centered sensitivity at real inputs; R2 at generated inputs. WGAN-GP moves its probe between them.">
 {[['WGAN-GP: λ(r−1)²', true], ['R1: (γ/2)r² at real', false], ['R2: (γ/2)r² at fake', false]].map(([label], i) => <g key={label}><text x="12" y={35 + i * 73} {...text}>{label}</text><path d={'M270 ' + (31 + i * 73) + ' H588'} stroke="#777" /><circle cx="270" cy={31 + i * 73} r="7" fill={i !== 2 ? '#e6b854' : '#666'} /><rect x="580" y={23 + i * 73} width="16" height="16" fill={i !== 1 ? '#ba9fd1' : '#666'} />{i === 0 && <><circle cx={270 + 318 * (1 - epsilon)} cy="31" r="6" fill="#eee" /><text x="425" y="53" {...text}>ε real + (1−ε) fake</text></>}<text x="270" y={60 + i * 73} textAnchor="middle" {...text}>real</text><text x="588" y={60 + i * 73} textAnchor="middle" {...text}>fake</text></g>)}
 </SpectralFigure><p>Current interpolant ε×0+(1−ε)×3 = {f(location, 6)}. For the declared critic f(x)=x+4 ReLU(x−2), its derivative is {probe.rows[0].slope === null ? 'undefined at the corner' : f(probe.rows[0].slope)}, and target-one penalty with λ=2 is {probe.penalty === null ? 'undefined' : f(probe.penalty)}. At real input 0, R1 with γ=2 gives 1; at generated input 3, R2 with γ=2 gives 25. A change in target and a change in location are different operations.</p><button onClick={() => setEpsilon(.5)}>Reset interpolation</button></>;
}
export function BatchGradientFigure() {
  return <><SpectralFigure title="Summing batch scores can cancel the derivatives you intended to measure" height={225} description="Both input coordinates enter both scores. These crossed dependencies are the issue: the gradient of the total score is a column sum, not a table of the self-derivatives.">
 <Box x={25} y={35} width={95} label="x₁" /><Box x={25} y={145} width={95} label="x₂" /><Box x={310} y={35} width={150} label="f₁=(x₁−x₂)/2" /><Box x={310} y={145} width={150} label="f₂=(x₂−x₁)/2" />
 <path d="M120 56 H310 M120 166 H310" stroke="#e6b854" /><path d="M120 56 L310 166 M120 166 L310 56" stroke="#ba9fd1" /><text x="210" y="41" {...text}>+½</text><text x="210" y="190" {...text}>+½</text><text x="170" y="99" {...text}>−½</text><text x="249" y="129" {...text}>−½</text><path d="M460 56 L520 110 M460 166 L520 110" stroke="#ddd" /><text x="555" y="105" {...text}>sum 0</text><text x="490" y="135" {...text}>gradient [0,0]</text>
 </SpectralFigure><NeuralTable caption="The full batch Jacobian and its column sum" headers={['Output', '∂/∂x₁', '∂/∂x₂']} rows={[['f₁', .5, -.5], ['f₂', -.5, .5], ['sum', 0, 0]]} /></>;
}
export function SpectralGroupSortFigure() {
  const [a, setA] = useState(-.6),
    [b, setB] = useState(.9);
  const sorted = [a, b].sort((x, y) => x - y);
  return <>
 <div className="neural-controls"><NeuralNumber label="GroupSort input a" value={a} min={-2} max={2} onChange={setA} /><NeuralNumber label="GroupSort input b" value={b} min={-2} max={2} onChange={setB} /></div>
 <SpectralFigure title="Sorting switches routes; ReLU discards a negative coordinate" height={160} description="This pairwise construction is a local Euclidean example. The approximation theorem in the text uses its own specified norms and architecture.">{[a, b].map((value, i) => <g key={i}><Box x={20} y={25 + i * 72} width={125} label={'input ' + i + ': ' + f(value)} /><Box x={445} y={25 + i * 72} width={175} label={'sorted ' + i + ': ' + f(sorted[i])} /><path d={'M145 ' + (46 + i * 72) + ' L445 ' + (46 + (a <= b ? i : 1 - i) * 72)} stroke={i ? '#ba9fd1' : '#e6b854'} /></g>)}</SpectralFigure><p>Input length {f(Math.hypot(a, b))}; sorted length {f(Math.hypot(...sorted))}; ReLU length {f(Math.hypot(Math.max(0, a), Math.max(0, b)))}. At a tie the routing derivative is not uniquely described by one fixed permutation.</p>
 </>;
}
export function SpectralOdeFigure() {
  const [rate, setRate] = useState(.5);
  return <>
 <NeuralNumber label="Linear ODE rate a in h′=a h" value={rate} min={-1} max={1} onChange={setRate} /><SpectralPlot title="A Lipschitz growth bound need not be contraction" xLabel="time" yLabel="distance between trajectories" xDomain={[0, 3]} yDomain={[0, .1 * Math.exp(Math.abs(rate) * 3) * 1.06]} series={[[rate, 'Actual distance 0.1 exp(a t)', '#e6b854'], [Math.abs(rate), 'Bound 0.1 exp(|a| t)', '#ddd']].map(([value, label, color]) => ({
      label,
      color,
      values: Array.from({
        length: 61
      }, (_, i) => [i / 20, .1 * Math.exp(value * i / 20)]),
      dashed: label.startsWith('Bound')
    }))} /><p>The field has Lipschitz constant |a|={f(Math.abs(rate))}. Negative a contracts; positive a separates trajectories. The bound is valid for both and does not choose a solver step size.</p>
 </>;
}
