import { useState } from 'react';
import { NeuralLab, NeuralSelect, NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { rbmDefault, tinyDistribution, hiddenProbabilities, statistics, updateTiny, probabilityFlow, particleTrace, tinyDraw, persistenceTrace, rbmMetrics, binaryStates } from '../../data/rbm-models.js';
import { RbmParameters, SwitchGraph, ProbabilityBars, RbmFigure, RbmStateTable, stateName, vector } from './RbmElements.jsx';
import libraryModel from '../../data/rbm-library-model.js';
import './rbm-labs.css';
import './neural-lesson-neutral.css';
export { RbmProgram } from './RbmElements.jsx';
export { RbmDigitLab, RbmSampleGallery, RbmStudyFigure } from './RbmDigitLabs.jsx';
export { RbmFamilyFigure, RbmAisFigure } from './RbmStructureFigures.jsx';
export function RbmGeneralFigure() {
  return <div className="rbm-two"><SwitchGraph model={rbmDefault()} visible={[1, 1]} hidden={0} general /><div><ProbabilityBars title="Direct visible interaction: masses 1, 1, 1, 3" rows={['00', '01', '10', '11'].map((label, i) => ({
        label,
        current: i === 3 ? .5 : 1 / 6
      }))} /><p>Each visible switch is on with probability 2/3, but both are on with probability 1/2. A direct interaction already creates dependence. The next graph removes that direct edge and uses a hidden switch.</p></div></div>;
}
export function RbmEnergyLab() {
  const [model, setModel] = useState(rbmDefault),
    [v, setV] = useState([1, 1]),
    [h, setH] = useState(1),
    [offset, setOffset] = useState(0);
  const dist = tinyDistribution(model, offset),
    base = tinyDistribution(rbmDefault()),
    index = v[0] * 2 + v[1];
  const terms = [-model.a[0] * v[0], -model.a[1] * v[1], -model.b[0] * h, -model.w[0][0] * v[0] * h, -model.w[1][0] * v[1] * h, offset];
  const event = dist.joint[index * 2 + h];
  return <NeuralLab id="rbm-energy" title="Move energy; watch the whole probability budget respond">
    <p>Two binary visible variables and one binary hidden variable. Current energy, every joint probability and all visible marginals are available before any edit.</p>
    <RbmParameters model={model} onChange={setModel} /><NeuralNumber label="Common energy offset" value={offset} min={-100} max={100} onChange={setOffset} />
    <div className="neural-buttons">{[0, 1].map(i => <button key={i} aria-pressed={Boolean(v[i])} onClick={() => setV(old => old.map((x, j) => i === j ? 1 - x : x))}>Toggle visible {i + 1}: {v[i]}</button>)}<button aria-pressed={Boolean(h)} onClick={() => setH(x => 1 - x)}>Toggle inspected hidden state: {h}</button></div>
    <div className="rbm-two"><SwitchGraph model={model} visible={v} hidden={h} /><NeuralTable caption="Selected event's dimensionless energy ledger" headers={['Term', 'Contribution']} rows={['−a₁v₁', '−a₂v₂', '−bh', '−W₁v₁h', '−W₂v₂h', 'common offset'].map((label, i) => [label, f(terms[i], 6)]).concat([['Total E(v,h)', f(event.energy, 6)], ['exp(−E)', Math.exp(-event.energy).toExponential(6)], ['Normalized p(v,h)', f(event.probability, 8)]])} /></div>
    <ProbabilityBars title="Add hidden alternatives, then normalize" rows={dist.states.map((state, i) => ({
      label: stateName(state),
      current: dist.p[i],
      baseline: base.p[i]
    }))} />
    <NeuralTable caption="Hidden sum and normalized visible state" headers={['Visible', 'Mass h=0', 'Mass h=1', 'Sum', 'p(v)']} rows={dist.states.map((state, i) => {
      const masses = dist.joint.slice(i * 2, i * 2 + 2).map(row => Math.exp(-row.energy));
      return [stateName(state), ...masses.map(x => x.toExponential(4)), masses.reduce((a, b) => a + b, 0).toExponential(4), f(dist.p[i], 8)];
    })} />
    <p>For visible {stateName(v)}, p(h=1 | v)={f(hiddenProbabilities(v, model)[0], 6)}. Marginals p(v₁=1)={f(dist.marginals[0], 6)}, p(v₂=1)={f(dist.marginals[1], 6)}. Covariance p(11)−p(v₁=1)p(v₂=1)={f(dist.covariance, 8)}. log Z={f(dist.logZ, 6)}.</p>
    <p>Selecting h changes the inspected event while p(v) adds both alternatives. A common energy offset changes absolute masses and log Z together, so the normalized bars stay fixed. To construct independent fair visible variables, inspect both covariance and the two marginal residuals: {vector([dist.covariance, dist.marginals[0] - .5, dist.marginals[1] - .5])}.</p>
    <div className="neural-buttons"><button onClick={() => setModel(rbmDefault())}>Restore original parameters</button><button onClick={() => setModel({
        w: [[0], [0]],
        a: [0, 0],
        b: [0]
      })}>Zero-interaction comparison</button><button onClick={() => {
        setModel(rbmDefault());
        setV([1, 1]);
        setH(1);
        setOffset(0);
      }}>Reset energy investigation</button></div>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect all eight normalized joint events</h4><RbmStateTable distribution={dist} /></section>
  </NeuralLab>;
}
export function RbmEnumerationFigure() {
  const [visible, setVisible] = useState(64),
    [hidden, setHidden] = useState(8);
  return <NeuralLab id="rbm-enumeration" title="Sum one layer analytically; enumerate the smaller layer">
    <div className="neural-controls"><NeuralNumber label="Visible binary units D" value={visible} min={1} max={100} integer onChange={setVisible} /><NeuralNumber label="Hidden binary units H" value={hidden} min={1} max={40} integer onChange={setHidden} /></div>
    <RbmFigure title="One hidden configuration stands for every visible alternative" width={520} height={200} description="For fixed h, visible choices factor. Multiply the two-state sums for each pixel, then add over hidden configurations. The controls calculate counts only; they never trigger these large enumerations.">
      <text x="20" y="28" fill="#eee" fontSize="14">one h →</text>{[0, 1, 2].map(i => <g key={i}><rect x={105 + i * 125} y="45" width="105" height="64" fill="#222" stroke="#e6b854" /><text x={157 + i * 125} y="70" textAnchor="middle" fill="#eee" fontSize="13">v{i + 1}: 0 or 1</text><text x={157 + i * 125} y="94" textAnchor="middle" fill="#eee" fontSize="13">1 + exp(logit)</text></g>)}<text x="225" y="86" fill="#eee">×</text><text x="350" y="86" fill="#eee">×</text><text x="500" y="86" textAnchor="middle" fill="#eee">…</text><path d="M105 125 V143 H460 V125" stroke="#aaa" fill="none" /><text x="280" y="175" textAnchor="middle" fill="#eee" fontSize="14">multiply D local sums; add the hidden-state masses</text>
    </RbmFigure>
    <p>Visible enumeration: 2^{visible} = {(2n ** BigInt(visible)).toLocaleString('en-US')} states. Hidden enumeration: 2^{hidden} = {(2n ** BigInt(hidden)).toLocaleString('en-US')} states. The smaller route enumerates <strong>{Math.min(visible, hidden)} bits</strong>. Each retained digit model actually enumerates only 256 hidden states.</p>
  </NeuralLab>;
}
export function RbmGradientLab() {
  const [model, setModel] = useState(rbmDefault),
    [counts, setCounts] = useState([0, 0, 0, 1]),
    [rate, setRate] = useState(.1);
  const valid = counts.some(x => x > 0),
    result = valid ? statistics(model, counts) : null;
  const updated = result ? updateTiny(model, result.gradient, rate) : model,
    newResult = result ? statistics(updated, counts) : null;
  return <NeuralLab id="rbm-gradient" title="Data and model co-occurrences compete for probability">
    <RbmParameters model={model} onChange={setModel} /><fieldset><legend>Observed four-state counts, normalized explicitly by their sum</legend><div className="neural-controls">{counts.map((value, i) => <NeuralNumber key={i} label={'Count / mass for ' + stateName(binaryStates(2)[i])} value={value} min={0} max={100} range={false} onChange={x => setCounts(old => old.map((c, j) => i === j ? x : c))} />)}</div></fieldset>
    <NeuralNumber label="Simultaneous ascent step size" value={rate} min={0} max={.5} step={.01} onChange={setRate} />
    {result ? <><NeuralTable caption="Both phases use the same current model before updating" headers={['Parameter', 'Data expectation', 'Model expectation', 'Difference', 'Current', 'Proposed']} rows={['W₁', 'W₂', 'a₁', 'a₂', 'b'].map((name, i) => [name, f(result.positive[i], 6), f(result.negative[i], 6), f(result.gradient[i], 6), f([...model.w.flat(), ...model.a, ...model.b][i]), f([...updated.w.flat(), ...updated.a, ...updated.b][i])])} />
      <RbmFigure title="An edge update is the signed difference of two expectations" width={460} height={145} description="The two bar lengths use a common 0–1 scale. Increasing a weight rewards a co-occurrence only when the data statistic exceeds the model's current statistic.">
        {['data v₁h', 'model v₁h'].map((label, i) => <g key={label}><text x="10" y={40 + i * 48} fill="#eee" fontSize="13">{label}</text><rect x="110" y={23 + i * 48} width={(i ? result.negative[0] : result.positive[0]) * 300} height="22" fill={i ? '#aaa' : '#e6b854'} /></g>)}<text x="110" y="135" fill="#eee" fontSize="13">difference = {f(result.gradient[0], 6)}</text>
      </RbmFigure><p>Mean data log likelihood: {f(result.logLikelihood, 8)} → {f(newResult.logLikelihood, 8)} nats. All five proposed parameters use the old statistics together. Step size zero keeps parameters fixed while count edits still change the data phase.</p></> : <p role="status">Counts have zero total. Enter a positive count to define an empirical distribution; no likelihood or update is defined.</p>}
    <div className="neural-buttons"><button onClick={() => setCounts(tinyDistribution(model).p)}>Use current model mass: zero-gradient control</button><button onClick={() => {
        setModel(rbmDefault());
        setCounts([0, 0, 0, 1]);
        setRate(.1);
      }}>Reset co-occurrence ledgers</button></div>
    <p>The zero-gradient control uses probability masses rather than integer observations. Editing any field replaces that entry; all nonnegative entries are normalized by their visible sum.</p>
  </NeuralLab>;
}
export function RbmChainLab() {
  const [model, setModel] = useState(rbmDefault),
    [initial, setInitial] = useState(3),
    [stationary, setStationary] = useState(false),
    [steps, setSteps] = useState(1),
    [source, setSource] = useState(3),
    [seed, setSeed] = useState(19);
  const [uniforms, setUniforms] = useState([.8, .3, .6]);
  const equilibrium = tinyDistribution(model).p,
    q0 = stationary ? equilibrium : equilibrium.map((_, i) => Number(i === initial));
  const flow = probabilityFlow(model, q0, steps),
    current = flow.traces.at(-1),
    next = current.incoming[0].map((_, j) => current.incoming.reduce((sum, row) => sum + row[j], 0));
  const sampled = particleTrace(binaryStates(2)[initial], model, Math.max(1, steps), seed),
    draw = tinyDraw([1, 0], model, uniforms);
  const ph = binaryStates(2).map(v => hiddenProbabilities(v, model)[0]),
    meanV = [0, 1].map(i => current.mass.reduce((sum, p, j) => sum + p * binaryStates(2)[j][i], 0));
  const dataStats = statistics(model, q0),
    negative = statistics(model, current.mass).positive,
    expectedGradient = dataStats.positive.map((x, i) => x - negative[i]);
  return <NeuralLab id="rbm-chain" title="Move probability mass and follow a sampled particle">
    <RbmParameters model={model} onChange={setModel} /><div className="neural-controls"><NeuralSelect label="Initial visible state" value={initial} onChange={v => {
        setInitial(Number(v));
        setStationary(false);
      }} options={binaryStates(2).map((v, i) => [i, stateName(v)])} /><NeuralNumber label="Completed full Gibbs transitions" value={steps} min={0} max={100} integer step={1} onChange={setSteps} /><NeuralSelect label="Inspect mass leaving source" value={source} onChange={v => setSource(Number(v))} options={binaryStates(2).map((v, i) => [i, stateName(v)])} /><NeuralNumber label="Sample stream seed" value={seed} min={0} max={100000} integer range={false} onChange={setSeed} /></div>
    <RbmFigure title={'Current distribution q' + steps + ' → next distribution'} width={470} height={300} description="Every source sends mass q(source) × T(source,destination). Only the selected source's four ribbons are drawn; the destination bars include contributions from all sources. Width is probability mass, not a trajectory count.">
      {[0, 1, 2, 3].map(i => <path key={i} d={'M125 ' + (45 + source * 62) + ' C220 ' + (45 + source * 62) + ' 240 ' + (45 + i * 62) + ' 335 ' + (45 + i * 62)} fill="none" stroke="#e6b854" opacity=".65" strokeWidth={30 * current.incoming[source][i]} />)}
      {[0, 1, 2, 3].map(i => <g key={i}><text x="12" y={50 + i * 62} fill="#eee" fontSize="14">{stateName(binaryStates(2)[i])}</text><rect x="40" y={30 + i * 62} width={80 * current.mass[i]} height="28" fill="#aaa" /><text x="125" y={74 + i * 62} textAnchor="end" fill="#eee" fontSize="12">{f(current.mass[i], 4)}</text><rect x="340" y={30 + i * 62} width={80 * next[i]} height="28" fill="#e6b854" /><text x="438" y={50 + i * 62} fill="#eee" fontSize="14">{stateName(binaryStates(2)[i])}</text><text x="340" y={74 + i * 62} fill="#eee" fontSize="12">{f(next[i], 4)}</text></g>)}<text x="235" y="294" textAnchor="middle" fill="#eee" fontSize="13">bar scale 0–1; next q includes every incoming route</text>
    </RbmFigure>
    <ProbabilityBars title="Current mass versus the current model's stationary distribution" baselineLabel="the current model's stationary distribution" rows={binaryStates(2).map((v, i) => ({
      label: stateName(v),
      current: current.mass[i],
      baseline: equilibrium[i]
    }))} />
    <NeuralTable caption="Exact one-step transition matrix T" headers={['From / to', '00', '01', '10', '11']} rows={flow.transition.map((row, i) => [stateName(binaryStates(2)[i]), ...row.map(x => f(x, 6))])} />
    <p>Total variation from equilibrium {f(current.tv, 8)}. Expected finite-chain gradient {vector(expectedGradient)}; exact likelihood gradient {vector(dataStats.gradient)}. Chain steps keep parameters fixed; they are not parameter updates.</p>
    <div className="neural-buttons"><button onClick={() => setSteps(n => Math.max(0, n - 1))}>Back one mass step</button><button onClick={() => setSteps(n => Math.min(100, n + 1))}>Advance one mass step</button><button onClick={() => setStationary(true)}>Initialize mass at equilibrium</button></div>
    <h4>One particle is a draw, not a fractional state</h4><p>Mulberry32 stream, seed {seed}; current sampled state after {steps} steps: <strong>{steps ? stateName(sampled.at(-1).next) : stateName(binaryStates(2)[initial])}</strong>. The particle begins at the selected binary state even when the separate mass view is initialized at equilibrium.</p>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Read every sampled probability, uniform draw and state</h4><NeuralTable caption="Deterministic replay of this JavaScript random stream" headers={['Step', 'Previous v', 'p(h=1)', 'u hidden', 'h', 'p(v=1 | h)', 'u visible', 'Next v']} rows={sampled.slice(0, steps).map((r, i) => [i + 1, stateName(r.previous), f(r.ph), f(r.uniforms[0]), r.hidden, vector(r.pv), vector(r.uniforms.slice(1)), stateName(r.next)])} /></section>
    <div className="neural-controls">{uniforms.map((value, i) => <NeuralNumber key={i} label={['Hand draw u hidden', 'Hand draw u visible 1', 'Hand draw u visible 2'][i]} value={value} min={0} max={.999999} range={false} onChange={x => setUniforms(old => old.map((u, j) => i === j ? x : u))} />)}</div><p>Separate hand draw from 10: p(h=1)={f(draw.ph)}; u={f(uniforms[0])} gives h={draw.hidden}. Visible probabilities {vector(draw.pv)}, draws {vector(uniforms.slice(1))} give <strong>{stateName(draw.next)}</strong>.</p>
    <p>Current E[p(h=1 | v)]={f(current.mass.reduce((sum, p, i) => sum + p * ph[i], 0), 8)}. Hidden sigmoid at the mean visible vector {vector(meanV)} gives {f(hiddenProbabilities(meanV, model)[0], 8)}. Their difference is the nonlinear mean-replacement effect.</p>
    <button onClick={() => {
      setModel(rbmDefault());
      setInitial(3);
      setStationary(false);
      setSteps(1);
      setSource(3);
      setSeed(19);
      setUniforms([.8, .3, .6]);
    }}>Reset probability flow</button>
  </NeuralLab>;
}
export function RbmPersistenceLab() {
  const [model, setModel] = useState(rbmDefault),
    [seed, setSeed] = useState(31),
    [a, setA] = useState([[1, 1], [1, 1], [1, 0], [0, 1]]),
    [b, setB] = useState([[0, 0], [0, 1], [1, 0], [1, 0]]);
  const trace = persistenceTrace(model, [a, b], seed);
  return <NeuralLab id="rbm-persistence" title="A new minibatch changes CD starts; PCD carries particles forward">
    <RbmParameters model={model} onChange={setModel} compact /><NeuralNumber label="Shared draw stream seed" value={seed} min={0} max={100000} integer range={false} onChange={setSeed} />
    <fieldset><legend>Edit the first minibatch</legend><div className="neural-controls">{a.map((v, i) => <NeuralSelect key={i} label={'Minibatch A example ' + (i + 1)} value={stateName(v)} onChange={value => setA(old => old.map((r, j) => i === j ? value.split('').map(Number) : r))} options={binaryStates(2).map(v => [stateName(v), stateName(v)])} />)}</div></fieldset>
    <fieldset><legend>Edit the second minibatch</legend><div className="neural-controls">{b.map((v, i) => <NeuralSelect key={i} label={'Minibatch B example ' + (i + 1)} value={stateName(v)} onChange={value => setB(old => old.map((r, j) => i === j ? value.split('').map(Number) : r))} options={binaryStates(2).map(v => [stateName(v), stateName(v)])} />)}</div></fieldset>
    <RbmFigure title="Follow particle 0 across the minibatch boundary" width={550} height={210} description="The PCD path connects the previous sampled state to the next start. CD's second start instead comes from minibatch B. These displayed states are calculated from the current parameters and random stream.">
      {['cd', 'pcd'].map((method, row) => <g key={method}><text x="10" y={52 + row * 100} fill="#eee" fontSize="14">{method.toUpperCase()}</text>{[trace[0][method][0].previous, trace[0][method][0].next, trace[1][method][0].previous, trace[1][method][0].next].map((state, i) => <g key={i}><rect x={70 + i * 120} y={27 + row * 100} width="65" height="38" fill="#222" stroke="#e6b854" /><text x={102 + i * 120} y={52 + row * 100} textAnchor="middle" fill="#eee" fontSize="16">{stateName(state)}</text><text x={102 + i * 120} y={86 + row * 100} textAnchor="middle" fill="#ccc" fontSize="12">{['A start', 'A result', 'B start', 'B result'][i]}</text>{i < 3 && <path d={'M' + (135 + i * 120) + ' ' + (46 + row * 100) + ' H' + (190 + i * 120)} stroke={i === 1 && row === 0 ? '#888' : '#e6b854'} strokeDasharray={i === 1 && row === 0 ? '4 4' : undefined} />}</g>)}</g>)}
    </RbmFigure>
    {trace.map(row => <section key={row.step}><h4>Minibatch {row.step ? 'B' : 'A'}: {row.batch.map(stateName).join(', ')}</h4><NeuralTable caption="Same uniforms isolate the different starting-state rule" headers={['Particle', 'CD start → next', 'PCD start → next', 'Hidden u', 'Visible u₁/u₂']} rows={row.cd.map((cd, i) => [i, stateName(cd.previous) + ' → ' + stateName(cd.next), stateName(row.pcd[i].previous) + ' → ' + stateName(row.pcd[i].next), f(row.draws[i][0]), vector(row.draws[i].slice(1))])} /></section>)}
    <p>PCD begins with particles 00, 01, 10, 11. Its B starts are exactly its A results. CD starts from the edited examples in each minibatch. Parameters stay fixed here so persistence is isolated from learning: changing data alone does not change the PCD particles in this view. The full study separately updates parameters from positive and negative means.</p>
    <button onClick={() => {
      setModel(rbmDefault());
      setSeed(31);
      setA([[1, 1], [1, 1], [1, 0], [0, 1]]);
      setB([[0, 0], [0, 1], [1, 0], [1, 0]]);
    }}>Reset particle comparison</button>
  </NeuralLab>;
}
export function RbmReconstructionLab() {
  const [index, setIndex] = useState(3),
    v = binaryStates(2)[index];
  const models = [{
    w: [[20], [20]],
    a: [-10, -10],
    b: [-20]
  }, {
    w: [[0], [0]],
    a: [Math.log(9), Math.log(9)],
    b: [0]
  }];
  return <NeuralLab id="rbm-reconstruction" title="A return path and a probability budget can rank models differently"><NeuralSelect label="Binary input for both models" value={index} onChange={v => setIndex(Number(v))} options={binaryStates(2).map((v, i) => [i, stateName(v)])} /><div className="rbm-two">{models.map((model, i) => {
        const metrics = rbmMetrics(v, model),
          p = tinyDistribution(model).p;
        return <section key={i}><h4>Constructed model {i ? 'B: independent' : 'A: two sharp modes'}</h4><RbmFigure title="Deterministic mean reconstruction" width={340} height={115} description={'Input ' + stateName(v) + ' → hidden conditional mean → reconstructed visible mean ' + vector(metrics.reconstructed)}><text x="10" y="45" fill="#eee">{stateName(v)}</text><path d="M40 40 H120 M170 40 H240" stroke="#e6b854" /><text x="145" y="45" textAnchor="middle" fill="#eee">σ</text><text x="290" y="45" textAnchor="middle" fill="#eee">σ</text><text x="170" y="92" textAnchor="middle" fill="#eee" fontSize="13">MSE {metrics.mse.toExponential(6)} · NLL {f(metrics.nll, 6)}</text></RbmFigure><ProbabilityBars title="Normalized probabilities of every visible state" rows={binaryStates(2).map((s, j) => ({
            label: stateName(s),
            current: p[j]
          }))} /></section>;
      })}</div><p>MSE averages the two squared coordinate errors. NLL is −log p(input) and includes the whole state space. Choose input 00 to see a case where model A wins both, then 11 to separate the two criteria.</p></NeuralLab>;
}
export function RbmLibraryLab() {
  const [bias, setBias] = useState(0),
    model = {
      ...libraryModel,
      b: [libraryModel.b[0] + bias]
    },
    dist = tinyDistribution(model);
  const transition = probabilityFlow(model, dist.p, 1);
  return <NeuralLab id="rbm-library" title="Match the library's fitted parameters to the exact tiny oracle">
    <p>These are the actual retained parameters fitted by the complete scikit-learn 1.9.1 bridge. The browser recomputes the deterministic probability equations from those weights.</p>
    <NeuralNumber label="Added hidden bias" value={bias} min={-3} max={3} onChange={setBias} />
    <RbmFigure title="Transpose the parameter axes, preserve the interaction" width={470} height={145} description="The library stores one row per hidden unit and one column per visible unit. The scratch model stores visible rows and hidden columns; it is the same two interactions.">
      <text x="105" y="25" textAnchor="middle" fill="#eee">components_ [H=1, D=2]</text>{libraryModel.w.map((row, i) => <g key={i}><rect x={25 + i * 82} y="43" width="82" height="35" fill="#222" stroke="#e6b854" /><text x={66 + i * 82} y="66" textAnchor="middle" fill="#eee" fontSize="12">{f(row[0], 5)}</text></g>)}<path d="M212 62 H283" stroke="#e6b854" /><text x="249" y="48" textAnchor="middle" fill="#eee" fontSize="12">transpose</text><text x="373" y="25" textAnchor="middle" fill="#eee">W [D=2, H=1]</text>{libraryModel.w.map((row, i) => <g key={i}><rect x="330" y={43 + i * 35} width="85" height="35" fill="#222" stroke="#e6b854" /><text x="373" y={66 + i * 35} textAnchor="middle" fill="#eee" fontSize="12">{f(row[0], 5)}</text></g>)}
    </RbmFigure>
    <NeuralTable caption="Current transform probabilities and normalized likelihood" headers={['Visible', 'p(h=1 | v)', 'Exact p(v)', 'Exact log p(v)']} rows={dist.states.map((v, i) => [stateName(v), f(hiddenProbabilities(v, model)[0], 8), f(dist.p[i], 8), f(Math.log(dist.p[i]), 8)])} />
    <p>Stationary distribution error max |pT−p|={Math.max(...dist.p.map((p, i) => Math.abs(p - transition.traces[1].mass[i]))).toExponential(3)}. A bias increment ln 2 doubles hidden on/off odds. <code>transform</code> returns probabilities, <code>gibbs</code> draws states, and <code>score_samples</code> reports random-bit pseudo-likelihood; the complete native program prints these separately.</p>
    <div className="neural-buttons"><button onClick={() => setBias(Math.log(2))}>Add ln 2</button><button onClick={() => setBias(0)}>Reset fitted-library bias</button></div>
  </NeuralLab>;
}
