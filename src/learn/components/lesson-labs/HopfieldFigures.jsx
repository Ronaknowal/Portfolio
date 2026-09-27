import { useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable } from './NeuralLessonElements.jsx';
import { MemoryFigure, Arrow, Node, MemoryPlane, MemoryPlot, Bars, Values, f, vec } from './HopfieldPrimitives.jsx';
import { binaryRecall, continuousRead, continuousTrace, energyContours, queryTraining, marginBound, storageCost, higherOrderEnergy, softmax } from '../../data/hopfield-memory-models.js';
export function LookupFigure() {
  return <MemoryFigure title="An address chooses a record; a cue asks for a related record" height={265}><Node x={10} y={15} lines={['address: row 27']} /><Arrow x1={145} y1={37} x2={235} y2={37} /><Node x={235} y={15} lines={['exact stored record']} /><Node x={10} y={90} lines={['partial / noisy cue']} /><Arrow x1={145} y1={112} x2={235} y2={112} /><Node x={235} y={90} lines={['compare with bank']} /><Arrow x1={370} y1={112} x2={460} y2={112} /><Node x={460} y={90} lines={['read an association']} />{[['recall: exact memory', 15], ['reconstruction: pixels', 220], ['classification: label', 420]].map(([t, x]) => <g key={t}><Arrow x1={527} y1={134} x2={x + 85} y2={192} /><Node x={x} y={192} width={180} lines={[t]} /></g>)}</MemoryFigure>;
}
export function NormTrapFigure() {
  return <><MemoryFigure title="A longer memory can win the dot product without being closer" width={570} height={245}><line x1="50" y1="180" x2="480" y2="180" stroke="#777" /><line x1="80" y1="210" x2="80" y2="25" stroke="#777" /><Arrow x1={80} y1={180} x2={180} y2={180} color="#87bdf1" /><Arrow x1={80} y1={180} x2={380} y2={80} /><circle cx="180" cy="180" r="7" fill="none" stroke="white" /><text x="160" y="210">q = x₁ = (1,0)</text><text x="350" y="62">x₂ = (3,1)</text><line x1="180" y1="180" x2="380" y2="80" stroke="#e9979f" strokeDasharray="5 4" /><text x="220" y="115">distance √5</text><text x="80" y="235">One unit =100 pixels on both axes</text></MemoryFigure><NeuralTable caption="Same cue, two comparison criteria" headers={['Memory', 'Length', 'Dot score', 'Distance to cue']} rows={[[vec([1, 0]), 1, 1, 0], [vec([3, 1]), '√10', 3, '√5']]} /></>;
}
export function SignedMemoryFigure({
  state = [1, 1, -1, -1],
  weights = null,
  active = 1
}) {
  const w = weights || binaryRecall([[1, 1, -1, -1]], [1, -1, -1, -1]).w,
    positions = [[90, 50], [280, 50], [90, 225], [280, 225]];
  return <><MemoryFigure title="Every other feature casts a signed vote" height={300} description="Solid + and dashed − edges carry symmetric weights. The highlighted row multiplies each weight by the current source bit.">{positions.flatMap((p, i) => positions.slice(i + 1).map((q, j0) => {
        const j = i + j0 + 1,
          v = w[i][j];
        return <g key={`${i}${j}`}><line x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} stroke={v < 0 ? '#e9979f' : '#e6bb60'} strokeDasharray={v < 0 ? '5 4' : undefined} /><text x={(p[0] + q[0]) / 2 + (i + j === 3 ? i * 13 : 0)} y={(p[1] + q[1]) / 2 - 8}>{v >= 0 ? '+' : ''}{f(v, 2)}</text></g>;
      }))}{positions.map((p, i) => <g key={i}><circle cx={p[0]} cy={p[1]} r="26" fill="#191919" stroke={active === i ? 'white' : '#888'} strokeWidth={active === i ? 3 : 1} /><text x={p[0]} y={p[1] + 5} textAnchor="middle">{state[i] > 0 ? '+1' : '−1'}</text><text x={p[0]} y={p[1] + 46} textAnchor="middle">feature {i + 1}</text></g>)}<text x="370" y="50">Row {active + 1} votes</text>{w[active].map((v, j) => <text key={j} x="370" y={85 + j * 35}>{f(v, 2)} × ({state[j]}) = {f(v * state[j], 2)}</text>)}<text x="370" y="248">field = {f(w[active].reduce((s, v, j) => s + v * state[j], 0), 3)}</text></MemoryFigure><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Complete symmetric weight matrix</h4><NeuralTable caption="W: source feature across columns" headers={['Target', 1, 2, 3, 4]} rows={w.map((row, i) => [i + 1, ...row.map(x => f(x, 3))])} /></section></>;
}
export function BinaryWorkedFigure() {
  const r = binaryRecall([[1, 1, -1, -1]], [1, -1, -1, -1]);
  return <><MemoryFigure title="The second update uses the latest four signs" height={205}><Node x={10} y={30} width={160} lines={['initial cue', '[+1, −1, −1, −1]']} /><Arrow x1={170} y1={52} x2={220} y2={52} /><Node x={220} y={30} width={160} lines={['visit feature1', 'field +0.25: keep +1']} /><Arrow x1={380} y1={52} x2={440} y2={52} /><Node x={440} y={30} width={170} lines={['visit feature2', 'field +0.75: −1→+1']} /><text x="215" y="120">+0.25×(+1) + (−0.25)×(−1) + (−0.25)×(−1)</text><Arrow x1={525} y1={75} x2={525} y2={145} /><text x="370" y="172">[+1,+1,−1,−1] → energy −1.5</text></MemoryFigure><NeuralTable caption="Actual first sweep" headers={['Update', 'State', 'Field', 'Energy']} rows={r.trace.slice(0, 5).map((t, i) => [i, vec(t.state), t.field === null ? '—' : f(t.field), f(t.energy)])} /></>;
}
export function BinaryEnergyFigure() {
  return <MemoryPlot title="Individual coordinate updates retain flat energy steps" xLabel="coordinate updates" yLabel="energy" series={[{
    label: 'E(s)',
    color: '#e6bb60',
    points: [[0, 0], [1, 0], [2, 0], [2, -1.5], [3, -1.5], [4, -1.5]]
  }]} />;
}
export function BinaryCapacityFigure() {
  const rows = [[2, 16, 16, 16], [6, 48, 48, 47], [10, 80, 61, 56], [16, 128, 49, 29], [24, 192, 12, 2]];
  return <><MemoryFigure title="Two different success criteria on the same eight sampled banks" width={600} height={290} description="Measured at d=64; exactly six cue flips. Gold: stored pattern stays fixed. Blue: corrupted cue returns exactly. Each bar shows its own count/denominator.">{rows.map(([p, n, a, b], i) => <g key={p}><text x="5" y={30 + i * 48}>P={p}</text><rect x="65" y={15 + i * 48} width={400 * a / n} height="15" fill="#e6bb60" /><rect x="65" y={32 + i * 48} width={400 * b / n} height="12" fill="#87bdf1" /><text x="480" y={28 + i * 48}>{a}/{n}</text><text x="480" y={44 + i * 48}>{b}/{n}</text></g>)}<text x="65" y="277">0</text><text x="260" y="277">0.5</text><text x="455" y="277">1</text></MemoryFigure></>;
}
export function ContinuousWorkedFigure() {
  const m = [[1, 0], [-1, 0]],
    q = [.2, .4],
    r = continuousRead(m, q, 2);
  return <><div className="hm-two"><MemoryPlane title="The read lies inside the memory segment" memories={m} q={q} read={r.read} /><Bars title="Dot scores become positive mixing weights" labels={['M1: dot +0.2', 'M2: dot −0.2']} values={r.weights} /></div><Values caption="One complete read, β=2" rows={[["Scores", vec(r.scores)], ["Scaled logits", vec(r.logits)], ["Weights", vec(r.weights)], ["Read", vec(r.read)]]} /></>;
}
export function CobwebFigure() {
  return <div className="hm-two">{[.5, 2].map(beta => {
      let x = .2,
        points = [[x, 0]];
      for (let i = 0; i < 10; i++) {
        const y = Math.tanh(beta * x);
        points.push([x, y], [y, y]);
        x = y;
      }
      return <MemoryPlot key={beta} title={`β=${beta}: repeated y=tanh(βx)`} xLabel="current x" yLabel="next x" series={[{
        label: 'identity',
        color: '#999',
        points: [[-1, -1], [1, 1]]
      }, {
        label: 'retrieval',
        color: '#e6bb60',
        points: Array.from({
          length: 101
        }, (_, i) => {
          const x = -1 + i / 50;
          return [x, Math.tanh(beta * x)];
        })
      }, {
        label: 'computed iterates',
        color: '#87bdf1',
        points
      }]} />;
    })}</div>;
}
const workedMemories = [[1, 0], [-1, 0]],
  workedTrace = continuousTrace(workedMemories, [.2, .4], 2, 6),
  workedContours = energyContours(workedMemories, 2);
export function EnergyLandscapeFigure() {
  return <><div className="hm-two"><MemoryPlane title="Actual energy contours and downhill iterates" memories={workedMemories} q={[.2, .4]} read={workedTrace[1].q} path={workedTrace.map(x => x.q)} contours={workedContours} /><MemoryPlot title="Energy falls at every displayed nonzero step" xLabel="updates" yLabel="E" series={[{
        label: 'continuous energy',
        color: '#e6bb60',
        points: workedTrace.map((t, i) => [i, t.energy])
      }]} /></div><p>The central point (0,0) is a saddle for β=2: the horizontal curvature is negative and the vertical curvature is positive. The two stable fixed points lie inside the stored endpoints.</p></>;
}
export function SensitivityFigure() {
  const [beta, setBeta] = useState(2),
    [x, setX] = useState(0),
    r = continuousRead(workedMemories, [x, 0], beta);
  return <><div className="neural-controls"><NeuralNumber label="Sensitivity β" value={beta} min={.1} max={8} onChange={setBeta} /><NeuralNumber label="Sensitivity cue x" value={x} min={-2} max={2} onChange={setX} /></div><MemoryFigure title="Weighted spread controls local amplification" width={520} height={190}><text x="10" y="25">horizontal cue perturbation δ</text><Arrow x1={30} y1={65} x2={90} y2={65} color="#87bdf1" /><Node x={130} y={43} width={210} lines={['β × weighted variance', f(r.jacobian[0][0], 6)]} /><Arrow x1={90} y1={65} x2={130} y2={65} /><Arrow x1={340} y1={65} x2={370 + Math.min(130, r.sensitivity * 30)} y2={65} /><text x="10" y="125">DF = diag({f(r.jacobian[0][0], 6)}, 0); local norm {f(r.sensitivity, 6)}</text><text x="10" y="156">A local norm above1 can amplify a small displacement.</text></MemoryFigure></>;
}
export function KeyValueFigure() {
  return <MemoryFigure title="One weight connects a key comparison to a value payload" height={285}><Node x={10} y={95} width={110} lines={['q:1×2', '[0.6,−0.2]']} /><Arrow x1={120} y1={117} x2={160} y2={117} /><text x="170" y="30">Keys:2×2</text><text x="320" y="30">softmax β=1</text><text x="485" y="30">Values:2×1</text>{[['[1,0]', '0.689974', '10'], ['[0,1]', '0.310026', '−2']].map((r, i) => <g key={i}><Node x={160} y={65 + i * 85} width={110} lines={[r[0]]} /><Arrow x1={270} y1={87 + i * 85} x2={320} y2={87 + i * 85} /><Node x={320} y={65 + i * 85} width={105} lines={[r[1]]} /><Arrow x1={425} y1={87 + i * 85} x2={480} y2={87 + i * 85} /><Node x={480} y={65 + i * 85} width={95} lines={[r[2]]} /></g>)}<Arrow x1={530} y1={194} x2={530} y2={233} /><text x="185" y="255">0.689974×10 + 0.310026×(−2) = 6.279694</text></MemoryFigure>;
}
export function ModuleOwnershipFigure() {
  return <>{[['Input-dependent Q', 'Input-dependent K,V', 'sequence-to-sequence read'], ['Persistent learned Q', 'Input-dependent K,V', 'pool a variable-sized input'], ['Input-dependent Q', 'Persistent learned K,V', 'compare against prototypes']].map(([q, k, out], i) => <MemoryFigure key={q} title={`Module ${i + 1}: ${out}`} width={600} height={145}><Node x={10} y={15} width={190} lines={[q]} color={q.startsWith('Persistent') ? '#e6bb60' : '#87bdf1'} /><Node x={10} y={85} width={190} lines={[k]} color={k.startsWith('Persistent') ? '#e6bb60' : '#87bdf1'} /><Arrow x1={200} y1={37} x2={265} y2={70} /><Arrow x1={200} y1={107} x2={265} y2={70} /><Node x={265} y={48} width={135} lines={['softmax(QKᵀ)', 'then ×V']} /><Arrow x1={400} y1={70} x2={450} y2={70} /><Node x={450} y={48} width={140} lines={['returned values']} /></MemoryFigure>)}</>;
}
export function QueryTrainingFigure() {
  const r = queryTraining([[1, 0], [0, 1]], [.2, -.1], 1, 1, .1),
    after = queryTraining([[1, 0], [0, 1]], r.next, 1, 1, .1);
  return <><MemoryFigure title="Changing a query parameter differs from retrieving a new state" height={200}><Node x={10} y={25} width={175} lines={['trainable q=[.2,−.1]']} /><Arrow x1={185} y1={47} x2={240} y2={47} /><Node x={240} y={25} width={160} lines={['class2 loss', '−log p₂=0.854355']} /><Arrow x1={400} y1={47} x2={460} y2={47} /><Node x={460} y={25} width={150} lines={['gradient', '[+.57444,−.57444]']} /><Arrow x1={535} y1={70} x2={535} y2={130} /><Arrow x1={535} y1={130} x2={185} y2={130} /><Node x={10} y={108} width={175} lines={['q −0.1×gradient', '[.142556,−.042556]']} /><text x="205" y="165">Training follows the label objective; refinement returns Xᵀp.</text></MemoryFigure><Bars title="Label-directed training moves mass toward class2" labels={['class1', 'class2']} values={after.weights} reference={r.weights} referenceLabel="before training" /><Values caption="Training quantities" rows={[["Before loss", f(r.loss, 8)], ["After loss", f(after.loss, 8)], ["Gradient", vec(r.gradient)]]} /></>;
}
export function DigitRolesFigure() {
  return <MemoryFigure title="The original test writers remain outside fitting and selection" width={640} height={290}><rect x="8" y="15" width="420" height="255" fill="none" stroke="#87bdf1" /><text x="20" y="40">Training source:3823 images,30 writers</text>{[['memory:200', 70], ['fit queries:800', 120], ['validation:300', 170], ['unused:2523', 220]].map(([t, y]) => <Node key={t} x={25} y={y} width={150} height={32} lines={[t]} />)}<Node x={257} y={96} width={155} lines={['fit projection']} /><Node x={257} y={166} width={155} lines={['select checkpoint']} /><Arrow x1={175} y1={86} x2={257} y2={118} /><Arrow x1={175} y1={136} x2={257} y2={118} /><Arrow x1={175} y1={186} x2={257} y2={188} /><rect x="449" y="15" width="183" height="255" fill="none" stroke="#e6bb60" /><text x="462" y="42">Original test group</text><text x="462" y="72">1797 images</text><text x="462" y="97">13 other writers</text><Node x={462} y={167} width={155} lines={['locked evaluation']} /><Arrow x1={412} y1={188} x2={462} y2={188} /></MemoryFigure>;
}
export function DigitArchitectureFigure() {
  return <MemoryFigure title="The same learned map shapes cues and all200 memory keys" width={660} height={300}><Node x={5} y={15} width={130} lines={['query:64 pixels']} /><Node x={5} y={110} width={130} lines={['200×64 pixels']} /><Node x={190} y={60} width={145} height={72} lines={['shared W:16×64', 'then unit length']} /><Arrow x1={135} y1={37} x2={190} y2={82} /><Arrow x1={135} y1={132} x2={190} y2={110} /><Arrow x1={335} y1={96} x2={385} y2={96} /><Node x={385} y={73} width={125} lines={['200 scores', 'softmax β=16']} /><Arrow x1={510} y1={96} x2={545} y2={37} /><Arrow x1={510} y1={96} x2={545} y2={167} /><Node x={545} y={15} width={110} lines={['10 label sums']} /><Node x={545} y={145} width={110} lines={['64 pixel sums']} /><path d="M70 154 V230 H600 V189" fill="none" stroke="#87bdf1" /><text x="95" y="251">Original memory pixels take the value route (no projection).</text><text x="95" y="278">Only class-label loss trains W; reconstruction is an inspected side read.</text></MemoryFigure>;
}
export function DigitMetricsFigure() {
  const rows = [['Nearest', 25, 156, 663], ['Fixed β64', 18, 136, 667], ['Seed17', 13, 100, 564], ['Seed41', 16, 113, 740]];
  return <MemoryFigure title="Clean accuracy and occlusion robustness rank the fits differently" width={640} height={300} description="Actual error counts; both test bars have denominator1797. Gold: clean. Blue: occluded. Clean validation selects fixed β64 and each learned checkpoint.">{rows.map(([name, v, a, b], i) => <g key={name}><text x="5" y={42 + i * 54}>{name}</text><rect x="105" y={25 + i * 54} width={a / 800 * 370} height="15" fill="#e6bb60" /><rect x="105" y={43 + i * 54} width={b / 800 * 370} height="15" fill="#87bdf1" /><text x="490" y={38 + i * 54}>{a} / {b}</text><text x="490" y={58 + i * 54}>val {v}/300</text></g>)}<line x1="105" y1="250" x2="475" y2="250" stroke="#777" />{[0, 200, 400, 600, 800].map(n => <text key={n} x={105 + n / 800 * 370} y="275" textAnchor="middle">{n}</text>)}</MemoryFigure>;
}
export function MarginFigure() {
  const [gap, setGap] = useState(3),
    [beta, setBeta] = useState(2),
    r = marginBound(100, beta, gap, 1),
    mass = 99 * Math.exp(-beta * gap);
  return <><div className="neural-controls"><NeuralNumber label="Assumed minimum score gap" value={gap} min={0} max={6} onChange={setGap} /><NeuralNumber label="Bound β" value={beta} min={.1} max={8} onChange={setBeta} /></div><Bars title="Ninety-nine competitors share one denominator" labels={['target exp score', 'combined upper mass']} values={[1, mass]} /><p>Under the stated gap assumption, target mass ≥{f(r.targetMass, 6)} and read distance ≤{f(r.errorBound, 6)} for maximum norm1. These are score assumptions, not a constructed unit-vector arrangement.</p></>;
}
export function CapacityAxesFigure() {
  const [count, setCount] = useState(1000000),
    [dim, setDim] = useState(64),
    r = storageCost(count, dim, 4);
  return <><div className="neural-controls"><NeuralNumber label="Stored rows" value={count} min={1} max={10000000} integer onChange={setCount} /><NeuralNumber label="Coordinates per row" value={dim} min={1} max={4096} integer onChange={setDim} /></div><MemoryFigure title="Storage, query work and attractor success are distinct quantities" width={600} height={190}><rect x="20" y="20" width="165" height="120" fill="#191919" stroke="#87bdf1" />{[1, 2, 3, 4].map(i => <line key={i} x1="20" y1={20 + i * 24} x2="185" y2={20 + i * 24} stroke="#555" />)}<text x="25" y="170">P rows × d columns</text><Node x={230} y={50} width={145} lines={['one query: d']} /><Arrow x1={375} y1={72} x2={420} y2={72} /><Node x={420} y={50} width={160} lines={['P dot scores', 'P×d multiplies']} /></MemoryFigure><Values caption="Float32 key storage; exact arithmetic counts" rows={[["Key bytes", r.bytes.toLocaleString()], ["Dot-product multiplications", r.multiplications.toLocaleString()], ["Scores per query", r.scores.toLocaleString()], ["Attractor/task success", 'Requires separate assumptions and measurement']]} /></>;
}
export function ParityFigure() {
  const [power, setPower] = useState(3);
  return <><div className="hm-controls"><button onClick={() => setPower(2)}>Quadratic energy</button><button onClick={() => setPower(3)}>Cubic energy</button></div><MemoryFigure title={`Power ${power}: clamp inputs, compare only the output sign`} width={560} height={275}>{[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([a, b], i) => {
        const es = [-1, 1].map(z => higherOrderEnergy([a, b, z], power)),
          minimum = Math.min(...es);
        return <g key={i}><Node x={5} y={10 + i * 61} width={130} lines={[`clamped (${a},${b})`]} /><Arrow x1={135} y1={32 + i * 61} x2={200} y2={32 + i * 61} />{[-1, 1].map((z, j) => <g key={z}><rect x={200 + j * 180} y={10 + i * 61} width="165" height="43" fill={es[j] === minimum ? '#3b311e' : '#191919'} stroke="#777" /><text x={210 + j * 180} y={37 + i * 61}>z={z}: E={es[j]}</text></g>)}</g>;
      })}</MemoryFigure></>;
}
export function ChangingBankFigure() {
  const [age, setAge] = useState(2),
    p = softmax([1, 1 - .5 * age]);
  return <><NeuralNumber label="Second record age (arbitrary time units)" value={age} min={0} max={10} onChange={setAge} /><Bars title="Equal similarity, different age: score = similarity −0.5×age" labels={['record1: age0', 'record2: edited age']} values={p} /><MemoryFigure title="When state changes both query and keys, both derivative paths matter" height={165}><Node x={10} y={60} lines={['state z']} /><Arrow x1={145} y1={82} x2={225} y2={32} /><Arrow x1={145} y1={82} x2={225} y2={123} /><Node x={225} y={10} lines={['query Q(z)']} /><Node x={225} y={101} lines={['keys K(z)']} /><Arrow x1={360} y1={32} x2={435} y2={82} /><Arrow x1={360} y1={123} x2={435} y2={82} /><Node x={435} y={60} width={175} lines={['energy / objective', 'differentiate both uses']} /></MemoryFigure></>;
}
export function BagPoolingFigure() {
  return <MemoryFigure title="A bag label supervises pooled evidence, not each instance" height={260}>{[3, 5, 2].map((n, i) => <g key={i}>{Array.from({
        length: n
      }, (_, j) => <rect key={j} x={12 + j * 20} y={25 + i * 65} width="15" height="30" fill="#333" stroke="#87bdf1" />)}<Arrow x1={125} y1={40 + i * 65} x2={195} y2={105} /></g>)}<Node x={195} y={75} width={150} height={60} lines={['shared sequence', 'encoder']} /><Arrow x1={345} y1={105} x2={395} y2={105} /><Node x={395} y={75} width={145} height={60} lines={['learned-query', 'pool over instances']} /><Node x={395} y={195} width={145} lines={['one bag label']} /><Arrow x1={467} y1={195} x2={467} y2={135} color="#e9979f" /><text x="5" y="237">Illustrative bags of sizes3,5,2; no instance-level labels are claimed.</text></MemoryFigure>;
}
export function HopularFigure() {
  return <MemoryFigure title="Alternate sample retrieval and original-feature retrieval" width={660} height={320}><rect x="10" y="15" width="170" height="110" fill="none" stroke="#87bdf1" />{[1, 2, 3].map(i => <line key={i} x1="10" y1={15 + i * 27} x2="180" y2={15 + i * 27} stroke="#555" />)}<text x="18" y="145">fixed training samples</text><text x="18" y="167">Hs: sample axis ↓</text><Node x={230} y={65} width={170} lines={['sample memory read']} /><Arrow x1={180} y1={70} x2={230} y2={87} /><Node x={10} y={235} width={170} lines={['query state', 'target masked']} /><Arrow x1={180} y1={257} x2={230} y2={87} /><Node x={455} y={65} width={190} lines={['feature memory read']} /><Arrow x1={400} y1={87} x2={455} y2={87} /><rect x="455" y="180" width="190" height="65" fill="none" stroke="#e6bb60" />{[1, 2, 3, 4].map(i => <line key={i} x1={455 + i * 38} y1="180" x2={455 + i * 38} y2="245" stroke="#666" />)}<text x="455" y="270">original embedded input</text><text x="455" y="292">Hf: feature axis →</text><Arrow x1={550} y1={180} x2={550} y2={109} /><path d="M645 87 H653 V310 H205 V257 H180" fill="none" stroke="#e9979f" /><text x="220" y="235">residual refinement ↻</text><text x="220" y="260">current state changes;</text><text x="220" y="283">memory sources stay distinct</text></MemoryFigure>;
}
