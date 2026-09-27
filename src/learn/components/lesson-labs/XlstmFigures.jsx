import { useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable } from './NeuralLessonElements.jsx';
import { XFigure, XArrow, XNode, XLedger, XMatrix, XValues, XPlot, XCausalGrid, XChunkJoin, XImage, f, vec } from './XlstmPrimitives.jsx';
import { scalarScan, matrixScan, denseMatrixRead, chunkMatrixRead, gateLearning, recurrentStorage, sigmoid } from '../../data/xlstm-memory-models.js';
import { xlstmExamples as examples } from '../../data/xlstm-example-inputs.js';
export const workedScalar = [.2, -.6, .8].map((value, i) => ({
  value,
  writeLog: Math.log([1, 3, 9][i]),
  retention: .5
}));
export const workedMatrix = [[1, 0], [0, 1], [1, 0]].map((query, i) => ({
  query,
  key: [...query],
  value: [[2, -1], [0, 3], [4, 1]][i],
  writeLog: Math.log([1, 1, 2][i]),
  retention: .5
}));
export const chunkRows = examples.chunk.queries.map((query, i) => ({
  query,
  key: examples.chunk.keys[i],
  value: examples.chunk.values[i],
  writeLog: examples.chunk.write_logs[i],
  retention: Math.exp(examples.chunk.forget_logs[i])
}));
export function XOpeningFigure() {
  return <XFigure title="Rows update one bounded state; an explicit bank retains separate records" height={275}>{Array.from({
      length: 8
    }, (_, i) => <g key={i}><rect x={12 + i * 22} y="15" width="16" height="65" fill="#242424" stroke="#87bdf1" /><text x={20 + i * 22} y="101" textAnchor="middle">{i + 1}</text></g>)}<XArrow x1={197} y1={47} x2={265} y2={47} /><XNode x={265} y={25} width={145} lines={['recurrent state']} /><path d="M330 25 V8 H420 V90 H330 V69" fill="none" stroke="#e6bb60" /><XArrow x1={410} y1={47} x2={460} y2={47} /><XNode x={460} y={25} width={160} lines={['final classifier']} /><text x="12" y="135">Only the processed prefix has reached this state.</text><XNode x={10} y={190} width={125} lines={['query']} />{[0, 1, 2].map(i => <g key={i}><XNode x={260 + i * 120} y={170} width={105} height={70} lines={['retained', `record${i + 1}`]} /><XArrow x1={135} y1={212} x2={260 + i * 120} y2={170} /></g>)}</XFigure>;
}
export function OrdinaryCellFigure() {
  return <XFigure title="Two signed contributions form content before the output gate" height={225}><text x="10" y="26">old:0.5×0.8</text><rect x="145" y="10" width="160" height="22" fill="#e6bb60" /><text x="320" y="26">+0.4</text><text x="10" y="72">new:0.25×(−0.4)</text><rect x="105" y="56" width="40" height="22" fill="#e9979f" /><text x="320" y="72">−0.1</text><line x1="145" y1="5" x2="145" y2="83" stroke="white" /><XArrow x1={370} y1={26} x2={435} y2={60} /><XArrow x1={370} y1={72} x2={435} y2={60} /><XNode x={435} y={38} width={160} lines={['c=+0.3']} /><XArrow x1={515} y1={82} x2={515} y2={130} /><XNode x={430} y={130} width={170} lines={['tanh then ×o', 'h=o×0.291313']} /><text x="10" y="200">The hidden output needs an output-gate value; c and h are different.</text></XFigure>;
}
export function ScalarWorkedFigure() {
  const trace = scalarScan(workedScalar, .75, false);
  return <><XLedger rows={workedScalar} trace={trace} step={2} output={.75} scaled={false} /><NeuralTable caption="Complete worked scalar trace" headers={['Step', 'Content c', 'Mass n', 'Output h']} rows={trace.map((x, i) => [i + 1, f(x.cell, 8), f(x.normalizer, 8), f(x.hidden, 8)])} /></>;
}
export function ScalarMixingFigure() {
  return <XFigure title="Channels mix while generating gates; the memory update is pointwise" height={290}>{[0, 1, 2].map(i => <g key={i}><circle cx="50" cy={40 + i * 65} r="18" fill="#242424" stroke="#87bdf1" /><text x="50" y={45 + i * 65} textAnchor="middle">h{i + 1}</text>{[0, 1, 2].map(j => <XArrow key={j} x1={68} y1={40 + i * 65} x2={242} y2={40 + j * 65} color="#555" />)}<XNode x={245} y={18 + i * 65} width={160} lines={[`gates/candidate ${i + 1}`]} /><XArrow x1={405} y1={40 + i * 65} x2={460} y2={40 + i * 65} /><XNode x={460} y={18 + i * 65} width={170} lines={[`c${i + 1},n${i + 1} update`]} /></g>)}<text x="10" y="239">Crossing edges: schematic recurrent weights within one head.</text><text x="10" y="267">Separate heads restrict mixing; no detector meaning is assigned to a channel.</text></XFigure>;
}
export function ScalarScaleFigure() {
  const raw = scalarScan(workedScalar, .75, false),
    stable = scalarScan(workedScalar, .75);
  return <><XFigure title="Rescale both totals together at each time" height={245}>{raw.map((r, i) => <g key={i}><XNode x={10} y={12 + i * 72} width={180} lines={[`raw c=${f(r.cell, 5)}`, `raw n=${f(r.normalizer, 5)}`]} /><XArrow x1={190} y1={34 + i * 72} x2={325} y2={34 + i * 72} /><text x="216" y={24 + i * 72}>÷ {1 === i ? 3 : 2 === i ? 9 : 1}</text><XNode x={325} y={12 + i * 72} width={170} lines={[`c′=${f(stable[i].cell, 5)}`, `n′=${f(stable[i].normalizer, 5)}`]} /><XArrow x1={495} y1={34 + i * 72} x2={530} y2={34 + i * 72} /><text x="535" y={39 + i * 72}>{f(r.hidden, 6)}</text></g>)}</XFigure><XPlot title="Raw and stabilized output traces coincide" xLabel="step" yLabel="output" series={[{
      label: 'raw',
      color: '#e6bb60',
      points: raw.map((r, i) => [i + 1, r.hidden])
    }, {
      label: 'stabilized',
      color: '#87bdf1',
      dashed: true,
      points: stable.map((r, i) => [i + 1, r.hidden])
    }]} /><NeuralTable caption="Changing scale and effective gates" headers={['Step', 'm', 'write i′', 'retain f′']} rows={stable.map((r, i) => [i + 1, f(r.log_scale, 8), f(r.write, 8), f(r.retain, 8)])} /></>;
}
export function GateLearningFigure() {
  const r = gateLearning(0),
    next = gateLearning(r.next);
  return <><XPlot title="An actual derivative changes the weight of the newer observation" xLabel="write log-weight θ" yLabel="half squared loss" points={[{
      x: 0,
      y: r.loss,
      label: 'Before update',
      selected: true
    }, {
      x: r.next,
      y: next.loss,
      label: 'After update',
      selected: true,
      color: '#87bdf1'
    }]} series={[{
      label: 'analytic L(θ)',
      color: '#e6bb60',
      points: Array.from({
        length: 81
      }, (_, i) => {
        const x = -.4 + i * .01;
        return [x, gateLearning(x).loss];
      })
    }]} /><XFigure title="The target reaches the write gate through the normalized estimate" height={120}><XNode x={5} y={25} width={135} lines={['θ→exp(θ)']} /><XArrow x1={140} y1={47} x2={205} y2={47} /><XNode x={205} y={25} width={180} lines={['content / mass', 'y:0.466667→0.472403']} /><XArrow x1={385} y1={47} x2={450} y2={47} /><XNode x={450} y={25} width={180} lines={['target0.7 loss', '0.027222→0.025900']} /><text x="5" y="106">dL/dθ={f(r.gradient, 9)}; learning rate0.5; next θ={f(r.next, 9)}</text></XFigure></>;
}
export function OuterProductFigure() {
  return <><XFigure title="A key coordinate chooses a row; a value coordinate supplies its write" height={160}><XNode x={10} y={15} width={150} lines={['key k=[1,0]']} /><XNode x={10} y={100} width={150} lines={['value v=[2,−1]']} /><XArrow x1={160} y1={37} x2={260} y2={77} /><XArrow x1={160} y1={122} x2={260} y2={77} /><XNode x={260} y={55} width={160} lines={['outer product kvᵀ']} /><text x="445" y="54">C₁₁=1×2=2</text><text x="445" y="86">C₁₂=1×(−1)=−1</text><text x="445" y="118">C₂₁=C₂₂=0</text></XFigure><XMatrix title="The first matrix write" values={[[2, -1], [0, 0]]} /></>;
}
export function MatrixWorkedFigure() {
  const trace = matrixScan(workedMatrix, {
    stabilized: false
  });
  return <><div className="xl-fit-grid">{trace.map((r, i) => <div key={i}><XMatrix title={`After write ${i + 1}: query ${vec(workedMatrix[i].query)}`} values={r.cell} /><p>n={vec(r.normalizer)}, numerator={vec(r.numerator)}, denominator={f(r.denominator)}; read={vec(r.read)}.</p></div>)}</div><XFigure title="The third query selects C rows before dividing" height={205}><XNode x={10} y={18} width={185} lines={["C key row1",vec(trace[2].cell[0])]} /><XNode x={10} y={112} width={185} lines={["C key row2",vec(trace[2].cell[1])]} /><XArrow x1={195} y1={40} x2={320} y2={85} /><text x="219" y="33">×q1=1</text><XArrow x1={195} y1={134} x2={320} y2={85} color="#777" dashed /><text x="219" y="159">×q2=0</text><XNode x={320} y={63} width={180} lines={["sum = Cᵀq",vec(trace[2].numerator)]} /><XArrow x1={500} y1={85} x2={536} y2={85}/><text x="544" y="90">÷2.25</text><text x="12" y="190">The second row stays stored; this query gives it zero read coefficient.</text></XFigure><XFigure title="At the first address, older and newer writes still coexist" height={150}><XNode x={5} y={15} width={195} lines={['old write survives0.25', '0.25×[2,−1]']} /><XNode x={5} y={95} width={195} lines={['new write weight2', '2×[4,1]']} /><XArrow x1={200} y1={37} x2={285} y2={75} /><XArrow x1={200} y1={117} x2={285} y2={75} /><XNode x={285} y={53} width={340} lines={['[8.5,1.75] / (0.25+2)', 'read [3.777778,0.777778]']} /></XFigure></>;
}
export function SignedReadFigure() {
  return <XFigure title="Opposite keys can cancel normalization mass while adding value" height={285}><line x1="25" y1="85" x2="325" y2="85" stroke="#888" /><XArrow x1={175} y1={85} x2={55} y2={85} color="#e9979f" /><XArrow x1={175} y1={85} x2={295} y2={85} /><text x="30" y="58">k₂=−1, value−1</text><text x="205" y="58">k₁=+1, value2</text><XArrow x1={175} y1={113} x2={295} y2={113} color="#87bdf1" /><text x="200" y="140">query+1</text><text x="360" y="42">numerator: (+1)×2 + (−1)×(−1)=3</text><text x="360" y="76">signed mass: +1−1=0</text><text x="360" y="110">denominator: max(|0|,1)=1</text><text x="360" y="144">read=3</text><line x1="115" y1="213" x2="490" y2="213" stroke="#777" /><rect x="130" y="197" width="225" height="32" fill="#87bdf133" stroke="#87bdf1" /><circle cx="430" cy="213" r="7" fill="#e9979f" /><text x="125" y="252">−1</text><text x="345" y="252">2</text><text x="418" y="252">3</text><text x="165" y="276">convex interval of values</text><text x="410" y="276">matrix read</text></XFigure>;
}
export function MatrixFloorFigure() {
  return <><XFigure title="The unit floor must move into the same scale as the numerator" height={270}><text x="10" y="25">q=.5; k=.25; v=4; write log=2</text><XNode x={10} y={55} width={190} lines={['raw numerator=e²/2', 'raw floor=1']} /><XArrow x1={200} y1={77} x2={295} y2={77} /><text x="216" y="62">×e⁻²</text><XNode x={295} y={55} width={180} lines={['scaled numerator=.5', 'scaled floor=e⁻²']} /><XArrow x1={475} y1={77} x2={525} y2={77} /><text x="530" y="82">3.694528</text><XNode x={295} y={165} width={180} lines={['incorrect floor1', '.5 /1']} /><XArrow x1={385} y1={99} x2={385} y2={165} color="#e9979f" dashed /><XArrow x1={475} y1={187} x2={525} y2={187} color="#e9979f" /><text x="530" y="192">0.5</text><text x="10" y="243">Scaled signed mass=.125; the correct floor e⁻²≈.135335 is active.</text></XFigure></>;
}
export function CausalWorkedFigure() {
  const result = denseMatrixRead(chunkRows),
    t = 4;
  return <><XCausalGrid rows={chunkRows} coefficients={result.map(x => x.coefficients)} selected={t} /><NeuralTable caption="Selected query5: its source write excludes that write’s own retention" headers={['Source j', 'Query5·keyj', 'Log write', 'Later retention product', 'Scaled signed coefficient']} rows={chunkRows.slice(0, t + 1).map((r, j) => [j + 1, f(r.key.reduce((sum, v, k) => sum + v * chunkRows[t].query[k], 0), 7), f(r.writeLog, 7), f(chunkRows.slice(j + 1, t + 1).reduce((p, x) => p * x.retention, 1), 7), f(result[t].coefficients[j], 7)])} /></>;
}
export function ChunkWorkedFigure() {
  const result = chunkMatrixRead(chunkRows, 3, examples.incoming);
  return <><XChunkJoin result={result.outputs[4]} /><XMatrix title="Full matrix carried into the second chunk" values={result.boundaries[1].incoming.cell} /><p>The accompanying normalizer is {vec(result.boundaries[1].incoming.normalizer)}. This moderate-input calculation uses raw C,n; a stabilized boundary must also carry its log scale.</p></>;
}
export function ReaderBlockFigure() {
  return <XFigure title="A complete trainable row reader, including both identity bypasses" height={490}><XNode x={210} y={8} width={200} lines={['8 pixels → Linear8→16']} /><XArrow x1={310} y1={52} x2={310} y2={82} /><XNode x={210} y={82} width={200} lines={['pre RMSNorm ε=10⁻⁶']} /><XArrow x1={310} y1={126} x2={310} y2={156} /><XNode x={210} y={156} width={200} lines={['sequence cell →16']} /><XArrow x1={310} y1={200} x2={310} y2={228} /><circle cx="310" cy="242" r="14" fill="#191919" stroke="#e6bb60" /><text x="310" y="247" textAnchor="middle">+</text><path d="M310 67 H120 V242 H296" fill="none" stroke="#87bdf1" /><text x="18" y="150">embedded identity</text><XArrow x1={310} y1={256} x2={310} y2={282} /><XNode x={200} y={282} width={220} height={62} lines={['post RMSNorm', 'SiLU(16→32) × (16→32)', 'contract32→16']} /><XArrow x1={310} y1={344} x2={310} y2={372} /><circle cx="310" cy="386" r="14" fill="#191919" stroke="#e6bb60" /><text x="310" y="391" textAnchor="middle">+</text><path d="M310 268 H490 V386 H324" fill="none" stroke="#87bdf1" /><text x="495" y="323">residual identity</text><XArrow x1={310} y1={400} x2={310} y2={429} /><XNode x={205} y={429} width={210} lines={['16→10 class logits']} /><XArrow x1={415} y1={451} x2={460} y2={451} /><text x="465" y="445">final row only</text><text x="465" y="466">cross-entropy loss</text></XFigure>;
}
export function DigitScanFigure() {
  return <><div className="xl-images"><XImage title="UCI training source row3451; true label1" pixels={examples.worked.pixels} selectedRow={0} /></div><XFigure title="The selected spatial scan turns one real image into eight inputs" height={180}>{examples.worked.pixels.map((row, i) => <g key={i}><rect x={10 + i * 78} y="20" width="64" height="30" fill="#222" stroke="#777" />{row.map((v, j) => <rect key={j} x={10 + i * 78 + j * 8} y="20" width="8" height="30" fill={`rgb(${v / 16 * 255},${v / 16 * 255},${v / 16 * 255})`} />)}<text x={42 + i * 78} y="76" textAnchor="middle">row{i + 1}</text>{i < 7 && <XArrow x1={74 + i * 78} y1={35} x2={88 + i * 78} y2={35} />}</g>)}<text x="10" y="119">First row counts {vec(examples.worked.pixels[0])}</text><text x="10" y="148">Network input {vec(examples.worked.pixels[0].map(v => v / 16))}</text></XFigure><p>Source: E.Alpaydin, C.Kaynak and UCI, CC BY4.0. Row scanning is this experiment’s modeling choice, not an acquisition timestamp.</p></>;
}
export function RowDataRolesFigure() {
  return <XFigure title="Writer-group assessment stays outside parameter fitting and selection" height={285}><rect x="10" y="10" width="405" height="260" fill="none" stroke="#87bdf1" /><text x="20" y="35">3823 training images:30 writers</text>{[['1000 fitting', 70], ['300 validation', 135], ['2523 unused', 210]].map(([name, y]) => <XNode key={name} x={25} y={y} width={155} lines={[name]} />)}<XArrow x1={180} y1={92} x2={240} y2={92} /><XNode x={240} y={70} width={155} lines={['learn parameters']} /><XArrow x1={180} y1={157} x2={240} y2={157} /><XNode x={240} y={135} width={155} lines={['select checkpoint']} /><rect x="435" y="10" width="195" height="260" fill="none" stroke="#e6bb60" /><text x="448" y="37">1797 test images</text><text x="448" y="63">13 separate writers</text><XNode x={450} y={135} width={170} lines={['measure once']} /><XArrow x1={395} y1={157} x2={450} y2={157} /><text x="448" y="235">No per-writer IDs</text><text x="448" y="256">in internal roles</text></XFigure>;
}
export function RowMetricsFigure() {
  return <><XFigure title="All six fits: clean and reversed-row error counts" height={360} description="Gold circles: clean; blue diamonds: reversed rows. Every test denominator is1797. Parameter counts and selected epochs are retained below.">{examples.metrics.map((r, i) => <g key={r.id}><text x="5" y={37 + i * 47}>{r.kind} seed{r.seed}</text><line x1={145 + r.testErrors / 1200 * 320} y1={32 + i * 47} x2={145 + r.reverseErrors / 1200 * 320} y2={32 + i * 47} stroke="#555" /><circle cx={145 + r.testErrors / 1200 * 320} cy={32 + i * 47} r="5" fill="#e6bb60" /><path d={`M${145 + r.reverseErrors / 1200 * 320},${25 + i * 47}l7,7 -7,7 -7,-7Z`} fill="#87bdf1" /><text x="485" y={36 + i * 47}>{r.testErrors} / {r.reverseErrors}</text></g>)}<line x1="145" y1="310" x2="465" y2="310" stroke="#888" />{[0, 300, 600, 900, 1200].map(v => <text key={v} x={145 + v / 1200 * 320} y="339" textAnchor="middle">{v}</text>)}</XFigure><NeuralTable caption="Predeclared six runs; clean validation chooses the epoch" headers={['Model/seed', 'Parameters', 'Epoch', 'Validation errors/300']} rows={examples.metrics.map(r => [r.kind + '/' + r.seed, r.parameters, r.epoch, r.validationErrors])} /></>;
}
export function WorkedDigitTraceFigure() {
  const w = examples.worked;
  return <><div className="xl-images"><XImage title="Original source3451" pixels={w.pixels} /><XImage title="Original rows6–8 set to zero" pixels={w.pixels.map((row, i) => i >= 5 ? row.map(() => 0) : row)} /></div><XPlot title="Seed19 scalar model: only the changed suffix can alter the computation" xLabel="processed image rows" yLabel="class1 model score" series={[{
      label: 'original',
      color: '#e6bb60',
      points: w.clean.probabilities.map((p, i) => [i + 1, p[1]])
    }, {
      label: 'rows6–8 zero',
      color: '#87bdf1',
      points: w.bottomZero.probabilities.map((p, i) => [i + 1, p[1]])
    }]} /><NeuralTable caption="Actual prefixes and learned channel0 state" headers={['Rows', 'Original class', 'Edited class', 'h₀', 'c′₀', 'n′₀', 'm₀']} rows={w.channel0.map((r, i) => [i + 1, w.clean.predictions[i], w.bottomZero.predictions[i], ...r.map(v => f(v, 6))])} /></>;
}
export function VersionBlocksFigure() {
  return <>
{[['Original sLSTM', 'pre LayerNorm', 'sLSTM + per-head norm', 'pre LayerNorm', 'expanded gated MLP → contract'], ['xLSTM-7B', 'pre RMSNorm', 'mLSTM within model width', 'post RMSNorm', 'SwiGLU expansion → contract']].map(([name, norm, memory, post, ffn]) => <XFigure key={name} title={name + ': two residual additions'} width={640} height={365}>
<XNode x={35} y={65} width={150} lines={[norm]} /><XArrow x1={185} y1={87} x2={220} y2={87} /><XNode x={220} y={65} width={250} lines={[memory]} /><XArrow x1={470} y1={87} x2={549} y2={87} /><circle cx="562" cy="87" r="13" fill="#222" stroke="#e6bb60" /><text x="562" y="92" textAnchor="middle">+</text><path d="M15 87 H35 M15 87 V25 H562 V74" fill="none" stroke="#87bdf1" /><text x="170" y="19">unchanged input bypass</text><path d="M562 100 V163 H15 V257 H35 M15 257 V196 H562 V244" fill="none" stroke="#87bdf1" /><text x="180" y="190">first residual bypass</text><XNode x={35} y={235} width={150} lines={[post]} /><XArrow x1={185} y1={257} x2={220} y2={257} /><XNode x={220} y={235} width={270} lines={[ffn]} /><XArrow x1={490} y1={257} x2={549} y2={257} /><circle cx="562" cy="257" r="13" fill="#222" stroke="#e6bb60" /><text x="562" y="262" textAnchor="middle">+</text><XArrow x1={562} y1={270} x2={562} y2={318} /><text x="12" y="346">Expansion belongs to the separate feed-forward branch after memory.</text></XFigure>)}
<XFigure title="Original mLSTM: expansion precedes the memory branch" width={640} height={415}>
<XNode x={25} y={55} width={155} lines={['pre LayerNorm']} /><XArrow x1={180} y1={77} x2={220} y2={77} /><XNode x={220} y={55} width={165} lines={['expanded projection']} /><XArrow x1={385} y1={77} x2={435} y2={77} /><circle cx="435" cy="77" r="3" fill="#ddd" /><XArrow x1={435} y1={77} x2={435} y2={127} /><XNode x={350} y={127} width={180} lines={['causal convolution', 'then Q/K projections']} /><path d="M435 77 H575 V213 H530" fill="none" stroke="#87bdf1" /><text x="545" y="158">V path</text><XArrow x1={435} y1={171} x2={435} y2={191} /><XNode x={350} y={191} width={180} lines={['matrix read + norm']} /><XArrow x1={435} y1={235} x2={435} y2={269} /><XNode x={330} y={269} width={210} lines={['output gate + contract']} /><XArrow x1={435} y1={313} x2={435} y2={345} /><circle cx="435" cy="358" r="13" fill="#222" stroke="#e6bb60" /><text x="435" y="363" textAnchor="middle">+</text><path d="M10 77 H25 M10 77 V358 H422" fill="none" stroke="#87bdf1" /><text x="22" y="339">input bypass</text><text x="18" y="402">Value information bypasses the local convolution; expanded work precedes C.</text></XFigure></>;
}
export function SigmoidVariantFigure() {
  const [a, setA] = useState(-3),
    exp = Math.exp(a),
    sig = sigmoid(a),
    raw = [exp, 2 * exp],
    normalized = raw.map(x => x / Math.sqrt((raw[0] ** 2 + raw[1] ** 2) / 2 + 1e-6)),
    other = [sig, 2 * sig],
    otherNormalized = other.map(x => x / Math.sqrt((other[0] ** 2 + other[1] ** 2) / 2 + 1e-6));
  return <><NeuralNumber label="Variant write preactivation" value={a} min={-12} max={6} onChange={setA} /><XFigure title="A different write activation changes the operator" height={190}><XNode x={10} y={15} width={170} lines={['exponential write', f(exp, 8)]} /><XNode x={10} y={109} width={170} lines={['sigmoid write', f(sig, 8)]} /><XArrow x1={180} y1={37} x2={245} y2={37} /><XArrow x1={180} y1={131} x2={245} y2={131} /><XNode x={245} y={15} width={375} lines={['illustrative raw two-coordinate signal', vec(raw)]} /><XNode x={245} y={109} width={375} lines={['illustrative raw two-coordinate signal', vec(other)]} /></XFigure><XValues caption="Illustrative RMS response with epsilon10⁻⁶; unit learned scale" rows={[["Normalized exp-scaled signal", vec(normalized)], ["Normalized sigmoid-scaled signal", vec(otherNormalized)]]} /><p>This isolates activation and RMS scale sensitivity on [1,2]; it is not a conversion of a fitted checkpoint or a claim of full operator equivalence. The sigmoid variant removes n,m and uses its stated unnormalized matrix read.</p></>;
}
export function StateAccountingFigure() {
  const [layers, setLayers] = useState(32),
    [heads, setHeads] = useState(8),
    [kd, setKd] = useState(256),
    [vd, setVd] = useState(512),
    [length, setLength] = useState(8192),
    [kvHeads, setKvHeads] = useState(8),
    [headWidth, setHeadWidth] = useState(128),
    r = recurrentStorage(layers, heads, kd, vd),
    cache = BigInt(layers) * BigInt(kvHeads) * BigInt(headWidth) * BigInt(length) * 2n * 4n;
  return <><div className="neural-controls">{[['State layers', layers, setLayers, 128], ['Matrix heads', heads, setHeads, 128], ['Key width', kd, setKd, 2048], ['Value width', vd, setVd, 2048], ['Attention KV heads', kvHeads, setKvHeads, 128], ['Attention K/V width', headWidth, setHeadWidth, 2048], ['Processed / cached tokens', length, setLength, 1000000]].map(([label, value, onChange, max]) => <NeuralNumber key={label} label={label} value={value} min={1} max={max} integer onChange={onChange} />)}</div><XFigure title="Exact persistent allocations under the entered float32 configurations" height={190}><XNode x={10} y={20} width={180} height={100} lines={['C: key×value', r.matrixBytes.toLocaleString() + ' bytes']} /><XNode x={215} y={20} width={180} height={100} lines={['n: key vector', r.normalizerBytes.toLocaleString() + ' bytes']} /><XNode x={420} y={20} width={205} height={100} lines={['m: one scalar/head', r.scaleBytes.toLocaleString() + ' bytes']} /><text x="10" y="160">C,n,m total: {r.totalBytes.toLocaleString()} bytes; length does not enter.</text></XFigure><XValues caption="Allocated state arithmetic, not measured device memory" rows={[["Recurrent state", `${r.totalBytes.toLocaleString()} bytes; ${f(Number(r.totalBytes) / 1048576, 7)} MiB`], ["Entered attention K/V cache", `${cache.toLocaleString()} bytes; ${f(Number(cache) / 1048576, 7)} MiB`], ["Cache formula", 'layers × KV heads × head width × tokens ×2 ×4 bytes']]} /><XPlot title="Length changes cache bytes while recurrent dimensions stay fixed" xLabel="tokens" yLabel="MiB" series={[{
      label: 'entered recurrent state',
      color: '#e6bb60',
      points: [[0, Number(r.totalBytes) / 1048576], [length, Number(r.totalBytes) / 1048576]]
    }, {
      label: 'entered attention K/V',
      color: '#87bdf1',
      points: [[0, 0], [length, Number(cache) / 1048576]]
    }]} /></>;
}
export function ApplicationFlowsFigure() {
  return <><XFigure title="Image directions change available spatial context" height={150}>{[0, 1, 2, 3].map(i => <XNode key={i} x={10 + i * 155} y={20} width={125} lines={[`patch${i + 1}`]} />)}{[0, 1, 2].map(i => <g key={i}><XArrow x1={135 + i * 155} y1={42} x2={165 + i * 155} y2={42} /><XArrow x1={165 + i * 155} y1={95} x2={135 + i * 155} y2={95} color="#87bdf1" /></g>)}<text x="10" y="135">Odd block →; even block ←. The complete image exists before classification.</text></XFigure><XFigure title="Forecast presence distinguishes an observed zero from an unknown value" height={205}>{[['observed0', 'mask1'], ['observed2', 'mask1'], ['missing0', 'mask0'], ['future0', 'mask0']].map((lines, i) => <g key={i}><XNode x={5 + i * 160} y={20} width={145} height={65} lines={lines} />{i < 3 && <XArrow x1={150 + i * 160} y1={52} x2={165 + i * 160} y2={52} />}</g>)}<XArrow x1={555} y1={85} x2={555} y2={135} /><XNode x={445} y={135} width={190} lines={['future quantiles']} /><text x="10" y="128">Training includes contiguous missing patches.</text><text x="10" y="174">Presence is supplied alongside normalized patch values.</text></XFigure><XFigure title="xLSTM-Mixer scans encoded variables after initial forecasts" height={290}>{['variate A', 'variate B', 'variate C'].map((name, i) => <g key={name}><XNode x={10 + i * 215} y={15} width={185} lines={[name + ' history']} /><XArrow x1={102 + i * 215} y1={59} x2={102 + i * 215} y2={91} /><XNode x={10 + i * 215} y={91} width={185} lines={['shared linear forecast']} /><XArrow x1={102 + i * 215} y1={135} x2={102 + i * 215} y2={178} /><XNode x={10 + i * 215} y={178} width={185} lines={['encoded variate token']} />{i < 2 && <XArrow x1={195 + i * 215} y1={200} x2={225 + i * 215} y2={200} />}</g>)}<text x="10" y="256">Recurrence across variates mixes their forecast representations.</text><text x="10" y="280">This is a separate forecasting architecture, not TiRex missing-patch inference.</text></XFigure><XFigure title="An action precedes the reward it will cause" height={190}><XNode x={5} y={20} width={185} height={70} lines={['available observation', 'desired return', 'previous rewards']} /><XArrow x1={190} y1={55} x2={245} y2={55} /><XNode x={245} y={33} width={125} lines={['recurrent state']} /><XArrow x1={370} y1={55} x2={415} y2={55} /><XNode x={415} y={33} width={95} lines={['action t']} /><XArrow x1={510} y1={55} x2={545} y2={55} /><XNode x={545} y={33} width={90} lines={['reward t']} /><path d="M590 77 V127 H160" fill="none" stroke="#87bdf1" /><text x="165" y="149">Reward enters a later decision, never its own earlier action.</text><text x="10" y="178">Reset complete state at an independent episode boundary.</text></XFigure></>;
}
