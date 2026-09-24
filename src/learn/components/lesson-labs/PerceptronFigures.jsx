import { useId, useState } from 'react';
import { perceptronData } from '../../data/perceptron-data.js';
import { activationNames, sigmoid, triangle } from '../../data/perceptron-models.js';
import { CurvePlot, fmt, Frame, PerceptronTable } from './PerceptronShared.jsx';
const seriesColors = ['#e8c36f', '#80c6b4', '#c49be2', '#e99b88', '#9bb9ec', '#cfcea0'];
export function DigitFigure() {
  const [label, setLabel] = useState(0),
    examples = Array.from({
      length: 10
    }, (_, digit) => perceptronData.digits.find(row => row.digit === digit)),
    selected = examples[label];
  return <Frame id="perceptron-digits-figure" title="An observed image becomes 64 input coordinates" kind="Observed UCI data"><div className="perceptron-actions">{examples.map(row => <button key={row.digit} aria-pressed={row.digit === label} onClick={() => setLabel(row.digit)}><span className="perceptron-thumbnail" aria-hidden="true">{row.pixels.map((value,i)=><span key={i} style={{background:`rgb(${value/16*255},${value/16*255},${value/16*255})`}}/>)}</span>Digit {row.digit}<small className="perceptron-source-id">ID {row.sourceId}</small></button>)}</div><p>Digit {selected.digit} · source ID {selected.sourceId}. Each selector uses the first retained source ID for that label.</p><div className="perceptron-grid-two"><div><div className="perceptron-digit" role="img" aria-label={`Digit ${selected.digit}, source ID ${selected.sourceId}. Actual 8 by 8 pixel counts; exact rows below.`}>{selected.pixels.map((value, i) => <span key={i} style={{
            background: `rgb(${value / 16 * 255},${value / 16 * 255},${value / 16 * 255})`
          }} />)}</div><p className="perceptron-note">Fixed grayscale: 0 = black, 16 = white. Gold outlines in the flattened view identify image row 0.</p><p>Image row 0: [{selected.pixels.slice(0, 8).join(', ')}]<br />→ input features pixel_0 through pixel_7.</p></div><div className="perceptron-pixels" aria-label="Flattened row-major vector">{selected.pixels.map((value, i) => <span className={i < 8 ? 'first-row' : ''} key={i}><small>p{i}</small>{value}</span>)}</div></div><PerceptronTable caption={`Exact 8 × 8 pixels for source ID ${selected.sourceId}; rows map left to right into the flattened vector`} headers={['Row', '0', '1', '2', '3', '4', '5', '6', '7']} rows={Array.from({
      length: 8
    }, (_, row) => [row, ...selected.pixels.slice(row * 8, (row + 1) * 8)])} /><div className="perceptron-flow"><div className="perceptron-node"><strong>280 training images</strong>280 × 64 input matrix<br />Divide each count by 16</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>32 hidden neurons</strong>280 × 32 activations<br />Same bias for every example</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>10 digit logits</strong>280 × 10 unrestricted scores<br />Compare with 280 labels</div></div><p className="perceptron-note">E. Alpaydin and C. Kaynak, UCI Optical Recognition of Handwritten Digits, CC BY 4.0. Row-major flattening preserves these values; a dense layer does not automatically know pixel adjacency. <a href="/learn-assets/perceptrons/digits-400.csv" download>Download the 400-image CSV</a> · <a href="/learn-assets/perceptrons/data-provenance.md">Selection, attribution and provenance</a>.</p></Frame>;
}
export function DigitComparisonFigure() {
  const id = useId(),
    [seed, setSeed] = useState(1),
    [zoom, setZoom] = useState(false),
    [compare, setCompare] = useState('relu'),
    runs = perceptronData.runs.filter(row => row.seed === seed),
    reference = runs.find(row => row.activation === 'sigmoid'),
    candidate = runs.find(row => row.activation === compare),
    byId = new Map(perceptronData.digits.map(row => [row.sourceId, row])),
    paired = perceptronData.splits.validationSourceIds.map((sourceId, i) => ({
      sourceId,
      actual: byId.get(sourceId).digit,
      a: reference.validationPredictions[i],
      b: candidate.validationPredictions[i]
    })).filter(row => row.a !== row.actual || row.b !== row.actual),
    xmin = zoom ? 116 : 0;
  return <Frame id="perceptron-comparison-figure" title="Matched starts, different training losses, tied validation counts" kind="Measured CPU results · recorded, not retrained here"><p>400 selected images; 280 train / 120 validation; split seed 22; 64 → 32 → 10; Adam 0.01; 200 full-batch updates; float32, one CPU thread. Each seed resets the initial affine parameters for all six activations.</p><div className="perceptron-actions"><label htmlFor={`${id}-seed`}>Recorded initialization seed</label><select id={`${id}-seed`} value={seed} onChange={event => setSeed(Number(event.target.value))}>{[1, 2, 3].map(value => <option key={value}>{value}</option>)}</select></div><CurvePlot title={`Training cross-entropy · recorded seed ${seed}`} curves={runs.map((row, i) => ({
      data: row.trace.map(point => [point.step, point.trainLoss]),
      color: seriesColors[i],
      dash: i === 3 ? '5 3' : i === 4 ? '2 3' : undefined
    }))} xDomain={[0, 200]} xTicks={[0, 50, 100, 150, 200]} yDomain={[.001, 3]} yTicks={[.001, .01, .1, 1, 3]} log yLabel="Mean cross-entropy (log scale)" xLabel="Actual update number" /><div className="perceptron-legend">{runs.map((row, i) => <span key={row.activation} style={{
        '--series-color': seriesColors[i]
      }}>{activationNames[row.activation]}</span>)}</div><p className="perceptron-note">Lines connect the six recorded observations at updates 0, 1, 10, 50, 100 and 200; the horizontal spacing is proportional to update number. Intermediate losses were not recorded. Log scale makes late differences visible while retaining the initial loss.</p><h4>Validation correct out of 120</h4><div className="perceptron-actions"><label><input type="checkbox" checked={zoom} onChange={event => setZoom(event.target.checked)} /> Zoom count axis to 116–120 (truncated axis)</label></div><p className="perceptron-note">Count scale: {xmin}–120. Equal counts have equal positions and bar lengths. One image is 0.833 percentage points.</p>{runs.map(row => <div className="perceptron-counts" key={row.activation}><span>{row.activation}</span><div className="perceptron-count-track" role="img" aria-label={`${activationNames[row.activation]}: ${row.correct} correct out of 120; count axis ${xmin} to 120`}><span style={{
          width: `${(row.correct - xmin) / (120 - xmin) * 100}%`
        }} /></div><strong>{row.correct}/120</strong></div>)}<PerceptronTable caption={`Measured final rows for seed ${seed}; validation is already used for comparison`} headers={['Activation', 'Training loss', 'Correct / 120', 'Mistakes']} rows={runs.map(row => [activationNames[row.activation], fmt(row.trainLoss), `${row.correct}/120`, 120 - row.correct])} /><details><summary>Inspect recorded intermediate losses and paired errors</summary><PerceptronTable caption={`All six sampled training losses, seed ${seed}`} headers={['Activation', '0', '1', '10', '50', '100', '200']} rows={runs.map(row => [row.activation, ...row.trace.map(point => fmt(point.trainLoss))])} /><div className="perceptron-actions"><label htmlFor={`${id}-pair`}>Compare sigmoid with</label><select id={`${id}-pair`} value={compare} onChange={event => setCompare(event.target.value)}>{runs.filter(row => row.activation !== 'sigmoid').map(row => <option key={row.activation} value={row.activation}>{activationNames[row.activation]}</option>)}</select></div><PerceptronTable caption="Union of wrong validation images, joined by stable source ID" headers={['Source ID', 'Actual digit', 'Sigmoid label', `${activationNames[compare]} label`]} rows={paired.map(row => [row.sourceId, row.actual, row.a, row.b])} /><p className="perceptron-note">These are retained class predictions, not new inference. No weights, probabilities or saliency were retained.</p></details><p className="perceptron-note">Recorded versions: Python {perceptronData.versions.python}, NumPy {perceptronData.versions.numpy}, PyTorch {perceptronData.versions.torch}, scikit-learn {perceptronData.versions.sklearn}. Counts and loss measure different things; ties are preserved.</p></Frame>;
}
export function SwiGluFigure() {
  const gates = [1, -1],
    values = [2, 3];
  return <Frame id="perceptron-swiglu-figure" title="SwiGLU multiplies two separately learned projections"><div className="perceptron-flow"><div className="perceptron-node"><strong>Same input x [d]</strong>Project through Wg [d × h]<br />and Wv [d × h]</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>Gate branch [h]</strong>Projected values (1, −1)<br />SiLU → (0.731059, −0.268941)</div><div className="perceptron-node"><strong>Value branch [h]</strong>Projected values (2, 3)<br />No activation on this branch</div></div><PerceptronTable caption="Coordinatewise multiplication joins corresponding entries from the two branches" headers={['Coordinate', 'Gate projection', 'SiLU(gate)', 'Value projection', 'Product']} rows={gates.map((gate, i) => [i, gate, fmt(gate * sigmoid(gate)), values[i], fmt(gate * sigmoid(gate) * values[i])])} /><div className="perceptron-flow"><div className="perceptron-node"><strong>Product h [h]</strong>(1.462117, −0.806824)<br />The second value changes sign</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>Output o [d]</strong>Multiply by Wo [h × d]<br />Output weights are unspecified</div></div><p className="perceptron-note">Numbers are projected values, not the raw input or probabilities. Only the gate branch passes through SiLU.</p></Frame>;
}
export function TriangleFigure() {
  const x = Array.from({
      length: 81
    }, (_, i) => -1 + i * .05),
    functions = [v => Math.max(0, v), v => -2 * Math.max(0, v - 1), v => Math.max(0, v - 2), triangle],
    titles = ['First ramp: ReLU(x)', 'Weighted delayed ramp: −2ReLU(x−1)', 'Restore slope: ReLU(x−2)', 'Sum: the triangular pulse'];
  return <Frame id="perceptron-triangle-figure" title="A new ramp changes the slope at each breakpoint"><div className="perceptron-grid-two">{functions.map((fn, i) => <CurvePlot key={titles[i]} title={titles[i]} curves={[{
        data: x.map(v => [v, fn(v)])
      }]} xDomain={[-1, 3]} xTicks={[-1, 0, 1, 2, 3]} yDomain={[-4, 3]} yTicks={[-4, -2, 0, 1, 3]} xLabel="Input x" yLabel={i === 3 ? 'Sum' : 'Signed contribution'} />)}</div><PerceptronTable caption="Exact values at the triangle's corners and midpoints" headers={['x', 'ReLU(x)', '−2ReLU(x−1)', 'ReLU(x−2)', 'Sum']} rows={[0, .5, 1, 1.5, 2].map(value => [value, ...functions.map(fn => fn(value))])} /><p>All four panels use the same axes. The summed slope is 0 before 0, +1 from 0 to 1, −1 from 1 to 2, and 0 after 2. The second contribution is negative because its weight is −2; each underlying ReLU output is nonnegative.</p></Frame>;
}
