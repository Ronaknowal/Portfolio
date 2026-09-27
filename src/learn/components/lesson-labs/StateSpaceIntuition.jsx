import { useState } from 'react';
import { NeuralNumber, NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { retainedTrace, impulseTrails, markedMemory, matrixWriteExample } from '../../data/state-space-intuition.js';

function StepChoices({ selected, onChange, count, label }) {
  return <div className="ssm-intuition-steps" role="group" aria-label={label}>
    {Array.from({ length: count }, (_, step) => <button key={step} type="button"
      aria-pressed={selected === step} onClick={() => onChange(step)}>Step {step}</button>)}
  </div>;
}

export function RetainWriteFigure() {
  const [retention, setRetention] = useState(.8), [step, setStep] = useState(1);
  const trace = retainedTrace(retention), current = trace[step];
  return <figure className="ssm-figure ssm-intuition" data-visual="retain-write">
    <figcaption>Keep some of the past, add some of the present</figcaption>
    <p>Constructed readings: <strong>5 → 0 → 0</strong>. Start with summary 0. The same rule is used at every step.</p>
    <NeuralNumber label="Fraction of old summary retained" value={retention} min={0} max={1} step={.01} onChange={setRetention} />
    <StepChoices selected={step} onChange={setStep} count={3} label="Inspect one update" />
    <div className="ssm-retain-arithmetic">
      <div><span>Carry from old summary</span><strong>{f(current.previous)} × {f(retention)} = {f(current.retained)}</strong>
        <div className="ssm-contribution-track"><span style={{ width: `${current.retained / 5 * 100}%` }} /></div></div>
      <div><span>Write from current reading</span><strong>{f(current.input)} × {f(1 - retention)} = {f(current.written)}</strong>
        <div className="ssm-contribution-track ssm-new-write"><span style={{ width: `${current.written / 5 * 100}%` }} /></div></div>
    </div>
    <p className="ssm-result">After step {step}: <strong>{f(current.retained)} + {f(current.written)} = {f(current.state)}</strong>.<br />Both bars share the scale 0–5 input units; a zero contribution has zero bar length.</p>
    <NeuralTable caption="The same readings, three different summaries" headers={['Step / reading', 'Latest only', 'Whole-history mean', 'Retain + write']} rows={trace.map(row => [`${row.step} / ${row.input}`, row.input, f(row.average), f(row.state)])} />
    <p>{retention === 1 ? 'At retention 1, writing has weight 0. A zero starting state stays zero even when the reading is 5.' : retention === 0 ? 'At retention 0, nothing old survives: the result is exactly the latest reading.' : `Retention ${f(retention)} leaves writing weight ${f(1 - retention)}. At the next zero reading, the old contribution fades; it is not erased all at once.`}</p>
    <button type="button" onClick={() => { setRetention(.8); setStep(1); }}>Reset retention example</button>
  </figure>;
}

export function ImpulseTrailsFigure() {
  const [step, setStep] = useState(2);
  const inputs = [2, 0, 1, 0], result = impulseTrails(inputs);
  return <figure className="ssm-figure ssm-intuition" data-visual="impulse-trails">
    <figcaption>Follow one input across a row; sum a column to get an output</figcaption>
    <p>Rule: keep half of the state, then add the input. A unit input leaves the trail <strong>{result.kernel.map(value => f(value)).join(' → ')}</strong>.</p>
    <StepChoices selected={step} onChange={setStep} count={4} label="Inspect output column" />
    <div className="ssm-trail-scroll" tabIndex={0} role="region" aria-label="Input contribution trails">
      <table className="ssm-trail-table"><thead><tr><th scope="col">Input arrival</th>{inputs.map((_, i) => <th key={i} scope="col" className={i === step ? 'is-current' : ''}>Output {i}</th>)}</tr></thead>
        <tbody>{result.contributions.map((row, birth) => <tr key={birth}><th scope="row">At {birth}: {inputs[birth]}</th>{row.map((value, i) => <td key={i} className={i === step ? 'is-current' : ''}>{value === null ? <span className="ssm-not-arrived">not yet</span> : <><span>{f(value)}</span><div className="ssm-trail-bar"><i style={{ width: `${value / 2 * 100}%` }} /></div></>}</td>)}</tr>)}</tbody>
        <tfoot><tr><th scope="row">Column sum</th>{result.outputs.map((value, i) => <td key={i} className={i === step ? 'is-current' : ''}>{f(value)}</td>)}</tr></tfoot>
      </table>
    </div>
    <p className="ssm-result">Output {step}: {result.contributions.slice(0, step + 1).map(row => f(row[step])).join(' + ')} = <strong>{f(result.outputs[step])}</strong>.</p>
    <p>Bars share 0–2 contribution units. “Not yet” means the input has not arrived; it cannot contribute to an earlier output. Zero means an arrived input contributes nothing.</p>
  </figure>;
}

export function MarkedMemoryFigure() {
  const rows = markedMemory(), constant = retainedTrace(.5, rows.map(row => row.input));
  return <figure className="ssm-figure ssm-intuition" data-visual="marked-memory">
    <figcaption>Remembering the last marked item is different from remembering a fixed age</figcaption>
    <div className="ssm-marked-events">{rows.map((row, i) => <div key={i} className={row.marked ? 'is-marked' : ''}><small>Step {i}</small><strong>{row.input}</strong><span>{row.marked ? 'Marked: replace' : 'Routine: retain'}</span></div>)}</div>
    <NeuralTable caption="Exact rules on the same available inputs and markers" headers={['Step', 'Last marked value', 'Two steps ago', 'Half-retention average']} rows={rows.map((row, i) => [i, row.selected, row.delayed, f(constant[i].state)])} />
    <p>The marked-value rule keeps 4 through both distractions, then replaces it with 6. A fixed delay instead returns the input from two steps earlier. The marker gates here are supplied; learning useful gates is the next step.</p>
  </figure>;
}

function TinyMatrix({ title, matrix }) {
  return <div className="ssm-tiny-matrix"><strong>{title}</strong><table aria-label={title}><tbody>{matrix.map((row, i) => <tr key={i}>{row.map((value, j) => <td key={j}>{f(value)}</td>)}</tr>)}</tbody></table></div>;
}

export function MatrixWriteFigure() {
  const data = matrixWriteExample();
  return <figure className="ssm-figure ssm-intuition" data-visual="matrix-write">
    <figcaption>One shared retention factor, one new write, then a read</figcaption>
    <p>Rows are two memory coordinates; columns are the two value features. Every matrix uses that same arrangement.</p>
    <div className="ssm-two">
      <div><h4>Carry the existing memory</h4><TinyMatrix title="Old state" matrix={data.old} /><p>Multiply every entry by {data.decay}.</p><TinyMatrix title="Retained state" matrix={data.retained} /></div>
      <div><h4>Write the new value [3, −1]</h4><p>Row 0 gets 0 × [3, −1].<br />Row 1 gets 1 × [3, −1].</p><TinyMatrix title="Outer-product write" matrix={data.write} /><p>The write weights [0, 1] specify which rows receive the value.</p></div>
    </div>
    <div className="ssm-matrix-sum"><span>Retained state + new write</span><span aria-hidden="true">↓</span><TinyMatrix title="New state" matrix={data.state} /></div>
    <p>Read weights [1, 1] add the rows: column 0 gives 1 + 3 = <strong>{f(data.read[0])}</strong>; column 1 gives .5 − 1 = <strong>{f(data.read[1])}</strong>.</p>
    <p className="ssm-result">Output = <strong>[{data.read.map(value => f(value)).join(', ')}]</strong>. Reading does not alter the stored matrix.</p>
  </figure>;
}
