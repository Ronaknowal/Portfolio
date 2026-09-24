import { DataStepControls } from "./DataLabControls.jsx";
import { useId, useState } from "react";
import { LessonTable } from "./LessonElements";
import { joinModel, transactionTrace } from "../../data/sql-models.js";
import './data-foundations.css';
import './scientific-concept-visuals.css';
export function SqlJoinLab() {
  const uid = useId();
  const [join, setJoin] = useState('left');
  const [duplicate, setDuplicate] = useState(false);
  const [cutoff, setCutoff] = useState(10);
  const [placement, setPlacement] = useState('on');
  const [selected, setSelected] = useState(0);
  const model = joinModel({
    join,
    duplicate,
    cutoff,
    placement
  });
  const current = model.pairs[Math.min(selected, model.pairs.length - 1)];
  const change = fn => e => {
    fn(e.target.value);
    setSelected(0);
  };
  return <section className="lesson-lab data-lab" data-lab="sql-join" aria-labelledby={`${uid}-title`}>
    <p className="lesson-eyebrow">TRACE THE ROWS, THEN COUNT THEM</p>
    <h3 id={`${uid}-title`}>Why did three sensors become four result rows?</h3>
    <p>A has two readings, B has one reading whose value is NULL, and C has no reading. Inspect the output count while changing a control, then select an output row to find its sources.</p>
    <div className="data-controls">
      <label>Join type<select value={join} onChange={change(setJoin)}><option value="left">LEFT JOIN: preserve sensors</option><option value="inner">INNER JOIN: matches only</option></select></label>
      <label>Include readings through minute<select value={cutoff} onChange={change(value => setCutoff(Number(value)))}><option value={10}>10</option><option value={0}>0</option></select></label>
      <label>Time condition lives in<select value={placement} onChange={change(setPlacement)}><option value="on">ON: decide which readings match</option><option value="where">WHERE: filter the joined rows</option></select></label>
      <label>Sensor keys<select value={duplicate ? 'duplicate' : 'unique'} onChange={change(value => setDuplicate(value === 'duplicate'))}><option value="unique">Unique: A, B, C</option><option value="duplicate">Staging error: another A</option></select></label>
    </div>
    <div className="data-source-columns">
      <div><h4>Source 1 · sensors</h4><ul className="data-source-rows">{model.sensors.map((sensor, index) => <li key={index} className={current?.sensorIndex === index ? 'is-linked' : ''}><code>{sensor.id}</code><span>{sensor.room}</span><small>{current?.sensorIndex === index ? '→ selected source' : `source row ${index + 1}`}</small></li>)}</ul></div>
      <div><h4>Source 2 · readings</h4><ul className="data-source-rows">{model.readings.map(reading => <li key={reading.id} className={current?.reading?.id === reading.id ? 'is-linked' : ''}><code>r{reading.id} · {reading.sensor}</code><span>{reading.value === null ? 'NULL' : `${reading.value} °C`}</span><small>{reading.minute} min{current?.reading?.id === reading.id ? ' → selected source' : ''}</small></li>)}</ul></div>
    </div>
    <p className="data-row-count" aria-live="polite"><strong>{model.pairs.length} output rows</strong> from {model.sensors.length} sensor rows. {duplicate ? 'The extra A repeats every match for A; this staging table has no enforced key.' : 'The primary sensor key is unique, but readings.sensor_id is allowed to repeat.'}</p>
    <div className="data-lineage" aria-label="Select a result row to trace">{model.pairs.map((pair, index) => <button key={`${pair.sensorIndex}-${pair.reading?.id ?? 'null'}`} type="button" aria-pressed={selected === index} onClick={() => setSelected(index)}><span>Row {index + 1}</span><strong>{pair.sensor.id} · {pair.sensor.room}</strong><span>+ {pair.reading ? `r${pair.reading.id}: ${pair.reading.value ?? 'NULL'}` : 'no match → NULL fields'}</span></button>)}</div>
    <p className="data-verdict" aria-live="polite">{current?.reading ? `This row pairs sensor source ${current.sensorIndex + 1} with reading r${current.reading.id}. Matching uses the sensor ID, not the displayed temperature.` : 'C has no matching reading. LEFT JOIN creates one row with NULL reading fields so the sensor is still represented.'} {placement === 'where' && 'The time predicate in WHERE removes null-extended rows because NULL <= cutoff is not true.'}</p>
    <LessonTable caption="Aggregate the result by sensor ID" headers={['Sensor', 'Rows', 'Readings', 'Measured', 'Mean']} rows={model.grouped.map(group => [group.id, group.rows, group.readings, group.measured, group.mean ?? 'NULL'])} />
    <p className="data-aggregate-key"><strong>Rows</strong> = <code>COUNT(*)</code>, every joined row. <strong>Readings</strong> = <code>COUNT(reading_id)</code>, matched observations. <strong>Measured</strong> = <code>COUNT(value)</code>, non-NULL values. <strong>Mean</strong> = <code>AVG(value)</code>, averaging only those measured values.</p>
    <button className="data-reset" type="button" onClick={() => {
      setJoin('left');
      setDuplicate(false);
      setCutoff(10);
      setPlacement('on');
      setSelected(0);
    }}>Reset join</button>
    <p className="lesson-note">Compare ON versus WHERE with the cutoff fixed. Then add the duplicate A: an unchanged mean can hide duplicated training rows. This fixed-data relational model does not parse SQL; the runnable SQLite queries below verify its results.</p>
  </section>;
}
function CreditState({
  title,
  values
}) {
  return <div className="data-state"><h4>{title}</h4>{values.map((value, index) => <div className="data-credit" key={index}><span>Account {index === 0 ? 'A' : 'B'}</span><div className="data-credit-track"><i style={{
          width: `${value * 10}%`
        }} /></div><strong>{value}</strong></div>)}<p>Total: <strong>{values[0] + values[1]} credits</strong></p></div>;
}
export function SqlTransactionLab() {
  const uid = useId();
  const [atomic, setAtomic] = useState(true);
  const [fail, setFail] = useState(true);
  const [step, setStep] = useState(0);
  const trace = transactionTrace({
    atomic,
    fail
  });
  const state = trace[step];
  return <section className="lesson-lab data-lab" data-lab="sql-transaction" aria-labelledby={`${uid}-title`}>
    <p className="lesson-eyebrow">ONE CHANGE, TWO STATEMENTS</p>
    <h3 id={`${uid}-title`}>Can a failed transfer lose two credits?</h3>
    <p>Account A starts with 6 credits and B with 4. Transfer 2 from A to B. The total should stay 10; nonnegative balances alone do not enforce that rule.</p>
    <div className="data-controls">
      <label>Transaction boundary<select value={atomic ? 'atomic' : 'separate'} onChange={e => {
          setAtomic(e.target.value === 'atomic');
          setStep(0);
        }}><option value="atomic">One explicit transaction</option><option value="separate">Commit each statement separately</option></select></label>
      <label>Second statement<select value={fail ? 'fail' : 'success'} onChange={e => {
          setFail(e.target.value === 'fail');
          setStep(0);
        }}><option value="fail">Bug: invalid negative balance</option><option value="success">Correct credit of 2</option></select></label>
    </div>
    <p className="lesson-note">Change the transaction boundary and failure point. Step through both account balances to see how rollback changes the final state.</p>
    <div className="data-state-columns" aria-live="polite"><CreditState title="Writer's current view" values={state.writer} /><CreditState title="Independently committed state" values={state.committed} /></div>
    <p className="data-verdict" aria-live="polite"><strong>{state.label}.</strong> {state.note}</p>
    <DataStepControls step={step} count={trace.length} setStep={setStep} reset={() => setStep(0)} />
    <p className="lesson-note">Compare both boundaries at the debit step and after failure. This model separates pending and committed values; real concurrent readers may block or retain snapshots depending on the database and isolation settings. It is not a locking simulator.</p>
  </section>;
}
