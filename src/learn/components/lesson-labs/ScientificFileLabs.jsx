import { DataStepControls, DataPrediction } from "./DataLabControls.jsx";
import { useId, useState } from "react";
import { LessonTable } from "./LessonElements";
import { measurementCases, measurementFields, measurementModel, publicationTrace } from "../../data/scientific-file-models.js";



import './data-foundations.css';
import './scientific-concept-visuals.css';





const showValue = value => value === null ? 'None (missing)' : Number.isNaN(value) ? 'NaN (not finite)' : String(value);

export function FileSchemaLab() {
  const uid = useId();
  const [caseId, setCaseId] = useState('valid');
  const [step, setStep] = useState(0);
  const model = measurementModel(caseId);
  const stages = ['Text', 'Parsed fields', 'Typed records', 'Validation'];
  const blocked = step >= 2 && model.conversionErrors.length > 0;
  const errors = blocked ? model.conversionErrors : step === 3 ? model.validationErrors : [];
  return <section className="lesson-lab data-lab" data-lab="file-schema" aria-labelledby={`${uid}-title`}>
    <p className="lesson-eyebrow">FOLLOW MEANING THROUGH THE FILE</p>
    <h3 id={`${uid}-title`}>Which gate should stop this batch?</h3>
    <p>The sample ID <code>001</code> must stay a string. A blank temperature means missing; <code>0</code> is a real reading. Every row must use Celsius and a unique ID.</p>
    <div className="data-controls"><label>File variation<select value={caseId} onChange={e => { setCaseId(e.target.value); setStep(0); }}>
      {measurementCases.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
    </select></label></div>
    <DataPrediction key={caseId} id={`${uid}-prediction`}>Will it pass parsing, conversion and validation?</DataPrediction>
    <ol className="data-pipeline" aria-label="Import stages">{stages.map((name,index) => {
      const failed = index <= step && ((index >= 2 && model.conversionErrors.length > 0) || (index === 3 && model.validationErrors.length > 0));
      return <li key={name} aria-current={step === index ? 'step' : undefined} className={failed ? 'is-blocked' : index <= step ? 'is-reached' : ''}><span>{index + 1}</span>{name}{failed && <strong className="data-stage-status">Blocked</strong>}</li>;
    })}</ol>
    <div className="data-stage" aria-live="polite">
      {step === 0 && <><p><strong>The file is text, not yet measurements.</strong> Quotes keep the comma in <code>room,north</code> inside one field.</p><pre tabIndex={0} aria-label="CSV source">{model.csv}</pre></>}
      {step === 1 && <><p><strong>Parsing found four fields in each row.</strong> The empty field is the string <code>""</code>. Quotes used by CSV are removed; data inside them is retained.</p><LessonTable caption="Parsed strings — quotes here show the Python string type" headers={measurementFields} rows={model.rows.map(row => row.map(value => JSON.stringify(value)))} /></>}
      {step >= 2 && <>
        <p>{step === 2 ? 'Apply the schema: preserve the ID, map blank to None, and convert other temperatures to numbers. Conversion alone does not enforce units or uniqueness.' : 'Check the application contract across the whole batch. This importer rejects the batch if any record is invalid.'}</p>
        <LessonTable caption="Typed records — source values retain their meaning" headers={['sample_id · str', 'temperature · float or None', 'unit · str', 'site · str']} rows={model.typed.map(row => [JSON.stringify(row.id), row.conversionError ? 'Conversion failed' : showValue(row.value), row.unit, row.site])} />
      </>}
      {errors.length > 0 && <div className="data-verdict is-error"><strong>Stop. Do not publish this batch.</strong><ul>{errors.map(error => <li key={error}>{error}</li>)}</ul><p>{blocked ? 'Repair the numeric source field before checking domain rules.' : 'Parsing and conversion succeeded, but those steps cannot prove the measurement is valid.'}</p></div>}
      {step === 3 && model.valid && <p className="data-verdict"><strong>Ready for a checked round trip:</strong> 3 records, 2 measured values, 1 missing value. The average of measured temperatures is (18.5 + 0) / 2 = 9.25 °C.</p>}
    </div>
    <DataStepControls step={step} count={4} setStep={setStep} reset={() => setStep(0)} />
    <p className="lesson-note">Try a variation: why does NaN pass numeric conversion but fail validation? These fixed CSV fixtures are already parsed by the teaching model; this is not a general CSV reader. Verify the same cases with the Python importer below.</p>
  </section>;
}

function FileArtifact({ contents, name }) {
  if (contents === null) return <div className="sci-file-artifact sci-file-artifact--absent"><strong>{name}</strong><p>No file at this name</p></div>;
  const rows = contents.trim().split('\n').slice(1).filter(Boolean).map(line => line.split(','));
  return <div className="sci-file-artifact"><strong>{name}</strong><p><small>{contents === '' ? 'Empty file · header absent' : 'Header: sample_id, temperature'}</small></p><ol>{rows.map(([id, value]) => <li key={id}><code>{id}</code><strong>{value} °C</strong></li>)}</ol>{rows.length < 2 && <p className="sci-missing-slot">{2 - rows.length} expected record{rows.length === 0 ? 's' : ''} not written</p>}<small>{rows.length} of 2 expected records</small></div>;
}

export function FilePublicationLab() {
  const uid = useId();
  const [strategy, setStrategy] = useState('replace');
  const [fail, setFail] = useState(true);
  const [step, setStep] = useState(0);
  const trace = publicationTrace(strategy, fail);
  const state = trace[Math.min(step, trace.length - 1)];
  return <section className="lesson-lab data-lab" data-lab="file-publication" aria-labelledby={`${uid}-title`}>
    <p className="lesson-eyebrow">SEPARATE BUILDING FROM PUBLISHING</p>
    <h3 id={`${uid}-title`}>What would a new reader find after a failed write?</h3>
    <p>Replace two old readings, 18 and 20, with 22 and 24. Watch the published name separately from the file being built.</p>
    <div className="data-controls">
      <label>Write strategy<select value={strategy} onChange={e => { setStrategy(e.target.value); setStep(0); }}><option value="replace">Stage, validate, then replace</option><option value="direct">Write directly to destination</option></select></label>
      <label>Writer outcome<select value={fail ? 'fail' : 'success'} onChange={e => { setFail(e.target.value === 'fail'); setStep(0); }}><option value="fail">Fail after one record</option><option value="success">Finish successfully</option></select></label>
    </div>
    <DataPrediction key={`${strategy}-${fail}`} id={`${uid}-prediction`}>Will the published file be old, new, empty or partial?</DataPrediction>
    <div className="data-state-columns" aria-live="polite">
      <div className="data-state"><h4>Published path → measurements.csv</h4><p className="lesson-note">What a fresh reader opening this path sees</p><FileArtifact contents={state.destination} name="measurements.csv" /><details><summary>Inspect exact published text</summary><pre tabIndex={0}>{state.destination || '(empty file)'}</pre></details></div>
      <div className="data-state"><h4>Staging path → measurements.tmp</h4><p className="lesson-note">Unpublished work in the same directory</p><FileArtifact contents={state.temporary} name="measurements.tmp" /><details><summary>Inspect exact staging text</summary><pre tabIndex={0}>{state.temporary === null ? '(no staging file)' : state.temporary || '(empty staging file)'}</pre></details></div>
    </div>
    <p className="data-verdict" aria-live="polite"><strong>{state.label}.</strong> {state.note}</p>
    <DataStepControls step={step} count={trace.length} setStep={setStep} reset={() => setStep(0)} />
    <p className="lesson-note">Now switch strategies and repeat the same failure. This model covers ordinary writes and successful local replacement, not power loss, an already-open file handle, concurrent writers or remote storage.</p>
  </section>;
}
