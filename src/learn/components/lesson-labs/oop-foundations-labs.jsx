import { useState } from "react";
import { CodeBlock } from "../content";
import { initialObjectState, methodCallFrames, initialLookupState, lookupValues, lookupAction, readingCandidates, validateReading, compositionFrames } from "../../data/oop-foundations-model";
import "./oop-foundations.css";

const valuesText = values => `[${values.join(", ")}]`;

function StepControls({ step, setStep, last = 4, reset, extra }) {
  return <div className="oop-controls">
    <button type="button" onClick={() => setStep(step - 1)} disabled={step === 0}>Back one step</button>
    <button type="button" onClick={() => setStep(step + 1)} disabled={step === last}>Step forward</button>
    <button type="button" onClick={reset}>Reset investigation</button>
    {extra}
  </div>;
}

function Arrow({ children, returning = false }) {
  return <div className={`oop-arrow${returning ? " oop-arrow--return" : ""}`}><span aria-hidden="true" className="oop-arrow__wide">{returning ? "←" : "→"}</span><span aria-hidden="true" className="oop-arrow__narrow">↓</span><small>{children}</small></div>;
}

export function OopBindingLab() {
  const [state, setState] = useState(initialObjectState);
  const [receiver, setReceiver] = useState("morning");
  const [value, setValue] = useState(18);
  const [step, setStep] = useState(0);
  const frames = methodCallFrames(state, receiver, value);
  const frame = frames[step];
  const change = (setter, value) => { setter(value); setStep(0); };
  const reset = () => { setState(initialObjectState()); setReceiver("morning"); setValue(18); setStep(0); };
  return <section className="oop-lab" aria-labelledby="oop-binding-title" data-oop-lab="binding">
    <p className="lesson-eyebrow">INVESTIGATE · WHICH OBJECT CHANGES?</p>
    <h3 id="oop-binding-title">Follow the receiver into self</h3>
    <p>There are two log objects and two lists. The names <code>morning</code> and <code>alias</code> both point to A. Predict which list changes, then follow the four stages of one method call.</p>
    <div className="oop-controls oop-controls--inputs">
      <label>Receiving name<select aria-label="Receiving name" value={receiver} onChange={e => change(setReceiver, e.target.value)}><option>morning</option><option>evening</option><option>alias</option></select></label>
      <label>Reading to add<select aria-label="Reading to add" value={value} onChange={e => change(setValue, Number(e.target.value))}>{[18, 24, 30].map(number => <option key={number} value={number}>{number}</option>)}</select></label>
    </div>
    <div className="oop-reference-map" aria-label="Names refer to log objects; each log refers to its own list">
      {["A", "B"].map(id => <div className={`oop-reference-row ${step > 0 && frame.object === id ? "oop-reference-row--selected" : ""}`} key={id}>
        <div className="oop-names"><span className="oop-label">NAMES</span>{(id === "A" ? ["morning", "alias"] : ["evening"]).map(name => <code key={name} className={name === receiver ? "oop-selected-name" : ""}>{name}</code>)}</div>
        <span className="oop-reference-arrow" aria-hidden="true">→</span>
        <div className="oop-object"><span className="oop-label">LOG OBJECT {id}</span><strong>ReadingLog</strong><div className="oop-owned-list"><code>values</code><span aria-hidden="true"> → </span><span>list {id} <strong>{valuesText(frame.state[id])}</strong></span></div></div>
      </div>)}
    </div>
    <div className="oop-call-strip">
      <div className={`oop-node ${step >= 2 ? "oop-node--active" : ""}`}><span className="oop-label">BOUND METHOD</span><code>{receiver}.add</code><p>function: <code>ReadingLog.add</code><br />receiver: object {frame.object}</p></div>
      <Arrow>call with {value}</Arrow>
      <div className={`oop-node ${step >= 3 ? "oop-node--active" : ""}`}><span className="oop-label">LOCAL NAMES IN THIS CALL</span><code>self → object {frame.object}</code><br /><code>value → {value}</code><p>The same log, reached inside the function.</p></div>
    </div>
    <p className="oop-command"><code>{step >= 3 ? `ReadingLog.add(${receiver}, ${value})` : `${receiver}.add(${value})`}</code>{step >= 3 && <span> equivalent call for this ordinary instance method</span>}</p>
    <StepControls step={step} setStep={setStep} reset={reset} extra={<button type="button" disabled={step !== 4} onClick={() => { setState(frames[4].state); setStep(0); }}>Keep result; prepare another call</button>} />
    <div className="oop-feedback" aria-live="polite" aria-atomic="true"><strong>{frame.title}</strong><p>{frame.note}</p></div>
    <details className="oop-deeper"><summary>Transfer: add through alias, then through evening</summary><p>Keep the first result. Select alias and add 24; then keep that result and add 30 through evening. Before each call, identify self. The final lists should be A = [18, 24] and B = [30]. Alias adds no new object.</p></details>
    <p className="lesson-note">Model boundary: these are the fixed methods shown in the example, not a Python interpreter. Boxes identify objects; their area does not measure memory. Changing a selector starts a fresh call from the last kept state. Back reverses the displayed trace; it is not a Python undo operation.</p>
  </section>;
}

export function OopLookupLab() {
  const [state, setState] = useState(() => initialLookupState());
  const [receiver, setReceiver] = useState("A");
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState("Predict: if A appends 18, what will B read?");
  const chosen = lookupValues(state, receiver);
  const act = action => {
    setHistory([...history, state]);
    setState(lookupAction(state, receiver, action));
    setMessage(action === "assign"
      ? `${receiver.toLowerCase()}.values = [99] creates an instance attribute on object ${receiver}. The class list and the other object's attribute do not change.`
      : `${receiver.toLowerCase()}.values was found on the ${chosen.source}. append(18) mutates list ${chosen.listId}; it does not assign a new attribute.`);
  };
  const mode = nextMode => { setState(initialLookupState(nextMode)); setHistory([]); setMessage(nextMode === "class" ? "Reset: both objects must look up values on the class." : "Reset: __init__ has created one values list per instance."); };
  return <section className="oop-lab" aria-labelledby="oop-lookup-title" data-oop-lab="lookup">
    <p className="lesson-eyebrow">INVESTIGATE · SHARED STATE</p><h3 id="oop-lookup-title">A lookup is different from an assignment</h3>
    <p>Start with the class-list bug. Append through A, inspect B, then assign a new list through B. Which arrow changes on assignment? Compare with the per-instance version.</p>
    <div className="oop-controls oop-controls--inputs">
      <label>Where lists begin<select aria-label="Where lists begin" value={state.mode} onChange={e => mode(e.target.value)}><option value="class">One class list · bug for independent logs</option><option value="instance">One list per instance · intended design</option></select></label>
      <label>Receiving object<select aria-label="Receiving object" value={receiver} onChange={e => setReceiver(e.target.value)}><option value="A">a · object A</option><option value="B">b · object B</option></select></label>
    </div>
    <CodeBlock language="python">{state.mode === "class" ? "class Log:\n    values = []\n\na = Log()\nb = Log()" : "class Log:\n    def __init__(self):\n        self.values = []\n\na = Log()\nb = Log()"}</CodeBlock>
    <div className="oop-lookup-map">
      <div className="oop-lookup-map__instances">{["A", "B"].map(id => { const found = lookupValues(state, id); return <div className={`oop-node ${receiver === id ? "oop-node--active" : ""}`} key={id}><span className="oop-label">OBJECT {id} · name {id.toLowerCase()}</span><p>Own <code>values</code>: {found.source === "instance" ? <><strong>list {id}</strong> {valuesText(found.values)}</> : <strong>absent</strong>}</p><p className="oop-lookup-route">{found.source === "class" ? "↓ not here: follow class lookup" : `↳ found here: stop at list ${id}`}</p><p><code>{id.toLowerCase()}.values</code> reads <strong>{valuesText(found.values)}</strong></p></div>; })}</div>
      <div className="oop-node oop-class-node"><span className="oop-label">CLASS Log · ordinary attribute fallback</span>{state.mode === "class" ? <p><code>values</code> <span aria-hidden="true">→ </span> list C <strong>{valuesText(state.shared)}</strong></p> : <p>No <code>values</code> attribute on the class. Both instances already own their own list.</p>}<p>{state.mode === "class" ? `Objects using C: ${["A", "B"].filter(id => lookupValues(state, id).source === "class").join(", ") || "neither"}.` : "No shared list exists in this version."}</p></div>
    </div>
    <div className="oop-controls">
      <button type="button" onClick={() => act("append")}>Append 18 through {receiver.toLowerCase()}</button>
      <button type="button" onClick={() => act("assign")}>Assign [99] to {receiver.toLowerCase()}.values</button>
      <button type="button" disabled={!history.length} onClick={() => { setState(history.at(-1)); setHistory(history.slice(0, -1)); setMessage("Previous model state restored. Re-read where each object finds values."); }}>Undo action</button>
      <button type="button" onClick={() => { setReceiver("A"); mode("class"); }}>Reset investigation</button>
    </div>
    <div className="oop-feedback" aria-live="polite" aria-atomic="true"><strong>Follow the lookup, then the operation</strong><p>{message}</p></div>
    <details className="oop-deeper"><summary>Check a changed case: both objects shadow the class list</summary><p>Assign [99] through A and B in the class-list version. Appending through either now changes only its own list. Log.values still refers to the original class list. Assignment changed the references; it did not delete the class attribute.</p></details>
    <p className="lesson-note">This model covers plain data attributes and list append/assignment only. Properties and other descriptors can change lookup and assignment behaviour. The property later in this lesson is one such deliberate exception.</p>
  </section>;
}

export function OopValidationLab() {
  const [candidate, setCandidate] = useState("bool");
  const [revealed, setRevealed] = useState(false);
  const result = validateReading(candidate);
  return <section className="oop-lab" aria-labelledby="oop-validation-title" data-oop-lab="validation">
    <p className="lesson-eyebrow">INVESTIGATE · A VALID STATE</p><h3 id="oop-validation-title">Where must the operation stop?</h3>
    <p>The log starts with [18.0]. Predict the first failing check and the final list for each input. Every attempt uses that same starting state, so the comparison isolates validation.</p>
    <div className="oop-controls oop-controls--inputs"><label>Candidate reading<select aria-label="Candidate reading" value={candidate} onChange={e => { setCandidate(e.target.value); setRevealed(false); }}>{readingCandidates.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><button type="button" onClick={() => setRevealed(true)}>Run validation</button><button type="button" onClick={() => { setCandidate("bool"); setRevealed(false); }}>Reset investigation</button></div>
    <p className="oop-command"><code>log.add({result.candidate.code})</code></p>
    <ol className="oop-gates" aria-label="Checks before mutation">{result.gates.map((gate, index) => <li key={gate.label} className={revealed ? `oop-gate--${gate.status}` : ""}><span className="oop-gate-number">{index + 1}</span><div><strong>{gate.label}</strong><span>{revealed ? gate.status === "passed" ? "✓ Passed" : gate.status === "blocked" ? "✕ Stop here" : "Not reached" : "Awaiting your prediction"}</span></div>{index < 3 && <span className="oop-gate-arrow" aria-hidden="true">↓</span>}</li>)}</ol>
    <div className="oop-before-after"><div><span className="oop-label">BEFORE · internal list</span><code>[18.0]</code></div><Arrow>successful checks permit mutation</Arrow><div><span className="oop-label">AFTER · internal list</span><code>{revealed ? result.error ? "[18.0]" : "[18.0, 24.0]" : "Predict first"}</code></div></div>
    <div className="oop-feedback" aria-live="polite" aria-atomic="true">{revealed ? <><strong>{result.error || "Accepted: the method returns None"}</strong><p>{result.error ? "The exception ends add before append. The original reading remains, and count and mean are unchanged." : "24 becomes 24.0 in float storage. Only after all checks pass does append change the list; the count becomes 2 and the mean 21.0."}</p></> : <p>Is a boolean a valid measurement merely because Python can treat it numerically?</p>}</div>
    <details className="oop-deeper"><summary>Explain the different failures</summary><p>True and the text "24" violate this class's accepted-type contract. NaN and infinity have float type, but fail the finite-value requirement. The huge integer passes the type check and fails float conversion. Validation tests meaning as well as type.</p></details>
    <p className="lesson-note">Six fixed Python inputs are modelled; the dropdown is not a general expression evaluator. The browser reproduces the checked method's error policy. It does not measure sensor quality or choose a scientifically valid temperature range.</p>
  </section>;
}

export function OopCompositionLab() {
  const [formatter, setFormatter] = useState("celsius");
  const [value, setValue] = useState(20);
  const [step, setStep] = useState(0);
  const frame = compositionFrames(value, formatter)[step];
  const name = formatter === "celsius" ? "CelsiusFormatter" : "FahrenheitFormatter";
  const change = (setter, value) => { setter(value); setStep(0); };
  return <section className="oop-lab" aria-labelledby="oop-composition-title" data-oop-lab="composition">
    <p className="lesson-eyebrow">INVESTIGATE · ONE JOB PER OBJECT</p><h3 id="oop-composition-title">Change a collaborator, keep the report</h3>
    <p>The caller has a temperature in degrees Celsius. The report adds a label; its formatter turns the number into text. Predict the result before stepping into and back out of those two calls.</p>
    <div className="oop-controls oop-controls--inputs"><label>Formatter object<select aria-label="Formatter object" value={formatter} onChange={e => change(setFormatter, e.target.value)}><option value="celsius">CelsiusFormatter</option><option value="fahrenheit">FahrenheitFormatter</option></select></label><label>Input in degrees Celsius<select aria-label="Input in degrees Celsius" value={value} onChange={e => change(setValue, Number(e.target.value))}>{[0, 20, 30].map(number => <option key={number}>{number}</option>)}</select></label></div>
    <div className="oop-collaboration-map">
      <div className={`oop-node ${step === 1 || step === 4 ? "oop-node--active" : ""}`}><span className="oop-label">REPORT OBJECT</span><code>render({value})</code><p><code>self</code> here refers to <strong>report</strong>.</p><code>formatter → {name}</code><p>Responsibility: add <code>Reading:</code></p></div>
      <Arrow>calls format({value})</Arrow>
      <div className={`oop-node ${step === 2 || step === 3 ? "oop-node--active" : ""}`}><span className="oop-label">FORMATTER OBJECT</span><strong>{name}</strong><p><code>self</code> here refers to the <strong>formatter</strong>.</p><code>{formatter === "celsius" ? `${value} → ${value.toFixed(1)} C` : `${value} × 9 / 5 + 32`}</code><p>Responsibility: return temperature text.</p></div>
    </div>
    <div className="oop-return-flow"><span aria-hidden="true">↩</span><p>{step >= 3 ? <>Return to report: <code>{compositionFrames(value, formatter)[3].output}</code></> : "The formatter's return value will travel back to Report.render."}</p></div>
    <p className="oop-command">Caller receives: <code>{step === 4 ? frame.output : "waiting for the report to return"}</code></p>
    <StepControls step={step} setStep={setStep} reset={() => { setFormatter("celsius"); setValue(20); setStep(0); }} />
    <div className="oop-feedback" aria-live="polite" aria-atomic="true"><strong>{frame.title}</strong><p>{frame.note}</p></div>
    <details className="oop-deeper"><summary>Transfer: plug in a Kelvin formatter</summary><p>Keep render unchanged. A KelvinFormatter with format(celsius) should return a string containing celsius + 273.15 and unit K. For 0°C, a two-decimal implementation returns "273.15 K"; the report then returns "Reading: 273.15 K". A method returning a bare number would violate this formatter contract even if its name were format.</p></details>
    <p className="lesson-note">The fixed formatters use modest exact inputs and one decimal place. This diagram models nested synchronous calls, not threads. The arrows show references and call/return direction, not inheritance or copying.</p>
  </section>;
}

export function OopRecordDiagram() {
  return <figure className="oop-static" aria-labelledby="oop-record-caption"><figcaption id="oop-record-caption"><strong>Same field values, different objects</strong></figcaption><div className="oop-record-pair"><div className="oop-node"><span className="oop-label">NAME a → READING A</span><code>celsius = 20.0</code><p><code>tags</code> → list A <strong>[]</strong></p></div><div className="oop-node"><span className="oop-label">NAME b → READING B</span><code>celsius = 20.0</code><p><code>tags</code> → list B <strong>[]</strong></p></div></div><p><code>a is b</code> is <strong>False</strong>: these names reach different objects. With the dataclass equality generated here, <code>a == b</code> is <strong>True</strong>: the same-type records have equal fields. Append to list A and equality becomes False; identity is still different.</p></figure>;
}
