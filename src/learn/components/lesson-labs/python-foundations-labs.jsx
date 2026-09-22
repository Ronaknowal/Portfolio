import { useId, useState } from "react";
import { referenceTrace, selectionTrace, functionTrace, loopReadings } from "../../data/python-foundations-model";
import "./python-foundations.css";

const py = value => value === null ? "None" : String(value);

function Steps({ step, setStep, length }) {
  return <div className="pyf-controls">
    <button type="button" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>Back</button>
    <button type="button" onClick={() => setStep(Math.min(length - 1, step + 1))} disabled={step === length - 1}>Next step</button>
    <button type="button" onClick={() => setStep(0)}>Reset</button>
    <span aria-label={`State ${step + 1} of ${length}`}>{step + 1} / {length}</span>
  </div>;
}
function CodeTrace({ code, line }) {
  return <div className="pyf-code" role="region" aria-label="Python code with active instruction" tabIndex={0}>
    {code.map((text, index) => <div key={index} className={index === line ? "is-active" : ""} aria-current={index === line ? "step" : undefined}><span aria-hidden="true">{index === line ? "▶" : index + 1}</span><code>{text}</code></div>)}
  </div>;
}
function Intro({ eyebrow, title, children }) {
  return <><p className="pyf-eyebrow">{eyebrow}</p><h3>{title}</h3><p className="pyf-predict">{children}</p></>;
}

export function PythonReferencesLab() {
  const [copy, setCopy] = useState(false), [action, setAction] = useState("append"), [step, setStep] = useState(0);
  const marker = `pyf-arrow-${useId().replace(/:/g, "")}`;
  const { code, states } = referenceTrace(copy, action), state = states[step];
  const objects = [...new Set(Object.values(state.names))];
  const yOf = id => objects.length === 1 ? 90 : 40 + objects.indexOf(id) * 108;
  return <section className="lesson-lab pyf-lab" data-pyf-lab="references" aria-label="Names and objects explorer">
    <Intro eyebrow="INVESTIGATION · SHARED STATE" title="Which list actually changes?">Follow each instruction and inspect whether <code>readings</code> contain 24? What about <code>backup</code>? Test assignment first, then change only how the backup is made.</Intro>
    <div className="pyf-options"><label>Make the backup<select value={copy ? "copy" : "alias"} onChange={e => { setCopy(e.target.value === "copy"); setStep(0); }}><option value="alias">Assign the same list</option><option value="copy">Make a shallow copy</option></select></label><label>Then change backup<select value={action} onChange={e => { setAction(e.target.value); setStep(0); }}><option value="append">Append 24 to its list</option><option value="rebind">Reassign it to [0]</option></select></label></div>
    <CodeTrace code={code} line={state.line} />
    <div className="pyf-object-map">
      <svg viewBox="0 0 370 235" role="img" aria-label={Object.entries(state.names).map(([name, id]) => `${name} points to list ${id} containing ${state.objects[id].join(", ")}`).join(". ") || "No names or list objects exist yet"}>
        <defs><marker id={marker} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" /></marker></defs>
        <text x="8" y="18" className="pyf-svg-label">NAMES</text><text x="177" y="18" className="pyf-svg-label">LIST OBJECTS</text>
        {["readings", "backup"].map((name, i) => <g key={name}><rect x="5" y={48 + i * 102} width="95" height="40" rx="4" className="pyf-svg-name" /><text x="52" y={73 + i * 102} textAnchor="middle">{name}</text>{state.names[name] ? <path d={`M103,${68 + i * 102} C138,${68 + i * 102} 143,${yOf(state.names[name]) + 30} 173,${yOf(state.names[name]) + 30}`} className="pyf-svg-arrow" markerEnd={`url(#${marker})`} /> : <text x="111" y={73 + i * 102} className="pyf-svg-label">unbound</text>}</g>)}
        {objects.map(id => <g key={id}><rect x="179" y={yOf(id)} width="184" height="67" rx="5" className="pyf-svg-object" /><text x="192" y={yOf(id) + 18} className="pyf-svg-label">LIST {id}</text>{state.objects[id].map((value, i) => <g key={i}><rect x={192 + i * 51} y={yOf(id) + 27} width="42" height="28" className="pyf-svg-cell" /><text x={213 + i * 51} y={yOf(id) + 47} textAnchor="middle">{value}</text></g>)}</g>)}
      </svg>
    </div>
    <Steps step={step} setStep={setStep} length={states.length} />
    <p className="pyf-feedback" role="status">{state.explanation}</p>
    {step === states.length - 1 && <div className="pyf-output"><span>PRINTED OUTPUT</span><code>{`[${state.objects[state.names.readings].join(", ")}] [${state.objects[state.names.backup].join(", ")}]`}</code></div>}
    <p className="pyf-transfer"><strong>Try a different cause:</strong> choose “Reassign it to [0]”. Does that alter <code>readings</code>, with or without a copy? Follow the arrow that moves.</p>
    <p className="pyf-boundary">A, B and C label object identity; they are not memory addresses. This fixed model shows outer lists of numbers, not nested copying, garbage-collection timing or arbitrary Python execution.</p>
  </section>;
}

export function PythonFlowLab() {
  const [threshold, setThreshold] = useState(20), [step, setStep] = useState(0);
  const { code, states } = selectionTrace(threshold), state = states[step];
  return <section className="lesson-lab pyf-lab" data-pyf-lab="flow" aria-label="Branch and loop explorer">
    <Intro eyebrow="INVESTIGATION · CONTROL FLOW" title="Follow one reading through the gates">Inspect which values reach <code>selected</code>. Then follow the cursor through one present value, the missing value and zero. Which instruction sends control back to the loop?</Intro>
    <div className="pyf-options"><label>Keep readings at or above<select value={threshold} onChange={e => { setThreshold(Number(e.target.value)); setStep(0); }}><option value="0">0 °C</option><option value="20">20 °C</option><option value="30">30 °C</option></select></label></div>
    <div className="pyf-input-strip" aria-label="Original input, unchanged">
      {loopReadings.map((v, i) => <div key={i} className={`${i === state.index ? "is-current" : ""} ${state.decisions[i] || ""}`}><small>index {i}</small><strong>{py(v)}</strong><span>{i === state.index ? "← current" : state.decisions[i] || "waiting"}</span></div>)}
    </div>
    <div className="pyf-flow-map" aria-label="Decision path">
      <div className={state.phase === "take" ? "is-active" : ""}><small>TAKE NEXT</small><strong>{state.index >= 0 && state.index < loopReadings.length ? `value = ${py(loopReadings[state.index])}` : state.phase === "done" ? "no items left" : "await first item"}</strong></div><span aria-hidden="true">↓</span>
      <div className={["missing", "skip"].includes(state.phase) ? "is-active" : ""}><small>GATE 1</small><strong>value is None?{["missing", "skip"].includes(state.phase) && <em>{loopReadings[state.index] === null ? "True" : "False"}</em>}</strong><p>Yes: continue, take next<br />No: compare the number</p></div><span aria-hidden="true">↓</span>
      <div className={["compare", "reject"].includes(state.phase) ? "is-active" : ""}><small>GATE 2</small><strong>value ≥ {threshold}?{["compare", "reject", "append"].includes(state.phase) && <em>{`${loopReadings[state.index]} ≥ ${threshold}: ${loopReadings[state.index] >= threshold ? "True" : "False"}`}</em>}</strong><p>False: take next<br />True: append</p></div><span aria-hidden="true">↓</span>
      <div className={state.phase === "append" || state.phase === "done" ? "is-active" : ""}><small>SELECTED · NEW LIST</small><strong>[{state.accepted.join(", ")}]</strong><p>After append → take next</p></div>
    </div>
    <CodeTrace code={code} line={state.line} />
    <Steps step={step} setStep={setStep} length={states.length} />
    <p className="pyf-feedback" role="status">{state.explanation}</p>
    <p className="pyf-transfer"><strong>Transfer:</strong> set the threshold to 0. Compare whether zero survives and explain why <code>if not value</code> would be the wrong missing-value test.</p>
    <p className="pyf-boundary">Manual trace of the displayed loop, using five fixed readings. Colors are backed by labels; the input is never mutated. This models control flow, not the speed of Python.</p>
  </section>;
}

export function PythonCallLab() {
  const [celsius, setCelsius] = useState(20), [mode, setMode] = useState("return"), [step, setStep] = useState(0);
  const { code, states } = functionTrace(celsius, mode), state = states[step];
  const assigned = Object.hasOwn(state, "result");
  return <section className="lesson-lab pyf-lab" data-pyf-lab="calls" aria-label="Function call and return explorer">
    <Intro eyebrow="INVESTIGATION · TWO DIFFERENT DESTINATIONS" title="A returned value is not a printed message">Track two destinations separately: what does <code>result</code> hold, and what does the console show? Step once through “Return the number”, then switch the function to “Print the number”.</Intro>
    <div className="pyf-options"><label>Input temperature<select value={celsius} onChange={e => { setCelsius(Number(e.target.value)); setStep(0); }}><option value="0">0 °C</option><option value="20">20 °C</option><option value="100">100 °C</option></select></label><label>Function's last instruction<select value={mode} onChange={e => { setMode(e.target.value); setStep(0); }}><option value="return">Return the number</option><option value="print">Print the number</option></select></label></div>
    <CodeTrace code={code} line={state.line} />
    <div className="pyf-call-map">
      <div className="pyf-frame"><small>CALLER · SCRIPT</small><strong>result</strong><p>{assigned ? <code>{py(state.result)}</code> : state.stage === "defined" ? "not assigned yet" : "waiting for convert(...)"}</p></div>
      <div className="pyf-call-arrow"><span>{state.stage === "call" || state.stage === "calculate" || state.stage === "print" ? "argument →" : step >= 3 ? "← call result" : "call →"}</span><strong>{step >= 3 && state.stage !== "print" ? py(state.returned) : `${celsius} °C`}</strong></div>
      <div className={`pyf-frame ${state.frame ? "is-active" : ""}`}><small>LOCAL CALL FRAME · convert</small>{state.frame ? <><p><code>celsius = {celsius}</code></p><p><code>fahrenheit = {state.fahrenheit === undefined ? "not assigned" : `${state.fahrenheit} °F`}</code></p></> : <p>{step === 0 ? "Created when called" : "Invocation ended; local names no longer available to this caller"}</p>}</div>
    </div>
    <div className="pyf-output"><span>CONSOLE · TEXT OUTPUT ONLY</span><pre>{state.output.length ? state.output.map(v => v === null ? "None" : `${v}.0`).join("\n") : "(nothing printed yet)"}</pre></div>
    <Steps step={step} setStep={setStep} length={states.length} />
    <p className="pyf-feedback" role="status">{state.explanation}</p>
    <p className="pyf-transfer"><strong>Transfer:</strong> choose 100 °C. After the call, could the caller calculate <code>result + 1</code>? Compare the return value and console for both modes.</p>
    <p className="pyf-boundary">The conversion is °F = °C × 9/5 + 32. This fixed model represents one ordinary call, local names, its return and printed output. It does not simulate recursive calls, exceptions or interpreter memory layout.</p>
  </section>;
}

export function PythonModuleDiagram() {
  return <figure className="pyf-module-figure"><figcaption>One program, two responsibilities</figcaption><div className="pyf-module-map"><div><small>report.py · ENTRY POINT</small><p>Provide text readings</p><p>Call <code>summarize(raw)</code></p><p>Format and print the report</p></div><div className="pyf-module-link"><span>list of strings →</span><span>← result dictionary</span></div><div><small>readings.py · REUSABLE MODULE</small><p>Define parsing and summary</p><p>Reject unusable input</p><p><code>return</code> count and mean</p></div></div><p>Import makes the function available. Calling it computes a result; only the entry point decides how to display that result.</p></figure>;
}
