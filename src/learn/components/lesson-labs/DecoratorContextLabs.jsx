import { useState } from "react";
import { Investigation, Stepper } from "./LessonInvestigation.jsx";
import { decoratorOrderTrace, contextTrace, exitStackTrace } from "../../data/decorator-context-models.js";
import './iteration-decorators.css';
import { ContextRouteMap } from "./DecoratorContextFigures.jsx";
export function DecoratorOrderLab() {
  const [outer, setOuter] = useState('cap'),
    [value, setValue] = useState(8),
    [step, setStep] = useState(0),
    trace = decoratorOrderTrace(outer, value),
    wrapperState = trace[step];
  return <Investigation id="decorator-order" kicker="CALL INWARD · RETURN OUTWARD" title="Does swapping two wrappers change the answer?">
    <p>reading returns its input. double multiplies the returned result by 2. cap limits the returned result to at most 10. Neither changes the arguments on the way in.</p><p className="lesson-note">Swap wrapper order for input 8 and follow both the call and return paths. Compare where doubling and capping change the value.</p>
    <div className="nt-controls"><label>Outer decorator<select value={outer} onChange={e => {
          setOuter(e.target.value);
          setStep(0);
        }}><option value="cap">cap outside double</option><option value="double">double outside cap</option></select></label><label>Reading value<select value={value} onChange={e => {
          setValue(Number(e.target.value));
          setStep(0);
        }}>{[3, 8, 12].map(v => <option key={v}>{v}</option>)}</select></label></div>
    <div className="id-spatial"><div className="id-call-layer" data-active={wrapperState.active === 'caller'}>Caller · reading({value})<div className="id-call-layer" data-active={wrapperState.active === outer}>{outer} · outer wrapper<div className="id-call-layer" data-active={wrapperState.active === wrapperState.order[1]}>{wrapperState.order[1]} · inner wrapper<div className="id-call-layer" data-active={wrapperState.active === 'body'}>original body → {value}</div></div></div></div><p className="id-call-result">Current returned value: {wrapperState.result ?? 'body not called yet'}</p></div>
    <p className="nt-feedback" aria-live="polite">{wrapperState.note}</p><Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">Two synchronous result wrappers with no errors or side effects. Layout nesting means retained function references, not source indentation at runtime. Transfer: input 3 gives the same final value in both orders—does one agreeing input prove wrappers commute for every input?</p>
  </Investigation>;
}
export function ContextLifetimeLab() {
  const [path, setPath] = useState('body-fails'),
    [suppress, setSuppress] = useState(false),
    [step, setStep] = useState(0),
    trace = contextTrace(path, suppress),
    contextState = trace[step];
  return <Investigation id="context-lifetime" kicker="ACQUIRE → USE → RELEASE" title="Which path still closes the resource?">
    <p>The manager opens an in-memory text file only after successful entry. Its exit method closes that file; choose whether a body error is suppressed. The failure preset raises before acquiring anything during enter.</p><p className="lesson-note">Switch between a normal body, failed body, failed enter and suppression. Follow which exit operation runs and where execution resumes.</p>
    <div className="nt-controls"><label>Execution path<select value={path} onChange={e => {
          setPath(e.target.value);
          setStep(0);
        }}><option value="success">Body succeeds</option><option value="body-fails">Body raises ValueError</option><option value="enter-fails">Enter raises ValueError</option></select></label><label>Exit returns<select value={String(suppress)} onChange={e => {
          setSuppress(e.target.value === 'true');
          setStep(0);
        }}><option value="false">False · preserve body error</option><option value="true">True · suppress body error</option></select></label></div>
    <ContextRouteMap state={contextState} path={path} suppress={suppress} />
    <p className="id-output">Observed events: {contextState.events.join(' → ') || 'none'}</p>
    <p className="nt-feedback" aria-live="polite">{contextState.note}</p><Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">Cleanup itself succeeds in this model. Truthy exit results suppress body exceptions, not errors raised by that same manager's enter. Ordinary return/break also leave entered contexts through exit; abrupt process termination is outside this guarantee.</p>
  </Investigation>;
}
export function ExitStackLab() {
  const [fail, setFail] = useState('C'),
    [step, setStep] = useState(0),
    trace = exitStackTrace(fail),
    cleanupState = trace[step];
  return <Investigation id="context-exit-stack" kicker="REGISTER SUCCESS · UNWIND IN REVERSE" title="What closes if the third acquisition fails?">
    <p>Acquire resources A, B and C in that order. Each successful entry adds one exit action. The top displayed action will run first when leaving the block.</p><p className="lesson-note">Trigger failure while acquiring B and follow the cleanup stack. Compare resources acquired successfully with those that never opened.</p>
    <div className="nt-controls"><label>Acquisition failure<select value={fail} onChange={e => {
          setFail(e.target.value);
          setStep(0);
        }}>{['A', 'B', 'C'].map(v => <option key={v}>{v}</option>)}<option value="none">No failure</option></select></label></div>
    <div className="id-spatial"><strong>Registered exit actions · most recent at top</strong><div className="id-stack">{cleanupState.stack.length ? cleanupState.stack.map((n, i) => <div key={n}>release {n}{i === cleanupState.stack.length - 1 ? ' ← next to run' : ''}</div>) : <p>No registered actions remain.</p>}</div><p className="id-output">{cleanupState.events.join(' → ') || 'No acquisition attempted.'}</p></div>
    <p className="nt-feedback" aria-live="polite">{cleanupState.note}</p><Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">A stack here means last entered, first exited. This introduces the ordering without requiring the later data-structure lesson. No suppression or failing cleanup callbacks are modeled; real ExitStack also passes updated exception state to outer exits.</p>
  </Investigation>;
}
