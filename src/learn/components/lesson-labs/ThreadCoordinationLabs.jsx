import { MechanismLab, StatePanel, ExecutionLog } from "./MechanismLab.jsx";
import { useState } from "react";
import { initialRace, raceStep, initialLocks, lockStep, deadlocked, waitEdges, conditionTrace } from "../../data/thread-coordination-models.js";






import { Stepper } from "./LessonInvestigation.jsx";
import { ThreadRacePicture } from "./ThreadFigures.jsx";
import "./workflow-labs.css";







export function ThreadRaceLab() {
  const [state,setState]=useState(()=>initialRace(false));
  const done=Object.values(state.workers).every(w=>w.phase===3);
  return <MechanismLab id="thread-race" title="Make two increments lose one update">
    <p>Start with A read, B read, A compute, B compute, A write, B write. Predict the total. Reset with a lock and try the same interference.</p>
    <label>Protocol <select value={String(state.locked)} onChange={e=>setState(initialRace(e.target.value==='true'))}><option value="false">Unprotected read → compute → write</option><option value="true">One lock covers all three steps</option></select></label>
    <ThreadRacePicture state={state} advance={id=>setState(raceStep(state,id))}/>
    <div className="wc-controls"><button type="button" onClick={()=>setState(initialRace(state.locked))}>Reset</button></div>
    <p className="wc-feedback" aria-live="polite">{done?`Finished: expected 2, observed ${state.value}. ${state.value===2?'Both updates survived.':'One update overwrote the other.'}`:state.log.at(-1)||'Choose a worker to execute one conceptual step.'}</p>
    <ExecutionLog items={state.log}/><p>Each button is one model step, not one CPython bytecode or measured time slice. A blocked attempt changes no shared data. Transfer: would locking only each write preserve the increment invariant?</p>
  </MechanismLab>;
}

export function ThreadConditionLab() {
  const [scenario,setScenario]=useState('empty'),[step,setStep]=useState(0),states=conditionTrace(scenario),conditionState=states[step];
  return <MechanismLab id="thread-condition" title="A notification is not an item">
    <p>Predict whether a consumer may take an item immediately after being notified. Watch both the queue and lock owner, not just the notification.</p>
    <label>Scenario <select value={scenario} onChange={e=>{setScenario(e.target.value);setStep(0);}}><option value="empty">Notify while empty</option><option value="item">Publish one item</option><option value="stolen">Another consumer takes it first</option></select></label>
    <Stepper step={step} count={states.length} setStep={setStep}/>
    <div className="wc-grid"><StatePanel title="Queue / predicate"><code>{JSON.stringify(conditionState.queue)}</code><p>Has item: {String(conditionState.queue.length>0)}</p></StatePanel><StatePanel title="Condition lock owner">{conditionState.owner}</StatePanel><StatePanel title="Original consumer">{conditionState.consumer}</StatePanel></div>
    <p className="wc-feedback" aria-live="polite">{conditionState.action}</p><p>These are possible schedules, not a fairness promise. Transfer: if publication happens before the consumer starts, why does checking stored state before waiting avoid a lost-notification bug?</p>
  </MechanismLab>;
}

export function ThreadDeadlockLab() {
  const [ordered,setOrdered]=useState(false),[state,setState]=useState(()=>initialLocks(false));
  const edges=waitEdges(state),blocked=deadlocked(state);
  return <MechanismLab id="thread-deadlock" title="Turn waiting into a cycle—and break it">
    <p>With opposite orders, advance A, B, A, B. Each has something the other needs. Then try both workers acquiring L1 before L2.</p>
    <label>Lock policy <select value={String(ordered)} onChange={e=>{const next=e.target.value==='true';setOrdered(next);setState(initialLocks(next));}}><option value="false">A: L1 → L2; B: L2 → L1</option><option value="true">Both: L1 → L2</option></select></label>
    <div className="wc-controls">{['A','B'].map(id=><button type="button" key={id} disabled={state.workers[id].pc===3||blocked} onClick={()=>setState(lockStep(state,id))}>Advance {id}</button>)}<button type="button" onClick={()=>setState(initialLocks(ordered))}>Reset</button></div>
    <svg viewBox="0 0 360 165" role="img" aria-label={`Wait-for graph: ${edges.map(([a,b,l])=>`${a} waits for ${b} via ${l}`).join('; ')||'no waiting edges'}`} style={{width:'100%',maxWidth:500}}>
      <defs><marker id="wc-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#d7b971"/></marker></defs>
      <rect x="10" y="45" width="110" height="70" rx="5" fill="#253121" stroke="#829565"/><rect x="240" y="45" width="110" height="70" rx="5" fill="#253121" stroke="#829565"/>
      <text x="65" y="86" textAnchor="middle" fill="#eee5c9" fontSize="20">Worker A</text><text x="295" y="86" textAnchor="middle" fill="#eee5c9" fontSize="20">Worker B</text>
      {edges.map(([a,b,l])=><g key={a}><path d={a==='A'?'M120,62 Q180,8 240,62':'M240,98 Q180,150 120,98'} stroke="#d7b971" fill="none" strokeWidth="2" markerEnd="url(#wc-arrow)"/><text x="180" y={a==='A'?24:160} textAnchor="middle" fill="#d7b971" fontSize="18">waits for {l}</text></g>)}
    </svg>
    <div className="wc-grid">{Object.entries(state.owners).map(([name,owner])=><StatePanel key={name} title={name}>{owner?`Held by ${owner}`:'Available'}</StatePanel>)}</div>
    <p className="wc-feedback" aria-live="polite">{blocked?'Deadlock: A waits for B and B waits for A. Neither can reach release.':state.log.at(-1)||'No locks are held yet.'}</p>
    <ExecutionLog items={state.log}/><p>Model: two exclusive locks, one instance each, no forced release. A cycle here is sufficient for deadlock. Ordered acquisition prevents this cycle; it does not promise scheduler fairness or prevent unrelated waits inside the critical section.</p>
  </MechanismLab>;
}
