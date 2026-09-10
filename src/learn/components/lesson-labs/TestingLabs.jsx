import { formatLabValue, LabChoices, LabFeedback } from "./LabControls.jsx";
import { useState } from "react";
import { Investigation, Predict, Stepper } from "./LessonInvestigation.jsx";
import { meanTrace, temperatureCases, testMatrix, dependencyModel } from "../../data/testing-models.js";



import './reliability.css';







export function DebugExecutionLab(){const[input,setInput]=useState('pair'),[early,setEarly]=useState(true),[step,setStep]=useState(0);const values={pair:[18,24],single:[18],zero:[0,24]}[input],states=meanTrace(values,early),executionState=states[step];const reset=()=>setStep(0);return <Investigation id="testing-execution" kicker="Follow control, not just output" title="Which readings reach the return?">
 <Predict>For [18,24], predict the total when a return inside the loop runs. Then predict whether [18] can expose that same bug.</Predict>
 <div className="nt-controls"><LabChoices label="Readings" value={input} onChange={v=>{setInput(v);reset();}} items={[["pair","[18, 24]"],["single","[18]"],["zero","[0, 24]"]]}/><LabChoices label="Return placement" value={String(early)} onChange={v=>{setEarly(v==='true');reset();}} items={[["true","Inside the loop (bug)"],["false","After the loop (fixed)"]]}/></div>
 <div className="rel-pipeline"><div><strong>Readings</strong><div className="rel-tokens">{values.map((v,i)=><span key={i} data-active={i<executionState.consumed}>{v}<small>{i<executionState.consumed?'added':'waiting'}</small></span>)}</div></div><b aria-hidden="true">→</b><div><strong>Total</strong><output>{executionState.total}</output></div><b aria-hidden="true">→</b><div><strong>Return / {values.length}</strong><output>{executionState.result===null?'not reached':executionState.result}</output></div></div>
 <LabFeedback>{executionState.note}</LabFeedback><Stepper step={step} count={states.length} setStep={setStep}/><p className="lesson-note">Fixed Python-like trace of these finite lists. Return exits the function, not merely the current iteration. Try [0,24]: the correct mean is 12.</p></Investigation>;}

export function TestDiscriminationLab(){const[selected,setSelected]=useState(['difference']);const rows=testMatrix(selected);return <Investigation id="testing-discrimination" kicker="Test the tests" title="Which bugs can your evidence distinguish?">
 <Predict>Does “adding 10°C adds 18°F” detect a missing +32? Choose a second case that separates the correct conversion from that bug.</Predict><fieldset className="rel-checks"><legend>Enable independently justified checks</legend>{temperatureCases.map(c=><label key={c.id}><input type="checkbox" checked={selected.includes(c.id)} onChange={()=>setSelected(selected.includes(c.id)?selected.filter(v=>v!==c.id):[...selected,c.id])}/>{c.label}</label>)}</fieldset>
 <div className="rel-candidates">{rows.map(row=><div key={row.id}><code>{row.label}</code><strong data-pass={row.checks.length>0&&row.checks.every(c=>c.pass)}>{!row.checks.length?'NO EVIDENCE':row.checks.every(c=>c.pass)?'SURVIVES':'DETECTED'}</strong><div className="rel-tokens">{row.checks.map(c=><span key={c.id} data-active={c.pass}>{c.id}<small>{formatLabValue(c.actual)} · {c.pass?'pass':'fail'}</small></span>)}</div></div>)}</div>
 <LabFeedback>{selected.length?`${rows.filter(r=>r.checks.every(c=>c.pass)).length} of four implementations survive the selected checks. Passing distinguishes only these cases; it does not prove all possible inputs.`:'No tests ran. A green exit with no checks is not correctness evidence.'}</LabFeedback><button type="button" onClick={()=>setSelected(['difference'])}>Reset</button><p className="lesson-note">These are deliberately changed implementations, called mutants. No mutation-testing package runs in the browser. All candidate/check combinations were compared with Python.</p></Investigation>;}

export function DependencyConstraintsLab(){const[modern,setModern]=useState(false),compatibility=dependencyModel(modern);return <Investigation id="testing-dependencies" kicker="Find a compatible intersection" title="Why can installing two packages fail?">
 <Predict>Plotter needs Core versions 2 or 3. If Reader needs version 1, can one environment satisfy both?</Predict><div className="nt-controls"><LabChoices label="Reader requirement" value={String(modern)} onChange={v=>setModern(v==='true')} items={[["false","Reader old: Core ≥1, <2"],["true","Reader newer: Core ≥3, <5"]]}/></div>
 <div className="rel-dependencies"><div><strong>Project requests</strong><p>Plotter → Core ≥2, &lt;4</p><p>Reader → Core {modern?'≥3, <5':'≥1, <2'}</p></div><div><strong>Candidate versions</strong>{[['Plotter',compatibility.a],['Reader',compatibility.b],['Both',compatibility.common]].map(([label,allowed])=><div className="rel-version-row" key={label}><span>{label}</span>{compatibility.versions.map(v=><span key={v} data-active={allowed.includes(v)} aria-label={`${label} ${allowed.includes(v)?'accepts':'rejects'} Core ${v}`}>{v}{allowed.includes(v)?' ✓':' ×'}</span>)}</div>)}</div></div>
 <LabFeedback>{compatibility.common.length?`Core ${compatibility.common[0]} meets both stated constraints. That is metadata compatibility; execute tests to check API behaviour.`:'No candidate meets both requirements. Creating another empty environment cannot make these contradictory ranges overlap.'}</LabFeedback><button type="button" onClick={()=>setModern(false)}>Reset</button><p className="lesson-note">Invented packages and four integer versions, no downloads. Real resolvers also consider Python/platform markers, artifacts and transitive dependencies. Changing a requirement needs an actually compatible release, not editing numbers to silence an error.</p></Investigation>;}
