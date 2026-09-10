import { formatLabValue, LabChoices, LabFeedback } from "./LabControls.jsx";
import { useState } from "react";
import { Investigation, Predict, Stepper } from "./LessonInvestigation.jsx";
import { scoreFixtures, apiBoundary, ownershipTrace, compatibilityModel } from "../../data/api-design-models.js";



import './reliability.css';







export function ApiBoundaryLab(){const[kind,setKind]=useState('ordinary'),[minimum,setMinimum]=useState(0.85),[step,setStep]=useState(0),states=apiBoundary(kind,minimum),boundaryState=states[step];return <Investigation id="api-boundary" kicker="Hints describe; checks decide" title="Can filtering conceal an invalid score?">
 <Predict>With minimum 0.85, should {'{ok: 0.9, bad: -0.1}'} quietly return only ok? Trace validation before deciding.</Predict><div className="nt-controls"><LabChoices label="Decoded data" value={kind} onChange={v=>{setKind(v);setStep(0);}} items={[["ordinary","0.81 and 0.86"],["zero","0 and 1"],["equal","Exactly 0.85"],["invalid","0.9 and −0.1"],["boolean","Boolean true"],["nan","Not a number (NaN)"]]}/><LabChoices label="Minimum (inclusive)" value={minimum} onChange={v=>{setMinimum(Number(v));setStep(0);}} items={[[0,'0'],[0.85,'0.85'],[1,'1']]}/></div>
 <div className="rel-gates">{['decoded','validate','filter'].map(stage=><div key={stage} data-active={boundaryState.stage===stage}><strong>{stage}</strong><span>{stage==='decoded'?'object values':stage==='validate'?'numeric → finite → range':'score ≥ minimum'}</span></div>)}</div><div className="rel-tokens">{scoreFixtures[kind].map(([n,v])=><span key={n} data-active={boundaryState.active===n}>{n}<small>{formatLabValue(v)}</small></span>)}</div><LabFeedback>{boundaryState.note} {boundaryState.stage==='filter'?`Returned: ${JSON.stringify(Object.fromEntries(boundaryState.accepted))}`:boundaryState.error?'ValueError; filtering never runs.':''}</LabFeedback><Stepper step={step} count={states.length} setStep={setStep}/><p className="lesson-note">Fixtures start after decoding; this does not simulate JSON parser behaviour or all schema errors. Python's bool is an int subclass, but this measurement contract deliberately rejects it. Try zero with minimum 0.</p></Investigation>;}

export function ApiOwnershipLab(){const[mode,setMode]=useState('shared'),[step,setStep]=useState(0),states=ownershipTrace(mode),ownershipState=states[step];return <Investigation id="api-ownership" kicker="Follow the object, not the variable name" title="Who observes an append?">
 <Predict>Two calls use a default list. After the second call, what does the first returned name show? Then compare aliasing a supplied list with copying it.</Predict><div className="nt-controls"><LabChoices label="Ownership policy" value={mode} onChange={v=>{setMode(v);setStep(0);}} items={[["shared","Shared mutable default"],["mutate","Alias caller's list"],["copy","Copy caller's outer list"]]}/></div>
 <div className="rel-reference-map"><div><h4>Names</h4>{ownershipState.refs.map(([name,id])=><p key={name}>{name} <strong>→ {id}</strong></p>)}</div><div><h4>Objects</h4>{ownershipState.objects.map(o=><div className="rel-object" key={o.id}><strong>{o.id}</strong><code>{JSON.stringify(o.values)}</code></div>)}</div></div><LabFeedback>{ownershipState.note}</LabFeedback><Stepper step={step} count={states.length} setStep={setStep}/><p className="lesson-note">Object labels are stable diagram IDs, not memory addresses. The strings are immutable. A shallow outer-list copy still shares nested mutable members; it is not a deep-copy promise.</p></Investigation>;}

export function ApiCompatibilityLab(){const[change,setChange]=useState('same'),compatibility=compatibilityModel(change);return <Investigation id="api-compatibility" kicker="An unchanged caller meets a changed contract" title="Can a change type-check and still break users?">
 <Predict>The caller expects edge: 0.85 at an inclusive threshold. Which change fails before execution, and which changes silently alter the answer?</Predict><div className="nt-controls"><LabChoices label="Library revision" value={change} onChange={setChange} items={[["same","Same contract"],["rename","Rename minimum to min_score"],["exclusive","Change ≥ to >"],["units","Return percentages"]]}/></div>
 <div className="rel-gates"><div><strong>Caller</strong><code>{compatibility.call}</code></div><div data-active={change==='rename'}><strong>Bind arguments</strong><span>{compatibility.binding}</span></div><div data-active={change!=='rename'}><strong>Return</strong><code>{compatibility.result===null?'body not reached':JSON.stringify(compatibility.result)}</code></div></div><LabFeedback>{compatibility.note}</LabFeedback><button type="button" onClick={()=>setChange('same')}>Reset</button><p className="lesson-note">Concrete revisions of one small interface, not automatic compatibility analysis. Transfer: changing a default can break a caller that omits the keyword even if this explicit call still works.</p></Investigation>;}
