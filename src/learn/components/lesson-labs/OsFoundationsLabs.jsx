import { useId, useState } from "react";
import { Investigation, Predict, Stepper } from "./LessonInvestigation.jsx";
import { LessonTable } from "./LessonElements";
import { scheduleTrace, translationModel, sharingTrace } from "../../data/os-foundations-model";
import './systems-structures.css';
import './mechanism-figures.css';

export function SchedulingLab(){
  const [quantum,setQuantum]=useState(1),[io,setIo]=useState(true),[step,setStep]=useState(0);
  const trace=scheduleTrace(quantum,io),state=trace[step];
  return <Investigation id="os-scheduler" kicker="ONE CPU · TWO SAVED CONTEXTS" title="Where does a paused calculation keep its place?">
    <p>A adds 1 to its own counter; B adds 10 to its own. One CPU can execute only one of these instruction streams at a time. A may request I/O between its additions.</p>
    <Predict>If A must wait for a device, does B need to wait too? Will changing the time slice change which counter belongs to which process?</Predict>
    <div className="nt-controls"><label>Time slice<select value={quantum} onChange={e=>{setQuantum(Number(e.target.value));setStep(0);}}><option value="1">One instruction interval</option><option value="2">Two instruction intervals</option></select></label><label>A's middle instruction<select value={String(io)} onChange={e=>{setIo(e.target.value==='true');setStep(0);}}><option value="true">Request I/O, then wait two intervals</option><option value="false">Add 1 without waiting</option></select></label></div>
    <p className="foundation-state">Time {state.time} · last interval: {state.cpu??(step?'idle':'not started')} · ready queue: {state.ready.join(' → ')||'empty'}</p>
    <ol className="foundation-timeline" aria-label="CPU execution timeline">{state.timeline.map((id,i)=><li className={id==='B'?'is-b':''} key={i}>{id}<small>{i}→{i+1}</small></li>)}</ol>
    <LessonTable caption="Saved process state after this interval" headers={['Process','Counter','Next instruction','State']} rows={Object.entries(state.jobs).map(([id,j])=>[id,j.value,j.code[j.pc]??'finished',j.state])}/>
    <div className="nt-columns">{Object.entries(state.jobs).map(([id,j])=><div key={id}><h4>{id}'s instruction stream</h4><ol>{j.code.map((code,i)=><li key={i} className={i===j.pc?'foundation-state':''}>{code}{i===j.pc?' ← next':i<j.pc?' · executed':''}</li>)}</ol></div>)}</div>
    <p className="nt-feedback" aria-live="polite">{state.note}</p><Stepper step={step} count={trace.length} setStep={setStep}/>
    <p className="lesson-note">A deterministic one-core round-robin model with zero switching overhead. Device completion is processed at the next interval boundary. Real scheduling includes priorities, interrupts, kernel work and multiple cores; this timeline does not predict Linux scheduling or elapsed performance.</p>
  </Investigation>;
}

export function AddressTranslationLab(){
  const uid=useId(),[process,setProcess]=useState('A'),[address,setAddress]=useState(22),[access,setAccess]=useState('read'),[step,setStep]=useState(0);
  const m=translationModel(process,address,access);
  const notes=[`Split ${address} into page ${m.page} and offset ${m.offset}: ${m.page} × 16 + ${m.offset}. Offset means the byte's position within its page.`,m.entry?`Process ${process}'s page ${m.page} is ${m.entry.resident?'resident in frame '+m.entry.frame:'valid but not resident'}, with ${m.entry.rights} permission.`:`There is no valid mapping for page ${m.page}. No physical address is available.`,m.kind==='resident'?`The mapping is resident and ${access} is permitted. Keep the offset and substitute frame ${m.frame}.`:m.kind==='demand'?'This is a valid anonymous page. The kernel can allocate a zero-filled frame, update the mapping and retry; this example needs no disk read.':m.kind==='protection'?'The requested write is forbidden. This read-only page is not marked copy-on-write, so copying it is not a valid repair.':'The access faults because the address is outside the permitted mappings. The kernel cannot treat every arbitrary address as new memory.',m.physical===null?'No access completes. A normal unhandled invalid/protection access may terminate the process.':`${m.frame} × 16 + ${m.offset} = physical byte ${m.physical}. ${access==='write'?'Write 99':'Read value '+m.value}${m.kind==='demand'?' after servicing the fault and retrying':''}.`];
  return <Investigation id="os-translation" kicker="VIRTUAL BYTE → PAGE ENTRY → PHYSICAL BYTE" title="Can the same address identify different data?">
    <p>Select A or B while holding virtual byte 22 fixed. These toy pages contain 16 bytes; real base pages are usually much larger. Page 0 is read-only, page 1 is private data, page 2 is valid but not resident, and page 3 is unmapped.</p><Predict>What must stay unchanged during translation? Is reading a valid nonresident page the same kind of fault as writing read-only code?</Predict>
    <div className="nt-controls"><label>Selected process<select value={process} onChange={e=>{setProcess(e.target.value);setStep(0);}}><option>A</option><option>B</option></select></label><label>Virtual byte address<select value={address} onChange={e=>{setAddress(Number(e.target.value));setStep(0);}}>{[0,1,2,3].map(page=>page*16+m.offset).map(v=><option key={v} value={v}>{v} · page {Math.floor(v/16)}</option>)}</select></label><label>Requested access<select value={access} onChange={e=>{setAccess(e.target.value);setStep(0);}}><option value="read">Read</option><option value="write">Write 99</option></select></label></div>
    <p><strong>Choose a byte within page {m.page}.</strong> Each position below is one byte offset. Predict which part of the physical address changes when you choose a neighbor.</p>
    <div className="offset-strip" role="group" aria-label="Byte offset within the selected virtual page">{Array.from({length:16},(_,offset)=><button type="button" key={offset} aria-label={`Offset ${offset}`} aria-pressed={m.offset===offset} onClick={()=>{setAddress(m.page*16+offset);setStep(0);}}>{offset}</button>)}</div>
    <p className="offset-strip-note">Offsets 0–15 form one consecutive page; grid rows wrap to fit the screen and are not page boundaries. Offset {m.offset} stays fixed when you change the process or virtual page. Choosing a byte restarts the translation.</p>
    <svg className="nt-diagram foundation-diagram" viewBox="0 0 360 265" role="img" aria-label={`Process ${process}: virtual ${address}, page ${m.page}, offset ${m.offset}; ${step<2?'translation in progress':m.kind}; ${step===3&&m.physical!==null?'physical '+m.physical:'no completed access yet'}`}>
      <defs><marker id={uid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#d8b16a"/></marker></defs>
      <rect x="20" y="10" width="320" height="54" fill="#221d13" stroke="#bd9b55"/><text x="34" y="43">Virtual {address} = page {m.page} · offset {m.offset}</text>
      <path d="M96 66 V102" stroke="#c4a567" strokeWidth="2" markerEnd={`url(#${uid})`}/>
      <rect x="18" y="109" width="144" height="68" fill="#151b17" stroke="#8aac92"/><text x="28" y="134">{process}'s page {m.page}</text><text x="28" y="157" className="nt-small">{step<1?'inspect mapping':m.entry?m.entry.rights:'no mapping'}</text>
      <path d="M165 142 H208" stroke="#c4a567" strokeWidth="2" markerEnd={`url(#${uid})`}/>
      <rect x="218" y="109" width="122" height="68" fill="#221d13" stroke="#bd9b55"/><text x="230" y="135">{step<2?'Pending':m.physical===null?'Blocked':'Frame '+m.frame}</text><text x="230" y="158" className="nt-small">{step<2?'check first':m.physical===null?'no access':'offset '+m.offset}</text>
      <text x="20" y="209">{step===3&&m.physical!==null?'Physical address: '+m.physical:'Same offset · different mapping'}</text><text x="20" y="241" className="nt-small">{step===3&&m.physical!==null?'Resulting byte value: '+m.value:'The selected process supplies the page table.'}</text>
    </svg>
    <LessonTable caption={`Process ${process}: permitted virtual pages${m.kind==='demand'&&step>=2?' after fault service':''}`} headers={['Virtual page','Resident frame','Permission']} rows={m.table.map(p=>[p.page,m.kind==='demand'&&step>=2&&p.page===m.page?m.frame:p.frame??'not resident',p.rights])}/>
    <p className="nt-feedback" aria-live="polite">{notes[step]}</p><Stepper step={step} count={4} setStep={setStep}/><p className="lesson-note">One-level page-table model; no TLB, eviction, multilevel walk or allocation failure. Values in each demonstrated frame are fixed illustrative bytes. A demand-page write is retried after allocation; the zero-fill is its initial content, not a forced final zero.</p>
  </Investigation>;
}

export function CopyOnWriteLab(){
  const uid=useId(),[shared,setShared]=useState(false),[step,setStep]=useState(0),trace=sharingTrace(shared),s=trace[step];
  const y=frame=>frame===2?92:219;
  return <Investigation id="os-sharing" kicker="PRIVATE BEHAVIOR CAN START WITH SHARED STORAGE" title="Why can copying wait until the first write?">
    <p>A and B initially read 7 from the same physical frame. Choose whether their mapping promises private writes or deliberately shared updates; then let B write 9.</p><Predict>Will A observe 7 or 9? Does “separate address spaces” forbid all intentional physical sharing?</Predict>
    <div className="nt-controls"><label>Mapping contract<select value={String(shared)} onChange={e=>{setShared(e.target.value==='true');setStep(0);}}><option value="false">Private · copy on write</option><option value="true">Explicit shared mapping</option></select></label></div>
    <svg className="nt-diagram foundation-diagram" viewBox="0 0 360 285" role="img" aria-label={`A maps frame ${s.aFrame} and reads ${s.aValue}; B maps frame ${s.bFrame} and reads ${s.bValue}`}>
      <defs><marker id={uid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#d8b16a"/></marker></defs>
      {['A','B'].map((id,i)=><g key={id}><rect x="12" y={46+i*144} width="122" height="63" fill="#1c221a" stroke="#8eab91"/><text x="26" y={73+i*144}>{id} reads {s[id.toLowerCase()+'Value']}</text><text x="26" y={96+i*144} className="nt-small">virtual page 1</text><path d={`M137 ${79+i*144} C164 ${79+i*144},184 ${y(s[id.toLowerCase()+'Frame'])},216 ${y(s[id.toLowerCase()+'Frame'])}`} fill="none" stroke="#d8b16a" strokeWidth="2" markerEnd={`url(#${uid})`}/></g>)}
      {Object.entries(s.frames).map(([frame,value])=><g key={frame}><rect x="226" y={y(Number(frame))-35} width="122" height="69" fill="#292013" stroke="#e1b869"/><text x="240" y={y(Number(frame))-8}>Frame {frame}</text><text x="240" y={y(Number(frame))+20}>value {value}</text></g>)}
    </svg>
    <p className="foundation-state">{s.phase}</p><p className="nt-feedback" aria-live="polite">{s.note}</p><Stepper step={step} count={trace.length} setStep={setStep}/>
    <p className="lesson-note">Two processes, one illustrative page, one write, available memory. Real private mappings can need no copy when a page is exclusively owned. Shared memory still needs a coordination protocol; a visible new value alone does not make a multi-step update safe.</p>
  </Investigation>;
}
