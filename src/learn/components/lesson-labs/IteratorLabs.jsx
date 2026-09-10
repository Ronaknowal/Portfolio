import { useState } from "react";
import { Investigation, Predict, Stepper } from "./LessonInvestigation.jsx";
import { cursorTrace, generatorTrace, pipelineTrace } from "../../data/iterator-models.js";


import './iteration-decorators.css';
import { CursorOwnershipMap } from "./IteratorFigures.jsx";

export function CursorOwnershipLab() {
  const [shared,setShared]=useState(false),[empty,setEmpty]=useState(false),[step,setStep]=useState(0), trace=cursorTrace(shared,empty),cursorState=trace[step];
  return <Investigation id="iterator-ownership" kicker="ONE COLLECTION · HOW MANY POSITIONS?" title="Does b have its own next item?">
    <p>The list stays unchanged. We alternate next(a, "END") and next(b, "END"). Each arrow points to the item its cursor would return next; END means nothing remains.</p>
    <Predict>After a returns 18, does b return 18 or 21? Decide for both setup choices before stepping.</Predict>
    <div className="nt-controls"><label>Cursor setup<select value={String(shared)} onChange={e=>{setShared(e.target.value==='true');setStep(0)}}><option value="false">a = iter(values); b = iter(values)</option><option value="true">a = iter(values); b = a</option></select></label><label>Source list<select value={String(empty)} onChange={e=>{setEmpty(e.target.value==='true');setStep(0)}}><option value="false">[18, 21, 24]</option><option value="true">[]</option></select></label></div>
    <CursorOwnershipMap state={cursorState} shared={shared} />
    <p className="id-output">Returned so far: {cursorState.output.join(' → ')||'none'}</p><p className="nt-feedback" aria-live="polite">{cursorState.note}</p><Stepper step={step} count={trace.length} setStep={setStep}/>
    <p className="lesson-note">Positions describe these fixed list iterators, not a universal iterator layout or actual memory pointers. Both independent cursors read the same source; sharing a cursor also shares its changing position. Transfer: choose empty input—why do both setups now produce the same returned values?</p>
  </Investigation>;
}

export function GeneratorFrameLab() {
  const [action,setAction]=useState('exhaust'),[step,setStep]=useState(0),trace=generatorTrace(action),generatorState=trace[step];
  return <Investigation id="generator-frame" kicker="CALLER ⇄ SUSPENDED FUNCTION" title="Where does execution wait after yield?">
    <p>This generator assigns remaining = 2, yields it while positive, then subtracts 1 after each resumption. A finally block records cleanup. Each step is one caller operation: creation, next, or close.</p><Predict>Immediately after receiving 2, is remaining 2 or 1? Will receiving 1 already run cleanup?</Predict>
    <div className="nt-controls"><label>Caller actions<select value={action} onChange={e=>{setAction(e.target.value);setStep(0)}}><option value="exhaust">next until exhausted, then next again</option><option value="close-started">next once, then close</option><option value="close-created">close before first next</option></select></label></div>
    <div className="id-spatial"><div className="id-frame"><div><strong>Caller has received</strong><span>{generatorState.output.join(', ')||'no items'}</span></div><div><strong>Generator: {generatorState.status}</strong><span>{generatorState.remaining===null?'No active local value to inspect':'remaining = '+generatorState.remaining}</span></div></div><div className="id-flow"><div className="id-flow-node">Caller requests an item ↓</div><div className="id-flow-node" data-active={generatorState.status==='SUSPENDED'}>{generatorState.position}</div><div className="id-flow-node">↑ Yielded value or end signal returns control</div></div></div>
    <p className="nt-feedback" aria-live="polite">{generatorState.note}</p><Stepper step={step} count={trace.length} setStep={setStep}/><p className="lesson-note">The matching program below uses inspect.getgeneratorstate and real next/close calls. This picture abstracts Python's execution frame; it is not a memory-address diagram. Transfer: add a third initial count and predict the extra suspension before running it.</p>
  </Investigation>;
}

export function PullPipelineLab() {
  const [limit,setLimit]=useState(2),[bad,setBad]=useState(false),[step,setStep]=useState(0),trace=pipelineTrace(limit,bad),pipelineState=trace[step];
  return <Investigation id="iterator-pipeline" kicker="DEMAND GOES UPSTREAM · DATA COMES BACK" title="Can one next call read several lines?">
    <p>A consumer wants one or two numeric readings. Its source has four lines; line 2 is blank. Select malformed line 3 to see when a delayed error appears.</p><Predict>How many source lines supply two readings? With only one requested result, is bad on line 3 ever parsed?</Predict>
    <div className="nt-controls"><label>Results requested<select value={limit} onChange={e=>{setLimit(Number(e.target.value));setStep(0)}}><option value="1">1 reading</option><option value="2">2 readings</option></select></label><label>Line 3<select value={String(bad)} onChange={e=>{setBad(e.target.value==='true');setStep(0)}}><option value="false">24 · valid</option><option value="true">bad · invalid</option></select></label></div>
    <div className="id-spatial"><div className="id-slots">{pipelineState.lines.map((l,i)=><div key={i} className="id-slot" data-current={i===pipelineState.read-1}><strong>{l||'blank'}</strong><small>line {i+1} · {i<pipelineState.read?'read':'unread'}</small></div>)}</div><div className="id-flow">{[['source','Source · next line'],['filter','Strip and keep nonblank'],['parse','Convert string to number'],['consumer','Consumer · receive result']].map(([id,label],i)=><div key={id}><div className="id-flow-node" data-active={pipelineState.active===id}>{label}{pipelineState.active===id&&pipelineState.item!==''?' · '+pipelineState.item:''}</div>{i<3&&<div className="id-flow-arrow">↓ data · ↑ next request</div>}</div>)}</div></div>
    <p className="id-output">Read: {pipelineState.read} / 4 lines · Results: [{pipelineState.received.join(', ')}]{pipelineState.error?' · '+pipelineState.error:''}</p><p className="nt-feedback" aria-live="polite">{pipelineState.note}</p><Stepper step={step} count={trace.length} setStep={setStep}/><p className="lesson-note">A synchronous pull pipeline, with no background producer or network buffering. The fixture is already in memory; counts measure reads and results, not RAM bytes. A blank may require more upstream work for the same requested output.</p>
  </Investigation>;
}
