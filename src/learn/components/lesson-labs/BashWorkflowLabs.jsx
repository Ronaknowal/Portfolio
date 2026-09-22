import { MechanismLab, StatePanel } from "./MechanismLab.jsx";
import { useState } from "react";
import { argumentTrace, pipelineStatus, publicationTrace } from "../../data/bash-workflow-models.js";


import { Stepper } from "./LessonInvestigation.jsx";
import { BashArgumentPicture, BashPipelinePicture } from "./BashFigures.jsx";

import "./workflow-labs.css";





export function BashArgumentsLab() {
  const [fixture,setFixture]=useState('spaces'),[quoted,setQuoted]=useState(true),[step,setStep]=useState(0);
  const trace=argumentTrace(fixture,quoted);
  return <MechanismLab id="bash-arguments" title="Where does one argument end?">
    <p>Explore how many arguments a program receives from <code>show_args $value</code>. Then change the quotes. A filename with a space is one value; the shell can accidentally turn it into two.</p>
    <label>Value <select value={fixture} onChange={e=>{setFixture(e.target.value);setStep(0);}}><option value="spaces">run alpha.csv</option><option value="wildcard">*.csv</option><option value="empty">Empty string</option></select></label>
    <label>Expansion <select value={String(quoted)} onChange={e=>{setQuoted(e.target.value==='true');setStep(0);}}><option value="true">Quoted: "$value"</option><option value="false">Unquoted: $value</option></select></label>
    <Stepper step={step} count={4} setStep={setStep}/>
    <BashArgumentPicture trace={trace} quoted={quoted} step={step}/>
    <p className="wc-feedback" aria-live="polite">{trace.phases[step]}</p>
    <p>Bounded model: default IFS, no custom shell options; the wildcard fixture has exactly a.csv and run alpha.csv. Quoting prevents splitting/globbing, but a leading dash can still be interpreted as a command option. Transfer: what would happen to the empty value with and without quotes?</p>
  </MechanismLab>;
}

export function BashStatusLab() {
  const [upstream,setUpstream]=useState(4),[downstream,setDownstream]=useState(0),[pipefailEnabled,setPipefailEnabled]=useState(false);
  const status=pipelineStatus([upstream,downstream],pipefailEnabled);
  return <MechanismLab id="bash-status" title="Data can arrive even when a producer fails">
    <p>A producer prints two rows, then fails. The consumer reads those rows and exits successfully. Inspect the pipeline status before enabling pipefail.</p>
    <label>Producer status <select value={upstream} onChange={e=>setUpstream(Number(e.target.value))}><option value={0}>0 · success</option><option value={4}>4 · partial output then failure</option></select></label>
    <label>Consumer status <select value={downstream} onChange={e=>setDownstream(Number(e.target.value))}><option value={0}>0 · success</option><option value={2}>2 · consumer error</option></select></label>
    <label><input type="checkbox" checked={pipefailEnabled} onChange={e=>setPipefailEnabled(e.target.checked)}/> Enable pipefail</label>
    <BashPipelinePicture upstream={upstream} downstream={downstream} strict={pipefailEnabled} status={status}/>
    <p className="wc-feedback" aria-live="polite">Pipeline status: <strong>{status}</strong>. {pipefailEnabled?'Rightmost nonzero status, or zero when all succeed.':'Status of the last command only.'} No output has been rolled back.</p>
    <p>Transfer: if both commands fail, why is the result 2 rather than 4? pipefail selects the rightmost failure, not the earliest failure in time. An expected no-match status still needs your explicit policy.</p>
  </MechanismLab>;
}

export function BashPublicationLab() {
  const [failure,setFailure]=useState(false),[step,setStep]=useState(0),states=publicationTrace(failure),state=states[step];
  return <MechanismLab id="bash-publication" title="Which version can a reader see?">
    <p>A previously complete report exists. Inspect whether a failed replacement should change it, then follow staging and publication.</p>
    <label>Producer outcome <select value={String(failure)} onChange={e=>{setFailure(e.target.value==='true');setStep(0);}}><option value="false">Complete successfully</option><option value="true">Fail after partial output</option></select></label>
    <Stepper step={step} count={states.length} setStep={setStep}/>
    <div className="wc-grid"><StatePanel title="Staging file">{state.staged}</StatePanel><StatePanel title="Public destination">{state.visible}</StatePanel><StatePanel title="Command status">{state.status}</StatePanel></div>
    <p className="wc-feedback" aria-live="polite">{state.step}</p>
    <p>Model assumes a successful same-filesystem rename. It does not simulate crashes, fsync durability, concurrent publishers or network-filesystem behavior. Directly redirecting into the public file would truncate it before the producer succeeds.</p>
  </MechanismLab>;
}
