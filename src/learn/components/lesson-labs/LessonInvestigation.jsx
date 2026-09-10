import { useId } from 'react';
import "./lesson-investigations.css";

export function Investigation({id,kicker,title,children}) {
  const uid=useId();
  return <section className="lesson-lab nt-lab" data-investigation={id} aria-labelledby={uid}><p className="lesson-eyebrow">{kicker}</p><h3 id={uid}>{title}</h3>{children}</section>;
}
export function Stepper({step,count,setStep,onReset}) {
  return <div className="nt-stepper"><button type="button" disabled={!step} onClick={()=>setStep(step-1)}>Back</button><span>Step {step+1} of {count}</span><button type="button" disabled={step===count-1} onClick={()=>setStep(step+1)}>Next step</button><button type="button" onClick={onReset||(()=>setStep(0))}>Reset</button></div>;
}
export function Predict({children}) {
  return <details className="nt-predict"><summary>Predict before exploring</summary><p>{children}</p></details>;
}
export function LearningResources({children}) {
  return <div className="nt-resources"><h4>Another explanation or guided practice</h4><ul>{children}</ul></div>;
}
