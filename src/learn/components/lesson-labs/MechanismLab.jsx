import { useId } from "react";
import './workflow-labs.css';

export function MechanismLab({id,title,children}) { const uid=useId(); return <section className="wc-lab" data-investigation={id} aria-labelledby={uid}><h3 id={uid}>{title}</h3>{children}</section>; }

export function StatePanel({title,children}) {return <div className="wc-state"><strong>{title}</strong>{children}</div>;}

export function ExecutionLog({items}) {return <ol className="wc-timeline" aria-label="Execution trace">{items.map((text,i)=><li key={i}>{text}</li>)}</ol>;}
