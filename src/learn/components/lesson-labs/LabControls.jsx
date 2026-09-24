import { useId } from 'react';

export const formatLabValue=v=>v===null?'not defined':typeof v==='object'?JSON.stringify(v):String(v);

export function LabChoices({label,value,onChange,items}){const id=useId();return <label><span id={id}>{label}</span><select aria-labelledby={id} value={value} onChange={e=>onChange(e.target.value)}>{items.map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label>;}

export function LabFeedback({children}){return <p className="rel-feedback" aria-live="polite">{children}</p>;}
