import { useId, useState } from "react";
import { Investigation, Predict, Stepper } from "./LessonInvestigation.jsx";
import { LessonTable } from "./LessonElements";
import { alignmentModel, cleaningModel, groupInputs, groupingModel } from "../../data/pandas-foundations-model";

export function PandasAlignmentLab() {
  const uid=useId();
  const [mode,setMode]=useState('labels'),[reversed,setReversed]=useState(false),[incomplete,setIncomplete]=useState(false),[selected,setSelected]=useState(0);
  const model=alignmentModel(mode,reversed,incomplete),row=model.rows[selected];
  return <Investigation id="pandas-alignment" kicker="FOLLOW THE LABEL" title="Which fee reaches this order?">
    <p>The table's visible order is b, a. Fees arrive in a different order. Predict whether an order's total should change when you reorder the rows.</p>
    <Predict>With label alignment, b receives fee 2 even though 2 is the second fee. What happens when you remove the fee for a?</Predict>
    <div className="nt-controls"><label>Assignment rule<select value={mode} onChange={e=>setMode(e.target.value)}><option value="labels">Match index labels</option><option value="positions">Use values by position</option></select></label><label>Order row arrangement<select value={String(reversed)} onChange={e=>{setReversed(e.target.value==='true');setSelected(0);}}><option value="false">b, then a</option><option value="true">a, then b</option></select></label><label>Incoming fee labels<select value={String(incomplete)} onChange={e=>setIncomplete(e.target.value==='true')}><option value="false">a: 1 and b: 2</option><option value="true">b: 2 and c: 7 (a missing)</option></select></label><label>Inspect order row<select value={selected} onChange={e=>setSelected(Number(e.target.value))}>{model.orders.map((r,i)=><option key={r.label} value={i}>{r.label} — amount {r.value}</option>)}</select></label></div>
    <svg className="nt-diagram" viewBox="0 0 360 230" role="img" aria-label={`Order ${row.label} receives ${row.fee===null?'no matching fee':'fee '+row.fee} using ${mode}`}>
      <defs><marker id={uid} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#deb665"/></marker></defs>
      <text x="14" y="22">Incoming fees</text><text x="218" y="22">Order {row.label}</text>
      {model.fees.map((f,i)=><g key={f.label}><rect x="14" y={45+i*92} width="116" height="62" rx="3" fill={row.source===i?'#302616':'#161510'} stroke={row.source===i?'#e0b456':'#554b37'}/><text x="26" y={70+i*92} className="nt-emphasis">{f.label}: {f.value}</text><text x="26" y={92+i*92} className="nt-small">position {i}</text></g>)}
      {row.source>=0?<path d={`M133,${76+row.source*92} C180,${76+row.source*92} 173,116 216,116`} fill="none" stroke="#deb665" strokeWidth="2" markerEnd={`url(#${uid})`}/>:<text x="142" y="115" className="nt-small">no match</text>}
      <rect x="220" y="79" width="128" height="88" rx="3" fill="#1c241d" stroke="#78a581"/><text x="230" y="108">{row.value} + {row.fee??'NA'}</text><text x="230" y="140" className="nt-emphasis">= {row.total??'NA'}</text>
    </svg>
    <p className="nt-feedback" aria-live="polite">{mode==='labels'?`Look for label ${row.label}; position does not decide the match.`:`Take fee position ${selected}; its label is ignored.`} {row.source<0?'No matching label means an unknown total, not a zero fee.':`Order ${row.label} receives ${row.fee}, giving ${row.total}.`}</p>
    <LessonTable caption="All destination rows" headers={['Label','Amount','Fee','Total']} rows={model.rows.map(r=>[r.label,r.value,r.fee??'NA',r.total??'NA'])}/>
    <button type="button" onClick={()=>{setMode('labels');setReversed(false);setIncomplete(false);setSelected(0);}}>Reset alignment</button>
    <p className="lesson-note">Bounded model: two unique row labels and a two-value Series. The labelled result depicts assignment back to the order table. Series arithmetic can also produce the fee-only label c in an intermediate result; assigning back to this table retains only its destination rows.</p>
  </Investigation>;
}

export function PandasCleaningLab() {
  const [step,setStep]=useState(0),[policy,setPolicy]=useState('known');
  const rows=cleaningModel(policy);
  return <Investigation id="pandas-cleaning" kicker="KEEP THE REASON FOR MISSINGNESS" title="Can a successful conversion hide a bad input?">
    <p>Four text readings arrive: "10", an empty string, "bad", and "0". Coercion lets the pipeline continue, but the original text tells you why a value is absent.</p>
    <Predict>Which rows survive “known values”? Which survive “strictly positive”? Does a zero belong with missing data?</Predict>
    <div className="nt-controls"><label>Selection policy<select value={policy} onChange={e=>{setPolicy(e.target.value);setStep(0);}}><option value="known">Keep known numbers, including zero</option><option value="positive">Keep strictly positive numbers</option></select></label></div>
    <div className="nt-flow" aria-label="Cleaning stages">{['Raw text','Nullable numbers','Selection mask','Selected rows'].map((s,i)=><span key={s} aria-current={step===i?'step':undefined}>{i+1}. {s}<small>{step===i?'Current step':i<step?'Inspected':'Not yet inspected'}</small></span>)}</div>
    <ul className="nt-records">{rows.map(row=><li key={row.index} className={step===3&&row.keep?'is-linked':''}><strong>Row {row.index}: {JSON.stringify(row.raw)}</strong>{step>=1&&<span> → {row.value??'NA'}</span>}{step>=2&&<span> → {row.mask===null?'NA':String(row.mask)}</span>}{step===3&&<strong> → {row.keep?'kept':'excluded'}</strong>}<small>{row.reason}</small></li>)}</ul>
    <p className="nt-feedback" aria-live="polite">{[
      'These are strings. Preserve raw text alongside converted values so that an invalid token does not become indistinguishable from an empty field.',
      'to_numeric(errors="coerce") turns unparseable text into missing. Casting to Float64 gives a nullable numeric Series. Coercion has not validated or repaired the source.',
      policy==='known'?'notna() returns True for 10 and 0, False for both absent numeric values.':'The comparison value > 0 yields True, NA, NA, False. With this nullable mask, .loc treats NA as not selected.',
      `Selected source rows: ${rows.filter(r=>r.keep).map(r=>r.index).join(', ')}. Keep excluded rows and their reasons for inspection; do not silently describe them as all zero.`
    ][step]}</p><Stepper step={step} count={4} setStep={setStep}/>
    <p className="lesson-note">The four fixed inputs illustrate conversion and nullable masking, not a general numeric validator. Units, finite values and accepted ranges require separate checks.</p>
  </Investigation>;
}

export function PandasGroupingLab() {
  const [keepMissing,setKeepMissing]=useState(true),[fillZero,setFillZero]=useState(false),[key,setKey]=useState('North');
  const result=groupingModel(keepMissing,fillZero),group=result.groups.find(g=>g.key===key)||result.groups[0];
  return <Investigation id="pandas-grouping" kicker="SPLIT → REDUCE → RETURN" title="Two rows: why is the mean 10 rather than 5?">
    <p>North has one observed amount, 10, and one unknown amount. Select a group to follow the records that contribute to its summary and to a transform returned to the original rows.</p>
    <Predict>Count rows separately from measured values. Then decide whether filling an unknown with zero changes the question being answered.</Predict>
    <div className="nt-controls"><label>Missing group key<select value={String(keepMissing)} onChange={e=>{setKeepMissing(e.target.value==='true');setKey('North');}}><option value="true">Keep it: dropna=False</option><option value="false">Omit it: dropna=True</option></select></label><label>Unknown amount policy<select value={String(fillZero)} onChange={e=>setFillZero(e.target.value==='true')}><option value="false">Preserve unknown values</option><option value="true">Assume unknown means zero</option></select></label><label>Trace group<select value={group.key??'missing'} onChange={e=>setKey(e.target.value==='missing'?null:e.target.value)}>{result.groups.map(g=><option key={g.key??'missing'} value={g.key??'missing'}>{g.key??'Unknown region'}</option>)}</select></label></div>
    <h4>Source rows → group membership</h4><ul className="nt-records">{groupInputs.map(r=><li key={r.id} className={group.members.includes(r.id)?'is-linked':''}><strong>{r.id}</strong> · {r.region??'Unknown region'} · {r.amount??'NA'}<small>{group.members.includes(r.id)?'↓ contributes to this group':'Outside selected group'}</small></li>)}</ul>
    <div className="nt-reducer" aria-live="polite"><h4>{group.key??'Unknown region'} reducer</h4><p>{group.size} rows → {group.count} known amounts → sum {group.sum}</p><p className="nt-big-value">{group.sum} ÷ {group.count} = {group.mean??'NA'}</p><p>{fillZero?'The zero assumption increased the known-value denominator. It did not discover the missing measurement.':'The missing amount does not count as an observed zero.'}</p></div>
    <div className="nt-columns"><div><h4>agg: one row per group</h4><ul className="nt-records">{result.groups.map(g=><li key={g.key??'missing'}>{g.key??'Unknown'} → mean {g.mean??'NA'}</li>)}</ul></div><div><h4>transform: return to each source row</h4><ul className="nt-records">{result.transformed.map(r=><li key={r.id}>{r.id} ← {r.mean??'NA'}<small>Original row count retained</small></li>)}</ul></div></div>
    <button type="button" onClick={()=>{setKeepMissing(true);setFillZero(false);setKey('North');}}>Reset grouping</button>
    <p className="lesson-note">The display models mean transformation only. Omitting unknown group keys leaves their transform values absent; it does not remove the original rows. More general apply operations have different shape contracts.</p>
  </Investigation>;
}
