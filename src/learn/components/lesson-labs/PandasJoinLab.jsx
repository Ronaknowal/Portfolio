import { useState } from "react";
import { LessonTable } from "./LessonElements";
import { joinOrders, lookupRows, modelJoin } from "../../data/pandas-join-model";
import "./python-trace.css";

export default function PandasJoinLab() {
  const [how, setHow] = useState("left");
  const [duplicate, setDuplicate] = useState(false);
  const [validate, setValidate] = useState(true);
  const [selected, setSelected] = useState(0);
  const result = modelJoin(how, duplicate, validate);
  const active = result.rows[Math.min(selected, Math.max(0,result.rows.length-1))];
  return <section className="lesson-lab pandas-join-lab" aria-label="Pandas join explorer">
    <h3>Explore how many rows survive the join</h3>
    <p>These are the tables in the merge example below. C9 has an order but no customer record; C3 has a customer record but no order. This illustrates join behaviour, not a full Pandas interpreter.</p>
    <div className="lesson-controls">
      <label>Join type<select aria-label="Join type" value={how} onChange={e => setHow(e.target.value)}>
        <option value="left">Left: preserve orders</option><option value="inner">Inner: matched keys only</option><option value="outer">Outer: preserve both sides</option>
      </select></label>
      <button type="button" onClick={() => { setHow("left"); setDuplicate(false); setValidate(true); setSelected(0); }}>Reset join</button>
    </div>
    <div className="join-options">
      <label><input type="checkbox" checked={duplicate} onChange={e => setDuplicate(e.target.checked)} />Add a second C1 lookup row</label>
      <label><input type="checkbox" checked={validate} onChange={e => setValidate(e.target.checked)} />Enforce many-to-one validation</label>
    </div>
    <LessonTable caption="Left table: orders (one row per order)" headers={["Order ID", "Customer"]} rows={joinOrders.map(row => [row.order, row.customer])} />
    <LessonTable caption="Right table: customer lookup" headers={["Customer", "Region"]} rows={lookupRows(duplicate).map(row => [row.customer, row.region])} />
    <div className="lesson-results" aria-live="polite">
      {result.error ? <p><strong>{result.error}</strong> The join is rejected before producing a misleading order report. Turn validation off only to inspect how the duplicate multiplies rows.</p> :
        <p><strong>{result.rows.length} result rows</strong> from 3 orders and {lookupRows(duplicate).length} lookup rows. {duplicate ? "Order 101 appears twice: each C1 lookup row contributes a match." : "Matched orders appear once because lookup keys are unique."}</p>}
    </div>
    {!result.error && <><h4>Select an output row to trace its origin</h4><div className="nt-records">{result.rows.map((row,i)=><button type="button" key={i} aria-pressed={active===row} onClick={()=>setSelected(i)}>{row.order??'No order'} · {row.customer} · {row.region??'No region'}<small>{row.match}</small></button>)}</div>
      <div className="nt-columns" aria-live="polite"><div className="nt-reducer"><h4>Left source →</h4><p>{active?.order?`Order ${active.order}, customer ${active.customer}`:'No left record: outer join kept a lookup-only customer.'}</p></div><div className="nt-reducer"><h4>← Right source</h4><p>{active?.region?`Customer ${active.customer}, region ${active.region}`:'No right record: the preserved order has no customer match.'}</p></div></div>
      <p className="nt-feedback">{active?.match==='both'?`Customer key ${active.customer} connects these two source records. ${duplicate&&active.customer==='C1'?'There is another C1 lookup row, so the same order produces another output row.':'Other orders can reuse this lookup without changing the right key’s uniqueness.'}`:'A preserved unmatched record produces one row with absent fields from the other side.'}</p>
      <LessonTable caption="Join result: read the match indicator" headers={["Order ID", "Customer", "Region", "Match"]} rows={result.rows.map(row => [row.order ?? "missing", row.customer, row.region ?? "missing", row.match])} /></>}
    <p className="lesson-note">“Missing” represents an absent value, not the literal text stored by Pandas. Display order is arranged for explanation; sort a real result explicitly when row order is part of the contract.</p>
  </section>;
}
