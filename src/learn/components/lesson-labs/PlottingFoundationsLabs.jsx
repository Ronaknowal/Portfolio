import { useState } from "react";
import { Investigation, Predict } from "./LessonInvestigation.jsx";
import { LessonTable } from "./LessonElements";
import { coordinateModel, histogramModel, histogramValues, intervalModel } from "../../data/plotting-foundations-model";

export function PlotCoordinateLab() {
  const [view,setView]=useState('full'),[selected,setSelected]=useState(0);
  const model=coordinateModel(view),point=model.points[selected];
  const ticks=model.scale==='log'?[1,10,100]:[model.min,(model.min+model.max)/2,model.max];
  const position=value=>235-190*(model.scale==='log'?(Math.log10(value)-Math.log10(model.min))/(Math.log10(model.max)-Math.log10(model.min)):(value-model.min)/(model.max-model.min));
  return <Investigation id="plot-coordinates" kicker="DATA → SCALE → POSITION" title="Can a point move without its measurement changing?">
    <p>A measured 10 ms and B measured 20 ms. Only the y-axis mapping changes. Follow one value from the data into the height of its plotted marker.</p><Predict>Zoom the limits, then use a log scale. Is B still twice A? Are equal vertical distances still equal differences?</Predict>
    <p>A base-10 logarithm answers “which power of 10 gives this value?” Thus log₁₀ 1 = 0, log₁₀ 10 = 1 and log₁₀ 100 = 2. On that scale, 1→10 and 10→100 occupy equal distances because both are tenfold changes.</p>
    <div className="nt-controls"><label>Axis mapping<select value={view} onChange={e=>setView(e.target.value)}><option value="full">Linear: 0 to 30 ms</option><option value="zoom">Zoom: 9 to 21 ms</option><option value="log">Log: 1 to 100 ms</option></select></label><label>Trace measurement<select value={selected} onChange={e=>setSelected(Number(e.target.value))}><option value="0">A: 10 ms</option><option value="1">B: 20 ms</option></select></label></div>
    <svg className="nt-diagram" viewBox="0 0 360 290" role="img" aria-label={`Dot plot of 10 and 20 milliseconds on a ${model.scale} axis from ${model.min} to ${model.max}`}>
      <rect x="58" y="32" width="280" height="214" fill="#14130f" stroke="#4b4333"/><text x="60" y="20" className="nt-small">Axes: one plotting panel · time (ms)</text>
      {ticks.map(t=><g key={t}><line x1="60" x2="332" y1={position(t)} y2={position(t)} stroke="#39382f"/><text x="48" y={position(t)+5} textAnchor="end">{t}</text></g>)}
      {model.points.map((p,i)=><g key={i}><line x1="61" x2={155+i*112} y1={position(p.value)} y2={position(p.value)} stroke={selected===i?'#e5bc63':'#82775f'} strokeDasharray="4 5"/><circle cx={155+i*112} cy={position(p.value)} r={selected===i?9:6} fill={selected===i?'#e5bc63':'#83b292'}/><text x={155+i*112} y="270" textAnchor="middle">{i?'B':'A'}</text></g>)}
    </svg>
    <p className="nt-feedback" aria-live="polite"><strong>Source value: {point.value} ms.</strong> {model.scale==='linear'?`(${point.value} − ${model.min}) ÷ (${model.max} − ${model.min})`:`(log₁₀ ${point.value} − log₁₀ ${model.min}) ÷ (log₁₀ ${model.max} − log₁₀ ${model.min})`} = {point.fraction.toFixed(3)} of the panel height from its lower limit. {model.scale==='log'?'Equal ratios occupy equal distances.':'Equal differences occupy equal distances.'}</p>
    <div className="nt-flow"><span>Figure<small>Canvas contains the panel</small></span><span>Axes + Axis<small>Panel, scale, ticks and units</small></span><span>Artist<small>The visible marker uses mapped coordinates</small></span></div>
    <button type="button" onClick={()=>{setView('full');setSelected(0);}}>Reset mapping</button><p className="lesson-note">This bounded model illustrates coordinate mapping for two fixed values. Dot positions are shown; a truncated bar would also hide part of its zero-based length. Real figures can contain more panels, transforms and annotations.</p>
  </Investigation>;
}

export function PlotHistogramLab() {
  const [layout,setLayout]=useState('three'),[density,setDensity]=useState(false),[selected,setSelected]=useState(0);
  const bins=histogramModel(layout,density),bin=bins[Math.min(selected,bins.length-1)],maxHeight=Math.max(...bins.map(b=>b.height))*1.2;
  return <Investigation id="plot-histogram" kicker="OBSERVATIONS → BINS → AREA" title="Does a taller bin always contain more observations?">
    <p>The six latencies are 1, 2, 2, 3, 7 and 9 ms. Bin width changes how many observations share a rectangle; n above a bin is its observation count. Density adjusts height so rectangle area represents the fraction.</p><Predict>With edges 0, 2, 10, the first bin has one value and the second has five. Will density make their heights differ fivefold?</Predict>
    <div className="nt-controls"><label>Bin edges<select value={layout} onChange={e=>{setLayout(e.target.value);setSelected(0);}}><option value="three">0, 3, 6, 10 ms</option><option value="two">0, 2, 10 ms</option></select></label><label>Rectangle height<select value={String(density)} onChange={e=>setDensity(e.target.value==='true')}><option value="false">Observation count</option><option value="true">Probability density (1/ms)</option></select></label><label>Trace bin<select value={bins.indexOf(bin)} onChange={e=>setSelected(Number(e.target.value))}>{bins.map((b,i)=><option key={b.lo} value={i}>{b.lo} to {b.hi} ms</option>)}</select></label></div>
    <svg className="nt-diagram" viewBox="0 0 360 280" role="img" aria-label={`${density?'Density':'Count'} histogram; selected bin ${bin.lo} to ${bin.hi} contains ${bin.count} of six observations`}>
      <text x="40" y="22">{density?'Density (1/ms)':'Count'}</text><line x1="40" x2="335" y1="218" y2="218" stroke="#a49a84"/>
      {[0,maxHeight/2,maxHeight].map((v,i)=><text key={i} x="34" y={218-175*v/maxHeight+5} textAnchor="end" className="nt-small">{density?v.toFixed(2):v.toFixed(1)}</text>)}
      {bins.map((b,i)=><g key={b.lo}><rect x={40+b.lo*29} y={218-175*b.height/maxHeight} width={b.width*29} height={175*b.height/maxHeight} fill={b===bin?'#8e7138':'#384b3d'} stroke="#d9c18a"/><text x={40+(b.lo+b.width/2)*29} y={210-175*b.height/maxHeight} textAnchor="middle">n={b.count}</text><text x={40+b.lo*29} y="239" textAnchor="middle">{b.lo}</text></g>)}<text x="330" y="239" textAnchor="middle">10</text><text x="182" y="271" textAnchor="middle">Latency (ms)</text>
    </svg>
    <p>Source observations: {histogramValues.map((value,i)=><span className="nt-chip" key={i} style={{borderColor:bin.indices.includes(i)?'#e5bd66':'#454033',color:bin.indices.includes(i)?'#f0cc79':'#aaa293'}}>{value}{bin.indices.includes(i)?' ↓':''}</span>)}</p>
    <p className="nt-feedback" aria-live="polite">{bin.count} observations ÷ 6 total = {(bin.mass).toFixed(3)} of the sample. {density?<>Height = {bin.count} ÷ (6 × {bin.width}) = {bin.height.toFixed(4)} per ms. Area = height × {bin.width} ms = {bin.mass.toFixed(3)}.</>:<>Count height = {bin.height}; unequal-width rectangle areas are not probabilities in this mode.</>}</p>
    <LessonTable caption="Exact bin values" headers={['Bin (ms)','Count','Width','Height']} rows={bins.map((b,i)=>[`${b.lo}–${b.hi}${i===bins.length-1?' inclusive':''}`,b.count,b.width,Number(b.height.toFixed(4))])}/>
    <button type="button" onClick={()=>{setLayout('three');setDensity(false);setSelected(0);}}>Reset histogram</button><p className="lesson-note">Bins include the lower edge and exclude the upper edge, except the last upper edge is included. Repeated values are separate observations. All six points lie inside these edges; other edges could exclude data.</p>
  </Investigation>;
}

export function PlotIntervalLab() {
  const [kind,setKind]=useState('sd'),[repeated,setRepeated]=useState(false);
  const model=intervalModel(kind,repeated),x=v=>34+(v-8)*36;
  return <Investigation id="plot-interval" kicker="LABEL THE QUANTITY, NOT JUST THE BAR" title="Same points and mean: why did the interval shrink?">
    <p>Three runs produced 10, 12 and 14 ms. A bar labelled sample SD describes their spread. A bar labelled SEM uses SD divided by √n to estimate variability of the mean under an independent-sampling model.</p><Predict>Does choosing SEM make individual runs less variable? Would copying every row create additional independent evidence?</Predict>
    <p>Here the mean is 12. Deviations are −2, 0 and 2; squaring gives 4, 0 and 4. Sample standard deviation (SD) is √((4 + 0 + 4) / (3 − 1)) = 2 ms. Standard error of the mean (SEM) is 2 / √3 ≈ 1.155 ms. They describe different quantities, so the label must name which you show.</p>
    <div className="nt-controls"><label>Interval shown<select value={kind} onChange={e=>setKind(e.target.value)}><option value="sd">Mean ± sample SD</option><option value="sem">Mean ± SEM</option></select></label><label>Dataset<select value={String(repeated)} onChange={e=>setRepeated(e.target.value==='true')}><option value="false">Three independent-run observations</option><option value="true">Copy the same rows twice</option></select></label></div>
    <svg className="nt-diagram" viewBox="0 0 360 260" role="img" aria-label={`Mean ${model.mean}, ${kind} interval ${model.low.toFixed(3)} to ${model.high.toFixed(3)}, ${model.n} rows`}>
      <text x="24" y="23">Individual recorded values</text>{model.values.map((v,i)=><circle key={i} cx={x(v)} cy={55+(i>=3?22:0)} r="6" fill={i>=3?'#7fa68d':'#e3b963'}/>)}
      <line x1={x(model.low)} x2={x(model.high)} y1="144" y2="144" stroke="#e7c579" strokeWidth="3"/>{[model.low,model.high].map(v=><line key={v} x1={x(v)} x2={x(v)} y1="131" y2="157" stroke="#e7c579" strokeWidth="2"/>)}<circle cx={x(model.mean)} cy="144" r="7" fill="#e7c579"/><text x="180" y="114" textAnchor="middle">Mean ± {kind.toUpperCase()}</text>
      <line x1="34" x2="322" y1="200" y2="200" stroke="#918b7c"/>{[8,10,12,14,16].map(v=><g key={v}><line x1={x(v)} x2={x(v)} y1="200" y2="207" stroke="#918b7c"/><text x={x(v)} y="229" textAnchor="middle">{v}</text></g>)}<text x="175" y="255" textAnchor="middle">Time (ms)</text>
    </svg>
    <p className="nt-feedback" aria-live="polite">n = {model.n}, mean = {model.mean}, sample SD = {model.sd.toFixed(3)}. Half-width = {model.halfWidth.toFixed(3)} ms; endpoints {model.low.toFixed(3)} and {model.high.toFixed(3)}. {repeated?<strong>These extra rows are copies. A smaller computed SEM would not establish more independent evidence.</strong>:'Changing the bar definition leaves the three observed values unchanged.'}</p>
    <button type="button" onClick={()=>{setKind('sd');setRepeated(false);}}>Reset interval</button><p className="lesson-note">Neither choice here is labelled a 95% confidence interval. Sample SD uses n−1. Whether independent sampling is justified is a separate scientific question.</p>
  </Investigation>;
}
