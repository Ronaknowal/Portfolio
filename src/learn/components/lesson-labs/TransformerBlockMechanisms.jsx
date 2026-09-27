import { useId } from 'react';
const amber = '#e6b854';
const tones = [amber, '#b8a0d2', '#9cbfe2'];
function Diagram({ title, width, height, children }) {
  return <div className="block-diagram-scroll" role="region" aria-label={title} tabIndex={0}><svg style={{ minWidth: width, width: '100%', height: 'auto' }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>{children}</svg></div>;
}
function Label({ x, y, children, size = 14, anchor = 'middle' }) {
  return <text x={x} y={y} textAnchor={anchor} fill="#eee" fontSize={size}>{children}</text>;
}
function Box({ x, y, width, label, dashed = false }) {
  return <g><rect x={x} y={y} width={width} height="34" rx="3" fill="#202020" stroke={amber} strokeDasharray={dashed ? '4 3' : undefined} /><Label x={x + width / 2} y={y + 22}>{label}</Label></g>;
}
function Mask({ x, y, causal, cross = false }) {
  return <g><Label x={x + 64} y={y - 12}>K / V positions →</Label><Label x={x - 10} y={y + 44} anchor="end">Q ↓</Label>{[0, 1, 2].flatMap(row => [0, 1, 2, 3].map(col => {
    const legal = col < 3 && (!causal || col <= row);
    return <g key={`${row}-${col}`}><rect x={x + col * 31} y={y + row * 29} width="27" height="25" fill={legal ? '#756032' : '#202020'} stroke="#777" /><Label x={x + col * 31 + 13} y={y + row * 29 + 17} size={12}>{legal ? '✓' : '×'}</Label></g>;
  }))}<Label x={x + 108} y={y + 106} size={12}>{cross ? 'source pad' : 'pad'}</Label></g>;
}
export function BlockTaskDiagrams() {
  return <div className="block-three">{['Encoder', 'Decoder-only', 'Encoder–decoder'].map((type, index) => <section key={type}><h4>{type}</h4><Diagram title={`${type}: inputs, query/key/value origins and legal positions`} width={290} height={index === 2 ? 470 : 385}>
    {index < 2 ? <><Box x={28} y={20} width={234} label={index === 0 ? 'source rows: s₁  s₂  s₃  PAD' : 'input rows: BOS  a  b  PAD'} /><path d="M145 54 V82 M145 65 H80 V82 M145 65 H210 V82" stroke={amber} fill="none" /><Label x={80} y={103}>Q</Label><Label x={145} y={103}>K</Label><Label x={210} y={103}>V</Label><Mask x={100} y={150} causal={index === 1} /><path d="M160 240 V281" stroke={amber} /><Box x={40} y={282} width={210} label="attention → FFN → readout" />{index === 1 ? <><Label x={145} y={345}>targets: a  b  c</Label><Label x={145} y={370} size={12}>Input b predicts c; c is never a key.</Label></> : <Label x={145} y={350}>pool → class, or one output per row</Label>}</> : <><Box x={15} y={20} width={260} label="source → encoder → fixed states" /><path d="M200 54 V151" stroke={amber} /><Label x={220} y={84}>K, V</Label><Box x={15} y={100} width={154} label="target causal block" /><path d="M90 134 V170 H128" stroke={amber} fill="none" /><Label x={85} y={159}>Q</Label><Box x={130} y={152} width={144} label="cross-attention" /><path d="M202 186 V216" stroke={amber} /><Mask x={99} y={254} cross /><Label x={145} y={385} size={12}>Every decoder Q can read valid source K/V.</Label><Label x={145} y={410} size={12}>Source padding is excluded independently.</Label><Label x={145} y={440} size={12}>Target self-attention uses the causal mask.</Label></>}
  </Diagram></section>)}</div>;
}
export function BlockWriteStacks({ contributions, output, selected }) {
  const ext = Math.max(.01, ...output.map((_, d) => contributions.reduce((sum, row) => sum + Math.abs(row[d]), 0)));
  const px = value => 220 + value / ext * 165;
  return <Diagram title="Each output coordinate adds signed writes from hidden responses" width={440} height={245}><Label x={220} y={20}>Signed contributions; common scale ±{ext.toFixed(3)}</Label><path d="M220 35 V208" stroke="#aaa" />{output.map((value, d) => {
    let negative = 0, positive = 0;
    return <g key={d}><Label x={38} y={65 + d * 43} size={12}>out {d + 1}</Label>{contributions.map((row, h) => {
      const start = row[d] < 0 ? negative : positive;
      if (row[d] < 0) negative += row[d]; else positive += row[d];
      return <rect key={h} x={Math.min(px(start), px(start + row[d]))} y={47 + d * 43} width={Math.abs(row[d]) / ext * 165} height="21" fill={tones[h % 3]} stroke={h === selected ? '#fff' : 'none'} strokeWidth="2"><title>Hidden {h + 1}: {row[d].toFixed(6)}</title></rect>;
    })}<path d={`M${px(value)} ${72 + d * 43} l-4 6 h8 Z`} fill="#fff" /></g>;
  })}<Label x={220} y={238} size={12}>White marker = net sum; outlined segment = selected response.</Label></Diagram>;
}
export function BlockSystemsFigure() {
  const id = useId().replaceAll(':', '');
  return <figure className="block-circuit"><figcaption>Store boundaries; split features; preserve the computation</figcaption>
    <Diagram title="Checkpointed forward stores boundaries, backward recomputes missing interior states" width={600} height={245}>
      <defs><marker id={id} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill={amber} /></marker></defs>
      <Label x={300} y={22}>Forward: keep boundaries, discard selected intermediates</Label>
      {[['saved x', 15, false], ['Norm', 160, true], ['Attention', 280, true], ['saved z', 485, false]].map(([name, x, dashed]) => <Box key={name} x={x} y={50} width={name==='Attention'?125:100} label={name} dashed={dashed} />)}
      <path d="M115 67 H160 M260 67 H280 M405 67 H421 M449 67 H485" fill="none" stroke={amber} markerEnd={`url(#${id})`} />
      <circle cx="435" cy="67" r="14" fill="#111" stroke={amber} /><Label x={435} y={72}>+</Label><path d="M65 84 V103 H435 V82" fill="none" stroke={amber} markerEnd={`url(#${id})`} /><Label x={300} y={96} size={12}>saved x bypass</Label>
      <path d="M65 84 V160 H210 M210 160 H360 M360 160 H505" fill="none" stroke={amber} strokeDasharray="4 4" markerEnd={`url(#${id})`} />
      <Label x={300} y={134}>Backward: replay operations from saved x</Label><Label x={300} y={190}>Recreated intermediates supply the same derivatives.</Label><Label x={300} y={221} size={12}>Retain matching randomness when replaying stochastic operations.</Label>
    </Diagram>
    <Diagram title="Tensor parallel FFN partitions hidden coordinates and sums partial output vectors" width={600} height={350}>
      <Box x={15} y={147} width={75} label="row x" /><path d="M90 164 H115 V73 H150 M115 164 V248 H150" fill="none" stroke={amber} />
      {[0,1].map(device => <g key={device}><Label x={300} y={device ? 220:45}>Device {device}: matched hidden coordinates</Label><Box x={150} y={device?231:56} width={150} label={`up columns ${device?'3–4':'1–2'}`} /><path d={`M300 ${device?248:73} H325`} stroke={amber} /><Box x={325} y={device?231:56} width={140} label={`down rows ${device?'3–4':'1–2'}`} /><path d={`M465 ${device?248:73} H510 V164 H534`} stroke={amber} fill="none" /><Label x={370} y={device?295:119} size={12}>local activations → partial d-vector</Label></g>)}
      <circle cx="549" cy="164" r="14" fill="#111" stroke={amber} /><Label x={549} y={169}>+</Label><Label x={549} y={195} size={12}>sum</Label><Label x={300} y={333} size={12}>Every device receives x; the two partial output vectors must communicate and add.</Label>
    </Diagram>
    <p>Partitioning hidden coordinates preserves the algebra when each up column stays paired with its down row. A gated FFN also keeps matching gate coordinates together. Pipeline partitioning instead sends each block’s output to the next device.</p>
  </figure>;
}
export function BlockPairedScores({ fits }) {
  const low = Math.min(...fits.map(r=>r.test.macro_f1))-.025, high=Math.max(...fits.map(r=>r.test.macro_f1))+.025;
  const x = score => 85+300*(score-low)/(high-low);
  return <Diagram title="All three paired pre-norm and post-norm test macro F1 results" width={440} height={245}>
    <Label x={235} y={20}>Test macro F1; validation-selected checkpoints</Label>
    {[101,102,103].map((seed,i)=>{const pair=fits.filter(r=>r.seed===seed), y=60+i*50;return <g key={seed}><Label x={40} y={y+5} size={12}>{seed}</Label><path d={`M${x(pair[0].test.macro_f1)} ${y} H${x(pair[1].test.macro_f1)}`} stroke="#aaa" />{pair.map(r=><g key={r.placement}><circle cx={x(r.test.macro_f1)} cy={y} r="6" fill={r.placement==='pre-norm'?amber:'#eee'} /><title>{r.placement}: {r.test.macro_f1}</title></g>)}</g>;})}
    <path d="M85 190 H385" stroke="#777" />{[0,.25,.5,.75,1].map(t=><Label key={t} x={85+300*t} y={211} size={12}>{(low+(high-low)*t).toFixed(3)}</Label>)}<Label x={235} y={237} size={12}>Amber: pre-norm · white: post-norm · row: same seed</Label>
  </Diagram>;
}
