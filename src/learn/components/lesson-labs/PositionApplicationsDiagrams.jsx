import { useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { xposPairTrace } from '../../data/position-convention-models.js';
function Diagram({
  title,
  width = 620,
  height,
  children
}) {
  return <figure className="position-figure"><figcaption>{title}</figcaption><div className="position-mechanism-scroll" role="region" aria-label={title} tabIndex={0}><svg width={width} style={{
        minWidth: width,
        width: '100%',
        height: 'auto'
      }} viewBox={'0 0 ' + width + ' ' + height} role="img" aria-label={title}>{children}</svg></div></figure>;
}
const textStyle = {
  fill: '#ddd',
  fontSize: 13
};
export function PositionApplicationsDiagrams() {
  return <>
    <Diagram title="Equal ordinal steps can hide unequal elapsed time" height={225}>
      <text x="12" y="28" {...textStyle}>Ordinal ID</text><path d="M150 55 H585" stroke="#999" />
      {[0, 1, 2].map((value, i) => <g key={i}><circle cx={150 + i * 217.5} cy="55" r="5" fill="#e6b854" /><text x={150 + i * 217.5} y="83" textAnchor="middle" {...textStyle}>{value}</text></g>)}
      <text x="12" y="117" {...textStyle}>Seconds</text><path d="M150 145 H585" stroke="#999" />
      {[0, 1, 20].map((value, i) => <g key={i}><path d={'M' + (150 + i * 217.5) + ' 92 L' + (150 + value * 21.75) + ' 130'} stroke="#555" fill="none" /><circle cx={150 + value * 21.75} cy="145" r="5" fill="#e6b854" /><text x={150 + value * 21.75} y={i === 1 ? 188 : 168} textAnchor="middle" {...textStyle}>{value}</text></g>)}
      <text x="310" y="214" textAnchor="middle" {...textStyle}>0.1 rad/s × 20 s = 0.0001 rad/ms × 20,000 ms = 2 rad</text>
    </Diagram>
    <p>Each connecting line preserves an event’s identity. Ordinal IDs omit the long second gap; timestamps retain it. Changing time units requires changing the frequency’s units as well.</p>
    <Diagram title="A row boundary in flattened storage is not horizontal adjacency" height={270}>
      {Array.from({
        length: 6
      }, (_, i) => <g key={i}><rect x={20 + i % 3 * 78} y={35 + Math.floor(i / 3) * 70} width="70" height="58" fill="#222" stroke={i === 2 || i === 3 ? '#e6b854' : '#777'} /><text x={55 + i % 3 * 78} y={59 + Math.floor(i / 3) * 70} textAnchor="middle" {...textStyle}>{'patch ' + i}</text><text x={55 + i % 3 * 78} y={80 + Math.floor(i / 3) * 70} textAnchor="middle" {...textStyle}>{'(' + i % 3 + ',' + Math.floor(i / 3) + ')'}</text></g>)}
      <path d="M210 97 L210 188 L55 188 L55 171" fill="none" stroke="#e6b854" strokeDasharray="4 3" /><text x="136" y="215" textAnchor="middle" {...textStyle}>2 → 3: Δx = −2, Δy = +1</text>
      <path d="M265 105 H300" stroke="#aaa" />
      {[0, 1].map((axis, i) => <g key={axis}><circle cx="395" cy={68 + i * 127} r="43" fill="none" stroke="#666" /><path d={'M345 ' + (68 + i * 127) + ' H445 M395 ' + (18 + i * 127) + ' V' + (118 + i * 127)} stroke="#444" /><line x1="395" y1={68 + i * 127} x2={395 + 40 * Math.cos(i ? 1 : -2)} y2={68 + i * 127 - 40 * Math.sin(i ? 1 : -2)} stroke={i ? '#ba9fd1' : '#e6b854'} strokeWidth="3" /><text x="457" y={60 + i * 127} {...textStyle}>{i ? 'pair (2,3): y' : 'pair (0,1): x'}</text><text x="457" y={81 + i * 127} {...textStyle}>{i ? 'relative angle +1' : 'relative angle −2'}</text></g>)}
    </Diagram>
    <p>This declared two-axis construction gives each spatial axis its own rotary pair at one radian per grid step. It represents the displacement across the row boundary; it does not claim to reproduce a particular vision checkpoint.</p>
    <Diagram title="Head count, cached positions and key coordinates are different axes" height={305}>
      <text x="20" y="25" {...textStyle}>One batch item and one layer; repeat for B × N</text>
      <text x="57" y="62" {...textStyle}>Hkv</text><text x="215" y="62" {...textStyle}>cached position L →</text>
      {[0, 1, 2].map(head => <g key={head}><text x="15" y={98 + head * 51} {...textStyle}>{'head ' + head}</text>{[0, 1, 2, 3].map(position => <rect key={position} x={82 + position * 65} y={77 + head * 51} width="57" height="40" fill={head === 1 && position === 2 ? '#393123' : '#222'} stroke={head === 1 && position === 2 ? '#e6b854' : '#666'} />)}</g>)}
      <path d="M241 128 L367 125 M241 168 L367 199" stroke="#e6b854" fill="none" />
      <text x="381" y="83" {...textStyle}>one record’s coordinate widths</text>
      <rect x="367" y="102" width="225" height="45" fill="#222" stroke="#e6b854" /><text x="480" y="129" textAnchor="middle" {...textStyle}>K: dk coordinates</text>
      <rect x="367" y="177" width="225" height="45" fill="#222" stroke="#ba9fd1" /><text x="480" y="205" textAnchor="middle" {...textStyle}>V: dv coordinates</text>
      <path d="M401 152 Q416 166 433 152 M449 152 Q464 166 481 152" fill="none" stroke="#e6b854" /><text x="386" y="246" {...textStyle}>RoPE transforms selected K pairs.</text>
      <text x="20" y="275" {...textStyle}>GQA changes Hkv; neither axis is the rotary coordinate index.</text>
      <text x="20" y="298" {...textStyle}>B × N × Hkv × L × (dk + dv) × bytes per element</text>
    </Diagram>
    <p>The exact example 1×32×8×4096×(128+128)×2 bytes is 536,870,912 bytes = 512 MiB, excluding metadata. This is a derived tensor count, not measured allocation or latency. Standard RoPE transforms keys before storing them; values remain unrotated here.</p>

  </>;
}
export function XposPositionFigure() {
  const [queryId, setQueryId] = useState(512),
    [keyId, setKeyId] = useState(0);
  const xpos = xposPairTrace(queryId, keyId);
  const extent = Math.max(1, xpos.queryAmplitude, xpos.keyAmplitude) * 1.08;
  return <figure className="position-figure"><figcaption>XPos multiplies two rotated vectors by reciprocal position scales</figcaption><div className="neural-controls"><NeuralNumber label="XPos query ID m" value={queryId} min={0} max={1024} integer onChange={setQueryId} /><NeuralNumber label="XPos key ID n" value={keyId} min={0} max={1024} integer onChange={setKeyId} /></div>
      <Diagram title="Same rotations, changed lengths" width={620} height={250}>
        {[[xpos.query, xpos.scaledQuery, 'q: ζ^(m/512)', xpos.queryAmplitude], [xpos.key, xpos.scaledKey, 'k: ζ^(−n/512)', xpos.keyAmplitude]].map(([raw, scaled, label, amplitude], i) => <g key={label}><text x={157 + 310 * i} y="23" textAnchor="middle" {...textStyle}>{label + ' = ' + f(amplitude, 5)}</text><path d={'M' + (65 + 310 * i) + ' 130 H' + (249 + 310 * i) + ' M' + (157 + 310 * i) + ' 38 V222'} stroke="#555" /><circle cx={157 + 310 * i} cy="130" r={85 / extent} fill="none" stroke="#666" strokeDasharray="4 3" /><line x1={157 + 310 * i} y1="130" x2={157 + 310 * i + raw[0] / extent * 85} y2={130 - raw[1] / extent * 85} stroke="#eee" strokeWidth="2" strokeDasharray="4 3" /><line x1={157 + 310 * i} y1="130" x2={157 + 310 * i + scaled[0] / extent * 85} y2={130 - scaled[1] / extent * 85} stroke="#e6b854" strokeWidth="3" /><circle cx={157 + 310 * i + scaled[0] / extent * 85} cy={130 - scaled[1] / extent * 85} r="3" fill="#e6b854" /></g>)}
        <text x="310" y="247" textAnchor="middle" {...textStyle}>{'Both planes share scale ±' + f(extent) + '; white = pure rotation, amber = scaled.'}</text>
      </Diagram>
      <p>Here ζ=2/7, scale denominator S=512, and raw q=k=[1,0] at frequency one. The product entering the dot is {f(xpos.queryAmplitude, 7)} × {f(xpos.keyAmplitude, 7)} = <strong>{f(xpos.product, 7)}</strong> = ζ^((m−n)/512). At m−n=512 it is 2/7. Individual lengths change; their common-shift factors cancel in the product.</p>
      <NeuralTable caption="Current XPos vectors and dot product" headers={['Quantity', 'Calculated value']} rows={[['Scaled query', xpos.scaledQuery.map(v => f(v, 7)).join(', ')], ['Scaled key', xpos.scaledKey.map(v => f(v, 7)).join(', ')], ['Pure rotary dot', f(xpos.rotaryDot, 7)], ['Scaled dot', f(xpos.scaledDot, 7)]]} />
      <button onClick={() => {
      setQueryId(512);
      setKeyId(0);
    }}>Reset XPos example</button>
    </figure>;
}
