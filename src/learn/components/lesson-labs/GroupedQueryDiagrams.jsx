import { useId } from 'react';

function Frame({ title, description, width = 680, height, children }) {
  const id = useId().replaceAll(':', '');
  return <figure className="gqa-figure" tabIndex={0}><figcaption>{title}</figcaption><svg className="gqa-mechanism" style={{ minWidth: width }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={description}>
    <defs><marker id={id} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#dcb764" /></marker></defs>
    {children(`url(#${id})`)}
  </svg><p>{description}</p></figure>;
}
const Edge = ({ d, marker, dashed = false }) => <path d={d} fill="none" stroke="#dcb764" strokeWidth="1.8" strokeDasharray={dashed ? '5 4' : undefined} markerEnd={marker} />;
function Cell({ x, y, width = 100, label, secondary = false }) {
  return <g><rect x={x} y={y} width={width} height="32" rx="3" fill={secondary ? '#292929' : '#202020'} stroke={secondary ? '#aaa' : '#dcb764'} /><text x={x + width / 2} y={y + 21} textAnchor="middle">{label}</text></g>;
}
export function GqaPromptDiagram() {
  return <Frame title="Known prompt first; generated inputs arrive one at a time" height={200} description="All prompt inputs are already known and can be projected together. Causal prompt queries still read only their legal prefixes. Each decode input is available only after the preceding token is chosen; the arrows describe dependency, not elapsed time.">{marker => <>
    <text x="185" y="22" textAnchor="middle">prefill: known prompt positions</text><text x="510" y="22" textAnchor="middle">decode: sequential availability</text>
    {[0, 1, 2].map(i => <g key={i}><Cell x={20 + i * 110} y={40} label={`input ${i}`} /><Edge d={`M${70 + i * 110} 72 V108`} marker={marker} /><Cell x={20 + i * 110} y={110} label={`KV${i}`} /></g>)}
    <path d="M20 150 V162 H340 V150" fill="none" stroke="#aaa" /><text x="180" y="187" textAnchor="middle">one layer's initial cache</text>
    <Edge d="M340 126 H375" marker={marker} /><Cell x={380} y={110} label="query 3" /><Edge d="M430 110 V73" marker={marker} /><Cell x={380} y={40} label="choose 4" /><Edge d="M480 56 H550" marker={marker} /><Cell x={555} y={40} label="input 4" /><Edge d="M605 72 V108" marker={marker} /><Cell x={555} y={110} label="query 4" />
    <text x="465" y="181" textAnchor="middle">append KV3, then KV4 as inputs arrive</text>
  </>}</Frame>;
}
export function GqaCompactWriteDiagram() {
  return <Frame title="Rotate each unique key, then append a compact record" height={310} description="This schematic uses four query heads and two KV heads. Position p rotates each query and each unique key. Values do not rotate. Only two key/value records append at p; four transient head outputs remain outside persistent storage. Old cache entries already follow the same rotary convention.">{marker => <>
    <Cell x={20} y={25} width={125} label="input at ID p" />
    <Edge d="M145 41 H175 V35 H205 M175 41 V107 H205 M175 107 V179 H205" marker={marker} />
    <Cell x={210} y={19} width={140} label="Q: 4 heads" /><Cell x={210} y={91} width={140} label="K: 2 heads" /><Cell x={210} y={163} width={140} label="V: 2 heads" />
    <Edge d="M350 35 H390 M350 107 H390 M350 179 H550" marker={marker} />
    <Cell x={395} y={19} width={125} label="rotate at p" /><Cell x={395} y={91} width={125} label="rotate at p" />
    <Edge d="M520 35 H592 V226 M520 107 H550 V179" marker={marker} /><circle cx="550" cy="179" r="3" fill="#dcb764" />
    <rect x="25" y="212" width="490" height="76" rx="4" fill="none" stroke="#aaa" strokeDasharray="5 3" />
    <text x="35" y="205">persistent cache in this layer</text>
    {[0, 1, 2].map(i => <g key={i}><Cell x={40 + i * 150} y={233} width={135} label={i === 2 ? 'new ID p: 2 KV' : `old ID ${i}: 2 KV`} secondary /></g>)}
    <Edge d="M550 179 V249 H477" marker={marker} /><Cell x={540} y={232} width={125} label="4 head reads" /><Edge d="M515 250 H538" marker={marker} dashed />
    <text x="540" y="294">transient outputs</text>
  </>}</Frame>;
}
export function GqaPayloadDiagram({ state }) {
  return <Frame title="A record has two widths; the other axes repeat it" height={290} description="The drawn cells are a schematic, not one cell per stored number. The live labels carry the actual dimensions: concatenate key and value fields within one head record, then repeat across KV heads, occupied tokens, layers and requests. Query heads are readers and add no extra stored field.">{marker => <>
    <text x="20" y="23">one token, one KV head</text>
    <Cell x={20} y={38} width={140} label={`K: ${state.keyWidth} numbers`} /><Cell x={160} y={38} width={140} label={`V: ${state.valueWidth} numbers`} secondary />
    <Edge d="M300 54 H350" marker={marker} /><text x="368" y="59">× {state.bytes} bytes per number</text>
    {[0, 1, 2].map(i => <g key={i}><rect x={20 + i * 8} y={112 + i * 8} width="150" height="42" fill="#202020" stroke="#dcb764" /><path d={`M${95 + i * 8} ${112 + i * 8} V${154 + i * 8}`} stroke="#aaa" /></g>)}
    <text x="20" y="197">{state.kvHeads} KV heads</text><Edge d="M195 150 H260" marker={marker} />
    <g>{[0, 1, 2, 3].map(i => <rect key={i} x={280 + 44 * i} y="121" width="40" height="55" fill="#202020" stroke="#aaa" />)}<text x="363" y="197" textAnchor="middle">{state.length} occupied tokens</text></g>
    <Edge d="M462 150 H508" marker={marker} />
    {[0, 1, 2].map(i => <rect key={i} x={524 + i * 8} y={112 + i * 8} width="110" height="42" fill="#202020" stroke="#dcb764" />)}
    <text x="578" y="197" textAnchor="middle">{state.layers} layers</text>
    <path d="M20 213 V228 H645 V213" fill="none" stroke="#aaa" /><text x="330" y="260" textAnchor="middle">repeat this collection for {state.batch} request{state.batch === 1 ? '' : 's'}</text>
  </>}</Frame>;
}
export function GqaOwnershipDiagram() {
  return <Frame title="Which memory grows, and which sublayer owns it?" height={325} description="An encoder-decoder example has a fixed source-side cross-attention cache and a growing causal decoder cache. Both supply attention. In an MoE Transformer block, routing to feed-forward experts changes the FFN branch; it does not by itself change attention's KV-head layout.">{marker => <>
    <text x="20" y="22">encoder–decoder cache lifetime</text><Cell x={20} y={42} width={165} label="source positions 0…S−1" />
    <Edge d="M185 58 H215" marker={marker} /><Cell x={220} y={42} width={175} label="fixed cross-attention KV" /><Edge d="M395 58 H465 V131" marker={marker} />
    <Cell x={20} y={118} width={165} label="decoder positions 0…t" /><Edge d="M185 134 H215" marker={marker} /><Cell x={220} y={118} width={175} label="growing self-attention KV" /><Edge d="M395 134 H420" marker={marker} /><Cell x={425} y={134} width={200} label="current decoder computation" />
    <text x="20" y="213">one example MoE block: separate ownership</text>
    <Cell x={20} y={239} width={95} label="stream" /><Edge d="M115 255 H145" marker={marker} /><Cell x={150} y={239} width={145} label="attention + KV" /><Edge d="M295 255 H335" marker={marker} /><Cell x={340} y={239} width={95} label="router" />
    {[0, 1, 2].map(i => <g key={i}><Edge d={`M435 255 H465 V${222 + 34 * i} H495`} marker={marker} /><Cell x={500} y={206 + 34 * i} width={150} label={`FFN expert ${i}`} secondary /></g>)}
  </>}</Frame>;
}
export function GqaDistributedDiagram() {
  return <Frame title="Two logical KV groups can occupy eight physical device copies" height={250} description="This declared replicated layout has 32 query heads, two logical KV groups and eight devices. Each device serves four local queries. Four devices copy group 0 and four copy group 1: eight physical KV-head copies in aggregate. A communicating layout would require a separate communication and scheduling cost model.">{marker => <>
    <Cell x={100} y={20} width={150} label="logical KV group 0" /><Cell x={430} y={20} width={150} label="logical KV group 1" />
    {Array.from({ length: 8 }, (_, i) => <g key={i}><Edge d={`M${i < 4 ? 175 : 505} 52 L${43 + i * 84} 109`} marker={marker} /><rect x={5 + i * 84} y="112" width="76" height="88" rx="4" fill="#202020" stroke="#aaa" /><text x={43 + i * 84} y="134" textAnchor="middle">device {i}</text><text x={43 + i * 84} y="156" textAnchor="middle">4 queries</text><text x={43 + i * 84} y="180" textAnchor="middle">KV {i < 4 ? 0 : 1}</text></g>)}
    <text x="340" y="231" textAnchor="middle">logical count 2 ≠ physical head-copy count 8</text>
  </>}</Frame>;
}
