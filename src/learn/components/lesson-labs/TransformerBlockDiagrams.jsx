import { useId } from 'react';
function ArrowDefinitions({
  id
}) {
  return <defs><marker id={id} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#e6b854" /></marker></defs>;
}
function Operation({
  x,
  y,
  text,
  width = 112,
  muted = false
}) {
  return <g><rect x={x - width / 2} y={y - 17} width={width} height="34" rx="5" fill={muted ? '#292929' : '#202020'} stroke={muted ? '#666' : '#e6b854'} /><text x={x} y={y + 5} textAnchor="middle" fill="#eee" fontSize="13">{text}</text></g>;
}
function Junction({
  x,
  y
}) {
  return <g><circle cx={x} cy={y} r="13" fill="#111" stroke="#e6b854" /><text x={x} y={y + 5} textAnchor="middle" fill="#fff" fontSize="18">+</text></g>;
}
export function BlockWiringDiagram({
  pre,
  zero
}) {
  const marker = useId().replaceAll(':', '');
  const arrow = {
    fill: 'none',
    stroke: '#e6b854',
    strokeWidth: 1.8,
    markerEnd: `url(#${marker})`
  };
  return <svg className="block-wiring-svg" viewBox="0 0 300 490" role="img" aria-label={pre ? 'Pre-norm circuit. Normalization is inside each update branch; the bypass reaches each sum without normalization.' : 'Post-norm circuit. Each bypass reaches its sum, then the combined stream passes through normalization.'}>
    <ArrowDefinitions id={marker} />
    {[0, 1].map(step => {
      const top = 28 + step * 222,
        center = 168,
        sum = top + 151;
      return <g key={step}>
        <text x={center} y={top - 7} textAnchor="middle" fill="#fff" fontSize="15">{step === 0 ? 'X' : 'Z'}</text>
        <path d={`M${center} ${top} V${top + 16} H40 V${sum} H${center - 14}`} {...arrow} />
        <text x="35" y={top + 112} transform={`rotate(-90 35 ${top + 112})`} textAnchor="middle" fill="#ddd" fontSize="12">saved {step === 0 ? 'X' : 'Z'} bypass</text>
        {pre ? <>
          <path d={`M${center} ${top} V${top + 32}`} {...arrow} />
          <Operation x={center} y={top + 50} text={`Norm ${step + 1}`} />
          <path d={`M${center} ${top + 67} V${top + 84}`} {...arrow} />
          <Operation x={center} y={top + 102} text={step === 0 ? 'Attention' : 'FFN'} muted={zero} />
          <path d={`M${center} ${top + 119} V${sum - 14}`} {...arrow} />
        </> : <>
          <path d={`M${center} ${top} V${top + 67}`} {...arrow} />
          <Operation x={center} y={top + 85} text={step === 0 ? 'Attention' : 'FFN'} muted={zero} />
          <path d={`M${center} ${top + 102} V${sum - 14}`} {...arrow} />
        </>}
        {zero && <text x="205" y={top + 134} fill="#ddd" fontSize="11">update 0</text>}
        <Junction x={center} y={sum} />
        {pre ? <path d={`M${center} ${sum + 14} V${top + 196}`} {...arrow} /> : <>
          <path d={`M${center} ${sum + 14} V${top + 169}`} {...arrow} />
          <Operation x={center} y={top + 187} text={`Norm ${step + 1}`} />
          <path d={`M${center} ${top + 204} V${top + 210}`} {...arrow} />
        </>}
      </g>;
    })}
    <text x="168" y="474" textAnchor="middle" fill="#fff" fontSize="15">Y</text>
  </svg>;
}
export function BlockCommunicationDiagram() {
  const marker = useId().replaceAll(':', '');
  const arrow = {
    fill: 'none',
    stroke: '#e6b854',
    strokeWidth: 1.4,
    markerEnd: `url(#${marker})`
  };
  const lanes = [75, 190, 305];
  return <div className="block-diagram-scroll" tabIndex={0} role="region" aria-label="Communication and local feature processing diagram"><svg className="block-communication-svg" viewBox="0 0 720 380" role="img" aria-label="Three input position rows each connect to every attention receiver. After each residual addition, FFN arrows stay in the same lane; the same FFN parameters are reused.">
    <ArrowDefinitions id={marker} />
    <text x="64" y="22" textAnchor="middle" fill="#ddd" fontSize="13">input rows</text><text x="265" y="22" textAnchor="middle" fill="#ddd" fontSize="13">read permitted positions</text><text x="505" y="22" textAnchor="middle" fill="#ddd" fontSize="13">same FFN in each lane</text><text x="677" y="22" textAnchor="middle" fill="#ddd" fontSize="13">output</text>
    {lanes.flatMap((from, i) => lanes.map((to, j) => <path key={`${i}-${j}`} d={`M113 ${from} C158 ${from} 175 ${to} 209 ${to}`} {...arrow} opacity={i === j ? 1 : .5} />))}
    {lanes.map((y, i) => <g key={y}>
      <Operation x={64} y={y} text={`position ${i + 1}`} width={98} />
      <Operation x={265} y={y} text="Attention" />
      <path d={`M321 ${y} H366`} {...arrow} /><Junction x={380} y={y} />
      <path d={`M64 ${y + 18} V${y + 40} H380 V${y + 14}`} {...arrow} />
      <path d={`M394 ${y} H449`} {...arrow} /><Operation x={505} y={y} text="d → f → d" />
      <path d={`M561 ${y} H591`} {...arrow} /><Junction x={605} y={y} />
      <path d={`M410 ${y} V${y + 40} H605 V${y + 14}`} {...arrow} />
      <path d={`M619 ${y} H655`} {...arrow} /><text x="677" y={y + 5} textAnchor="middle" fill="#eee" fontSize="13">d features</text>
    </g>)}
    <text x="64" y="373" textAnchor="middle" fill="#bbb" fontSize="12">sequence axis ↓</text><text x="505" y="373" textAnchor="middle" fill="#bbb" fontSize="12">feature expansion stays inside a position</text>
  </svg></div>;
}
export function BlockVariantDependencies() {
  const marker = useId().replaceAll(':', '');
  const arrow = {
    fill: 'none',
    stroke: '#e6b854',
    strokeWidth: 1.8,
    markerEnd: `url(#${marker})`
  };
  return <div className="block-two">{[false, true].map(parallel => <div key={String(parallel)} className="block-diagram-scroll" role="region" aria-label={parallel ? "Parallel dependency diagram" : "Sequential dependency diagram"} tabIndex={0}><svg className="block-wiring-svg" viewBox="0 0 300 380" role="img" aria-label={parallel ? 'Parallel block: x forks into attention and FFN branches, then both updates join the saved x.' : 'Sequential block: attention updates x to z before FFN reads normalized z.'}>
    <ArrowDefinitions id={`${marker}${parallel}`} />
    <g style={{
        '--unused': 0
      }}>
      <text x="150" y="22" textAnchor="middle" fill="#eee" fontSize="15">{parallel ? 'Parallel' : 'Sequential'}</text><text x="150" y="54" textAnchor="middle" fill="#fff" fontSize="14">x</text>
      {parallel ? <>
        <path d="M150 62 V76" {...arrow} markerEnd={`url(#${marker}${parallel})`} /><Operation x={150} y={94} text="Norm(x)" />
        <path d="M150 111 V135 H70 V168 M150 135 H230 V168" {...arrow} markerEnd={`url(#${marker}${parallel})`} />
        <Operation x={70} y={187} text="Attention" width={106} /><Operation x={230} y={187} text="FFN" width={90} />
        <path d="M70 204 V275 H136 M230 204 V275 H164" {...arrow} markerEnd={`url(#${marker}${parallel})`} />
        <path d="M150 62 H12 V322 H150 V289" {...arrow} markerEnd={`url(#${marker}${parallel})`} /><Junction x={150} y={275} />
        <path d="M150 289 V343" {...arrow} markerEnd={`url(#${marker}${parallel})`} />
      </> : <>
        <path d="M150 62 V74" {...arrow} markerEnd={`url(#${marker}${parallel})`} /><Operation x={150} y={92} text="Norm → Attention" width={154} />
        <path d="M150 110 V136 M150 62 H30 V150 H136" {...arrow} markerEnd={`url(#${marker}${parallel})`} /><Junction x={150} y={150} />
        <text x="180" y="180" fill="#eee" fontSize="14">z</text><path d="M150 164 V197" {...arrow} markerEnd={`url(#${marker}${parallel})`} /><Operation x={150} y={215} text="Norm → FFN" width={140} />
        <path d="M150 233 V277 M150 180 H30 V291 H136" {...arrow} markerEnd={`url(#${marker}${parallel})`} /><Junction x={150} y={291} /><path d="M150 305 V343" {...arrow} markerEnd={`url(#${marker}${parallel})`} />
      </>}
      <text x="150" y="365" textAnchor="middle" fill="#fff" fontSize="14">y</text>
    </g>
  </svg></div>)}</div>;
}
