import {useId} from 'react';
import './mechanism-figures.css';

export function ArrayAddressFigure() {
  return <figure className="mechanism-figure array-address" aria-label="Four consecutive eight-byte slots starting at address 1000">
    <figcaption><strong>Move by slot widths, not by searching earlier values.</strong> Each equally wide cell below represents an 8-byte slot.</figcaption>
    <div className="address-slots">{[0, 1, 2, 3].map(index => <div key={index} className={index === 3 ? 'is-chosen' : ''}><span>index {index}</span><strong>{1000 + index * 8}</strong><small>bytes {1000 + index * 8}–{1007 + index * 8}</small></div>)}</div>
    <div className="address-displacement"><span>3 slot widths × 8 bytes = 24 bytes</span><b>↓</b></div>
    <p><strong>1000 + 24 = 1024:</strong> the start of slot 3. These are illustrative byte addresses. The slot may store a value or a reference, depending on the array; this drawing does not locate the referenced objects.</p>
  </figure>;
}

export function SeparateAddressFigure() {
  return <figure className="mechanism-figure separate-address" aria-label="Same virtual byte 22 reaches physical byte 54 for A and 86 for B">
    <figcaption><strong>An address needs a process’s map.</strong> Hold page 1 and offset 6 fixed; change only the selected process.</figcaption>
    {['A', 'B'].map((name, i) => <div className="process-address-lane" key={name}><strong>Process {name}</strong><span>Virtual 22<small>page 1 | offset 6</small></span><b aria-hidden="true">→</b><span>Frame {i ? 5 : 3}<small>{i ? '5' : '3'} × 16 + <em>6</em></small></span><b aria-hidden="true">→</b><span>Physical <strong>{i ? 86 : 54}</strong></span></div>)}
    <p>The mapping changes the frame. The offset stays 6. This is the lab’s 16-byte-page model, with separate resident data frames; it is not a measured layout of your computer.</p>
  </figure>;
}

export function QueueRingFigure({state, capacity}) {
  const uid = useId();
  const points = capacity === 4 ? [[180, 45], [300, 150], [180, 255], [60, 150]] : [[180, 45], [290, 225], [70, 225]];
  return <div className="queue-ring-view">
    <svg viewBox="0 0 360 310" role="img" aria-label={`Circular storage, capacity ${capacity}, head ${state.head}, size ${state.size}. ${state.cells.map((v, i) => `Slot ${i}: ${v ?? 'empty'}`).join('; ')}. Increasing indices wrap from ${capacity - 1} to zero.`}>
      <defs><marker id={uid} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#baa886" /></marker></defs>
      {points.map(([x, y], i) => {
        const [nx, ny] = points[(i + 1) % capacity], dx = nx - x, dy = ny - y, length = Math.hypot(dx, dy);
        return <line key={i} x1={x + dx / length * 49} y1={y + dy / length * 49} x2={nx - dx / length * 51} y2={ny - dy / length * 51} stroke="#baa886" strokeWidth="2" markerEnd={`url(#${uid})`} />;
      })}
      {points.map(([x, y], i) => <g key={i}>
        <rect x={x - 45} y={y - 31} width="90" height="66" rx="4" fill={i === state.changed ? '#3a301c' : '#17241c'} stroke={i === state.head ? '#efc56a' : '#839a85'} strokeWidth={i === state.head ? 3 : 1} />
        <text x={x} y={y - 13} textAnchor="middle" className="ring-minor">slot {i}</text><text x={x} y={y + 8} textAnchor="middle" className="ring-value">{state.cells[i] ?? '·'}</text>
        <text x={x} y={y + 27} textAnchor="middle" className="ring-minor">{[i === state.head ? 'HEAD' : '', i === state.tail ? 'NEXT' : ''].filter(Boolean).join(' + ') || ' '}</text>
      </g>)}
      <text x="180" y="143" textAnchor="middle">{state.size} / {capacity}</text><text x="180" y="164" textAnchor="middle" className="ring-minor">{state.size === capacity ? 'full' : state.size === 0 ? 'empty' : 'occupied'}</text>
    </svg>
    <p className="lesson-note">Arrows show increasing slot indices and wraparound, not pointers stored in the buffer. NEXT is the insertion index; when full, enqueue must first wait for space or be rejected.</p>
  </div>;
}
