import { useId } from "react";
import "./python-mechanism-figures.css";

function ArrowMarker({ id }) {
  return <defs><marker id={id} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" /></marker></defs>;
}

export function CursorOwnershipMap({ state, shared }) {
  const marker = `pmf-cursor-${useId().replace(/:/g, "")}`;
  const items = [...state.values, "END"], width = 360 / items.length;
  const cursors = shared ? [{ label: "a, b → CURSOR A", position: state.positions[0], x: 180 }] : [{ label: "a → CURSOR A", position: state.positions[0], x: 88 }, { label: "b → CURSOR B", position: state.positions[1], x: 272 }];
  return <div className="pmf-cursor-map" data-python-figure="cursor-ownership">
    <p><strong>values → one unchanged source list</strong><span>Arrows mark each cursor's next item. END is a signal, not a list element.</span></p>
    <svg viewBox="0 0 370 235" role="img" aria-label={`${shared ? "a and b share one cursor" : "a and b refer to independent cursors"}. Source: [${state.values.join(", ")}]. ${cursors.map(c => `${c.label}: ${c.position === state.values.length ? "exhausted" : `next index ${c.position}, value ${state.values[c.position]}`}`).join(". ")}`}>
      <ArrowMarker id={marker}/>
      {items.map((value, i) => <g key={i}><rect x={5 + i * width} y="22" width={width - 8} height="60" className={i === state.values.length ? "pmf-end" : "pmf-cell"}/><text x={1 + i * width + width / 2} y="48" textAnchor="middle">{value}</text><text className="pmf-svg-label" x={1 + i * width + width / 2} y="69" textAnchor="middle">{i === state.values.length ? "end" : `index ${i}`}</text></g>)}
      {cursors.map((cursor, i) => {
        const target = 1 + cursor.position * width + width / 2 + (shared ? 0 : i === 0 ? -6 : 6);
        return <g key={cursor.label} className={i === 1 ? "pmf-cursor-b" : "pmf-cursor-a"}><path className="pmf-arrow" d={`M${cursor.x},165 C${cursor.x},125 ${target},128 ${target},87`} markerEnd={`url(#${marker})`}/><rect x={cursor.x - 78} y="169" width="156" height="59" rx="3" className="pmf-object"/><text x={cursor.x} y="191" textAnchor="middle" className="pmf-svg-label">{cursor.label}</text><text x={cursor.x} y="214" textAnchor="middle">{cursor.position < state.values.length ? `next index: ${cursor.position}` : "exhausted"}</text></g>;
      })}
    </svg>
  </div>;
}
