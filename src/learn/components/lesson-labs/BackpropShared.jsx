import { useId, useState } from 'react';
import './backprop-labs.css';

export function numberText(value) {
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 1e7) return value.toExponential(7);
  return Number(value.toPrecision(9)).toString();
}

export function BackpropNumber({ name, value, onChange, min, max, step = 0.1, resetKey = 0 }) {
  const id = useId();
  const [edit, setEdit] = useState(null);
  const [editKey, setEditKey] = useState(resetKey);
  if (editKey !== resetKey) {
    setEdit(null);
    setEditKey(resetKey);
  }
  const invalid = edit !== null && (edit.trim() === '' || !Number.isFinite(Number(edit)) || Number(edit) < min || Number(edit) > max);
  return <div className="backprop-control">
    <label htmlFor={`${id}-range`}>{name} — slider</label>
    <input id={`${id}-range`} type="range" min={min} max={max} step="any" value={value}
      onKeyDown={event => {
        const direction = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[event.key];
        if (direction) {
          event.preventDefault();
          setEdit(null);
          onChange(Math.max(min, Math.min(max, Number((value + direction * step).toPrecision(14)))));
        }
      }}
      onChange={event => { setEdit(null); onChange(Number(event.target.value)); }} />
    <label htmlFor={`${id}-number`}>{name} — exact value</label>
    <input id={`${id}-number`} type="number" min={min} max={max} step="any" value={edit ?? value}
      aria-invalid={invalid} aria-describedby={invalid ? `${id}-error` : undefined}
      onChange={event => {
        const text = event.target.value;
        setEdit(text);
        const numeric = Number(text);
        if (text.trim() !== '' && Number.isFinite(numeric) && numeric >= min && numeric <= max) onChange(numeric);
      }} />
    {invalid && <p id={`${id}-error`} className="backprop-invalid">Enter a number from {min} to {max}. Showing the last valid result for {name}: {value}.</p>}
  </div>;
}

export function BackpropTable({ caption, headers, rows }) {
  return <div className="backprop-table-shell"><p className="backprop-table-title">{caption}</p><p className="backprop-scroll-hint">Scroll the table horizontally to inspect every column.</p><div className="backprop-table" role="region" aria-label={caption} tabIndex={0}>
    <table><caption className="backprop-sr-only">{caption}</caption><thead><tr>{headers.map((header, index) => <th scope="col" key={index}>{header}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{row.map((value, j) => j === 0 ? <th scope="row" key={j}>{value}</th> : <td key={j}>{value}</td>)}</tr>)}</tbody>
    </table>
  </div></div>;
}

export function BackpropFigure({ title, children, caption, id }) {
  return <figure className="backprop-figure" data-backprop-figure={id} aria-label={title}>
    <h3>{title}</h3>{children}<figcaption>{caption}</figcaption>
  </figure>;
}

export function BackpropSteps({ index, setIndex, count, name = 'Trace' }) {
  return <div className="backprop-actions" aria-label={name}>
    <button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous {name.toLowerCase()} stage</button>
    <span>Stage {index + 1} / {count}</span>
    <button type="button" disabled={index === count - 1} onClick={() => setIndex(index + 1)}>Next {name.toLowerCase()} stage</button>
  </div>;
}
