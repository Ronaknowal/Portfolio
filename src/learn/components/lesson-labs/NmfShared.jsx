import { cloneElement, isValidElement, useId, useState } from 'react';
import './nmf-labs.css';

export const round = (value, digits = 6) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return String(value);
  if (value !== 0 && Math.abs(value) < 1e-4) {
    const [mantissa, exponent] = value.toExponential(3).split('e');
    const marks = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const superscript = [...exponent.replace('+', '')].map(character => marks[character] ?? character).join('');
    return `${mantissa} × 10${superscript}`;
  }
  const text = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return text === '-0' ? '0' : text;
};
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? value.toFixed(digits) : round(value));
export const signed = (value, digits = 6) => {
  if (value === 0) return '0';
  return value > 0 ? `+${round(value, digits)}` : round(value, digits);
};

// --------------------------------------------------------------- the palettes
/** Sequential intensity for nonnegative mass: black at zero, cream at the
 * stated maximum. The same ramp serves feature strips and image panels. */
export function sequential(level) {
  const clamped = Math.max(0, Math.min(1, level));
  const mix = (low, high) => Math.round(low + (high - low) * clamped ** 0.85);
  return `rgb(${mix(11, 244)}, ${mix(15, 238)}, ${mix(16, 222)})`;
}
/** Diverging scale for a signed residual: blue for excess reconstructed mass,
 * a neutral light grey at exactly zero, gold for missing mass. */
export function diverging(level) {
  const clamped = Math.max(-1, Math.min(1, level));
  const neutral = [214, 214, 208];
  const end = clamped >= 0 ? [224, 158, 46] : [98, 150, 200];
  const weight = Math.abs(clamped) ** 0.8;
  return `rgb(${end.map((value, index) => Math.round(neutral[index] + (value - neutral[index]) * weight)).join(', ')})`;
}
/** Text that stays readable on whichever end of the sequential ramp it sits. */
export const inkOn = level => (level > 0.55 ? '#14181a' : '#e6e9e5');

export const componentColours = ['#e7b94a', '#8eb9a5', '#91aecf', '#c8a2c8', '#d9a07a', '#7fb3a3', '#b0a7d8', '#cfc08a'];

// ---------------------------------------------------------------- the shell
export function Investigation({ title, question, note, children, onReset }) {
  const id = useId();
  return <section className="nm-investigation" aria-labelledby={id} data-live-exploration>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="nm-question">{question}</p>}
    {note && <p className="nm-note">{note}</p>}
    {children}
  </section>;
}

export function Figure({ title, caption, children, className = '' }) {
  return <figure className={`nm-figure ${className}`.trim()}>
    {title && <figcaption><strong>{title}</strong></figcaption>}
    {children}
    {caption && <p className="nm-caption">{caption}</p>}
  </figure>;
}

export function Field({ label, value, error, children }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': error ? `${id}-error` : undefined })
    : children;
  return <label className={`nm-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="nm-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field on a declared lattice. An unparsable, out-of-range or
 * off-lattice draft stays on screen with the supported range beside it and is
 * never applied; the model keeps its last good value instead of a substitute. */
export function NumberField({ label, value, onChange, min, max, step, suffix }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const text = draft.trim();
    if (text === '' || text === '-' || text === '−') return 'Type a number.';
    const parsed = Number(text);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Supported range ${min} to ${max}.`;
    if (Math.abs(parsed / step - Math.round(parsed / step)) > 1e-9) return `Use steps of ${step}.`;
    return null;
  })();
  return <div><Field label={label} error={problem} value={suffix}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
          && Math.abs(parsed / step - Math.round(parsed / step)) <= 1e-9) onChange(parsed);
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>{/^Amount [ab]/i.test(label) && Number.isFinite(value) && Number.isFinite(min) && Number.isFinite(max) && <input type="range" aria-label={label + ' slider'} min={min} max={max} step={step} value={value} disabled={false} style={{width:'100%',accentColor:'var(--accent, #e7b94a)'}} onChange={event => {setDraft(null);onChange(Number(event.target.value));}} />}</div>;
}

export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false }) {
  return <div className={`nm-table-scroll${scroll ? ' nm-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export function useInvestigation(initial) {
  const [draft, setDraft] = useState(initial);
  return { draft, active: draft,
    edit: update => setDraft(current => ({ ...current, ...update })),
    reset: () => setDraft(initial), load: inputs => setDraft(inputs),
  };
}

// ------------------------------------------------------------ the visual kit
/** A horizontal strip of equal-width feature cells. Cell width encodes feature
 * identity, intensity and the printed number encode the amount, and the numbers
 * stay visible so the picture never has to be read by colour alone. */
export function Strip({ label, values, maximum, highlight = [], describe, compact = false, tone = 'sequential' }) {
  const top = tone === 'signed' ? Math.max(...values.map(Math.abs), 1e-12) : maximum;
  return <div className={`nm-strip${compact ? ' is-compact' : ''}`}>
    <span className="nm-strip-label">{label}</span>
    <ol className="nm-cells" role="img" aria-label={describe ?? `${label}: ${values.map(value => round(value, 4)).join(', ')}`}>
      {values.map((value, index) => {
        const level = tone === 'signed' ? value / top : value / top;
        const background = tone === 'signed' ? diverging(level) : sequential(level);
        const ink = tone === 'signed' ? '#14181a' : inkOn(level);
        return <li key={index} className={`nm-cell${highlight.includes(index) ? ' is-marked' : ''}${value > top + 1e-12 ? ' is-over' : ''}`}
          style={{ background, color: ink }} aria-hidden="true">
          <span>{round(value, tone === 'signed' ? 3 : 3)}</span>
        </li>;
      })}
    </ol>
  </div>;
}

/** An image-shaped row drawn at a true square aspect ratio on a stated scale.
 * Every value is available as text beneath, because a picture of 64 numbers is
 * not an accessible substitute for the numbers. */
export function ImagePanel({ title, values, side = 8, maximum, tone = 'sequential', note, describe, selected = null, onSelect = null }) {
  const top = tone === 'signed' ? Math.max(...values.map(Math.abs), 1e-12) : maximum;
  // For a signed panel the interesting cell is the largest in magnitude, which
  // an all-negative panel would not have reported if we took the maximum.
  const extreme = tone === 'signed'
    ? values.reduce((best, value) => (Math.abs(value) > Math.abs(best) ? value : best), values[0])
    : Math.max(...values);
  const extremeIndex = values.indexOf(extreme);
  const rowSums = Array.from({ length: side }, (_, row) =>
    values.slice(row * side, row * side + side).reduce((sum, value) => sum + value, 0));
  const label = describe ?? `${title}. ${tone === 'signed'
    ? `Signed values on a symmetric scale from minus ${round(top, 4)} to plus ${round(top, 4)}, with a neutral grey at exactly zero. The largest value by magnitude is ${round(extreme, 4)}`
    : `Intensity from 0, black, to ${round(top, 4)}, cream. The largest value is ${round(extreme, 4)}`} at row ${Math.floor(extremeIndex / side) + 1}, column ${extremeIndex % side + 1}. Row totals are ${rowSums.map(value => round(value, 3)).join(', ')}.`;
  return <div className="nm-image-panel">
    <h4>{title}</h4>
    <div className={`nm-image${onSelect ? ' is-selectable' : ''}`} role="img" aria-label={label}
      style={{ gridTemplateColumns: `repeat(${side}, 1fr)` }}>
      {values.map((value, index) => {
        const level = value / top;
        const background = tone === 'signed' ? diverging(level) : sequential(level);
        const cell = <span key={index} className={`nm-pixel${value > top + 1e-9 ? ' is-over' : ''}${selected === index ? ' is-selected' : ''}`}
          style={{ background }} />;
        if (!onSelect) return cell;
        return <button key={index} type="button" style={{ background }}
          className={`nm-pixel is-button${selected === index ? ' is-selected' : ''}`}
          aria-label={`Row ${Math.floor(index / side) + 1}, column ${index % side + 1}, value ${round(value, 4)}`}
          aria-pressed={selected === index} onClick={() => onSelect(index)} />;
      })}
    </div>
    {note && <p className="nm-panel-note">{note}</p>}
  </div>;
}

/** The numeric backing for a set of image panels: one row of the grid per row
 * of the picture, so the exact values are always selectable as text. */
export function ImageValues({ summary, panels, side = 8 }) {
  return <details className="nm-values">
    <summary>{summary}</summary>
    <Table caption={summary}
      headings={['panel', 'row', ...Array.from({ length: side }, (_, index) => `c${index + 1}`)]}
      rows={panels.flatMap(panel => Array.from({ length: side }, (_, row) => [
        row === 0 ? panel.title : '', `r${row + 1}`,
        ...panel.values.slice(row * side, row * side + side).map(value => round(value, 3)),
      ]))} scroll />
  </details>;
}

/** A legend that states what the colours mean in numbers. */
export function ScaleKey({ tone = 'sequential', maximum, note }) {
  const stops = tone === 'signed'
    ? [-1, -0.5, 0, 0.5, 1].map(level => ({ level, value: level * maximum, background: diverging(level) }))
    : [0, 0.25, 0.5, 0.75, 1].map(level => ({ level, value: level * maximum, background: sequential(level) }));
  return <div className="nm-scale-key">
    <ol aria-hidden="true">{stops.map(stop => <li key={stop.level} style={{ background: stop.background }} />)}</ol>
    <p>{tone === 'signed'
      ? `Signed scale from −${round(maximum, 4)} (blue, excess reconstructed mass) through 0 (neutral grey) to +${round(maximum, 4)} (gold, missing mass).`
      : `Intensity scale from 0 (black) to ${round(maximum, 4)} (cream).`} {note}</p>
  </div>;
}

/** A framed plot with one shared scale for everything drawn on it. The viewBox
 * is close to the rendered width, so its type keeps a readable size instead of
 * being scaled down with the drawing. */
export function Plot({ caption, describe, width = 340, height = 220, padding = { left: 48, right: 16, top: 18, bottom: 38 }, domain, range, xTicks, yTicks, formatX = value => round(value, 2), formatY = value => round(value, 3), children }) {
  const scaleX = value => padding.left + (width - padding.left - padding.right) * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom) * (value - range[0]) / (range[1] - range[0]);
  return <figure className="nm-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {(yTicks ?? []).map(value => <g key={`y${value}`}>
        <line className="nm-grid" x1={padding.left} x2={width - padding.right} y1={scaleY(value)} y2={scaleY(value)} />
        <text x={padding.left - 6} y={scaleY(value) + 5} textAnchor="end">{formatY(value)}</text>
      </g>)}
      {(xTicks ?? []).map(value => <text key={`x${value}`} x={scaleX(value)} y={height - padding.bottom + 20} textAnchor="middle">{formatX(value)}</text>)}
      <line className="nm-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="nm-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
  </figure>;
}
