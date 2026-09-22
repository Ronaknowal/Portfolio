import { cloneElement, isValidElement, useId, useState } from 'react';
import './scaling-labs.css';

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
/** Keep every decimal place, for a column whose rows must line up. */
export const fixed = (value, digits = 4) => (Number.isFinite(value) ? value.toFixed(digits) : round(value));
/** A measurement that was never taken is not a number. */
export const measured = (value, unit) => (value === null || value === undefined ? 'not recorded' : `${round(value, 4)}${unit ? ` ${unit}` : ''}`);
/** Include a stated decimal boundary without rejecting subtraction roundoff. */
export const withinTolerance = (value, expected, tolerance) => Math.abs(value - expected)
  <= tolerance + 4 * Number.EPSILON * Math.max(1, Math.abs(value), Math.abs(expected));
/** An exact sixth, ninth or third is clearer as a fraction than as 0.555556. */
export function fraction(value, limit = 40) {
  for (let denominator = 1; denominator <= limit; denominator += 1) {
    const numerator = value * denominator;
    if (Math.abs(numerator - Math.round(numerator)) < 1e-9) {
      return denominator === 1 ? String(Math.round(numerator)) : `${Math.round(numerator)}/${denominator}`;
    }
  }
  return round(value, 6);
}

export function Investigation({ title, question, note, children, onReset }) {
  const id = useId();
  return <section className="sc-investigation" aria-labelledby={id} data-live-exploration>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="sc-question">{question}</p>}
    {note && <p className="sc-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, children }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': error ? `${id}-error` : undefined })
    : children;
  return <label className={`sc-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="sc-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation, and the model keeps
 * its last good state instead of being handed a silent substitute. When
 * `allowMissing` is set, an empty field means "not recorded", which is a
 * different state from zero. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, suffix, allowMissing = false }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? (value === null ? '' : String(value)) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '') return allowMissing ? null : 'Type a number.';
    if (trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <div><Field label={label} error={problem} value={suffix}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown}
      onChange={event => {
        const text = event.target.value;
        setDraft(text);
        if (text.trim() === '') { if (allowMissing) onChange(null); return; }
        const parsed = Number(text);
        if (Number.isFinite(parsed) && parsed >= min && parsed <= max
          && Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) <= 1e-9) {
          onChange(Number(parsed.toFixed(decimals)));
        }
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>{/divisor|smoothing/i.test(label) && Number.isFinite(value) && Number.isFinite(min) && Number.isFinite(max) && <input type="range" aria-label={label + ' slider'} min={min} max={max} step={step} value={value} disabled={false} style={{width:'100%',accentColor:'var(--accent, #e7b94a)'}} onChange={event => {setDraft(null);onChange(Number(event.target.value));}} />}</div>;
}

export function SelectField({ label, value, onChange, options }) {
  return <Field label={label}>
    <select value={value} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </Field>;
}

export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false }) {
  return <div className={`sc-table-scroll${scroll ? ' sc-rows' : ''}`} role="region" aria-label={caption} tabIndex={0}>
    <table>
      <caption>{caption}</caption>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>{row.map((cellValue, column) => <td key={column}>{cellValue}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

/** A framed plot with one shared scale for everything drawn on it. The viewBox
 * matches the width the figure is rendered at, so the text is the size it says
 * it is. */
export function Plot({ caption, describe, width = 340, height = 200, padding = { left: 44, right: 16, top: 16, bottom: 32 }, domain, range, ticks, rangeTicks, className = '', children }) {
  const scaleX = value => padding.left + (width - padding.left - padding.right) * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom) * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  return <figure className={`sc-plot ${className}`.trim()}>
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={value}>
        <line className="sc-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 15} textAnchor="middle">{round(value, 2)}</text>
      </g>)}
      {(rangeTicks ?? [range[0], range[1]]).map(value => <text key={`y${value}`} x={padding.left - 5} y={scaleY(value) + 4} textAnchor="end" style={{ fontSize: 9 }}>{round(value, 2)}</text>)}
      <line className="sc-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="sc-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
  </figure>;
}

/** A labelled number line: actual ticks at actual coordinates, never a set of
 * different ranges stretched to the same unlabelled width. */
/** A labelled number line. The viewBox width matches the width the figure is
 * rendered at, so 11-unit type really is 11 px on screen. */
export function NumberLine({ title, domain, ticks, width = 300, height = 84, points, describe, decimals = 2, highlight }) {
  const left = 16;
  const right = width - 16;
  const place = value => left + (right - left) * (value - domain[0]) / (domain[1] - domain[0]);
  const inside = value => Math.min(right, Math.max(left, place(value)));
  return <figure className="sc-line">
    <figcaption>{title}{highlight && <span className="sc-expanded"> · the bracketed interval is expanded below</span>}</figcaption>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {/* The bracket marks the interval the magnified line below expands. Its
          caption lives in the figcaption, where it is HTML and keeps a readable
          size at every width, rather than shrinking with the viewBox. */}
      {highlight && <path className="sc-bracket" d={`M${inside(highlight[0])},72 V78 H${inside(highlight[1])} V72`} />}
      <line className="sc-axis" x1={left} x2={right} y1="40" y2="40" />
      {ticks.map(tick => <g key={tick}>
        <line className="sc-axis" x1={place(tick)} x2={place(tick)} y1="40" y2="46" />
        <text x={place(tick)} y="62" textAnchor="middle">{round(tick, decimals)}</text>
      </g>)}
      {points.map(point => (
        <circle key={point.id} className={`sc-point${point.later ? ' is-later' : ''}`} cx={place(point.at)} cy="40" r="6" />
      ))}
      {/* Every observation keeps its own marker. Identifiers that would collide
          at this scale are joined into one range label; the magnified view
          below separates them. */}
      {clusterLabels(points.map(point => ({ ...point, x: place(point.at) })), 22).map(group => (
        <text key={group.id} className="sc-point-id" x={group.x} y="24" textAnchor="middle">{group.id}</text>
      ))}
    </svg>
  </figure>;
}

/** Join identifiers that would collide at this scale into one range label. */
function clusterLabels(placed, minimumGap) {
  const sorted = [...placed].sort((a, b) => a.x - b.x);
  const groups = [];
  sorted.forEach(point => {
    const last = groups[groups.length - 1];
    if (last && point.x - last.members[last.members.length - 1].x < minimumGap) last.members.push(point);
    else groups.push({ members: [point] });
  });
  return groups.map(group => {
    const first = group.members[0];
    const last = group.members[group.members.length - 1];
    return {
      id: group.members.length === 1 ? first.id : `${first.id}–${last.id}`,
      x: (first.x + last.x) / 2,
    };
  });
}
