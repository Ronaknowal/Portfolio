import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import './bayesnet-labs.css';
import { graphRoutes, layeredLayout, pathPolyline } from '../../data/bayesnet-models.js';



const sign = text => String(text).replace('-', '−');

export const round = (value, digits = 6) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (Number.isInteger(value)) return sign(String(value));
  if (value !== 0 && Math.abs(value) < 1e-4) {
    const [mantissa, exponent] = value.toExponential(3).split('e');
    const marks = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const superscript = [...exponent.replace('+', '')].map(character => marks[character] ?? character).join('');
    return `${sign(mantissa)} × 10${superscript}`;
  }
  const text = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return text === '-0' ? '0' : sign(text);
};

/** Keep every decimal place, for a column whose rows must line up. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));

/** A probability that has no value says so, and says which denominator is empty. */
export function Undefined({ because }) {
  return <span className="bn-undefined">undefined <span className="bn-caption">({because})</span></span>;
}

export const ratio = (value, because, digits = 6) =>
  (value === null || value === undefined ? <Undefined because={because} /> : round(value, digits));

/** A small exact fraction reads better than six decimals when it is exact. */
export function asFraction(value, maximumDenominator = 40) {
  if (value === null || !Number.isFinite(value)) return null;
  for (let denominator = 1; denominator <= maximumDenominator; denominator += 1) {
    const numerator = value * denominator;
    if (Math.abs(numerator - Math.round(numerator)) < 1e-12) {
      return denominator === 1 ? String(Math.round(numerator)) : `${Math.round(numerator)}/${denominator}`;
    }
  }
  return null;
}

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="bn-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="bn-question">{question}</p>}
    {role && <p className={`bn-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="bn-note">{note}</p>}
    {children}
  </section>;
}

export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`bn-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="bn-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="bn-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation naming the field, and
 * the model keeps its last good state rather than being handed a substitute.
 *
 * A probability editor here is a typed number, not a slider: the alarm prior is
 * .001, and a slider that cannot reach it would make the lesson's own opening
 * example unreachable in its own lab. */
export function NumberField({
  label, value, onChange, min, max, step = 'any', decimals = 3, suffix, hint,
  disabled = false, blind = false, placeholder,
}) {
  const [draft, setDraft] = useState(null);
  /* A blind field never displays the value it edits.
   *
   * Investigation 3 asks which of two unrevealed measurements is worth buying.
   * A field prefilled with the measurement's own number answers that question
   * before it is asked: a learner can read all four values straight off the
   * edit boxes. The control still publishes edits, so the "edit a hidden value
   * and watch nothing move" null is unaffected; it just does not show what it
   * is editing until the measurement is revealed. */
  const shown = draft === null ? (blind ? '' : String(value)) : draft;
  const scale = 10 ** decimals;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * scale - Math.round(parsed * scale)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown} disabled={disabled}
      placeholder={placeholder}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
          && Math.abs(parsed * scale - Math.round(parsed * scale)) <= 1e-9) {
          onChange(Number(parsed.toFixed(decimals)));
        }
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>;
}

export function Select({ label, value, onChange, options, hint, disabled = false }) {
  return <Field label={label} hint={hint}>
    <select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </Field>;
}

/** Three-state observation control: a variable is unknown, or observed at 0, or
 * observed at 1. Unknown is not state 0, and the control says so. */
export function StateChoice({ label, value, onChange, disabled = false, hint }) {
  const name = useId();
  return <div className="bn-state-choice">
    <span className="bn-state-label">{label}</span>
    <span className="bn-state-options">
      {[['unknown', 'unknown'], ['0', 'is 0'], ['1', 'is 1']].map(([key, text]) => (
        <label key={key}>
          <input type="radio" name={name} value={key} checked={value === key} disabled={disabled}
            onChange={() => onChange(key)} />
          <span>{text}</span>
        </label>
      ))}
    </span>
    {hint && <span className="bn-caption">{hint}</span>}
  </div>;
}

/** A table whose caption sits OUTSIDE the scroll box.
 *
 * Below the narrow breakpoint every cell becomes its own labelled row, which
 * needs `display: block`; that drops the implicit table roles, so they are
 * restored explicitly. A `<caption>` inside a blockified table collapses to the
 * width of its longest word, so the caption is a sibling paragraph instead.
 */
export function Table({ caption, headings, rows, rowClass = () => undefined, footnote }) {
  const id = useId();
  return <div className="bn-table">
    <p className="bn-caption" id={id}>{caption}</p>
    <div className="bn-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>
      <table role="table">
        <thead role="rowgroup"><tr role="row">
          {headings.map(heading => <th key={heading} role="columnheader" scope="col">{heading}</th>)}
        </tr></thead>
        <tbody role="rowgroup">{rows.map((row, index) => <tr key={index} role="row" className={rowClass(index)}>
          {row.map((cell, column) => (column === 0
            ? <th key={column} role="rowheader" scope="row" data-label={headings[column]}>{cell}</th>
            : <td key={column} role="cell" data-label={headings[column]}>{cell}</td>))}
        </tr>)}</tbody>
      </table>
    </div>
    {footnote && <p className="bn-caption">{footnote}</p>}
  </div>;
}


export const useInvestigation = useLiveInvestigation;





export function LiveResult({ state, calculateInputs, blocked, describe }) {
  const problem = useLiveResult(state, calculateInputs, blocked);
  return <div data-live-exploration="result">
    {problem ? <p role="status">{problem} The plots retain the last valid calculation; correct the inputs to update them.</p>
      : <p role="status">Live calculation for the current controls. {describe}</p>}
    <button type="button" disabled={Boolean(problem) || !state.result} onClick={state.snapshot}>Use current values as comparison baseline</button>
  </div>;
}

/* ============================================================ drawn graphs */

/**
 * One directed acyclic graph.
 *
 * Every coordinate comes from `bayesnet-models.js`, so the verifier asserts the
 * same numbers the browser draws. Three separate visual channels carry three
 * separate facts, and none of them is colour alone: an arrowhead carries
 * direction, a doubled ring plus the printed word "observed" carries the
 * observation set, and a square badge carries a query endpoint. A highlighted
 * path is drawn as a thick underlay along the node centres and is also written
 * out as text beneath the drawing.
 */
export function Dag({
  edges, positions, width = 300, height = 210, radius = 17, rowGap = 62, labels = {},
  observed = [], endpoints = [], highlight = null, highlightBlocked = false,
  dimmedEdges = [], caption, describe, legend = true, verdict,
}) {
  const useIdSafe = useId();
  const layout = positions
    ? { positions, radius, height }
    : layeredLayout(edges, { width, radius, rowGap });
  const place = layout.positions;
  const drawnRadius = layout.radius;
  const drawnHeight = layout.height;
  const observedSet = new Set(observed);
  const endpointSet = new Set(endpoints);
  const dimmed = new Set(dimmedEdges.map(([from, to]) => `${from}>${to}`));
  // The router is told which nodes carry a badge and which carry an observed
  // tag, because those are larger than the circle and are what an edge has to
  // miss on screen.
  const drawn = graphRoutes(edges, place, drawnRadius, { endpoints, observed });
  const trail = highlight ? pathPolyline(highlight, place) : null;
  const titleId = `${useIdSafe}-title`;
  const descriptionId = `${useIdSafe}-desc`;
  const identity = Object.keys(place)
    .filter(node => labels[node] && labels[node] !== node)
    .map(node => `${node} = ${labels[node]}`);
  return <figure className="bn-figure">
    {caption && <figcaption>{caption}</figcaption>}
    {/* The visual specification asks for accessible titles and descriptions.
        `role="img"` with an `aria-label` is functionally equivalent and was what
        this drew first, but it is not what the specification says; a `<title>`
        and `<desc>` referenced by id are. */}
    <svg className="bn-dag" viewBox={`0 0 ${width} ${drawnHeight}`} role="img"
      aria-labelledby={titleId} aria-describedby={descriptionId}
      style={{ maxWidth: `${width}px` }}>
      <title id={titleId}>{caption ?? `A directed graph on ${Object.keys(place).join(', ')}`}</title>
      <desc id={descriptionId}>{describe}</desc>
      {/* The highlighted path is an UNDERLAY: it follows node centres, so it
          necessarily runs under the labels of the nodes it passes through, and
          it is emitted before every node group so the circles and labels paint
          over it. The `bn-underlay` class marks that intent, and the browser
          review asserts the document order rather than taking it on trust. */}
      {/* Three states, not two. A trail drawn in the "active" gold while the
          verdict is withheld still hints at the answer to someone who has seen
          a blocked path drawn grey; `is-undecided` is neither. */}
      {trail && <polyline className={`bn-underlay bn-trail${highlightBlocked ? ' is-blocked' : ''}`
        + `${verdict === null ? ' is-undecided' : ''}`}
        points={trail.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')} />}
      {drawn.map(edge => <g key={edge.key} className={`bn-edge${dimmed.has(edge.key) ? ' is-dimmed' : ''}`}>
        <path d={edge.route.path} />
        <polygon points={edge.route.arrow.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')} />
      </g>)}
      {Object.entries(place).map(([node, point]) => <g key={node}
        className={`bn-node${observedSet.has(node) ? ' is-observed' : ''}${endpointSet.has(node) ? ' is-endpoint' : ''}`}>
        {endpointSet.has(node) && <rect x={point.x - drawnRadius - 5} y={point.y - drawnRadius - 5}
          width={2 * drawnRadius + 10} height={2 * drawnRadius + 10} rx="4" className="bn-endpoint-badge" />}
        <circle cx={point.x} cy={point.y} r={drawnRadius} />
        {observedSet.has(node) && <circle cx={point.x} cy={point.y} r={drawnRadius - 4} className="bn-observed-ring" />}
        <text x={point.x} y={point.y + 5} textAnchor="middle" className="bn-node-name">{node}</text>
        {observedSet.has(node) && <text x={point.x} y={point.y + drawnRadius + 13} textAnchor="middle"
          className="bn-node-tag">observed</text>}
      </g>)}
    </svg>
    {/* The identity list names only the nodes whose label adds something: a
        three-node example labelled X, Y and Z would otherwise print "X = X;
        Y = Y; Z = Z". The legend can be turned off for a figure that shows
        several small graphs, so it is stated once for the group instead of
        repeated beside each one. */}
    <p className="bn-caption">
      {identity.length ? `${identity.join('; ')}. ` : ''}
      {legend ? <>Arrowheads give direction. A doubled ring and the word “observed” mark the observation set
        {observed.length ? `: ${[...observed].sort().join(', ')}` : ', which is empty here'}.
        {endpoints.length ? ` A square badge marks the query endpoints ${endpoints.join(' and ')}.` : ''}</> : ''}
      {!legend && observed.length ? `Observed: ${[...observed].sort().join(', ')}.` : ''}
      {/* `verdict` is how a lab suppresses the one clause that would state the
          answer. Undefined keeps the figure default; null prints nothing; a
          string prints exactly that. */}
      {verdict === undefined
        ? (highlight ? ` The thick trail follows ${highlight.join('–')}, which is ${highlightBlocked ? 'blocked' : 'active'}.` : '')
        : (verdict ?? (highlight ? ` The thick trail follows ${highlight.join('–')}.` : ''))}
    </p>
  </figure>;
}

/** A row of probability bars with their exact values printed beside them. */
export function Distribution({ caption, labels, values, highlight = null, digits = 6, describe, className }) {
  return <div className={`bn-distribution${className ? ` ${className}` : ''}`} role="img"
    aria-label={describe ?? caption}>
    <p className="bn-caption">{caption}</p>
    <dl>
      {values.map((value, index) => <div key={labels[index]}
        className={`bn-bar-row${highlight === index ? ' is-leading' : ''}`}>
        <dt>{labels[index]}</dt>
        <dd>
          <span className="bn-bar-track"><span className="bn-bar-fill" style={{ width: `${100 * value}%` }} /></span>
          <span className="bn-bar-value">{value === 0 ? 'exactly 0' : round(value, digits)}</span>
        </dd>
      </div>)}
    </dl>
  </div>;
}

/** A single horizontal mixture whose segment widths are the weights actually used. */
export function MixtureLane({ caption, segments, totalLabel, describe }) {
  const weights = segments.map(segment =>
    `${segment.load === 0 ? 'Low' : 'High'} load: ${segment.share === null ? 'undefined' : round(segment.share, 6)}`).join(' · ');
  return <div className="bn-lane" role="img" aria-label={`${describe ?? caption} ${weights}.`}>
    <p className="bn-caption">{caption}</p>
    <div className="bn-lane-track" aria-hidden="true">
      {segments.map(segment => (segment.share > 0
        ? <span key={segment.load} className={`bn-lane-part is-load-${segment.load}`}
          style={{ width: `${100 * segment.share}%` }} />
        : null))}
    </div>
    <p className="bn-caption bn-lane-key">{weights}</p>
    <p className="bn-caption">{totalLabel}</p>
  </div>;
}
