import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useId, useState } from 'react';
import { MISSING, edgeWidth, edgeWidthRange, localLikelihood, trellis, trellisEdges } from '../../data/hmm-models.js';
import './hmm-labs.css';



/** Every printed number uses a typographic minus sign, matching the prose. */
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
/** An exact zero is said in words, never shaded, and a tiny value is never 0. */
export const exactly = (value, digits = 6) => (value === 0 ? 'exactly 0' : round(value, digits));

/** A quantity with no value says so and names why.
 *
 * The load-bearing case here is an impossible observation sequence: its
 * posterior is undefined, not zero and not uniform. Printing a number there
 * would teach exactly the mistake section 7 exists to prevent.
 */
export function Undefined({ because }) {
  return <span className="hmm-undefined" title={because}>undefined <span className="hmm-caption">({because})</span></span>;
}

/** A small exact fraction reads better than six decimals when it is exact. */
export function asFraction(value, maximumDenominator = 120) {
  if (value === null || !Number.isFinite(value)) return null;
  for (let denominator = 1; denominator <= maximumDenominator; denominator += 1) {
    const numerator = value * denominator;
    if (Math.abs(numerator - Math.round(numerator)) < 1e-12) {
      return denominator === 1 ? String(Math.round(numerator)) : `${Math.round(numerator)}/${denominator}`;
    }
  }
  return null;
}

/** A state's identity carried by name, index, position and a line pattern, so
 * nothing in this lesson depends on telling two colours apart. */
export const statePattern = index => ['solid', 'dashed', 'dotted', 'dot-dash'][index % 4];
export function StateKey({ names, prefix = 'state' }) {
  return <p className="hmm-legend">
    {names.map((name, index) => <span key={name}>
      <svg viewBox="0 0 34 12" aria-hidden="true">
        <line className={`hmm-key-line is-${statePattern(index)}`} x1="1" y1="6" x2="33" y2="6" />
      </svg>
      {prefix} {index} · {name}
    </span>)}
  </p>;
}

export function Investigation({ title, question, note, role, children, onReset }) {
  const id = useId();
  return <section className="hmm-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="hmm-question">{question}</p>}
    {role && <p className={`hmm-role${role.kind ? ` is-${role.kind}` : ''}`} role="status">{role.text}</p>}
    {note && <p className="hmm-note">{note}</p>}
    {children}
  </section>;
}

/** A labelled control. The id is explicit so the label describes the control
 * rather than the output that echoes its value. */
export function Field({ label, value, error, hint, children }) {
  const id = useId();
  const described = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean).join(' ');
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': described || undefined })
    : children;
  return <label className={`hmm-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {hint && <span className="hmm-caption" id={`${id}-hint`}>{hint}</span>}
    {error && <span className="hmm-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable or
 * out-of-range draft stays on screen with an explanation naming the field, and
 * the model keeps its last good state instead of being handed a substitute. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, suffix, hint, disabled = false }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix} hint={hint}>
    <input type="number" inputMode="decimal" min={min} max={max} step={step} value={shown} disabled={disabled}
      onChange={event => {
        setDraft(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
          && Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) <= 1e-9) {
          onChange(Number(parsed.toFixed(decimals)));
        }
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>;
}

/** A slider paired with the same value as a typed number: the keyboard route
 * and the precise route are both first class, neither is a fallback. */
export function SliderField({ label, value, onChange, min, max, step, decimals = 2, suffix, hint, disabled = false }) {
  return <div className="hmm-paired">
    <Field label={label} value={suffix ?? round(value, decimals)} hint={hint}>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={event => onChange(Number(Number(event.target.value).toFixed(decimals)))} />
    </Field>
    <NumberField label={`${label} — exact value`} value={value} onChange={onChange}
      min={min} max={max} step={step} decimals={decimals} disabled={disabled} />
  </div>;
}

export function Select({ label, value, onChange, options, hint, disabled = false }) {
  return <Field label={label} hint={hint}>
    <select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </Field>;
}

/** One probability row, edited whole and applied as one action.
 *
 * Editing a distribution entry by entry cannot work: either the row stops
 * summing to one between keystrokes, or some other entry is silently changed to
 * compensate, and a learner who did not touch that entry is then shown a
 * different model from the one they built. So the row is a text draft, checked
 * on demand, and applied only when it is a distribution. A structural zero
 * typed as 0 stays exactly 0; nothing here lifts it to an epsilon.
 */
export function ProbabilityRow({ label, values, names, onApply, disabled = false, tolerance = 1e-9 }) {
  const id = useId();
  const [draft, setDraft] = useState(null);
  // Editable values must round-trip through Number(), including tiny entries
  // and thirds. Display formatting would change the row or make it unparsable.
  const text = draft === null ? values.map(String).join(', ') : draft;
  const parsed = text.split(/[,\s]+/).filter(part => part !== '').map(part => Number(part.replace('−', '-')));
  const total = parsed.reduce((sum, value) => sum + value, 0);
  const problem = (() => {
    if (parsed.length !== values.length) return `Give ${values.length} numbers, separated by commas.`;
    if (parsed.some(value => !Number.isFinite(value))) return 'Every entry must be a number.';
    if (parsed.some(value => value < 0)) return 'A probability cannot be negative.';
    if (Math.abs(total - 1) > tolerance) return `These sum to ${round(total, 6)}; a row must sum to 1.`;
    return null;
  })();
  const changed = draft !== null && parsed.some((value, index) => value !== values[index]);
  return <div className={`hmm-row-editor${problem ? ' is-invalid' : ''}`}>
    <label className="hmm-field" htmlFor={id}>
      <span>{label}<output htmlFor={id}>sums to {round(total, 6)}</output></span>
      <input id={id} type="text" inputMode="decimal" value={text} disabled={disabled}
        aria-describedby={`${id}-help`} aria-invalid={Boolean(problem)}
        onChange={event => setDraft(event.target.value)} />
      <span className="hmm-caption" id={`${id}-help`}>
        One number per {names.join(', ')}. The row is applied only when it is a distribution, so nothing you did
        not type is quietly adjusted.
      </span>
      {problem && <span className="hmm-field-error">{problem}</span>}
    </label>
    <div className="hmm-row-actions">
      <button type="button" disabled={disabled || Boolean(problem) || !changed}
        onClick={() => { onApply(parsed.slice()); setDraft(null); }}>Validate and apply this row</button>
      <button type="button" disabled={disabled || draft === null} onClick={() => setDraft(null)}>Discard the edit</button>
    </div>
  </div>;
}

/** A table whose caption sits OUTSIDE the scroll box.
 *
 * Below the narrow breakpoint every cell becomes its own labelled row, which
 * needs `display: block`; that drops the implicit table roles, so they are
 * restored explicitly here. A `<caption>` inside a blockified table collapses to
 * the width of its longest word, so the caption is a sibling paragraph.
 */
export function Table({ caption, headings, rows, rowClass = () => undefined, cellClass, scroll = false, footnote }) {
  const id = useId();
  return <div className="hmm-table">
    <p className="hmm-caption" id={id}>{caption}</p>
    <div className={`hmm-table-scroll${scroll ? ' hmm-rows' : ''}`} role="region" aria-labelledby={id} tabIndex={0}>
      <table role="table">
        <thead role="rowgroup"><tr role="row">
          {headings.map(heading => <th key={heading} role="columnheader" scope="col">{heading}</th>)}
        </tr></thead>
        <tbody role="rowgroup">{rows.map((row, index) => <tr key={index} role="row" className={rowClass(index)}>
          {row.map((cell, column) => (column === 0
            ? <th key={column} role="rowheader" scope="row" data-label={headings[column]}>{cell}</th>
            : <td key={column} role="cell" data-label={headings[column]}
              className={cellClass ? cellClass(index, column) : undefined}>{cell}</td>))}
        </tr>)}</tbody>
      </table>
    </div>
    {footnote && <p className="hmm-caption">{footnote}</p>}
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

/** A construction task: start unsolved, grade the computed consequence.
 *
 * `grade(draft)` returns `{ solved, conditions: [{ met, text }] }`. The verdict
 * is never derived from which preset was pressed, so loading a setup that
 * happens to be a solution still has to be submitted and computed. Reset puts
 * the named fixture back and clears success.
 */
export function Construction({ attempts, onSubmit, task, title = 'Construct it yourself', submitLabel = 'Submit this attempt', disabledNote }) {
  const id = useId();
  const latest = attempts.at(-1);
  const solved = attempts.some(attempt => attempt.verdict.solved);
  return <section className="hmm-construction" aria-labelledby={id}>
    <h4 id={id}>{title}</h4>
    <p>{task}</p>
    {disabledNote && <p className="hmm-note">{disabledNote}</p>}
    <div className="hmm-buttons">
      <button type="button" className="is-primary" onClick={onSubmit}>{submitLabel}</button>
      <span>{attempts.length === 0
        ? 'Nothing is checked until you submit: the task starts unsolved.'
        : `${attempts.length} attempt${attempts.length === 1 ? '' : 's'} submitted.`}</span>
    </div>
    {latest && <div className={`hmm-verdict ${latest.verdict.solved ? '' : 'is-miss'}`} role="status">
      <p><strong>{latest.verdict.solved ? 'Solved.' : 'Not yet.'}</strong> Each condition is computed from the
        inputs you submitted, not from which setup you loaded.</p>
      <ul>{latest.verdict.conditions.map(condition => <li key={condition.text}>
        <span className="hmm-verdict-mark" aria-hidden="true">{condition.met ? '=' : '≠'}</span>
        {condition.text}
      </li>)}</ul>
    </div>}
    {solved && !latest.verdict.solved && <p className="hmm-caption">
      An earlier attempt did solve it. Reset to start again from the named fixture.
    </p>}
  </section>;
}

/** A framed plot with one shared scale for everything drawn on it. */
export function Plot({
  caption, describe, width = 340, height = 200,
  padding = { left: 52, right: 14, top: 16, bottom: 34 },
  domain, range, ticks, valueTicks,
  formatTick = value => round(value, 2), formatValue = value => round(value, 3), children,
}) {
  const scaleX = value => padding.left + (width - padding.left - padding.right)
    * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom)
    * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (unused, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const sideMarks = valueTicks ?? [range[0], (range[0] + range[1]) / 2, range[1]];
  return <figure className="hmm-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={`x${value}`}>
        <line className="hmm-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 18} textAnchor="middle">{formatTick(value)}</text>
      </g>)}
      {/* Clamped inside the drawn box, so the lowest value label never lands on
          the row of horizontal tick labels beneath the axis. */}
      {sideMarks.map(value => <text key={`y${value}`} x={padding.left - 6} textAnchor="end"
        y={Math.min(Math.max(scaleY(value) + 4, padding.top + 9), height - padding.bottom - 2)}>{formatValue(value)}</text>)}
      <line className="hmm-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="hmm-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg>
  </figure>;
}

/** A polyline through recorded points on the plot's own scale. */
export function polyline(points, scaleX, scaleY) {
  return points.map(([x, y]) => `${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`).join(' ');
}

/** Nonnegative quantities adding to one total, drawn as one track.
 *
 * Every part carries its own fill pattern and its own printed value, so the
 * parts are distinguishable without colour and a part that is exactly zero says
 * so instead of becoming an invisible sliver. */
export function Stack({ caption, describe, parts, totalLabel = 'total', digits = 6, unit = '' }) {
  const total = parts.reduce((accumulated, part) => accumulated + part.value, 0);
  const extent = Math.max(total, 1e-12);
  return <div className="hmm-stack" role="img" aria-label={describe}>
    <p className="hmm-caption">{caption}</p>
    <div className="hmm-stack-track">
      {parts.map(part => (part.value > 0
        ? <span key={part.name} className={`hmm-stack-part ${part.className}`}
          style={{ width: `${100 * part.value / extent}%` }} />
        : null))}
    </div>
    <dl className="hmm-stack-key">
      {parts.map(part => <div key={part.name} className="hmm-stack-row">
        <dt><span className={`hmm-swatch ${part.className}`} aria-hidden="true" />{part.name}</dt>
        <dd>{exactly(part.value, digits)}{part.value === 0 ? '' : unit}</dd>
      </div>)}
      <div className="hmm-stack-row is-total">
        <dt>{totalLabel}</dt>
        <dd>{round(total, digits)}{unit}</dd>
      </div>
    </dl>
  </div>;
}

/* ===================================== drawing primitives shared by both files */

/** The largest trellis this drawing renders before handing over to the table.
 *
 * Beyond it the nodes are too narrow to hold their own values without the type
 * shrinking, and shrinking the type instead would just move the failure
 * somewhere a reader notices later. The exact table carries every cell at any
 * length, so a long sequence loses the picture and keeps the numbers rather
 * than the reverse.
 */
export const maximumDrawnColumns = 6;

/** A frame around one drawing.
 *
 * On a wide screen it caps the drawing near its own viewBox so the type is not
 * magnified by the column. On a phone it lets the drawing keep its natural
 * width and scrolls, which is what the layout contract allows for a diagram,
 * rather than shrinking every label to two thirds of its intended size.
 */
export function Diagram({ children, describe, hint }) {
  return <div className="hmm-diagram-frame">
    <div className="hmm-diagram" role="group" aria-label={describe} tabIndex={0}>{children}</div>
    {/* Shown only at the width where the drawing actually scrolls. A scroll box
        with no affordance reads as a clipped drawing. The extra sentence is the
        caller's, because "each node keeps its state initial" is true of a
        trellis and false of a schematic topology. */}
    <p className="hmm-diagram-hint">Drag the drawing sideways to see the rest of it.{hint ? ` ${hint}` : ''}</p>
  </div>;
}

const observationLabel = (model, value) => (value === MISSING ? 'missing' : model.symbolNames[value]);

/** The trellis: nodes, incoming edges whose width encodes the share they carry,
 * the selected predecessor in max mode, and the observation card under each
 * column.
 *
 * Nothing here computes a proportion. `trellisEdges` supplies each edge's share
 * of its destination's incoming total and the stroke width that encodes it, and
 * `trellis` supplies every cell value. The focus step's arithmetic is printed in
 * reflowing HTML beneath the drawing rather than on the edges, because four
 * labels on four crossing edges collide at every width worth reading.
 */
export function Trellis({
  model, observations, mode = 'sum', focusTime = null, path = null, queryTime = null,
  showForbidden = false, showValues = true, label, describe,
}) {
  
  const columns = observations.length;
  if (columns > maximumDrawnColumns) return null;
  const states = model.start.length;
  const built = trellis(model, observations, mode);
  const edgeRows = trellisEdges(model, observations, mode);
  const width = 420;
  const left = 56;
  const right = 14;
  const top = 34;
  const rowStep = 46;
  const spacing = (width - left - right) / columns;
  const nodeWidth = Math.min(54, spacing - 10);
  const nodeHeight = 24;
  const centreX = time => left + spacing * (time + 0.5);
  const centreY = state => top + state * rowStep;
  /* The emission connector is a column-level bus: a short horizontal tick under
     the whole column, then one drop to the card. Drawn from the bottom node
     alone it read as "only the last state emits", and drawn from every node it
     would have to cross the nodes beneath it. */
  const busY = centreY(states - 1) + nodeHeight / 2 + 10;
  const cardY = busY + 16;
  const cardHeight = 26;
  const height = cardY + cardHeight + 26;
  const onPath = (time, state) => Boolean(path) && path[time] === state;
  return <Diagram describe={describe}
    hint="At this width each node keeps only its state initial, and the exact values are in the table beside it.">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
    <title>{label}</title>
    {model.stateNames.map((name, state) => <g key={name}>
      <line className={`hmm-key-line is-${statePattern(state)}`}
        x1="2" y1={centreY(state) - 11} x2="30" y2={centreY(state) - 11} />
      <text className="hmm-small" x="2" y={centreY(state) + 5}>{name}</text>
    </g>)}
    {edgeRows.map(row => row.edges.map(edge => {
      const from = { x: centreX(edge.time - 1) + nodeWidth / 2, y: centreY(edge.from) };
      const to = { x: centreX(edge.time) - nodeWidth / 2, y: centreY(edge.to) };
      const dimmed = focusTime !== null && focusTime !== edge.time;
      if (edge.forbidden) {
        /* An explicit width even here, so that every edge on the page carries a
           stroke-width attribute and none of them sits outside the guard that
           compares attributes against painted values. */
        return showForbidden
          ? <line key={`${edge.time}-${edge.from}-${edge.to}`} className="hmm-edge is-forbidden"
            x1={from.x} y1={from.y} x2={to.x} y2={to.y} strokeWidth={edgeWidthRange.zero} />
          : null;
      }
      const classes = ['hmm-edge'];
      if (edge.carriesNothing) classes.push('is-empty');
      else if (mode === 'max') classes.push(edge.chosen ? 'is-chosen' : 'is-rejected');
      else classes.push('is-carrying');
      return <line key={`${edge.time}-${edge.from}-${edge.to}`} className={classes.join(' ')}
        x1={from.x} y1={from.y} x2={to.x} y2={to.y}
        strokeWidth={edgeWidth(edge.share)}
        opacity={dimmed ? 0.35 : 1} />;
    }))}
    {built.columns.map(column => <g key={column.time}>
      {column.cells.map(cell => {
        const x = centreX(column.time) - nodeWidth / 2;
        const y = centreY(cell.state) - nodeHeight / 2;
        const classes = ['hmm-node'];
        if (onPath(column.time, cell.state)) classes.push('is-on-path');
        if (queryTime === column.time) classes.push('is-query');
        return <g key={cell.state}>
          <rect className={classes.join(' ')} x={x} y={y} width={nodeWidth} height={nodeHeight} rx="4" />
          {/* The cell value lives INSIDE its node. Printed beneath it, the
              bottom row's values sat exactly where the emission connector runs,
              and every one of them was struck through. The row labels on the
              left already name the states, so a node that carries a value loses
              nothing by not repeating the name.

              Two texts, one shown: an eight-character number inside a node that
              renders 37px wide on a phone is not a small label, it is an
              unreadable one. Below the narrow breakpoint the node carries its
              state initial and the exact values are read from the table beside
              the drawing. The stylesheet chooses; nothing is recomputed. */}
          <text className="hmm-small hmm-halo hmm-wide-only" x={centreX(column.time)}
            y={centreY(cell.state) + 4} textAnchor="middle">
            {showValues ? round(cell.value, 6) : model.stateNames[cell.state].slice(0, 5)}
          </text>
          <text className="hmm-small hmm-halo hmm-narrow-only" x={centreX(column.time)}
            y={centreY(cell.state) + 4} textAnchor="middle">
            {showValues ? model.stateNames[cell.state].slice(0, 1) : model.stateNames[cell.state].slice(0, 3)}
          </text>
        </g>;
      })}
      {/* The observation card is a different kind of object from a hidden state,
          so it gets its own outline and a dashed emission connector rather than
          a thinner version of a transition. The tick spans the column, so the
          card reads as the report observed at that time rather than as
          something the bottom state alone produced. */}
      <line className="hmm-emission" x1={centreX(column.time) - nodeWidth / 2} y1={busY}
        x2={centreX(column.time) + nodeWidth / 2} y2={busY} />
      <line className="hmm-emission" x1={centreX(column.time)} y1={busY}
        x2={centreX(column.time)} y2={cardY} />
      <rect className={`hmm-card${column.observation === MISSING ? ' is-missing' : ''}`}
        x={centreX(column.time) - nodeWidth / 2} y={cardY} width={nodeWidth} height={cardHeight} rx="3" />
      <text className="hmm-small hmm-halo" x={centreX(column.time)} y={cardY + 17} textAnchor="middle">
        {observationLabel(model, column.observation)}
      </text>
      <text className="hmm-small hmm-muted" x={centreX(column.time)} y={cardY + cardHeight + 15}
        textAnchor="middle">t = {column.time}</text>
    </g>)}
  </svg></Diagram>;
}

/** The arithmetic of one trellis step, in reflowing HTML beside the drawing.
 *
 * Four labels on four crossing edges collide at every width worth reading, so
 * the amounts live here instead. Each line is the whole factorisation a learner
 * has to follow: the previous cell, the transition it takes, the contribution
 * that carries, how the contributions combine, and the destination emission
 * applied last.
 */
export function TrellisStep({ model, observations, mode, time }) {
  if (time < 1 || time >= observations.length) return null;
  const built = trellis(model, observations, mode);
  const row = trellisEdges(model, observations, mode)[time - 1];
  const local = localLikelihood(model, observations[time]);
  return <div className="hmm-step">
    <p className="hmm-caption">
      Step t = {time}, report <b>{observationLabel(model, observations[time])}</b>. Each destination{' '}
      {mode === 'sum' ? 'adds' : 'keeps the largest of'} its incoming contributions, then multiplies by its own
      emission probability for that report.
    </p>
    <ul>
      {built.columns[time].cells.map(cell => {
        const incoming = row.edges.filter(edge => edge.to === cell.state);
        const chosen = incoming.find(edge => edge.chosen);
        const contributions = incoming.map(edge => (
          `${model.stateNames[edge.from]} ${round(edge.previousValue, 6)} × ${round(edge.transition, 6)} = ${round(edge.carried, 6)}`
        )).join('; ');
        const combined = mode === 'sum'
          ? `Their sum is ${round(cell.aggregate, 6)}`
          : `The largest is ${round(cell.aggregate, 6)}, from ${model.stateNames[chosen.from]}`
            + (cell.tiedPredecessor ? ' (an exact tie; the lower state index owns it)' : '');
        return <li key={cell.state}>
          <b>{model.stateNames[cell.state]}</b> {contributions}. {combined} × emission{' '}
          {round(local[cell.state], 6)} = <b>{round(cell.value, 6)}</b>.
        </li>;
      })}
    </ul>
  </div>;
}

/** The exact trellis as a table, which works at every sequence length. */
export function TrellisTable({ model, observations, mode, caption }) {
  const built = trellis(model, observations, mode);
  const valueHeadings = model.stateNames.map(name => (
    `${mode === 'sum' ? 'forward mass' : 'best prefix'} ${name}`
  ));
  const trailingHeadings = mode === 'max'
    ? model.stateNames.map(name => `${name} predecessor`)
    : ['column total'];
  return <Table caption={caption} scroll={observations.length > 6}
    headings={['time and report', ...valueHeadings, ...trailingHeadings]}
    rows={built.columns.map(column => {
      const trailing = mode === 'max'
        ? column.cells.map(cell => (cell.chosen === null ? '—' : model.stateNames[cell.chosen]))
        : [fixed(built.columnTotals[column.time], 8)];
      return [
        `${column.time} ${observationLabel(model, column.observation)}`,
        ...column.cells.map(cell => fixed(cell.value, 8)),
        ...trailing,
      ];
    })}
    footnote={mode === 'sum'
      ? 'The column total at the last time is the probability of the whole observed sequence.'
      : 'Backtracking starts at the largest cell of the last column and follows the stored predecessors.'} />;
}

/** The state graph: start probabilities and transitions, with emissions left out.
 *
 * The emission rows are deliberately NOT drawn as arcs. A transition and an
 * emission are different conditional factorisations, and drawing them in one
 * visual vocabulary is exactly the confusion section 1 exists to remove, so
 * emissions stay an exact table beside the graph.
 */
export function StateGraph({ model, label, describe }) {
  const states = model.start.length;
  const width = 340;
  const height = 158;
  const radius = 28;
  const spacing = states === 2 ? 132 : 98;
  const startX = width / 2 - spacing * (states - 1) / 2;
  const centre = state => ({ x: startX + spacing * state, y: 84 });
  return <Diagram describe={describe}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
    <title>{label}</title>
    {model.stateNames.map((name, origin) => model.stateNames.map((other, destination) => {
      const from = centre(origin);
      const to = centre(destination);
      const probability = model.transition[origin][destination];
      if (probability === 0) return null;
      if (origin === destination) {
        const loopY = from.y - radius;
        return <g key={`${name}-${other}`}>
          <path className="hmm-edge is-carrying" strokeWidth={edgeWidth(probability) ?? 1}
            d={`M ${from.x - 12} ${loopY} C ${from.x - 30} ${loopY - 40}, ${from.x + 30} ${loopY - 40}, ${from.x + 12} ${loopY}`} />
          {/* Above the arc's apex, not on it. The cubic's control points sit at
              loopY - 40, which puts its highest point at loopY - 30, so a label
              baseline at loopY - 28 straddled the curve and the arc travelled
              through a third of the text. A backplate hides a line crossing
              glyphs; it does not excuse a curve running the width of them. */}
          <text className="hmm-small hmm-halo" x={from.x} y={loopY - 36} textAnchor="middle">
            stay {round(probability, 4)}
          </text>
        </g>;
      }
      const lift = origin < destination ? 1 : -1;
      const midY = from.y + lift * 36;
      const exitX = from.x + (to.x > from.x ? radius : -radius);
      const entryX = to.x + (to.x > from.x ? -radius : radius);
      return <g key={`${name}-${other}`}>
        <path className="hmm-edge is-carrying" strokeWidth={edgeWidth(probability) ?? 1}
          d={`M ${exitX} ${from.y + lift * 8} Q ${(from.x + to.x) / 2} ${midY} ${entryX} ${to.y + lift * 8}`} />
        {/* Two arcs join the same pair of nodes and this lesson draws no
            arrowheads, so a bare number on each cannot say which direction it
            belongs to. Each label names its own. */}
        <text className="hmm-small hmm-halo" x={(from.x + to.x) / 2} y={midY + lift * 11} textAnchor="middle">
          {name.slice(0, 1)}→{other.slice(0, 1)} {round(probability, 4)}
        </text>
      </g>;
    }))}
    {model.stateNames.map((name, state) => {
      const point = centre(state);
      return <g key={name}>
        <circle className="hmm-node" cx={point.x} cy={point.y} r={radius} />
        <line className={`hmm-key-line is-${statePattern(state)}`}
          x1={point.x - 14} y1={point.y - 13} x2={point.x + 14} y2={point.y - 13} />
        <text className="hmm-small hmm-halo hmm-strong" x={point.x} y={point.y + 2} textAnchor="middle">{name}</text>
        <text className="hmm-small hmm-halo hmm-muted" x={point.x} y={point.y + 15} textAnchor="middle">
          start {round(model.start[state], 4)}
        </text>
      </g>;
    })}
  </svg></Diagram>;
}

/** Paired filtered and smoothed bars on one shared 0 to 1 scale.
 *
 * One scale is the point: these are two answers to two different questions
 * about the same state, and drawing each against its own maximum would make
 * every sequence look the same. Each bar carries its exact value.
 */
export function BeliefBars({ rows, stateName, label, describe }) {
  const width = 420;
  const left = 40;
  const right = 14;
  const top = 26;
  const plotHeight = 92;
  const spacing = (width - left - right) / rows.length;
  const barWidth = Math.min(20, spacing / 3);
  const height = top + plotHeight + 56;
  const base = top + plotHeight;
  return <Diagram describe={describe}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
    <title>{label}</title>
    {[0, 0.5, 1].map(value => <g key={value}>
      <line className="hmm-grid" x1={left} x2={width - right} y1={base - plotHeight * value} y2={base - plotHeight * value} />
      <text className="hmm-small" x={left - 5} y={base - plotHeight * value + 4} textAnchor="end">{value}</text>
    </g>)}
    {rows.map((row, index) => {
      const centre = left + spacing * (index + 0.5);
      return <g key={row.time}>
        <rect className="hmm-bar is-filtered" x={centre - barWidth - 2} y={base - plotHeight * row.filtered}
          width={barWidth} height={plotHeight * row.filtered} />
        <rect className="hmm-bar is-smoothed" x={centre + 2} y={base - plotHeight * row.smoothed}
          width={barWidth} height={plotHeight * row.smoothed} />
        <text className="hmm-small hmm-halo" x={centre} y={base + 15} textAnchor="middle">t = {row.time}</text>
        <text className="hmm-small hmm-halo hmm-muted" x={centre} y={base + 28} textAnchor="middle">{row.label}</text>
        <text className="hmm-small hmm-halo" x={centre - barWidth / 2 - 2} y={base + 42} textAnchor="middle">
          {round(row.filtered, 4)}
        </text>
        <text className="hmm-small hmm-halo hmm-strong" x={centre + barWidth / 2 + 2} y={base + 54} textAnchor="middle">
          {round(row.smoothed, 4)}
        </text>
      </g>;
    })}
    <line className="hmm-axis" x1={left} x2={width - right} y1={base} y2={base} />
    <text className="hmm-small hmm-muted" x="2" y={top - 12}>P({stateName}) · left bar filtered, right bar smoothed</text>
  </svg></Diagram>;
}
