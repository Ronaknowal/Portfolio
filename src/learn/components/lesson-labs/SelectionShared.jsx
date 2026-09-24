import { useLiveInvestigation, useLiveResult, useLiveStages } from './LiveInvestigationState.js';
import { cloneElement, isValidElement, useEffect, useId, useRef, useState } from 'react';
import { drawnThreshold } from '../../data/selection-models';
import './selection-labs.css';



/** Every printed number uses a typographic minus sign, matching the prose. */
const sign = text => text.replace('-', '−');
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
/** Keep every decimal place, for a column whose rows must line up and for a
 * value the prose quotes exactly. */
export const fixed = (value, digits = 6) => (Number.isFinite(value) ? sign(value.toFixed(digits)) : round(value));
export const signed = (value, digits = 6) => (value >= 0 ? `+${round(value, digits)}` : round(value, digits));
/** An exact zero is said, not shaded. A tiny nonzero value is never printed as 0. */
export const exactly = (value, digits = 6) => (value === 0 ? 'exactly 0' : round(value, digits));
/** A negative zero displays as 0 without changing the stored value. */
export const unsigned = value => (Object.is(value, -0) ? 0 : value);
/** Floating-point dust from an exact-arithmetic identity displays as the zero it
 * represents. The stored value is untouched, and the actual residual is always
 * printed separately beside it rather than hidden. */
export const settled = (value, tolerance = 1e-10) => (Math.abs(value) <= tolerance ? 0 : value);
export const percent = (value, digits = 1) => `${sign((100 * value).toFixed(digits))}%`;

export function Investigation({ title, question, note, children, onReset }) {
  const id = useId();
  return <section className="fs-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="fs-question">{question}</p>}
    {note && <p className="fs-note">{note}</p>}
    {children}
  </section>;
}

/** A labelled control. The id is explicit so the label describes the control
 * rather than the output that echoes its value. */
export function Field({ label, value, error, children }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string'
    ? cloneElement(children, { id, 'aria-describedby': error ? `${id}-error` : undefined })
    : children;
  return <label className={`fs-field${error ? ' is-invalid' : ''}`} htmlFor={id}>
    <span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>
    {control}
    {error && <span className="fs-field-error" id={`${id}-error`}>{error}</span>}
  </label>;
}

/** A number field that publishes only a valid value. An unparsable, blank or
 * out-of-range draft stays on screen with its local reason, and the model keeps
 * its last committed state instead of being handed a silent substitute. Blank
 * text is never coerced to zero. */
export function NumberField({ label, value, onChange, min, max, step = 'any', decimals = 2, integer = false, suffix, disabled = false }) {
  const [draft, setDraft] = useState(null);
  const shown = draft === null ? String(value) : draft;
  const problem = (() => {
    if (draft === null) return null;
    const trimmed = draft.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '−') return 'Type a number; a blank field is not zero.';
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return 'That is not a finite number.';
    if (integer && !Number.isInteger(parsed)) return 'Use a whole number.';
    if (parsed < min || parsed > max) return `Keep it between ${min} and ${max}.`;
    if (!integer && Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) > 1e-9) {
      return `Use at most ${decimals} decimal place${decimals === 1 ? '' : 's'}.`;
    }
    return null;
  })();
  return <Field label={label} error={problem} value={suffix}>
    <input type="number" inputMode={integer ? 'numeric' : 'decimal'} min={min} max={max} step={step}
      value={shown} disabled={disabled}
      onChange={event => {
        setDraft(event.target.value);
        const trimmed = event.target.value.trim();
        const parsed = Number(trimmed);
        if (trimmed !== '' && Number.isFinite(parsed) && parsed >= min && parsed <= max
          && (!integer || Number.isInteger(parsed))
          && (integer || Math.abs(parsed * 10 ** decimals - Math.round(parsed * 10 ** decimals)) <= 1e-9)) {
          onChange(integer ? parsed : Number(parsed.toFixed(decimals)));
        }
      }}
      onBlur={() => setDraft(null)} aria-invalid={Boolean(problem)} />
  </Field>;
}

/** Expose actual horizontal overflow without shading instructional content.
 * ResizeObserver follows the frame and its content; no scroll polling is used. */
export function ScrollRegion({ children, className = '', label = 'Diagram' }) {
  const ref = useRef(null);
  const hintId = useId();
  const [overflows, setOverflows] = useState(false);
  useEffect(() => {
    const frame = ref.current;
    if (!frame) return undefined;
    const measure = () => setOverflows(frame.scrollWidth > frame.clientWidth + 1);
    const observer = new ResizeObserver(measure);
    observer.observe(frame);
    if (frame.firstElementChild) observer.observe(frame.firstElementChild);
    measure();
    return () => observer.disconnect();
  }, [children]);
  return <div className="fs-scroll-group">
    <p id={hintId} className="fs-scroll-hint" hidden={!overflows}>↔ Scroll sideways for the full {label === 'Diagram' ? 'diagram' : 'table'}. Keyboard: focus the region and use the arrow keys.</p>
    <div ref={ref} className={`fs-scroll-content ${className}`} role="region" aria-label={label}
      aria-describedby={overflows ? hintId : undefined} tabIndex={overflows ? 0 : undefined}>{children}</div>
  </div>;
}

/** A data table whose caption sits outside the scrolling box.
 *
 * A <caption> inside a horizontally scrolling table takes the TABLE's width,
 * not the container's, so a long one is clipped mid-word instead of wrapping.
 * The caption is therefore an ordinary paragraph above it, and the scroll
 * region keeps it as its accessible name. */
export function Table({ caption, headings, rows, rowClass = () => undefined, scroll = false, stack = true, wrap = false }) {
  return <>
    <p className="fs-table-caption">{caption}</p>
    <ScrollRegion className={`fs-table-scroll${scroll ? ' fs-rows' : ''}${stack ? ' is-stackable' : ''}${wrap ? ' is-wrapping' : ''}`}
      label={caption}>
      <table>
        <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={index} className={rowClass(index)}>
          {row.map((cell, column) => <td key={column} data-heading={headings[column]}>{cell}</td>)}
        </tr>)}</tbody>
      </table>
    </ScrollRegion>
  </>;
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

/** A framed plot with one shared scale for everything drawn on it. The viewBox
 * matches the rendered width, so a 12px label renders at 12px. */
export function Plot({
  caption, describe, width = 340, height = 200,
  padding = { left: 46, right: 14, top: 16, bottom: 32 },
  domain, range, ticks, valueTicks, formatTick = value => round(value, 2),
  formatValueTick = value => round(value, 3), children,
}) {
  const scaleX = value => padding.left + (width - padding.left - padding.right) * (value - domain[0]) / (domain[1] - domain[0]);
  const scaleY = value => height - padding.bottom - (height - padding.top - padding.bottom) * (value - range[0]) / (range[1] - range[0]);
  const marks = ticks ?? Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  const sideMarks = valueTicks ?? [range[0], (range[0] + range[1]) / 2, range[1]];
  return <figure className="fs-plot">
    {caption && <figcaption>{caption}</figcaption>}
    <ScrollRegion><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      {marks.map(value => <g key={`x${value}`}>
        <line className="fs-grid" x1={scaleX(value)} x2={scaleX(value)} y1={padding.top} y2={height - padding.bottom} />
        <text x={scaleX(value)} y={height - padding.bottom + 18} textAnchor="middle">{formatTick(value)}</text>
      </g>)}
      {sideMarks.map(value => <text key={`y${value}`} x={padding.left - 6} textAnchor="end"
        y={Math.min(Math.max(scaleY(value) + 4, padding.top + 9), height - padding.bottom - 2)}>{formatValueTick(value)}</text>)}
      <line className="fs-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
      <line className="fs-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
      {children(scaleX, scaleY)}
    </svg></ScrollRegion>
  </figure>;
}

/** Signed bars around a common zero line, on ONE named scale. Two quantities
 * that are not in the same units must use two of these and say so. A value that
 * is exactly zero gets its own mark and its own word. */
export function SignedBars({ caption, describe, items, unit = '', digits = 6, highlight = () => false }) {
  const extent = Math.max(...items.map(item => Math.abs(item.value))) || 1;
  return <div className="fs-bars" role="img" aria-label={describe}>
    <p className="fs-caption">{caption}</p>
    {items.map((item, index) => {
      const share = 50 * Math.abs(item.value) / extent;
      const zero = item.value === 0;
      return <div className={`fs-bar-row${highlight(index) ? ' is-highlight' : ''}`} key={item.name}>
        <span className="fs-bar-name">{item.name}</span>
        <span className="fs-bar-track">
          <span className="fs-bar-zero" />
          {zero
            ? <span className="fs-bar-exact-zero" aria-hidden="true">0</span>
            : <span className={`fs-bar-fill ${item.value > 0 ? 'is-positive' : 'is-negative'}`}
              style={item.value > 0
                ? { left: '50%', width: `${share}%` }
                : { right: '50%', width: `${share}%` }} />}
        </span>
        <span className="fs-bar-value">{zero ? 'exactly 0' : `${signed(item.value, digits)}${unit}`}</span>
      </div>;
    })}
  </div>;
}

/** A waterfall drawn from computed segment endpoints. `model` is the result of
 * selection-models' waterfall(): its last segment's end IS the reconstructed
 * output, so the closing bar is arithmetic rather than a placed rectangle. */
/** A waterfall drawn from computed segment endpoints.
 *
 * Three columns that never overlap: a name, the bar track, and the value. The
 * caption is HTML below the drawing, because a sentence inside a 340-unit
 * viewBox runs past its right edge and is clipped mid-word. */
export function Waterfall({ model, names, baselineLabel = 'baseline', totalLabel = 'reconstructed output', unit = '', digits = 6, describe }) {
  const width = 340;
  const rowHeight = 26;
  const height = 16 + rowHeight * (model.segments.length + 2);
  const left = 120;
  const right = 276;
  const span = (model.maximum - model.minimum) || 1;
  const place = value => left + (right - left) * (value - model.minimum) / span;
  const bars = [
    { name: baselineLabel, low: Math.min(model.minimum, model.baseline), high: model.baseline, kind: 'base', value: model.baseline },
    ...model.segments.map((segment, index) => ({
      name: names[index], low: segment.low, high: segment.high,
      kind: segment.sign > 0 ? 'up' : segment.sign < 0 ? 'down' : 'zero', value: segment.value,
    })),
    { name: totalLabel, low: Math.min(model.minimum, model.reconstruction), high: model.reconstruction, kind: 'base', value: settled(model.reconstruction) },
  ];
  return <div className="fs-waterfall">
    <ScrollRegion><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
      <line className="fs-grid" x1={place(model.baseline)} x2={place(model.baseline)} y1="6" y2={height - 8} />
      {bars.map((bar, index) => {
        const y = 8 + index * rowHeight;
        const start = place(bar.low);
        const barWidth = Math.max(place(bar.high) - start, 0);
        return <g key={bar.name}>
          <text className="is-small" x="4" y={y + 14}>{bar.name}</text>
          {bar.kind === 'zero'
            ? <text className="fs-zero-text is-small" x={place(bar.low)} y={y + 15} textAnchor="middle">0</text>
            : <rect className={`fs-wf-bar is-${bar.kind}`} x={start} y={y + 4} width={barWidth} height={rowHeight - 12} />}
          <text className="is-small" x="336" y={y + 14} textAnchor="end">
            {bar.kind === 'base' ? round(unsigned(bar.value), digits) : (bar.value === 0 ? 'exactly 0' : signed(bar.value, digits))}
          </text>
        </g>;
      })}
    </svg></ScrollRegion>
    <p className="fs-caption">
      Values in {unit || 'output units'}; the dotted vertical line is the baseline. The two outer bars are levels, drawn from
      the axis floor: the first is the baseline and the last is the reconstructed output. Each contribution between them is a
      floating bar that starts where the one above it finished, so the closing level is the running total rather than a placed
      rectangle. A bar's length is the distance it spans on the axis, so an outer level and a contribution can happen to be the
      same length without meaning the same thing; the number beside each bar is the value.
    </p>
  </div>;
}

/** A closed detail block. Hints and worked explanations start closed and never
 * select the learner's answer. */
export function Folded({ summary, children }) {
  return <details className="fs-folded"><summary>{summary}</summary>{children}</details>;
}

/* ------------------------------------------------------ the saved tree */

/** Place the saved tree's nodes: leaves take equally spaced slots in visiting
 * order and an internal node sits at the mean of its two children. The layout
 * is drawing arithmetic; every label it carries — the split field, the exact
 * threshold, the leaf distribution and the fitted row count — is read from the
 * saved tree, and the highlighted path is computed by treeDecision. */
export function treeLayout(tree, { slot = 68, left = 34, top = 12, level = 50 } = {}) {
  const depth = new Array(tree.childrenLeft.length).fill(0);
  const order = [];
  const walk = (node, level_) => {
    depth[node] = level_;
    if (tree.childrenLeft[node] === -1) { order.push(node); return; }
    walk(tree.childrenLeft[node], level_ + 1);
    walk(tree.childrenRight[node], level_ + 1);
  };
  walk(0, 0);
  const x = new Array(tree.childrenLeft.length).fill(0);
  order.forEach((node, index) => { x[node] = left + index * slot; });
  const settle = node => {
    if (tree.childrenLeft[node] === -1) return x[node];
    x[node] = (settle(tree.childrenLeft[node]) + settle(tree.childrenRight[node])) / 2;
    return x[node];
  };
  settle(0);
  return {
    nodes: x.map((position, node) => ({
      node, x: position, y: top + depth[node] * level, depth: depth[node],
      leaf: tree.childrenLeft[node] === -1,
    })),
    leafOrder: order,
    width: left * 2 + (order.length - 1) * slot,
    height: top + Math.max(...depth) * level + 36 + 10,
  };
}

/** The label for a split in the drawing.
 *
 * NOT a rounding of the stored threshold. Rounding produces a rule that can
 * disagree with the tree at a value the learner can type — 1.5899999737739563
 * rounds to 1.59, and 1.59 takes the OTHER branch. The label is instead the
 * largest value on that control's own grid which the tree still sends left, so
 * the drawn rule and the applied rule agree at every enterable value. Where no
 * control is supplied there is no grid to be equivalent on, and the exact
 * stored double is shown. */
function thresholdLabel(threshold, control) {
  if (!control) return String(threshold);
  const drawn = drawnThreshold(threshold, control);
  if (drawn === null) return String(threshold);
  return drawn.value.toFixed(drawn.decimals);
}

/** The saved four-field tree, with one row's actual route highlighted.
 *
 * Both lines inside a node box are placed far enough apart that their glyph
 * boxes do not touch, and the branch labels sit under the parent rather than
 * midway, where they would land on the child's own text. */
export function TreeDiagram({ tree, shortLabels, controls, decision, classIndex = 0, describe, boxWidth = 64, boxHeight = 36 }) {
  const layout = treeLayout(tree);
  const onPath = new Set(decision ? [...decision.path.map(step => step.node), decision.leaf] : []);
  const takenEdge = new Set(decision ? decision.path.map(step => `${step.node}-${step.next}`) : []);
  const width = 340;
  return <ScrollRegion><svg viewBox={`0 0 ${width} ${layout.height}`} role="img" aria-label={describe}>
    {layout.nodes.filter(item => !item.leaf).flatMap(item => [tree.childrenLeft[item.node], tree.childrenRight[item.node]]
      .map(child => {
        const target = layout.nodes[child];
        const taken = takenEdge.has(`${item.node}-${child}`);
        const isLeft = child === tree.childrenLeft[item.node];
        return <g key={`${item.node}-${child}`}>
          <path className={taken ? 'fs-flow' : 'fs-flow is-alt'}
            d={`M${item.x},${item.y + boxHeight} C${item.x},${item.y + boxHeight + 10} ${target.x},${target.y - 10} ${target.x},${target.y}`} />
          <text className="is-small" x={item.x + (isLeft ? -13 : 13)} y={item.y + boxHeight + 11} textAnchor="middle">
            {isLeft ? '≤' : '>'}
          </text>
        </g>;
      }))}
    {layout.nodes.map(item => {
      const highlighted = onPath.has(item.node);
      const probability = tree.value[item.node][classIndex];
      return <g key={item.node}>
        <rect className={`fs-lane${highlighted ? ' is-refit' : ''}`} x={item.x - boxWidth / 2} y={item.y}
          width={boxWidth} height={boxHeight} rx="3" />
        <text className="is-small" x={item.x} y={item.y + 14} textAnchor="middle">
          {item.leaf ? `P₁ ${round(probability, 3)}` : shortLabels[tree.feature[item.node]]}
        </text>
        <text className="is-small" x={item.x} y={item.y + 29} textAnchor="middle">
          {item.leaf ? `n ${tree.samples[item.node]}` : `≤ ${thresholdLabel(tree.threshold[item.node], controls && controls[tree.feature[item.node]])}`}
        </text>
      </g>;
    })}
  </svg></ScrollRegion>;
}
