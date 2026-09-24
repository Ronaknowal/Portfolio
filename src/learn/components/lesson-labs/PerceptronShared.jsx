import { useId, useState } from 'react';
import { parseNumeric } from '../../data/perceptron-models.js';
import './perceptron-labs.css';
export const fmt = (value, digits = 6) => value === null ? 'undefined' : Number(value.toFixed(digits)).toString().replace(/-/g, '−');
export function PerceptronTable({
  caption,
  headers,
  rows
}) {
  return <div className="perceptron-table" tabIndex={0} role="region" aria-label={caption}><table><caption>{caption}</caption><thead><tr>{headers.map(h => <th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
export function NumberControl({
  name,
  value,
  onChange,
  min,
  max,
  step = .25,
  fractions = false
}) {
  const parseControl = text => {
    const result = parseNumeric(text, min, max);
    if (!result.error && !fractions && Math.abs((result.value - min) / step - Math.round((result.value - min) / step)) > 1e-8) {
      return { error: `Use increments of ${step} starting at ${min}; the last valid result remains visible.` };
    }
    return result;
  };
  const id = useId(),
    [draft, setDraft] = useState(null),
    parsed = draft === null ? {} : parseControl(draft);
  const edit = text => {
    setDraft(text);
    const next = parseControl(text);
    if (!next.error) onChange(next.value);
  };
  return <div className="perceptron-control"><label htmlFor={`${id}-range`}>{name}</label><output htmlFor={`${id}-range`}>{fmt(value)}</output><input id={`${id}-range`} aria-label={name} type="range" min={min} max={max} step={fractions ? 'any' : step} value={value} onChange={event => {
      setDraft(null);
      onChange(Number(event.target.value));
    }} onKeyDown={fractions ? event => {
      if (['ArrowLeft', 'ArrowDown', 'ArrowRight', 'ArrowUp'].includes(event.key)) {
        event.preventDefault();
        setDraft(null);
        onChange(Math.max(min, Math.min(max, value + (['ArrowLeft', 'ArrowDown'].includes(event.key) ? -step : step))));
      }
    } : undefined} /><label htmlFor={`${id}-number`} className="perceptron-visually-hidden">{name} exact value</label><input id={`${id}-number`} aria-describedby={parsed.error ? `${id}-error` : undefined} aria-invalid={!!parsed.error} type="text" inputMode="decimal" value={draft ?? String(value)} onChange={event => edit(event.target.value)} onBlur={() => {
      if (!parsed.error) setDraft(null);
    }} />{parsed.error && <p className="perceptron-error" id={`${id}-error`}>{parsed.error} Last valid result: {fmt(value)}.</p>}</div>;
}
export function Frame({
  id,
  title,
  children,
  kind = 'Calculated model'
}) {
  return <section className="perceptron-panel" id={id} aria-label={title}><p className="perceptron-kind">{kind}</p><h3>{title}</h3>{children}</section>;
}
export function SignedBars({
  labels,
  values,
  extent = Math.max(1, ...values.map(Math.abs))
}) {
  return <div className="perceptron-bars" aria-label="Signed contributions, shared proportional scale">{labels.map((label, index) => <div className="perceptron-bar-row" key={label}><span>{label}</span><div className="perceptron-bar-track"><span className="perceptron-bar-zero" /><span className={`perceptron-bar-fill color-${index % 3}`} style={{
          left: values[index] < 0 ? `${50 + values[index] / extent * 50}%` : '50%',
          width: `${Math.abs(values[index]) / extent * 50}%`
        }} /></div><strong>{fmt(values[index])}</strong></div>)}<p className="perceptron-note">Zero is the centre line. The displayed scale runs from {fmt(-extent)} to {fmt(extent)} score units.</p></div>;
}
export function CurvePlot({
  title,
  curves,
  xDomain = [-6, 6],
  yDomain,
  selected,
  points = [],
  hollowPoints = [],
  xLabel = 'Preactivation z',
  yLabel = 'Value',
  xTicks,
  yTicks,
  log = false
}) {
  const id = useId(),
    [xmin, xmax] = xDomain,
    [ymin, ymax] = yDomain,
    width = 340,
    height = 265,
    left = 55,
    right = 20,
    top = 24,
    bottom = 54;
  const px = x => left + (x - xmin) / (xmax - xmin) * (width - left - right),
    transform = y => log ? Math.log10(y) : y,
    py = y => height - bottom - (transform(y) - transform(ymin)) / (transform(ymax) - transform(ymin)) * (height - top - bottom);
  const xt = xTicks ?? [xmin, 0, xmax].filter((v, i, a) => a.indexOf(v) === i),
    yt = yTicks ?? [ymin, 0, ymax].filter((v, i, a) => a.indexOf(v) === i && v >= ymin && v <= ymax);
  return <figure className="perceptron-plot"><figcaption>{title}</figcaption><svg className="perceptron-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title`}><title id={`${id}-title`}>{title}. {xLabel} from {xmin} to {xmax}; {yLabel} from {ymin} to {ymax}.</title><defs><clipPath id={`${id}-clip`}><rect x={left} y={top} width={width - left - right} height={height - top - bottom} /></clipPath></defs>{yt.map(y => <g key={y}><line className="perceptron-grid" x1={left} x2={width - right} y1={py(y)} y2={py(y)} /><text x={left - 9} y={py(y) + 4} textAnchor="end">{fmt(y, 3)}</text></g>)}{xt.map(x => <g key={x}><line className="perceptron-grid" x1={px(x)} x2={px(x)} y1={top} y2={height - bottom} /><text x={px(x)} y={height - bottom + 21} textAnchor="middle">{fmt(x)}</text></g>)}<g clipPath={`url(#${id}-clip)`}>{selected !== undefined && <line className="perceptron-guide" x1={px(selected)} x2={px(selected)} y1={top} y2={height - bottom} />} {curves.map((curve, i) => <path key={i} className={`perceptron-line color-${i % 6}`} style={curve.color ? {
          stroke: curve.color
        } : undefined} strokeDasharray={curve.dash} d={curve.data.map((p, j) => `${j ? 'L' : 'M'}${px(p[0])},${py(p[1])}`).join(' ')} />)}{hollowPoints.map((p, i) => <circle key={`open-${i}`} cx={px(p[0])} cy={py(p[1])} r="4" fill="#181714" stroke="#e8c36f" strokeWidth="1.5"/>)}{points.map((p, i) => <circle key={i} cx={px(p[0])} cy={py(p[1])} r="4" className="perceptron-dot" />)}</g><text x={(left + width - right) / 2} y={height - 8} textAnchor="middle">{xLabel}</text><text x={left} y="13">{yLabel}</text></svg></figure>;
}
