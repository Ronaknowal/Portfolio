import { cloneElement, isValidElement, useId, useState } from 'react';
import { sixPoints, sixLabels, sixNames, silhouetteSamples, silhouetteBarOrder, pairAgreement, pairBoard, informationMeasures, fixedMarginNull, exactLineCenters, irisFit, irisRepresentation } from '../../data/clustering-evaluation-models';
import { irisRows, irisSpecies, irisSpeciesNames, irisFeatures, irisFits } from '../../data/clustering-evaluation-data';
import './clustering-evaluation-labs.css';

export const number = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Number.isInteger(value)) return String(value);
  const fixed = value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
  return fixed === '-0' ? '0' : fixed;
};
const ids8 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
const direction = (after, before) => Math.abs(after - before) <= 1e-9 * Math.max(1, Math.abs(before)) ? 'same' : after < before ? 'lower' : 'higher';
const sign = value => value === null ? 'undefined' : Math.abs(value) <= 1e-12 ? 'zero' : value < 0 ? 'negative' : 'positive';

export function Investigation({ title, question, children, onReset }) {
  const id = useId();
  return <section className="ce-investigation" aria-labelledby={id}>
    <header><h3 id={id}>{title}</h3><button type="button" onClick={onReset}>Reset</button></header>
    {question && <p className="ce-question">{question}</p>}
    {children}
  </section>;
}
export function Prediction({ prompt, options, value, onChange, answer, revealed, stale = false, onCheck, checkLabel = 'Apply and compare', explanation }) {
  const id = useId();
  const answerLabel = options.find(([key]) => key === answer)?.[1] ?? String(answer);
  return <div className="ce-prediction">
    <label htmlFor={id}><strong>Predict first:</strong> {prompt}</label>
    <div className="ce-prediction-row">
      <select id={id} value={value} onChange={event => onChange(event.target.value)}>
        <option value="">Choose a prediction</option>
        {options.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select>
      <button type="button" disabled={!value || (revealed && !stale)} onClick={onCheck}>{checkLabel}</button>
    </div>
    {stale && <p className="ce-feedback is-stale" role="status">The inputs changed after your last comparison. Record a new prediction for the current draft.</p>}
    {revealed && !stale && value && <p className={`ce-feedback ${value === answer ? 'is-match' : 'is-miss'}`} role="status">
      {value === answer ? `Your prediction matches: ${answerLabel}.` : `Not this time. The calculation gives: ${answerLabel}.`} {explanation}
    </p>}
  </div>;
}
export function Field({ label, children, value, invalid = false }) {
  const id = useId();
  const control = isValidElement(children) && typeof children.type === 'string' && ['input', 'select'].includes(children.type) ? cloneElement(children, { id }) : children;
  return <label className={`ce-field ${invalid ? 'is-invalid' : ''}`} htmlFor={id}><span>{label}{value !== undefined && <output htmlFor={id}>{value}</output>}</span>{control}</label>;
}
const bits = value => `${number(value)} ${Math.abs(value - 1) < 5e-5 ? 'bit' : 'bits'}`;
export function Table({ caption, headings, rows, highlight = () => false }) {
  return <div className="ce-table-scroll" role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index} className={highlight(index) ? 'is-selected' : undefined}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
/** Sorted-within-group silhouette bars on a fixed [−1, 1] axis with zero and mean markers. */
export function SilhouetteBars({ labels, values, names, mean, title, selected = null, onSelect = null, compact = false }) {
  const groups = [...new Set(labels)];
  const order = silhouetteBarOrder(labels, values);
  const barHeight = compact ? 1.4 : Math.max(6, Math.min(16, 200 / Math.max(1, order.length)));
  const gap = compact ? 0 : 2;
  const groupGap = compact ? 8 : 10;
  const height = 30 + order.length * (barHeight + gap) + groups.length * groupGap + 14;
  const x = value => 40 + 240 * (value + 1) / 2;
  let y = 24;
  const rows = [];
  groups.forEach(group => {
    order.filter(index => labels[index] === group).forEach(index => { rows.push({ index, y, group }); y += barHeight + gap; });
    y += groupGap;
  });
  return <figure className="ce-plot">
    {title && <figcaption className="ce-legend">{title}</figcaption>}
    <svg viewBox={`0 0 320 ${height}`} role="img" aria-label={`Silhouette bars for ${order.length} observations in ${groups.length} groups; mean ${number(mean)}.`}>
      {[-1, -0.5, 0, 0.5, 1].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="16" y2={height - 12} className={value === 0 ? 'ce-zero' : 'ce-grid'} /><text x={x(value)} y="12" textAnchor="middle">{value}</text></g>)}
      {rows.map(({ index, y: top, group }) => <g key={index} onClick={onSelect ? () => onSelect(index) : undefined} style={onSelect ? { cursor: 'pointer' } : undefined}>
        <rect x={Math.min(x(0), x(values[index]))} y={top} width={Math.abs(x(values[index]) - x(0))} height={barHeight} className={`ce-bar-group-${groups.indexOf(group) % 8}`} opacity={selected === null || selected === index ? 1 : 0.55} />
        {!compact && names && <text x={values[index] >= 0 ? x(0) - 4 : x(0) + 4} y={top + barHeight * 0.8} textAnchor={values[index] >= 0 ? 'end' : 'start'}>{names[index]}{selected === index ? ' ◂' : ''}</text>}
      </g>)}
      {mean !== null && <line x1={x(mean)} x2={x(mean)} y1="16" y2={height - 12} className="ce-mean" />}
      {mean !== null && <text x={x(mean)} y={height - 2} textAnchor="middle">mean {number(mean, 3)}</text>}
    </svg>
  </figure>;
}

// ---------------------------------------------------------------- L1
export function CeSilhouetteLab() {
  const initial = { coordinates: sixPoints.map(point => point[0]), memberships: [...sixLabels] };
  const [draft, setDraft] = useState(initial);
  const [applied, setApplied] = useState(initial);
  const [selected, setSelected] = useState(2);
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const result = silhouetteSamples(applied.coordinates.map(value => [value]), applied.memberships);
  const key = JSON.stringify([draft, selected]);
  const stale = committed !== null && committed.key !== key;
  const invalid = draft.coordinates.some(value => !Number.isFinite(value) || Math.abs(value) > 20);
  const apply = () => {
    const next = silhouetteSamples(draft.coordinates.map(value => [value]), draft.memberships);
    const previousMean = result.mean;
    setCommitted({ key, answer: prediction.startsWith('mean:') ? `mean:${next.undefined || previousMean === null ? 'undefined' : direction(next.mean, previousMean)}` : `sign:${next.undefined ? 'undefined' : sign(next.values[selected])}`, previousMean, nextMean: next.undefined ? null : next.mean, nextSelected: next.undefined ? null : next.values[selected] });
    setApplied(draft);
  };
  const options = [['sign:negative', `s(${sixNames[selected]}) will be negative`], ['sign:zero', `s(${sixNames[selected]}) will be zero`], ['sign:positive', `s(${sixNames[selected]}) will be positive`], ['sign:undefined', 'silhouette will be undefined'], ['mean:lower', 'overall mean lower than the applied state'], ['mean:same', 'overall mean the same'], ['mean:higher', 'overall mean higher'], ['mean:undefined', 'overall mean undefined']];
  const detail = result.undefined ? null : result.details[selected];
  return <Investigation title="Change a membership, then explain every moving bar" question="Six locations on a line, two groups. Edit one coordinate or move one observation to another group, record what you expect for the selected point or the overall mean, then apply. Every bar is recomputed, not only the edited one." onReset={() => { setDraft(initial); setApplied(initial); setSelected(2); setPrediction(''); setCommitted(null); }}>
    <div className="ce-controls is-compact">
      {sixNames.map((name, index) => <Field key={name} label={`${name} location`} invalid={!Number.isFinite(draft.coordinates[index]) || Math.abs(draft.coordinates[index]) > 20}><input type="number" step="0.5" min="-20" max="20" value={draft.coordinates[index]} onChange={event => { const raw = event.target.value.trim(); if (raw === '' || raw === '-') return; const value = Number(raw); setDraft({ ...draft, coordinates: draft.coordinates.map((old, i) => i === index ? value : old) }); }} /></Field>)}
    </div>
    <div className="ce-controls is-compact">
      {sixNames.map((name, index) => <Field key={name} label={`${name} group`}><select value={draft.memberships[index]} onChange={event => setDraft({ ...draft, memberships: draft.memberships.map((old, i) => i === index ? event.target.value : old) })}>{['L', 'R', 'S'].map(group => <option key={group} value={group}>{group}</option>)}</select></Field>)}
    </div>
    <div className="ce-controls">
      <Field label="Selected observation"><select value={selected} onChange={event => setSelected(Number(event.target.value))}>{sixNames.map((name, index) => <option key={name} value={index}>{name} at {number(applied.coordinates[index])}, group {applied.memberships[index]}</option>)}</select></Field>
      <div className="ce-buttons"><button type="button" onClick={() => setDraft({ ...draft, coordinates: draft.coordinates.map(value => value * 2) })} disabled={draft.coordinates.some(value => Math.abs(value * 2) > 20)}>Double every draft coordinate</button><button type="button" onClick={() => setDraft({ ...draft, memberships: draft.memberships.map((group, i) => i === 2 ? (group === 'L' ? 'R' : 'L') : group) })}>Move C to the other group</button></div>
    </div>
    {invalid && <p className="ce-error">Coordinates must be finite numbers within ±20. The applied state is unchanged until the draft is valid.</p>}
    <Prediction prompt="What will the applied draft produce?" options={options} value={prediction} onChange={setPrediction} answer={committed?.answer ?? ''} revealed={committed !== null} stale={stale} onCheck={invalid ? () => {} : apply} explanation={committed && !stale ? (committed.nextMean === null ? 'Fewer than two groups or all singletons: no silhouette is defined.' : `Mean moved from ${number(committed.previousMean)} to ${number(committed.nextMean)}; s(${sixNames[selected]}) is now ${number(committed.nextSelected)}.`) : ''} />
    <p className="ce-readout" aria-live="polite">{result.undefined ? `Silhouette is undefined for the applied state: ${result.reason}. The points and groups are kept; add a second group to score them.` : `Applied state: ${sixNames[selected]} has a = ${detail.singleton ? 'no other group member' : number(detail.a)}, b = ${number(detail.b)} (nearest group ${detail.nearest.join(' or ')}${detail.tie ? ', tied' : ''}), s = ${number(detail.s)}${detail.singleton ? ' by the singleton convention' : ''}. Overall mean ${number(result.mean)} over ${applied.coordinates.length} observations; ${result.negative} negative.`}</p>
    <NumberLineFan coordinates={applied.coordinates} memberships={applied.memberships} selected={selected} detail={detail} onSelect={setSelected} />
    {!result.undefined && <SilhouetteBars labels={applied.memberships} values={result.values} names={sixNames} mean={result.mean} selected={selected} onSelect={setSelected} title="Sorted within each group; click a bar to select its observation" />}
    <Table caption="Applied state: within-group average a, nearest foreign average b and silhouette s" headings={['ID', 'location', 'group', 'a', 'b', 's']} rows={sixNames.map((name, index) => [name, number(applied.coordinates[index]), applied.memberships[index], result.undefined ? '—' : result.details[index].singleton ? 'singleton' : number(result.details[index].a), result.undefined ? '—' : number(result.details[index].b), result.undefined ? 'undefined' : number(result.details[index].s)])} highlight={index => index === selected} />
    <p className="ce-caption">Moving C from L to R exchanges its two averages: a becomes 6 and b becomes 1.5, so s(C) = −0.75, and every other bar changes because their neighbours changed. Doubling every coordinate doubles every a and b and leaves each s exactly where it was. A group with one member has no own-group average, so its s is 0 by convention rather than a claim that a = 0.</p>
  </Investigation>;
}
function NumberLineFan({ coordinates, memberships, selected, detail, onSelect }) {
  const low = Math.min(...coordinates) - 1, high = Math.max(...coordinates) + 1;
  const x = value => 20 + 280 * (value - low) / (high - low);
  const groups = [...new Set(memberships)];
  return <figure className="ce-plot">
    <svg viewBox="0 20 320 110" role="img" aria-label={`Number line with ${coordinates.length} observations; the selected observation's distances to its own group are solid and to its nearest foreign group dashed.`}>
      <line x1="20" x2="300" y1="90" y2="90" className="ce-grid" strokeWidth="1.5" />
      {detail && !detail.singleton && detail.ownMembers.map(j => <path key={`own${j}`} className="ce-fan-own" fill="none" d={`M${x(coordinates[selected])},90 Q${(x(coordinates[selected]) + x(coordinates[j])) / 2},${90 - 24 - Math.abs(x(coordinates[selected]) - x(coordinates[j])) / 6} ${x(coordinates[j])},90`} />)}
      {detail && detail.foreign.filter(entry => detail.nearest.includes(entry.group)).flatMap(entry => entry.members).map(j => <path key={`for${j}`} className="ce-fan-foreign" fill="none" d={`M${x(coordinates[selected])},90 Q${(x(coordinates[selected]) + x(coordinates[j])) / 2},${90 + 24 + Math.abs(x(coordinates[selected]) - x(coordinates[j])) / 6} ${x(coordinates[j])},90`} />)}
      {coordinates.map((value, index) => <g key={index} onClick={() => onSelect(index)} style={{ cursor: 'pointer' }}>
        <circle cx={x(value)} cy="90" r="6" className={`ce-point ${index === selected ? 'is-selected' : ''} ce-bar-group-${groups.indexOf(memberships[index]) % 8}`} />
        <text x={x(value)} y={index % 2 === 0 ? 122 : 110} textAnchor="middle">{sixNames[index]}·{memberships[index]}</text>
      </g>)}
    </svg>
    {detail && <p className="ce-legend">Solid arcs: own-group distances{detail.singleton ? ' (none)' : `, a = ${number(detail.a)}`}. Dashed arcs: nearest foreign group, b = {number(detail.b)}.</p>}
  </figure>;
}

// ---------------------------------------------------------------- L2
const labelOptions = [0, 1, 2, 3, 4, 5, 6, 7];
export function CePairLab() {
  const initial = { u: [0, 0, 0, 0, 1, 1, 1, 1], v: [0, 0, 0, 1, 0, 1, 1, 1] };
  const [draft, setDraft] = useState(initial);
  const [applied, setApplied] = useState(initial);
  const [pair, setPair] = useState([0, 4]);
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const result = pairAgreement(applied.u, applied.v);
  const board = pairBoard(applied.u, applied.v);
  const key = JSON.stringify([draft, pair]);
  const stale = committed !== null && committed.key !== key;
  const apply = () => {
    const next = pairAgreement(draft.u, draft.v);
    const together = draft.v[pair[0]] === draft.v[pair[1]];
    setCommitted({ key, answer: prediction.startsWith('ari:') ? `ari:${direction(next.ARI, result.ARI)}` : `pair:${together ? 'together' : 'apart'}`, previous: result.ARI, next: next.ARI });
    setApplied(draft);
  };
  const rename = () => setDraft({ ...draft, v: draft.v.map(label => (label + 3) % 8) });
  const selectedPair = board.find(entry => entry.i === pair[0] && entry.j === pair[1]);
  return <Investigation title="Contingency cells are pair-count shortcuts" question="Eight observations, a reference grouping U and a candidate V. Change any candidate label, rename all of them, or split a group; predict how ARI moves, or whether a chosen pair stays together, then apply and read the 28 pair decisions." onReset={() => { setDraft(initial); setApplied(initial); setPair([0, 4]); setPrediction(''); setCommitted(null); }}>
    <div className="ce-controls is-compact">
      {ids8.map((name, index) => <Field key={name} label={`${name} in V`}><select value={draft.v[index]} onChange={event => setDraft({ ...draft, v: draft.v.map((old, i) => i === index ? Number(event.target.value) : old) })}>{labelOptions.map(label => <option key={label} value={label}>{label}</option>)}</select></Field>)}
    </div>
    <details><summary>Change the reference U as well</summary><div className="ce-controls is-compact">
      {ids8.map((name, index) => <Field key={name} label={`${name} in U`}><select value={draft.u[index]} onChange={event => setDraft({ ...draft, u: draft.u.map((old, i) => i === index ? Number(event.target.value) : old) })}>{labelOptions.map(label => <option key={label} value={label}>{label}</option>)}</select></Field>)}
    </div></details>
    <div className="ce-buttons"><button type="button" onClick={rename}>Rename every candidate label (+3 mod 8)</button><button type="button" onClick={() => setDraft({ ...draft, v: [0, 0, 1, 1, 2, 2, 3, 3] })}>Refine into four pairs</button><button type="button" onClick={() => setDraft({ ...draft, v: [0, 0, 1, 1, 0, 0, 1, 1] })}>Cross the labels</button>
      <Field label="Selected pair"><select value={pair.join('-')} onChange={event => setPair(event.target.value.split('-').map(Number))}>{board.map(entry => <option key={`${entry.i}-${entry.j}`} value={`${entry.i}-${entry.j}`}>{ids8[entry.i]}–{ids8[entry.j]}</option>)}</select></Field></div>
    <Prediction prompt={`After applying the draft, how does ARI compare with the applied value ${number(result.ARI)}, or will ${ids8[pair[0]]} and ${ids8[pair[1]]} be together in V?`} options={[['ari:lower', 'ARI lower'], ['ari:same', 'ARI the same'], ['ari:higher', 'ARI higher'], ['pair:together', `${ids8[pair[0]]} and ${ids8[pair[1]]} together in V`], ['pair:apart', `${ids8[pair[0]]} and ${ids8[pair[1]]} apart in V`]]} value={prediction} onChange={setPrediction} answer={committed?.answer ?? ''} revealed={committed !== null} stale={stale} onCheck={apply} explanation={committed && !stale ? `ARI moved from ${number(committed.previous)} to ${number(committed.next)}.` : ''} />
    <div className="ce-strip"><span>ID</span>{ids8.map(name => <span key={name} className="ce-chip">{name}</span>)}</div>
    <div className="ce-strip"><span>U</span>{applied.u.map((label, index) => <span key={index} className={`ce-chip ${pair.includes(index) ? 'is-selected' : ''}`}>{label}</span>)}</div>
    <div className="ce-strip"><span>V</span>{applied.v.map((label, index) => <span key={index} className={`ce-chip ${pair.includes(index) ? 'is-selected' : ''}`}>{label}</span>)}</div>
    <p className="ce-readout" aria-live="polite">S = {result.S}, A = {result.A}, B = {result.B}, M = {result.M} → TP {result.TP}, FN {result.FN}, FP {result.FP}, TN {result.TN}. RI = {number(result.RI)}; expected S under fixed margins = {number(result.expectedS)}; ARI = {number(result.ARI)}{result.degenerate ? ' by the degenerate convention' : ''}. Selected pair {ids8[pair[0]]}–{ids8[pair[1]]}: {selectedPair.category} ({selectedPair.togetherU ? 'together' : 'apart'} in U, {selectedPair.togetherV ? 'together' : 'apart'} in V).</p>
    <div className="ce-matrix" style={{ gridTemplateColumns: `repeat(${result.table.columns.length + 2}, auto)` }} role="table" aria-label="Contingency table of U rows against V columns with member IDs and pair counts">
      <span className="is-head">U \ V</span>{result.table.columns.map(column => <span key={column} className="is-head">V{column}</span>)}<span className="is-head">size · C(n,2)</span>
      {result.table.rows.map((row, r) => <span key={`row${r}`} style={{ display: 'contents' }}><span className="is-head">U{row}</span>{result.table.cells[r].map((ids, c) => <span key={c} className={ids.includes(pair[0]) && ids.includes(pair[1]) ? 'is-highlight' : undefined}>{ids.length ? `${ids.map(i => ids8[i]).join('')} · ${ids.length * (ids.length - 1) / 2}` : '·'}</span>)}<span>{result.table.rowSizes[r]} · {result.table.rowSizes[r] * (result.table.rowSizes[r] - 1) / 2}</span></span>)}
      <span className="is-head">size · C(n,2)</span>{result.table.columnSizes.map((size, c) => <span key={c}>{size} · {size * (size - 1) / 2}</span>)}<span className="is-head">n = {result.table.n}</span>
    </div>
    <div className="ce-board" role="group" aria-label="The 28 unordered pairs and their agreement category">
      {ids8.map((rowName, i) => ids8.map((columnName, j) => { if (j <= i) return <span key={`${i}${j}`} className="ce-tile is-blank" />; const entry = board.find(pairEntry => pairEntry.i === i && pairEntry.j === j); return <button type="button" key={`${i}${j}`} className={`ce-tile is-${entry.category} ${pair[0] === i && pair[1] === j ? 'is-selected' : ''}`} onClick={() => setPair([i, j])} aria-label={`Pair ${rowName} ${columnName}: ${entry.category}`}>{rowName}{columnName}<br />{entry.category}</button>; }))}
    </div>
    <p className="ce-caption">TP: together in both. FN: together in U, split by V. FP: apart in U, merged by V. TN: apart in both. The cell counts C(n,2) add up to the same S that the board shows as TP tiles, which is why the contingency table is a shortcut and not a different measure. Renaming labels changes no tile; refining U's groups into pairs gives RI 5/7 and ARI 4/11; crossing gives RI 3/7 and ARI −1/6. Both partitions constant, or both all singletons, are reported as ARI 1 by convention.</p>
  </Investigation>;
}

// ---------------------------------------------------------------- L3
export function CeChanceLab() {
  const initial = { u: [0, 0, 0, 0, 1, 1, 1, 1], v: [0, 0, 1, 1, 0, 0, 1, 1] };
  const [draft, setDraft] = useState(initial);
  const [applied, setApplied] = useState(initial);
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const info = informationMeasures(applied.u, applied.v);
  const nullModel = fixedMarginNull(applied.u, applied.v);
  const key = JSON.stringify(draft);
  const stale = committed !== null && committed.key !== key;
  const apply = () => {
    const next = informationMeasures(draft.u, draft.v);
    const nextNull = fixedMarginNull(draft.u, draft.v);
    setCommitted({ key, answer: prediction.startsWith('ami:') ? `ami:${next.degenerateNull ? 'degenerate' : sign(next.ami)}` : `null:${next.degenerateNull ? 'degenerate' : nextNull.meanNmi > 1e-9 ? 'positive' : 'zero'}`, next, nextNull });
    setApplied(draft);
  };
  const overlaps = Object.keys(nullModel.overlapCounts).map(Number).sort((a, b) => a - b);
  const maxCount = Math.max(...Object.values(nullModel.overlapCounts));
  const nmiValues = [...new Map(nullModel.scored.map(entry => [entry.nmi.toFixed(9), entry.nmi])).values()].sort((a, b) => a - b);
  return <Investigation title="A finite chance experiment, not a generic random slider" question="Two binary labelings of eight observations. Predict the sign of AMI, or whether the average NMI over every fixed-margin rearrangement is zero or positive, then apply. The experiment enumerates every assignment of V's labels with U held fixed; nothing is sampled." onReset={() => { setDraft(initial); setApplied(initial); setPrediction(''); setCommitted(null); }}>
    <div className="ce-controls is-compact">{ids8.map((name, index) => <Field key={name} label={`${name} in U`}><select value={draft.u[index]} onChange={event => setDraft({ ...draft, u: draft.u.map((old, i) => i === index ? Number(event.target.value) : old) })}><option value={0}>0</option><option value={1}>1</option></select></Field>)}</div>
    <div className="ce-controls is-compact">{ids8.map((name, index) => <Field key={name} label={`${name} in V`}><select value={draft.v[index]} onChange={event => setDraft({ ...draft, v: draft.v.map((old, i) => i === index ? Number(event.target.value) : old) })}><option value={0}>0</option><option value={1}>1</option></select></Field>)}</div>
    <div className="ce-buttons"><button type="button" onClick={() => setDraft({ ...draft, v: [0, 0, 0, 0, 1, 1, 1, 1] })}>Match U exactly</button><button type="button" onClick={() => setDraft({ ...draft, v: [0, 0, 0, 1, 1, 1, 1, 1] })}>Unbalanced 3/5 candidate</button><button type="button" onClick={() => setDraft({ ...draft, v: [0, 0, 0, 0, 0, 0, 0, 0] })}>Constant candidate</button></div>
    <Prediction prompt="What will the applied draft produce?" options={[['ami:negative', 'Observed AMI negative'], ['ami:zero', 'Observed AMI zero'], ['ami:positive', 'Observed AMI positive'], ['ami:degenerate', 'AMI has a degenerate null (a constant labeling)'], ['null:zero', 'Mean NMI over the fixed-margin null is zero'], ['null:positive', 'Mean NMI over the fixed-margin null is positive'], ['null:degenerate', 'The null is degenerate']]} value={prediction} onChange={setPrediction} answer={committed?.answer ?? ''} revealed={committed !== null} stale={stale} onCheck={apply} explanation={committed && !stale ? (committed.next.degenerateNull ? 'A constant labeling has zero entropy; the conventions give NMI = AMI = 1 for two constants and 0 for one constant, with no informative null.' : `Observed MI ${number(committed.next.I)} bits against expected ${number(committed.next.EMI)} bits; AMI ${number(committed.next.ami)}; mean NMI over ${committed.nextNull.count} assignments ${number(committed.nextNull.meanNmi)}.`) : ''} />
    <p className="ce-readout" aria-live="polite">Applied: H(U) = {bits(info.HU)}, H(V) = {bits(info.HV)}, I = {bits(info.I)}, E[I] under fixed margins = {bits(info.EMI)}. NMI (arithmetic) = {number(info.nmi)}; AMI = {number(info.ami)}{info.degenerateNull ? ' by convention (degenerate null)' : ''}. {nullModel.count} distinct assignments of V's labels; mean NMI {number(nullModel.meanNmi)}, mean AMI {number(nullModel.meanAmi)}, mean ARI {number(nullModel.meanAri)}.</p>
    <div className="ce-matrix" style={{ gridTemplateColumns: 'repeat(4, auto)' }} role="table" aria-label="Two-by-two contingency table">
      <span className="is-head">U \ V</span>{info.table.columns.map(column => <span key={column} className="is-head">V{column}</span>)}<span className="is-head">row size</span>
      {info.table.rows.map((row, r) => <span key={r} style={{ display: 'contents' }}><span className="is-head">U{row}</span>{info.table.cells[r].map((ids, c) => <span key={c} className={r === 0 && c === 0 ? 'is-highlight' : undefined}>{ids.map(i => ids8[i]).join('') || '·'} · {ids.length}</span>)}<span>{info.table.rowSizes[r]}</span></span>)}
      <span className="is-head">column size</span>{info.table.columnSizes.map((size, c) => <span key={c}>{size}</span>)}<span className="is-head">n = 8</span>
    </div>
    <div className="ce-bars" role="img" aria-label={`Number of assignments by overlap r: ${overlaps.map(r => `${r}: ${nullModel.overlapCounts[r]}`).join(', ')}.`}>
      {overlaps.map(r => <div key={r} className="ce-bar-row"><span>r = {r}</span><span className="ce-bar-track"><span className="ce-bar-fill" style={{ width: `${100 * nullModel.overlapCounts[r] / maxCount}%` }} /></span><span className="ce-bar-value">{nullModel.overlapCounts[r]} assignment{nullModel.overlapCounts[r] === 1 ? '' : 's'}</span></div>)}
    </div>
    <Table caption="Attainable NMI values across the enumerated null, with their multiplicities" headings={['NMI', 'assignments', 'ARI at that NMI', 'AMI at that NMI']} rows={nmiValues.map(value => { const matching = nullModel.scored.filter(entry => Math.abs(entry.nmi - value) < 1e-9); return [number(value), matching.length, [...new Set(matching.map(entry => number(entry.ari)))].join(', '), [...new Set(matching.map(entry => number(entry.ami)))].join(', ')]; })} />
    <p className="ce-caption">r counts the observations in U's first group that receive V's smaller label. With balanced 4/4 margins there are 70 assignments, with overlaps 0 to 4 occurring 1, 16, 36, 16 and 1 times. Overlap 2 is an exactly independent table with NMI 0 and AMI about −0.13; overlaps 0 and 4 give NMI = AMI = 1. The unadjusted mean over the null is positive, about 0.115, while the adjusted means are zero up to rounding. Change the margins and the whole histogram recomputes; the default 70-case picture is not reused.</p>
  </Investigation>;
}

// ---------------------------------------------------------------- L4
const representationNames = { raw4: 'raw measurements (cm)', scaled4: 'standardized four features', pca2: 'two unwhitened principal components', white2: 'two whitened components' };
export function CeIrisLab() {
  const [mode, setMode] = useState('compare');
  const [representation, setRepresentation] = useState('raw4');
  const [k, setK] = useState(2);
  const [specimen, setSpecimen] = useState(0);
  const [draftWeights, setDraftWeights] = useState([1, 1, 1, 1]);
  const [appliedWeights, setAppliedWeights] = useState([1, 1, 1, 1]);
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const fit = mode === 'compare' ? irisFit(representation, k) : irisFit('scaled4', 3, appliedWeights);
  const key = JSON.stringify([draftWeights, specimen]);
  const stale = committed !== null && committed.key !== key;
  const invalid = draftWeights.some(weight => !Number.isFinite(weight) || weight < 0.25 || weight > 4);
  const apply = () => {
    const next = irisFit('scaled4', 3, draftWeights);
    setCommitted({ key, answer: prediction.startsWith('mean:') ? `mean:${direction(next.silhouette.mean, fit.silhouette.mean)}` : `sign:${sign(next.silhouette.values[specimen])}`, previous: fit.silhouette.mean, next: next.silhouette.mean, negative: next.silhouette.negative });
    setAppliedWeights(draftWeights);
  };
  const detail = fit.silhouette.details[specimen];
  const clusterNames = [...new Set(fit.labels)].sort((a, b) => a - b);
  const contingency = irisSpeciesNames.map((name, species) => [name, ...clusterNames.map(cluster => fit.labels.filter((label, i) => label === cluster && irisSpecies[i] === species).length), 50]);
  return <Investigation title="Real specimens: geometry and reference agreement" question="Compare the eight declared fits of the 150 irises, or freeze the standardized three-group partition and change the weight of one measurement: the silhouette bars move while ARI and AMI cannot, because the memberships have not changed." onReset={() => { setMode('compare'); setRepresentation('raw4'); setK(2); setSpecimen(0); setDraftWeights([1, 1, 1, 1]); setAppliedWeights([1, 1, 1, 1]); setPrediction(''); setCommitted(null); }}>
    <div className="ce-controls">
      <Field label="Mode"><select value={mode} onChange={event => setMode(event.target.value)}><option value="compare">Compare declared fits</option><option value="rescore">Rescore frozen standardized k = 3</option></select></Field>
      {mode === 'compare' && <Field label="Representation"><select value={representation} onChange={event => setRepresentation(event.target.value)}>{Object.entries(representationNames).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></Field>}
      {mode === 'compare' && <Field label="Number of groups k"><select value={k} onChange={event => setK(Number(event.target.value))}><option value={2}>2</option><option value={3}>3</option></select></Field>}
      <Field label="Selected specimen (row ID)"><select value={specimen} onChange={event => setSpecimen(Number(event.target.value))}>{irisRows.map((_, index) => <option key={index} value={index}>row {index} · {irisSpeciesNames[irisSpecies[index]]}</option>)}</select></Field>
    </div>
    {mode === 'rescore' && <>
      <div className="ce-controls is-compact">{irisFeatures.map((feature, index) => <Field key={feature} label={`${feature.replace('_cm', '').replace(/_/g, ' ')} weight`} invalid={!Number.isFinite(draftWeights[index]) || draftWeights[index] < 0.25 || draftWeights[index] > 4}><input type="number" min="0.25" max="4" step="0.25" value={draftWeights[index]} onChange={event => { const raw = event.target.value.trim(); if (raw === '') return; setDraftWeights(draftWeights.map((old, i) => i === index ? Number(raw) : old)); }} /></Field>)}</div>
      <div className="ce-buttons"><button type="button" onClick={() => setDraftWeights([4, 4, 4, 4])}>Common weight 4 on every feature</button><button type="button" onClick={() => setDraftWeights([1, 1, 1, 0.25])}>Petal width ¼</button><button type="button" onClick={() => setDraftWeights([1, 1, 1, 4])}>Petal width 4</button></div>
      {invalid && <p className="ce-error">Weights must be numbers from 0.25 to 4.</p>}
      <p className="ce-caption">Each weight multiplies that measurement’s squared difference inside every pairwise distance; the frozen labels are never refitted.</p>
      <Prediction prompt={`With the draft weights, will the mean silhouette be lower, the same or higher than the applied ${number(fit.silhouette.mean)}, or what sign will row ${specimen} have?`} options={[['mean:lower', 'Mean lower'], ['mean:same', 'Mean the same'], ['mean:higher', 'Mean higher'], ['sign:negative', `Row ${specimen} negative`], ['sign:positive', `Row ${specimen} positive`]]} value={prediction} onChange={setPrediction} answer={committed?.answer ?? ''} revealed={committed !== null} stale={stale} onCheck={invalid ? () => {} : apply} explanation={committed && !stale ? `Mean moved from ${number(committed.previous)} to ${number(committed.next)}; ${committed.negative} specimens are negative. Labels, ARI and AMI are unchanged because nothing was refitted.` : ''} />
    </>}
    <p className="ce-readout" aria-live="polite">{mode === 'compare' ? `${representationNames[representation]}, k = ${k}: mean silhouette ${number(fit.silhouette.mean)}, ${fit.silhouette.negative} negative, sizes ${fit.sizes.join('/')}; species ARI ${number(fit.ari)}, AMI ${number(fit.ami)}.` : `Frozen standardized k = 3 labels rescored with weights ${appliedWeights.join(', ')}: mean silhouette ${number(fit.silhouette.mean)}, ${fit.silhouette.negative} negative. Species ARI ${number(fit.ari)} and AMI ${number(fit.ami)}: same memberships.`} Row {specimen} ({irisSpeciesNames[irisSpecies[specimen]]}, cluster {fit.labels[specimen]}): a = {number(detail.a)}, b = {number(detail.b)} to cluster {detail.nearest.join('/')}, s = {number(detail.s)}.</p>
    <SilhouetteBars labels={fit.labels.map(String)} values={fit.silhouette.values} mean={fit.silhouette.mean} selected={specimen} compact title="All 150 silhouettes, sorted within each cluster; the dashed line is the observation-weighted mean" />
    <IrisView labels={fit.labels} selected={specimen} onSelect={setSpecimen} />
    <Table caption="Species rows against cluster columns (species were never a fitting input)" headings={['species', ...clusterNames.map(cluster => `cluster ${cluster}`), 'total']} rows={contingency} />
    <Table caption={`Row ${specimen}: measurements in centimetres`} headings={irisFeatures.map(feature => feature.replace('_cm', '').replace(/_/g, ' '))} rows={[irisRows[specimen].map(value => number(value))]} />
    <p className="ce-caption">In raw centimetres, two groups have the higher silhouette (0.681 against 0.553) while three groups agree better with species (ARI 0.730 against 0.540). Standardized and two-component k = 3 give the same partition, ARI 1 between them, yet different silhouettes, because the distances differ while the labels do not. Petal-width weight ¼ lowers the frozen mean to about 0.454 and weight 4 raises it to about 0.471; a common weight of 4 leaves every bar exactly in place. The scatter is a two-component view; the silhouette uses its declared feature space.</p>
  </Investigation>;
}
function IrisView({ labels, selected, onSelect }) {
  const coordinates = irisRepresentation('pca2');
  const x = value => 160 + 40 * value, y = value => 110 - 40 * value;
  const marks = ['circle', 'square', 'triangle'];
  return <figure className="ce-plot">
    <svg viewBox="0 0 320 226" role="img" aria-label="Two-component view of the 150 irises; color is the active cluster label, shape is the species. The silhouette is computed in the declared feature space, not in this view." style={{ maxWidth: 480, margin: 'auto' }}>
      {[-2, 0, 2].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="10" y2="210" className="ce-grid" /><line x1="30" x2="290" y1={y(value)} y2={y(value)} className="ce-grid" /><text x={x(value)} y="222" textAnchor="middle">{value}</text><text x="24" y={y(value) + 3} textAnchor="end">{value}</text></g>)}
      {coordinates.map((point, index) => { const cx = x(point[0]), cy = y(point[1]); const cls = `ce-bar-group-${labels[index] % 8}`; const kind = marks[irisSpecies[index]]; const common = { onClick: () => onSelect(index), style: { cursor: 'pointer' }, opacity: index === selected ? 1 : 0.75, stroke: index === selected ? '#f2e7ca' : 'none', strokeWidth: 1.5 }; if (kind === 'square') return <rect key={index} x={cx - 3} y={cy - 3} width="6" height="6" className={cls} {...common} />; if (kind === 'triangle') return <path key={index} d={`M${cx},${cy - 4}l4,7h-8z`} className={cls} {...common} />; return <circle key={index} cx={cx} cy={cy} r="3" className={cls} {...common} />; })}
    </svg>
    <p className="ce-legend">View only: PC1 horizontal, PC2 vertical, from standardized data. Circle setosa, square versicolor, triangle virginica; color is the active cluster label.</p>
  </figure>;
}

// ---------------------------------------------------------------- L5
const probeNames = ['A', 'B', 'C', 'D', 'E', 'F'];
export function CeResampleLab() {
  const initial = { locations: [0, 1, 4, 5, 8, 9], weightsA: [3, 3, 1, 1, 1, 1], weightsB: [1, 1, 1, 1, 3, 3], k: 2 };
  const [draft, setDraft] = useState(initial);
  const [applied, setApplied] = useState(initial);
  const [pair, setPair] = useState([2, 3]);
  const [prediction, setPrediction] = useState('');
  const [committed, setCommitted] = useState(null);
  const invalidLocations = draft.locations.some((value, index) => !Number.isFinite(value) || Math.abs(value) > 40 || (index > 0 && value <= draft.locations[index - 1]));
  const fitA = exactLineCenters(applied.locations, applied.weightsA, applied.k), fitB = exactLineCenters(applied.locations, applied.weightsB, applied.k);
  const agreement = pairAgreement(fitA.probeLabels, fitB.probeLabels);
  const key = JSON.stringify([draft, pair]);
  const stale = committed !== null && committed.key !== key;
  const apply = () => {
    const nextA = exactLineCenters(draft.locations, draft.weightsA, draft.k), nextB = exactLineCenters(draft.locations, draft.weightsB, draft.k);
    const nextAgreement = pairAgreement(nextA.probeLabels, nextB.probeLabels);
    const together = nextB.probeLabels[pair[0]] === nextB.probeLabels[pair[1]];
    setCommitted({ key, answer: prediction.startsWith('ari:') ? `ari:${direction(nextAgreement.ARI, agreement.ARI)}` : `pair:${together ? 'together' : 'apart'}`, previous: agreement.ARI, next: nextAgreement.ARI, centersA: nextA.centers, centersB: nextB.centers });
    setApplied(draft);
  };
  const low = Math.min(...applied.locations) - 1, high = Math.max(...applied.locations) + 1;
  const position = value => `${100 * (value - low) / (high - low)}%`;
  const ProbeStrip = ({ labels, name }) => <div className="ce-strip"><span>{name}</span>{labels.map((label, index) => <span key={index} className={`ce-chip ${pair.includes(index) ? 'is-selected' : ''}`}>{probeNames[index]}·{label}</span>)}</div>;
  const weightEditor = (which, weights) => <div className="ce-controls is-compact">{probeNames.map((name, index) => <Field key={name} label={`${name} × in fit ${which}`}><select value={weights[index]} onChange={event => setDraft({ ...draft, [which === 'A' ? 'weightsA' : 'weightsB']: weights.map((old, i) => i === index ? Number(event.target.value) : old) })}>{[1, 2, 3, 4, 5].map(count => <option key={count} value={count}>{count}</option>)}</select></Field>)}</div>;
  return <Investigation title="Resampling moves the representatives; the probes stay fixed" question="Six fixed probe locations. Two fits use the same locations with different multiplicities, solved exactly by the best contiguous split. Predict whether a chosen pair of probes stays together under fit B, or how their agreement changes, then apply." onReset={() => { setDraft(initial); setApplied(initial); setPair([2, 3]); setPrediction(''); setCommitted(null); }}>
    <p className="ce-legend">Multiplicities for fit A</p>{weightEditor('A', draft.weightsA)}
    <p className="ce-legend">Multiplicities for fit B</p>{weightEditor('B', draft.weightsB)}
    <div className="ce-controls">
      <Field label="Groups k in both fits"><select value={draft.k} onChange={event => setDraft({ ...draft, k: Number(event.target.value) })}><option value={1}>1</option><option value={2}>2</option><option value={3}>3</option></select></Field>
      <Field label="Selected probe pair"><select value={pair.join('-')} onChange={event => setPair(event.target.value.split('-').map(Number))}>{probeNames.flatMap((a, i) => probeNames.slice(i + 1).map((b, offset) => <option key={`${i}-${i + 1 + offset}`} value={`${i}-${i + 1 + offset}`}>{a}–{b}</option>))}</select></Field>
    </div>
    <details><summary>Change the six probe locations (strictly increasing, within ±40)</summary><div className="ce-controls is-compact">{probeNames.map((name, index) => <Field key={name} label={`${name} location`} invalid={index > 0 && draft.locations[index] <= draft.locations[index - 1]}><input type="number" step="1" min="-40" max="40" value={draft.locations[index]} onChange={event => { const raw = event.target.value.trim(); if (raw === '' || raw === '-') return; setDraft({ ...draft, locations: draft.locations.map((old, i) => i === index ? Number(raw) : old) }); }} /></Field>)}</div>
      <div className="ce-buttons"><button type="button" onClick={() => setDraft({ ...draft, locations: [0, 1, 2, 8, 9, 10], weightsA: [1, 1, 1, 1, 1, 1], weightsB: [1, 1, 1, 1, 1, 1] })}>Two tight runs, uniform weights</button><button type="button" disabled={draft.locations.some(value => Math.abs(3 * value) > 40)} onClick={() => setDraft({ ...draft, locations: draft.locations.map(value => 3 * value) })}>Triple every location</button></div></details>
    {invalidLocations && <p className="ce-error">Locations must be finite, strictly increasing and within ±40. The applied fits are unchanged until the draft is valid.</p>}
    <Prediction prompt={`After applying, will probes ${probeNames[pair[0]]} and ${probeNames[pair[1]]} share a group under fit B, or how will the A/B agreement ARI ${number(agreement.ARI)} change?`} options={[['pair:together', `${probeNames[pair[0]]} and ${probeNames[pair[1]]} together under fit B`], ['pair:apart', `${probeNames[pair[0]]} and ${probeNames[pair[1]]} apart under fit B`], ['ari:lower', 'ARI lower'], ['ari:same', 'ARI the same'], ['ari:higher', 'ARI higher']]} value={prediction} onChange={setPrediction} answer={committed?.answer ?? ''} revealed={committed !== null} stale={stale} onCheck={invalidLocations ? () => {} : apply} explanation={committed && !stale ? `Fit A centers ${committed.centersA.map(value => number(value)).join(', ')}; fit B centers ${committed.centersB.map(value => number(value)).join(', ')}; ARI moved from ${number(committed.previous)} to ${number(committed.next)}.` : ''} />
    <p className="ce-readout" aria-live="polite">Fit A: centers {fitA.centers.map(value => number(value)).join(', ')}, cost {number(fitA.cost)}{fitA.ties > 1 ? ` (${fitA.ties} tied optima; the smaller ordered centers were kept)` : ''}. Fit B: centers {fitB.centers.map(value => number(value)).join(', ')}, cost {number(fitB.cost)}{fitB.ties > 1 ? ` (${fitB.ties} tied optima)` : ''}. Probe agreement ARI = {number(agreement.ARI)}{agreement.degenerate ? ' by convention' : ''}.</p>
    <div className="ce-line" role="img" aria-label={`Shared number line with fit A centers at ${fitA.centers.map(value => number(value)).join(', ')} and fit B centers at ${fitB.centers.map(value => number(value)).join(', ')}.`}>
      <span className="ce-axis" />
      {applied.locations.map((value, index) => <span key={index}><span className="ce-stack" style={{ left: position(value), marginLeft: -7 }}>{Array.from({ length: applied.weightsA[index] }, (_, i) => <span key={i} />)}</span><span className="ce-stack is-b" style={{ left: position(value), marginLeft: 7 }}>{Array.from({ length: applied.weightsB[index] }, (_, i) => <span key={i} />)}</span><span className={`ce-dot ${pair.includes(index) ? 'is-selected' : ''}`} style={{ left: position(value) }} /><span className="ce-tick" style={{ left: position(value) }}>{probeNames[index]}<br />{number(value)}</span></span>)}
      {fitA.centers.map((center, index) => <span key={`a${index}`} className="ce-center" style={{ left: position(center), background: '#91aecf' }} title={`fit A center ${number(center)}`} />)}
      {fitB.centers.map((center, index) => <span key={`b${index}`} className="ce-center" style={{ left: position(center), background: '#da9c86' }} title={`fit B center ${number(center)}`} />)}
      {fitA.boundaries.map((boundary, index) => <span key={`ba${index}`} className="ce-boundary" style={{ left: position(boundary), background: '#91aecf' }} />)}
      {fitB.boundaries.map((boundary, index) => <span key={`bb${index}`} className="ce-boundary" style={{ left: position(boundary), background: '#da9c86' }} />)}
    </div>
    <p className="ce-legend">Blue chips above each probe are fit A's multiplicity, red chips fit B's; thick ticks are the fitted centers and thin ticks the midpoint decision boundaries in the same colors.</p>
    <ProbeStrip labels={fitA.probeLabels} name="fit A" /><ProbeStrip labels={fitB.probeLabels} name="fit B" />
    <Table caption="Every contiguous split considered by fit B, best first" headings={['blocks', 'centers', 'weighted cost']} rows={fitB.alternatives.map(entry => [entry.blocks.map(block => block.map(i => probeNames[i]).join('')).join(' | '), entry.centers.map(value => number(value)).join(', '), number(entry.cost)])} highlight={index => index === 0} />
    <p className="ce-caption">Emphasizing A and B three times pulls the left center to 0.5 and moves the boundary so that C and D join the right group; emphasizing E and F does the opposite, and the two fits then agree on only some pairs, with ARI −1/14. With k = 1 any multiplicities give one group each and ARI 1: a trivially stable answer. Tripling every location triples the centers and the cost by nine but changes no membership. The probes never move; only the representatives do.</p>
  </Investigation>;
}
