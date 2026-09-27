import { NeuralNumber } from './NeuralNumberControl.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralPlot as BaseNeuralPlot, NeuralSelect, NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { alibiCompetition, alibiSlopes, cachePositionRead, extensionFrequencies, frequencies, positionCacheDefault, positionMovementForward, relativeBucket, rotate, sinusoidal } from '../../data/positional-encoding-models.js';
import { add, dot, maxDifference } from '../../data/sequence-tensor-operations.js';
import './positional-encoding-labs.css';
import './neural-lesson-neutral.css';
export { XposPositionFigure } from './PositionApplicationsDiagrams.jsx';
import { PositionApplicationsDiagrams } from './PositionApplicationsDiagrams.jsx';
import { AlibiDecompositionFigure, RotaryConventionFigure } from './PositionMechanismFigures.jsx';
const assetBase = '/learn-code/positional-encodings-sinusoidal-learned-rope-alibi/';
const colors = ['#e6b854', '#dddddd', '#ba9fd1', '#999999'];
function NeuralPlot(props) {
  return <div className="position-plot-scroll" role="region" aria-label={props.title} tabIndex={0}><div className="position-plot-size"><BaseNeuralPlot {...props} /></div></div>;
}
const vector = values => `[${values.map(value => f(value, 5)).join(', ')}]`;
function EditVector({
  label,
  values,
  onChange,
  min = -4,
  max = 4,
  integer = false
}) {
  return <fieldset className="position-vector"><legend>{label}</legend><div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={`${label} ${i + 1}`} value={value} min={min} max={max} integer={integer} step={integer ? 1 : 'any'} range={false} onChange={next => onChange(values.map((item, j) => i === j ? next : item))} />)}</div></fieldset>;
}
function Plane({
  title,
  arrows,
  extent = 1.2
}) {
  const x = value => 130 + value / extent * 95;
  const y = value => 130 - value / extent * 95;
  return <figure className="position-plane" tabIndex={0}><figcaption>{title}</figcaption><svg viewBox="0 0 260 260" role="img" aria-label={`${title}. Equal horizontal/vertical scales from ${-extent} to ${extent}. Arrow endpoints and values are listed below.`}><circle cx="130" cy="130" r={95 / extent} fill="none" stroke="#555" strokeDasharray="3 3" /><line x1="25" y1="130" x2="235" y2="130" stroke="#666" /><line x1="130" y1="25" x2="130" y2="235" stroke="#666" />{arrows.map((arrow, i) => {
        const endX = x(arrow.value[0]),
          endY = y(arrow.value[1]);
        const angle = Math.atan2(endY - 130, endX - 130),
          size = 7;
        const tip = `${endX},${endY} ${endX - size * Math.cos(angle - .45)},${endY - size * Math.sin(angle - .45)} ${endX - size * Math.cos(angle + .45)},${endY - size * Math.sin(angle + .45)}`;
        return <g key={arrow.label}><line x1="130" y1="130" x2={endX} y2={endY} stroke={colors[i]} strokeWidth="2.5" strokeDasharray={arrow.dashed ? '5 3' : undefined} /><polygon points={tip} fill={colors[i]} /></g>;
      })}<text x="235" y="147" textAnchor="end">x</text><text x="142" y="28">y</text><text x="124" y="146">0</text></svg><ul>{arrows.map((arrow, i) => <li key={arrow.label}><span style={{
          color: colors[i]
        }}>{arrow.label}</span>: {vector(arrow.value)}</li>)}</ul></figure>;
}
function Matrix({
  title,
  rows,
  labels = rows.map((_, i) => i),
  signed = true
}) {
  const extent = Math.max(1e-10, ...rows.flat().map(Math.abs));
  return <NeuralTable caption={title} headers={['Row / column', ...rows[0].map((_, i) => i)]} rows={rows.map((row, i) => [labels[i], ...row.map((value, j) => <span key={j} className="position-matrix-value" style={{
    background: `rgba(${signed && value < 0 ? '185,158,210' : '230,184,84'},${.07 + .4 * Math.abs(value) / extent})`
  }}>{f(value, 4)}</span>)])} />;
}
export function PositionJourneyFigure() {
  const points = [[45, 150], [75, 65], [160, 45], [215, 145]];
  return <figure className="position-figure"><figcaption>Same locations; different journeys</figcaption><div className="position-three">{['Forward journey', 'Reverse journey', 'Unordered set'].map((title, mode) => <figure key={title} className="position-small-diagram" tabIndex={0}><figcaption>{title}</figcaption><svg className="position-diagram" viewBox="0 0 260 200" role="img" aria-label={title}><polyline points={mode === 2 ? '' : points.map(point => point.join(',')).join(' ')} stroke="#777" strokeWidth="2" fill="none" />{points.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="6" fill="#e6b854" /><text x={x} y={y + 24} textAnchor="middle">{String.fromCharCode(65 + i)}{mode !== 2 ? ` · slot ${mode === 0 ? i : 3 - i}` : ''}</text></g>)}</svg><p>{mode === 0 ? 'A → B → C → D' : mode === 1 ? 'D → C → B → A' : '{A, B, C, D}'}</p></figure>)}</div><p>Moving complete (point, slot) records changes storage order. Keeping slots fixed and moving only points changes the journey.</p></figure>;
}
export function AdditivePositionLab() {
  const original = [[1, 2], [-1, 1], [.5, -.5]];
  const [content, setContent] = useState(original),
    [table, setTable] = useState([[.2, 0], [0, .3], [-.1, .1]]);
  const [order, setOrder] = useState([0, 1, 2]),
    [slots, setSlots] = useState([0, 1, 2]);
  const baseline = content.map((row, i) => add(row, table[i]));
  const combined = order.map((record, i) => add(content[record], table[slots[i]]));
  const difference = Math.max(...combined.map((row, i) => maxDifference(row, baseline[order[i]])));
  return <NeuralLab id="position-additive" title="Move the record or change its slot?">
    <p>These are declared teaching vectors, not trained embeddings. Each position table row and each content vector is editable.</p>
    <div className="position-three">{order.map((record, i) => <section className="position-card" key={record}><h4>Record {String.fromCharCode(65 + record)} · slot {slots[i]}</h4><p>{vector(content[record])} + {vector(table[slots[i]])}</p><strong>{vector(combined[i])}</strong></section>)}</div>
    <p data-result="additive">Largest change after matching record identity: <strong>{f(difference)}</strong>. {difference < 1e-12 ? 'The combined records are unchanged up to storage order.' : 'The content-to-slot association changes the combined input.'}</p>
    <div className="neural-buttons"><button onClick={() => {
        setOrder(previous => [...previous.slice(1), previous[0]]);
        setSlots(previous => [...previous.slice(1), previous[0]]);
      }}>Move first labeled record to end</button><button onClick={() => setSlots(previous => [previous[1], previous[0], previous[2]])}>Swap slot assignments of first two records</button><button onClick={() => {
        setContent(original);
        setTable([[.2, 0], [0, .3], [-.1, .1]]);
        setOrder([0, 1, 2]);
        setSlots([0, 1, 2]);
      }}>Reset additive records</button></div>
    <details><summary>Edit all content and position vectors</summary><div className="position-two">{content.map((row, i) => <EditVector key={`content${i}`} label={`Content ${String.fromCharCode(65 + i)}`} values={row} onChange={next => setContent(previous => previous.map((value, j) => i === j ? next : value))} />)}{table.map((row, i) => <EditVector key={`table${i}`} label={`Position table row ${i}`} values={row} onChange={next => setTable(previous => previous.map((value, j) => i === j ? next : value))} />)}</div></details>
  </NeuralLab>;
}
export function SinusoidalPositionLab() {
  const [width, setWidth] = useState(8),
    [base, setBase] = useState(10000),
    [position, setPosition] = useState(3),
    [pair, setPair] = useState(0),
    [span, setSpan] = useState(32),
    [reference, setReference] = useState({
      position: 3,
      width: 8,
      base: 10000,
      pair: 0
    });
  const frequency = frequencies(width, base),
    selected = Math.min(pair, width / 2 - 1),
    angle = position * frequency[selected];
  const referenceFrequency = frequencies(reference.width, reference.base)[reference.pair];
  const referenceAngle = reference.position * referenceFrequency;
  const domain = [Math.min(position, reference.position) - span / 2, Math.max(position, reference.position) + span / 2];
  const samples = Math.ceil((domain[1] - domain[0]) * 4) + 1;
  const series = ['sin', 'cos'].map((name, i) => ({
    label: 'Current ' + name,
    color: colors[i],
    values: Array.from({
      length: samples
    }, (_, t) => {
      const x = domain[0] + t * (domain[1] - domain[0]) / (samples - 1);
      return [x, Math[name](x * frequency[selected])];
    })
  }));
  if (referenceFrequency !== frequency[selected]) ['sin', 'cos'].forEach((name, i) => series.push({
    label: 'Pinned ' + name,
    color: colors[i + 2],
    dashed: true,
    values: Array.from({
      length: samples
    }, (_, t) => {
      const x = domain[0] + t * (domain[1] - domain[0]) / (samples - 1);
      return [x, Math[name](x * referenceFrequency)];
    })
  }));
  return <NeuralLab id="position-sinusoidal" title="Several clocks describe one position">
    <div className="neural-controls"><NeuralSelect label="Encoding width" value={String(width)} onChange={value => {
        setWidth(Number(value));
        setPair(0);
      }} options={[4, 8, 16, 32].map(value => [value, String(value)])} /><NeuralNumber label="Sinusoidal base" value={base} min={10} max={1000000} range={false} onChange={setBase} /><NeuralNumber label="Position coordinate" value={position} min={0} max={255} onChange={setPosition} /><NeuralNumber label="Selected frequency pair" value={selected} min={0} max={width / 2 - 1} integer step={1} onChange={setPair} /></div>
    <NeuralNumber label="Extra trace span around current and pinned positions" value={span} min={4} max={128} integer step={1} onChange={setSpan} />
    <div className="position-two"><Plane title={`Pair ${selected}: phase ${f(angle, 4)} radians`} arrows={[{
        label: '[cos phase, sin phase] clock hand',
        value: [Math.cos(angle), Math.sin(angle)]
      }, {
        label: 'Pinned reference clock hand',
        value: [Math.cos(referenceAngle), Math.sin(referenceAngle)],
        dashed: true
      }]} /><NeuralPlot title="Current and pinned clocks across their positions" xLabel="position (both markers always in view)" yLabel="encoding coordinate" xDomain={domain} yDomain={[-1, 1]} series={series} points={['sin', 'cos'].flatMap(name => [{
        id: 'current-' + name,
        x: position,
        y: Math[name](angle),
        label: 'Current ' + name + ' at position ' + position
      }, {
        id: 'pinned-' + name,
        x: reference.position,
        y: Math[name](referenceAngle),
        label: 'Pinned ' + name + ' at position ' + reference.position
      }])} /></div>
    <NeuralTable caption="Current encoding versus the pinned state" headers={['Quantity', 'Pinned', 'Current', 'Difference']} rows={[['Position', reference.position, position, f(position - reference.position)], ['Pair', reference.pair, selected, 'selection'], ['Radians per position', f(referenceFrequency, 8), f(frequency[selected], 8), f(frequency[selected] - referenceFrequency, 8)], ['sin phase', f(Math.sin(referenceAngle), 8), f(Math.sin(angle), 8), f(Math.sin(angle) - Math.sin(referenceAngle), 8)], ['cos phase', f(Math.cos(referenceAngle), 8), f(Math.cos(angle), 8), f(Math.cos(angle) - Math.cos(referenceAngle), 8)]]} />
    <p>Pinned input: position {reference.position}, width {reference.width}, base {reference.base}, pair {reference.pair}. Current and reference outputs retain their own frequencies; dashed reference curves appear when those frequencies differ. Negative positions in the surrounding trace evaluate the same mathematical formula.</p>
    <button onClick={() => setReference({
      position,
      width,
      base,
      pair: selected
    })}>Pin current clock as reference</button>
    <p>The encoding stores <strong>[sin, cos]</strong>, while the ordinary clock plane draws [cos, sin]. Frequency {f(frequency[selected], 8)} radians/position; wavelength {f(2 * Math.PI / frequency[selected], 3)} positions; total turns {f(angle / (2 * Math.PI))}.</p>
    <Matrix title={`Selected width ${width}: first four position signatures`} rows={[0, 1, 2, 3].map(value => sinusoidal(value, width, base))} />
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect the full 16-position, 32-coordinate reference</h4><Matrix title="Fixed base 10000, width 32; computed from its own frequencies" rows={Array.from({
        length: 16
      }, (_, i) => sinusoidal(i, 32))} /></section>
    <button onClick={() => {
      setWidth(8);
      setBase(10000);
      setPosition(3);
      setPair(0);
      setSpan(32);
      setReference({
        position: 3,
        width: 8,
        base: 10000,
        pair: 0
      });
    }}>Reset clocks</button>
  </NeuralLab>;
}
export function RotaryPositionLab() {
  const [resetVersion, setResetVersion] = useState(0);
  const defaultQuery = [.8, -.5, .3, 1.2],
    defaultKey = [1, .25, -.5, .75];
  const [query, setQuery] = useState(defaultQuery),
    [key, setKey] = useState(defaultKey);
  const [queryId, setQueryId] = useState(3),
    [keyId, setKeyId] = useState(7),
    [shift, setShift] = useState(0),
    [base, setBase] = useState(10000);
  const rotatedQuery = rotate(query, queryId + shift, base),
    rotatedKey = rotate(key, keyId + shift, base);
  const relative = rotate(key, keyId - queryId, base),
    score = dot(rotatedQuery, rotatedKey);
  const [matrixContent, setMatrixContent] = useState(Array.from({
    length: 4
  }, () => [...defaultQuery]));
  const [matrixRow, setMatrixRow] = useState(1);
  const rotatedMatrix = matrixContent.map((row, i) => rotate(row, i, base));
  return <NeuralLab id="position-rotary" title="Turn both arrows, preserve their relative comparison">
    <EditVector label="Content query q" values={query} onChange={setQuery} min={-3} max={3} /><EditVector label="Content key k" values={key} onChange={setKey} min={-3} max={3} />
    <div className="neural-controls"><NeuralNumber label="Query position m" value={queryId} min={-128} max={256} integer step={1} onChange={setQueryId} /><NeuralNumber label="Key position n" value={keyId} min={-128} max={256} integer step={1} onChange={setKeyId} /><NeuralNumber label="Shared position shift" value={shift} min={-128} max={256} integer step={1} onChange={setShift} /><NeuralNumber label="Rotary base" value={base} min={10} max={1000000} range={false} onChange={setBase} /></div>
    <div className="position-two">{[0, 1].map(pair => <Plane key={pair} title={`Pair ${pair}; frequency ${f(frequencies(4, base)[pair])}`} extent={Math.max(1.2, Math.hypot(...query.slice(2 * pair, 2 * pair + 2)), Math.hypot(...key.slice(2 * pair, 2 * pair + 2))) * 1.1} arrows={[{
        label: 'Raw query',
        value: query.slice(pair * 2, pair * 2 + 2),
        dashed: true
      }, {
        label: 'Raw key',
        value: key.slice(pair * 2, pair * 2 + 2),
        dashed: true
      }, {
        label: 'Rotated query',
        value: rotatedQuery.slice(pair * 2, pair * 2 + 2)
      }, {
        label: 'Rotated key',
        value: rotatedKey.slice(pair * 2, pair * 2 + 2)
      }]} />)}</div>
    <NeuralTable caption="Pair contributions reconcile to the attention logit" headers={['Pair', 'Rotated dot contribution', 'Relative-rotation contribution']} rows={[0, 1].map(pair => [pair, f(dot(rotatedQuery.slice(pair * 2, pair * 2 + 2), rotatedKey.slice(pair * 2, pair * 2 + 2))), f(dot(query.slice(pair * 2, pair * 2 + 2), relative.slice(pair * 2, pair * 2 + 2)))])} />
    <p data-result="rope">Dot product {f(score)}; relative-only calculation {f(dot(query, relative))}; logit after √4 division <strong>{f(score / 2)}</strong>. Query length {f(Math.hypot(...query))} → {f(Math.hypot(...rotatedQuery))}. The shared shift changes the drawn arrows while preserving this dot product.</p>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Change content to break a diagonal-constant score matrix</h4><NeuralNumber label="Edit content at position" value={matrixRow} min={0} max={3} integer step={1} onChange={setMatrixRow} /><EditVector label={`Matrix content row ${matrixRow}`} values={matrixContent[matrixRow]} onChange={next => setMatrixContent(previous => previous.map((row, i) => i === matrixRow ? next : row))} /><Matrix title="Unscaled RoPE scores: same array supplies Q and K" rows={rotatedMatrix.map(row => rotatedMatrix.map(other => dot(row, other)))} /><button onClick={() => setMatrixContent(Array.from({
        length: 4
      }, () => [...defaultQuery]))}>Restore equal content</button><p>Equal content makes diagonals constant. A content edit can break that pattern while every pair still obeys the rotary identity.</p></section>
    <NeuralPlot title="A concrete counterexample to monotone distance decay" xLabel="integer position offset" yLabel="unscaled score cos(offset), q=k=[1,0]" xDomain={[0, 12]} yDomain={[-1, 1]} series={[{
      label: 'Continuous cosine (reference curve)',
      color: colors[0],
      values: Array.from({
        length: 97
      }, (_, i) => [i / 8, Math.cos(i / 8)])
    }]} points={Array.from({
      length: 13
    }, (_, offset) => ({
      id: offset,
      x: offset,
      y: Math.cos(offset),
      label: `Offset ${offset}: ${f(Math.cos(offset))}`
    }))} />
    <p>Adjacent pairs [0,1] / [2,3] convert to half-split pairs through coordinate permutation [0,2,1,3]. The projection basis and inverse mapping must move together. A partial rotary head keeps a separate unrotated content dot product.</p>
    <RotaryConventionFigure key={resetVersion} query={query} keyVector={key} queryId={queryId + shift} keyId={keyId + shift} base={base} />
    <button onClick={() => {
      setQuery(defaultQuery);
      setKey(defaultKey);
      setQueryId(3);
      setKeyId(7);
      setShift(0);
      setBase(10000);
      setResetVersion(value => value + 1);
    }}>Reset rotary geometry</button>
  </NeuralLab>;
}
export function AlibiPositionLab() {
  const [scores, setScores] = useState([2, 0, 0, 0]),
    [keys, setKeys] = useState([0, 1, 2, 3]),
    [query, setQuery] = useState(3),
    [slope, setSlope] = useState(.5),
    [constant, setConstant] = useState(0),
    [heads, setHeads] = useState(3),
    [oddsFirst, setOddsFirst] = useState(0),
    [oddsSecond, setOddsSecond] = useState(1);
  const result = alibiCompetition(scores.map(value => value + constant), keys, query, slope);
  const schedules = alibiSlopes(heads);
  return <NeuralLab id="position-alibi" title="Can distant evidence overcome a distance penalty?">
    <div className="neural-controls"><NeuralNumber label="Query position" value={query} min={-16} max={32} integer step={1} onChange={setQuery} /><NeuralNumber label="ALiBi slope" value={slope} min={0} max={1} onChange={setSlope} /><NeuralNumber label="Common content-logit offset" value={constant} min={-8} max={8} onChange={setConstant} /></div>
    <EditVector label="Key position" values={keys} onChange={setKeys} min={-16} max={32} integer /><EditVector label="Scaled content score" values={scores} onChange={setScores} min={-8} max={8} />
    <AlibiDecompositionFigure scores={scores.map(value => value + constant)} keys={keys} query={query} slope={slope} result={result} />
    <div className="position-key-rail">{keys.map((key, i) => <section key={i} className={result.legal[i] ? '' : 'is-masked'}><strong>Key {i}: slot {key}</strong><span>content {f(scores[i] + constant)}</span><span>bias {f(result.bias[i])}</span><strong>{result.legal[i] ? `${f(result.weights?.[i] ?? 0)} attention` : 'future: prohibited'}</strong></section>)}</div>
    <NeuralTable caption="From content through bias to normalized weight" headers={['Logical position', 'Distance', 'Content', 'Bias', 'Final logit', 'Weight']} rows={keys.map((key, i) => [key, query - key, f(scores[i] + constant), f(result.bias[i]), result.legal[i] ? f(result.final[i]) : 'masked', result.weights ? f(result.weights[i]) : 'no legal key'])} />
    <div className="neural-controls"><NeuralSelect label="Odds numerator record" value={oddsFirst} options={keys.map((key, i) => [i, 'Record ' + i + ' at ' + key + (result.legal[i] ? '' : ' (masked)')])} onChange={value => {
        const i = Number(value);
        setOddsFirst(i);
        if (i === oddsSecond) setOddsSecond(i === 0 ? 1 : 0);
      }} /><NeuralSelect label="Odds denominator record" value={oddsSecond} options={keys.map((key, i) => [i, 'Record ' + i + ' at ' + key + (result.legal[i] ? '' : ' (masked)')]).filter(([i]) => i !== oddsFirst)} onChange={value => setOddsSecond(Number(value))} /></div>
    {result.weights ? <p data-result="alibi">For record {oddsFirst} versus record {oddsSecond}, the weight ratio is {result.legal[oddsFirst] && result.legal[oddsSecond] ? f(result.weights[oddsFirst] / result.weights[oddsSecond]) : 'undefined because one record is masked'}. For two legal records, the content factor is exp({f(scores[oddsFirst] - scores[oddsSecond])}) and the distance factor is exp({f(slope * (keys[oddsFirst] - keys[oddsSecond]))}). A tied-content pair isolates the distance effect. Adding another competitor changes probabilities but not this pair’s odds.</p> : <p role="status">No legal key. Move a key to or before the query; a fully masked row has no attention distribution.</p>}
    <button onClick={() => setScores(old => old.map((value, i) => i === oddsSecond ? old[oddsFirst] : value))}>Equalize selected pair's content scores</button>
    <div className="neural-buttons"><button disabled={keys.length >= 8} onClick={() => {
        setKeys(previous => [...previous, query]);
        setScores(previous => [...previous, 0]);
      }}>Add a legal competitor</button><button disabled={keys.length <= 2} onClick={() => {
        const remaining = keys.length - 1;
        const first = Math.min(oddsFirst, remaining - 1);
        const second = oddsSecond < remaining && oddsSecond !== first ? oddsSecond : first === 0 ? 1 : 0;
        setOddsFirst(first);
        setOddsSecond(second);
        setKeys(previous => previous.slice(0, -1));
        setScores(previous => previous.slice(0, -1));
      }}>Remove last key</button><button onClick={() => setSlope(0)}>No distance penalty</button><button onClick={() => {
        setScores([2, 0, 0, 0]);
        setKeys([0, 1, 2, 3]);
        setQuery(3);
        setSlope(.5);
        setConstant(0);
        setOddsFirst(0);
        setOddsSecond(1);
      }}>Reset score competition</button></div>
    <NeuralSelect label="Original slope schedule head count" value={String(heads)} onChange={value => setHeads(Number(value))} options={[2, 3, 4, 8].map(value => [value, `${value} heads`])} />
    <NeuralPlot title="Equal-content odds factor for each original head ID" xLabel="extra distance in positions" yLabel="odds multiplier exp(−slope × distance)" xDomain={[0, 32]} yDomain={[0, 1]} series={schedules.map((value, i) => ({
      label: `Head ${i}: slope ${f(value, 7)}`,
      color: colors[i % 4],
      dashed: i >= 4,
      values: Array.from({
        length: 33
      }, (_, distance) => [distance, Math.exp(-value * distance)])
    }))} />
  </NeuralLab>;
}
export function RelativeBucketFigure() {
  const [offset, setOffset] = useState(16);
  return <figure className="position-figure"><figcaption>A bucket is a category; a relation vector changes with the query</figcaption><NeuralNumber label="Signed key-minus-query offset" value={offset} min={-256} max={256} integer step={1} onChange={setOffset} /><p>T5-style bidirectional bucket: <strong>{relativeBucket(offset)}</strong>. Nearby distances have exact bins; large offsets share increasingly broad bins, saturating at 15 or 31. The bucket’s learned value can have either sign.</p><NeuralTable caption="One Shaw relation vector, two queries" headers={['Query', 'Relation vector', 'Unscaled added score']} rows={[[vector([2, 0]), vector([.5, 1]), 1], [vector([0, 2]), vector([.5, 1]), 2]]} /></figure>;
}
export function PositionCacheLab() {
  const [settings, setSettings] = useState(positionCacheDefault),
    [selected, setSelected] = useState(0);
  const actual = cachePositionRead(settings),
    full = cachePositionRead(settings, true);
  const change = (name, value) => setSettings(previous => ({
    ...previous,
    [name]: value
  }));
  const shift = () => setSettings(previous => ({
    ...previous,
    ids: previous.ids.map(id => id + 100),
    queryId: previous.queryId + 100,
    rotaryId: previous.rotaryId + 100,
    maskId: previous.maskId + 100
  }));
  return <NeuralLab id="position-cache" title="Logical positions are not physical cache slots">
    <div className="position-key-rail">{settings.ids.map((id, i) => <section key={i}><span>Physical slot {i}</span><strong>Logical ID {id}</strong><span>{actual.legal[i] ? 'allowed key' : 'masked key'}</span><span>{settings.mode === 'rope' ? 'Rotated at base ' + settings.cacheBase : 'Raw key; distance enters the logit'}</span></section>)}</div>
    <div className="neural-controls"><NeuralSelect label="Cache encoding" value={settings.mode} onChange={value => change('mode', value)} options={['rope', 'alibi'].map(name => [name, name.toUpperCase()])} /><NeuralNumber label="Correct logical query ID" value={settings.queryId} min={0} max={512} integer step={1} onChange={value => change('queryId', value)} />{settings.mode === 'rope' && <NeuralNumber label="Actual query rotation ID" value={settings.rotaryId} min={0} max={512} integer step={1} onChange={value => change('rotaryId', value)} />}<NeuralNumber label="Actual query mask ID" value={settings.maskId} min={0} max={512} integer step={1} onChange={value => change('maskId', value)} />{settings.mode === 'rope' ? <><NeuralNumber label="Current query base" value={settings.base} min={10} max={1000000} range={false} onChange={value => change('base', value)} /><NeuralNumber label="Stored key rotation base" value={settings.cacheBase} min={10} max={1000000} range={false} onChange={value => change('cacheBase', value)} /></> : <NeuralNumber label="ALiBi distance slope" value={settings.slope} min={0} max={2} onChange={value => change('slope', value)} />}</div>
    <EditVector label="Stored key logical ID" values={settings.ids} onChange={value => change('ids', value)} min={0} max={512} integer />
    <p data-result="cache">Consistent full-reference last row: {full.output ? vector(full.output) : 'no legal key'}. Current cached result: <strong>{actual.output ? vector(actual.output) : 'no legal key'}</strong>. {full.output && actual.output ? `Maximum difference ${f(maxDifference(full.output, actual.output))}.` : 'A fully masked row is rejected.'}</p>
    <NeuralTable caption="Selected query reads the actual cache records" headers={['Slot', 'Logical ID', settings.mode === 'rope' ? 'Rotated Q·K / √4' : 'Content Q·K / √4', 'Distance bias', 'Final logit', 'Allowed', 'Attention', 'Value']} rows={settings.ids.map((id, i) => [i, id, f(actual.content[i]), f(settings.mode === 'alibi' ? -settings.slope * (settings.queryId - id) : 0), actual.legal[i] ? f(actual.scores[i]) : '−∞ (masked)', actual.legal[i] ? 'yes' : 'no', actual.weights ? f(actual.weights[i]) : 'undefined', vector(settings.values[i])])} />
    <details><summary>Edit query, a cached key and its value</summary><EditVector label="New query content" values={settings.query} onChange={value => change('query', value)} min={-3} max={3} /><NeuralNumber label="Selected physical cache slot" value={selected} min={0} max={settings.ids.length - 1} integer step={1} onChange={setSelected} /><EditVector label="Stored key content" values={settings.keys[selected]} onChange={value => change('keys', settings.keys.map((row, i) => i === selected ? value : row))} min={-3} max={3} /><EditVector label="Stored value content" values={settings.values[selected]} onChange={value => change('values', settings.values.map((row, i) => i === selected ? value : row))} min={-3} max={3} /></details>
    <div className="neural-buttons"><button disabled={settings.ids.length >= 8} onClick={() => {
        setSettings(previous => ({
          ...previous,
          ids: [...previous.ids, previous.queryId],
          keys: [...previous.keys, [1, 0, 0, 1]],
          values: [...previous.values, [0, 1]]
        }));
        setSelected(settings.ids.length);
      }}>Add cache record</button><button disabled={settings.ids.length <= 3} onClick={() => {
        setSettings(previous => ({
          ...previous,
          ids: previous.ids.filter((_, i) => i !== selected),
          keys: previous.keys.filter((_, i) => i !== selected),
          values: previous.values.filter((_, i) => i !== selected)
        }));
        setSelected(Math.max(0, selected - 1));
      }}>Remove selected cache record</button><button disabled={Math.max(...settings.ids, settings.queryId, settings.rotaryId, settings.maskId) > 412} onClick={shift}>Shift all IDs by 100</button><button onClick={() => setSettings(previous => ({
        ...previous,
        ids: [...previous.ids].reverse(),
        keys: [...previous.keys].reverse(),
        values: [...previous.values].reverse()
      }))}>Reverse whole cache records</button>{settings.mode === 'rope' && <><button onClick={() => change('rotaryId', 0)}>Isolate wrong query angle</button><button onClick={() => change('cacheBase', settings.base)}>Rephase stored keys consistently</button></>}<button onClick={() => {
        setSettings(positionCacheDefault());
        setSelected(0);
      }}>Reset cache timeline</button></div>
    <p>The reference uses the current raw content, logical query ID and base consistently. {settings.mode === 'rope' ? 'The cached path uses the separately editable rotation/mask IDs and stored base.' : 'The ALiBi path uses the logical query ID for distance and the separately editable mask ID for legality; it does not rotate keys.'} Added records start at the current logical query ID, so duplicates remain separate competitors until you edit them. This is one fixed-content attention layer; it does not reconstruct hidden states from an entire decoder.</p>
  </NeuralLab>;
}
export function PositionExtensionLab() {
  const [width, setWidth] = useState(64),
    [base, setBase] = useState(10000),
    [context, setContext] = useState(4096),
    [factor, setFactor] = useState(8),
    [pair, setPair] = useState(0),
    [offset, setOffset] = useState(8);
  const [q, setQ] = useState([1, 0]),
    [k, setK] = useState([0, 1]);
  const result = extensionFrequencies(width, base, context, factor),
    chosen = Math.min(pair, width / 2 - 1);
  const schemes = [['original', 'Original'], ['pi', 'PI'], ['baseScaled', 'Base-scaled'], ['yarn', 'Paper-ramp YaRN']];
  const wavelengthLogs = schemes.flatMap(([name]) => result[name].map(value => Math.log10(2 * Math.PI / value)));
  const rotatePair = angle => [k[0] * Math.cos(angle) - k[1] * Math.sin(angle), k[0] * Math.sin(angle) + k[1] * Math.cos(angle)];
  return <NeuralLab id="position-extension" title="Stretch geometry, then ask a separate quality question">
    <div className="neural-controls"><NeuralSelect label="Rotary width" value={String(width)} onChange={value => {
        setWidth(Number(value));
        setPair(0);
      }} options={[4, 8, 16, 32, 64, 128].map(value => [value, String(value)])} /><NeuralNumber label="Frequency base" value={base} min={10} max={1000000} range={false} onChange={setBase} /><NeuralNumber label="Original nominal context" value={context} min={16} max={8192} integer step={1} onChange={setContext} /><NeuralNumber label="Extension factor s" value={factor} min={1} max={32} onChange={setFactor} /><NeuralNumber label="Selected rotary pair" value={chosen} min={0} max={width / 2 - 1} integer step={1} onChange={setPair} /><NeuralNumber label="Relative position offset" value={offset} min={0} max={262144} range={false} onChange={setOffset} /></div>
    <NeuralPlot title="Formula-derived wavelengths across rotary pairs" xLabel="pair index" yLabel="log10(wavelength in positions)" xDomain={[0, width / 2 - 1]} yDomain={[Math.floor(Math.min(...wavelengthLogs)), Math.ceil(Math.max(...wavelengthLogs)) + .1]} series={schemes.map(([name, label], i) => ({
      label,
      color: colors[i],
      values: result[name].map((value, pairIndex) => [pairIndex, Math.log10(2 * Math.PI / value)])
    }))} />
    <EditVector label="Geometry query pair" values={q} onChange={setQ} min={-3} max={3} /><EditVector label="Geometry key pair" values={k} onChange={setK} min={-3} max={3} />
    <Plane title={`Selected pair ${chosen}, relative rotation of k`} extent={Math.max(1.2, Math.hypot(...k)) * 1.1} arrows={schemes.map(([name, label]) => ({
      label,
      value: rotatePair(offset * result[name][chosen])
    }))} />
    <NeuralTable caption="Selected pair: actual geometry, not model performance" headers={['Rule', 'Frequency', 'Wavelength', 'Phase at offset', 'qᵀRΔk']} rows={schemes.map(([name, label]) => [label, f(result[name][chosen], 8), f(2 * Math.PI / result[name][chosen], 3), f(offset * result[name][chosen]), f(dot(q, rotatePair(offset * result[name][chosen])))])} />
    <p data-result="extension">YaRN q/k multiplier c={f(result.qkScale)}; completed-logit multiplier c²={f(result.qkScale ** 2)}. PI maps the last nominal target position {f(context * factor - 1)} to {f((context * factor - 1) / factor)}. Factor 1 returns the original geometry and unit multiplier.</p>
    <p>The paper ramp uses turns during the original context, clipped between 1 and 32; checkpoint index ramps can differ. The plot’s vertical coordinate is explicitly log10 wavelength: 3 means 1,000 positions. Geometry alone supplies no retrieval accuracy or perplexity measurement.</p>
    <button onClick={() => {
      setWidth(64);
      setBase(10000);
      setContext(4096);
      setFactor(8);
      setPair(0);
      setOffset(8);
      setQ([1, 0]);
      setK([0, 1]);
    }}>Reset extension geometry</button>
  </NeuralLab>;
}
export function PositionFrequencyFigure() {
  const original = frequencies(64, 10000),
    changed = frequencies(64, 500000);
  return <figure className="position-figure"><figcaption>Changing the base stretches the slower clocks most</figcaption><NeuralPlot title="Two declared bases at rotary width 64" xLabel="pair index" yLabel="log10(wavelength in positions)" xDomain={[0, 31]} yDomain={[0, 7]} series={[[original, 'Base 10000'], [changed, 'Base 500000']].map(([values, label], i) => ({
      label,
      color: colors[i],
      values: values.map((value, index) => [index, Math.log10(2 * Math.PI / value)])
    }))} /><p>Both first wavelengths are {f(2 * Math.PI)} positions. Last wavelengths: {f(2 * Math.PI / original.at(-1), 3)} and {f(2 * Math.PI / changed.at(-1), 3)} positions. Vertical coordinate 3 means 10³=1,000 positions; this is a geometric calculation.</p></figure>;
}
export function PositionApplicationsFigure() {
  return <PositionApplicationsDiagrams />;
}
function usePositionModel(mode) {
  const container = useRef(null),
    [visible, setVisible] = useState(false),
    [attempt, setAttempt] = useState(0);
  const [resource, setResource] = useState({
    data: null,
    error: null,
    mode: null
  });
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '240px'
    });
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return undefined;
    const controller = new AbortController();
    setResource({
      data: null,
      error: null,
      mode
    });
    fetch(`${assetBase}movement-${mode}.json`, {
      signal: controller.signal
    }).then(response => {
      if (!response.ok) throw new Error('The selected saved model could not be loaded.');
      return response.json();
    }).then(data => {
      if (!controller.signal.aborted) setResource({
        data,
        error: null,
        mode
      });
    }).catch(error => {
      if (!controller.signal.aborted) setResource({
        data: null,
        error: 'The saved model could not be loaded. Retry this model or select another one.',
        mode
      });
    });
    return () => controller.abort();
  }, [mode, visible, attempt]);
  return {
    ...resource,
    container,
    retry: () => setAttempt(value => value + 1)
  };
}
function FittedPositionExplorer({
  data
}) {
  const [points, setPoints] = useState(data.points),
    [positions, setPositions] = useState(data.positions),
    [point, setPoint] = useState(22),
    [head, setHead] = useState(0),
    [key, setKey] = useState(0),
    [paddingMode, setPaddingMode] = useState('none');
  const result = useMemo(() => positionMovementForward(data, paddingMode === 'none' ? points : [...points, ...Array.from({
    length: 5
  }, () => [.75, .75])], paddingMode === 'none' ? positions : [...positions, 0, 0, 0, 0, 0], paddingMode === 'masked' ? [...Array(45).fill(false), ...Array(5).fill(true)] : []), [data, points, positions, paddingMode]);
  const winner = result.probabilities.indexOf(Math.max(...result.probabilities)) + 1;
  const trace = result.heads[head];
  const logitDeltas = result.logits.map((value, i) => value - data.baseline.logits[i]);
  const changedClass = logitDeltas.map(Math.abs).indexOf(Math.max(...logitDeltas.map(Math.abs)));
  const path = values => values.map(([x, y]) => `${35 + 250 * x},${285 - 250 * y}`).join(' ');
  return <><p>Source row 77, class 4 (anticlockwise arc). All five original selected models misclassify it. The current model is <strong>{data.mode === 'alibi' ? 'symmetric bidirectional ALiBi' : data.mode}</strong>; edits use its fixed actual weights.</p><div className="neural-controls"><NeuralNumber label="Selected movement record" value={point + 1} min={1} max={45} integer step={1} onChange={value => setPoint(value - 1)} /><NeuralNumber label="Movement x" value={points[point][0]} min={0} max={1} onChange={value => setPoints(previous => previous.map((row, i) => i === point ? [value, row[1]] : row))} /><NeuralNumber label="Movement y" value={points[point][1]} min={0} max={1} onChange={value => setPoints(previous => previous.map((row, i) => i === point ? [row[0], value] : row))} /><NeuralNumber label="Movement position ID" value={positions[point]} min={0} max={44} integer step={1} onChange={value => setPositions(previous => previous.map((id, i) => i === point ? value : id))} /><NeuralSelect label="Movement padding" value={paddingMode} onChange={setPaddingMode} options={['none', 'masked', 'unmasked'].map(name => [name, name])} /></div><div className="position-two"><figure className="position-path" tabIndex={0}><figcaption>Current path with selected-head attention over its records</figcaption><svg viewBox="0 0 320 320" role="img" aria-label="Equal x and y scale from zero to one, original and edited movement. Purple disk area is proportional to the selected query attention probability; blue cross is the inspected key."><rect x="35" y="35" width="250" height="250" fill="none" stroke="#555" />{[0, .5, 1].map(value => <g key={value}><text x={35 + value * 250} y="309" textAnchor="middle">{value}</text><text x="25" y={289 - value * 250} textAnchor="end">{value}</text></g>)}<polyline points={path(data.points)} fill="none" stroke="#888" strokeDasharray="4 3" strokeWidth="2" /><polyline points={path(points)} fill="none" stroke="#e6b854" strokeWidth="2" />{points.map(([x, y], i) => <circle key={i} cx={35 + 250 * x} cy={285 - 250 * y} r={30 * Math.sqrt(trace.attention[point][i])} fill="#ba9fd1" fillOpacity=".38" stroke="#ba9fd1" strokeOpacity=".6" />)}<path d={'M' + (29 + 250 * points[key][0]) + ' ' + (285 - 250 * points[key][1]) + ' h12 M' + (35 + 250 * points[key][0]) + ' ' + (279 - 250 * points[key][1]) + ' v12'} stroke="#92c9e6" strokeWidth="3" /><circle cx={35 + 250 * points[0][0]} cy={285 - 250 * points[0][1]} r="5" fill="#fff" /><rect x={31 + 250 * points[44][0]} y={281 - 250 * points[44][1]} width="8" height="8" fill="#e6b854" /><circle cx={35 + 250 * points[point][0]} cy={285 - 250 * points[point][1]} r="8" fill="none" stroke="#fff" strokeWidth="2" /></svg><p>White circle: first storage row; amber square: last. Selected query ring is record {point + 1} carrying logical ID {positions[point]}. Purple disk area equals attention probability times 900π square SVG units; a zero weight has no disk. Blue cross marks key record {key + 1}, weight {f(trace.attention[point][key], 6)}. The dashed path is the original; the amber path is current. {data.mode === 'none' ? 'This model ignores position IDs; changing an ID alone cannot change its output.' : 'The selected positional method uses these logical IDs to encode absolute positions or relative relationships.'}</p></figure><div><p data-result="movement">Maximum absolute logit change <strong>{f(Math.abs(logitDeltas[changedClass]))}</strong> at class {changedClass + 1} (signed change {f(logitDeltas[changedClass])}); current top class {winner}. Original label applies only to the unedited source.</p><NeuralTable caption="All 15 outputs; same argmax need not mean unchanged logits" headers={['Class', 'Original probability', 'Current probability', 'Logit change']} rows={result.probabilities.map((value, i) => [i + 1, f(data.baseline.probabilities[i]), f(value), f(result.logits[i] - data.baseline.logits[i])])} /></div></div><div className="neural-buttons"><button onClick={() => setPoints(previous => [...previous].reverse())}>Reverse points at fixed slots</button><button onClick={() => {
        setPoints(previous => [...previous].reverse());
        setPositions(previous => [...previous].reverse());
      }}>Reverse complete point/ID records</button><button onClick={() => {
        setPoints(data.points);
        setPositions(data.positions);
        setPaddingMode('none');
        setPoint(22);
        setKey(0);
        setHead(0);
      }}>Reset fitted movement</button></div><div className="neural-controls"><NeuralSelect label="Inspect fitted attention head" value={String(head)} onChange={value => setHead(Number(value))} options={[[0, 'Head 0'], [1, 'Head 1']]} /><NeuralNumber label="Inspect key record" value={key + 1} min={1} max={45} integer step={1} onChange={value => setKey(value - 1)} /></div><NeuralPlot title={`Attention from selected record ${point + 1}`} xLabel="key storage row (one-based)" yLabel="attention probability" xDomain={[1, 45]} yDomain={[0, Math.max(.05, ...trace.attention[point]) * 1.05]} series={[{
      label: 'Selected actual head',
      color: colors[0],
      values: trace.attention[point].slice(0, 45).map((value, i) => [i + 1, value])
    }]} points={[{
      id: 'selected-key',
      selected: true,
      color: '#92c9e6',
      x: key + 1,
      y: trace.attention[point][key],
      label: 'Selected key ' + (key + 1) + ': ' + f(trace.attention[point][key], 6)
    }]} /><p>{paddingMode === 'none' ? 'The 45 path disks cover every key.' : 'The path shows 45 real records. The five padding keys hold total probability ' + f(trace.attention[point].slice(45).reduce((a, b) => a + b, 0), 6) + '; their locations are omitted from this real-record path.'}</p><p>Raw query {vector(trace.rawQuery[point])}; raw key {vector(trace.rawKey[key])}.</p>{data.mode === 'rope' ? <Plane title="Actual first rotary pair in the fitted model" extent={Math.max(1, Math.hypot(...trace.query[point].slice(0, 2)), Math.hypot(...trace.key[key].slice(0, 2))) * 1.1} arrows={[{
      label: 'Rotated query',
      value: trace.query[point].slice(0, 2)
    }, {
      label: 'Rotated key',
      value: trace.key[key].slice(0, 2)
    }]} /> : data.mode === 'alibi' ? <p>Selected pair content logit {f(trace.content[point][key])} + symmetric distance bias {f(trace.bias[point][key])} = {f(trace.content[point][key] + trace.bias[point][key])}. This encoder adaptation has no causal direction cue.</p> : <p>Selected position contribution: {vector(result.positionVectors[point])}. Combined input: {vector(result.combined[point])}.</p>}<section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect every key identity and attention weight</h4><NeuralTable caption="Storage row is distinct from logical position" headers={['Storage row', 'Logical ID', 'Attention probability']} rows={points.map((_, i) => [i + 1, positions[i], f(trace.attention[point][i], 6)])} /></section></>;
}
export function PositionMovementLab() {
  const [mode, setMode] = useState('rope');
  const resource = usePositionModel(mode);
  return <div ref={resource.container}><NeuralLab id="position-movement" title="Reattach a real trajectory to its slots"><NeuralSelect label="Saved positional model" value={mode} onChange={setMode} options={['none', 'sinusoidal', 'learned', 'rope', 'alibi'].map(name => [name, name === 'alibi' ? 'Symmetric bidirectional ALiBi' : name])} />{resource.error ? <p role="alert">{resource.error} <button onClick={resource.retry}>Retry selected model</button></p> : resource.data && resource.mode === mode ? <FittedPositionExplorer key={mode} data={resource.data} /> : <p role="status">Loading selected measured model…</p>}</NeuralLab></div>;
}
