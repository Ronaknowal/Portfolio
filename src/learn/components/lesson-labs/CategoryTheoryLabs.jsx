import { useId, useState } from 'react';
import { compositionSnapshot, schemaSnapshot, naturalitySnapshot, productSnapshot, stochasticCopySnapshot, tangentSnapshot, tangentFunctionChoices, adjunctionSnapshot } from '../../data/category-theory-models.js';
import './category-theory-labs.css';
const format = value => Number(value.toPrecision(7)).toString();
const functionOptions = [['cycle', 'Cycle: 0→1, 1→2, 2→0'], ['reverse', 'Flip: 0→2, 1→1, 2→0'], ['collapse', 'Merge: 0→0, 1→0, 2→2'], ['identity', 'Identity: each stays itself']];
function Choice({
  label,
  value,
  options,
  onChange,
  numeric = false
}) {
  return <label className="category-control">{label}<select aria-label={label} value={value} onChange={event => onChange(numeric ? Number(event.target.value) : event.target.value)}>{options.map(([key, title]) => <option key={key} value={key}>{title}</option>)}</select></label>;
}
function Range({
  label,
  value,
  min,
  max,
  step = 1,
  onChange
}) {
  return <label className="category-control">{label}: <strong>{format(value)}</strong><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Tokens({
  values,
  prefix = ''
}) {
  return <span className="category-tokens" aria-label={values.length ? values.join(', ') : 'empty list'}>{values.length ? values.map((value, index) => <span key={index} className={'category-token category-tone-' + (Number.isInteger(value) ? value % 3 : index % 3)}>{prefix}{value}</span>) : <span className="category-empty">[ ]</span>}</span>;
}
function Reset({
  onClick
}) {
  return <button className="category-reset" onClick={onClick}>Reset investigation</button>;
}
function Result({
  children
}) {
  return <p className="category-result" aria-live="polite">{children}</p>;
}
export function CompositionLab() {
  const marker = useId().replaceAll(':', '');
  const [input, setInput] = useState(0);
  const [names, setNames] = useState(['cycle', 'collapse', 'reverse']);
  const [group, setGroup] = useState('first');
  const result = compositionSnapshot(input, ...names);
  const columns = [25, 102, 179, 256];
  const row = value => 68 + value * 58;
  const update = (index, value) => setNames(old => old.map((name, position) => position === index ? value : name));
  return <section className="category-lab" aria-label="Typed composition investigation">
    <h3>Follow one element through typed arrows</h3>
    <p>The pale lines show the entire finite function. The solid gold path follows your selected input. A, B, C and D are separately labelled three-element sets.</p>
    <div className="category-controls"><Choice label="Starting element in A" value={input} onChange={setInput} numeric options={[0, 1, 2].map(value => [value, String(value)])} />{names.map((name, index) => <Choice key={index} label={['f: A → B', 'g: B → C', 'h: C → D'][index]} value={name} onChange={value => update(index, value)} options={functionOptions} />)}<Choice label="Group the calculation" value={group} onChange={setGroup} options={[['first', 'First g∘f, then h'], ['last', 'First h∘g, then f enters it']]} /></div>
    <svg className="category-path" viewBox="0 0 282 242" role="img" aria-label={'Selected path: ' + result.path.join(' to ')}>
      <defs><marker id={marker} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="currentColor" /></marker></defs>
      {result.maps.flatMap((mapping, column) => mapping.map((target, source) => <line key={column + '-' + source} x1={columns[column] + 14} y1={row(source)} x2={columns[column + 1] - 16} y2={row(target)} className={result.path[column] === source ? 'category-active-edge' : 'category-edge'} markerEnd={'url(#' + marker + ')'} />))}
      {columns.map((x, column) => <g key={column}><text x={x} y={26} textAnchor="middle">{['A', 'B', 'C', 'D'][column]}</text>{[0, 1, 2].map(value => <g key={value}><circle cx={x} cy={row(value)} r={14} className={result.path[column] === value ? 'category-active-node' : 'category-node'} /><text x={x} y={row(value) + 6} textAnchor="middle">{value}</text></g>)}</g>)}
      <path d={group === 'first' ? 'M25 218v9h154v-9' : 'M102 218v9h154v-9'} className="category-bracket" />
    </svg>
    <Result>{group === 'first' ? 'h ∘ (g ∘ f)' : '(h ∘ g) ∘ f'} sends {input} to {result.path[3]}. Both parenthesizations have the same full lookup table [{result.left.join(', ')}]. The bracket changes; arrow order stays fixed.</Result>
    <p>As a separate comparison on the common index set, g∘f has table [{result.intermediate.join(', ')}], while f∘g has [{result.swapped.join(', ')}]. Such a swap uses relabelled common domains; the original typed path does not itself permit f after g.</p>
    <Reset onClick={() => {
      setInput(0);
      setNames(['cycle', 'collapse', 'reverse']);
      setGroup('first');
    }} />
  </section>;
}
export function SchemaFunctorLab() {
  const [sensor, setSensor] = useState(1);
  const [direct, setDirect] = useState([0, 1, 1]);
  const result = schemaSnapshot(sensor, direct);
  return <section className="category-lab" aria-label="Schema functor investigation">
    <h3>Does the data respect the schema triangle?</h3>
    <p>The schema requires directSite = siteOf ∘ deviceOf for every sensor. Start with s1: its stored shortcut disagrees with its device's site.</p>
    <div className="category-schema-map" aria-label="Sensor to Device to Site, with a direct Sensor to Site route"><div>Sensor <strong>deviceOf →</strong> Device <strong>siteOf →</strong> Site</div><div className="category-shortcut">Sensor <strong>directSite →</strong> Site</div></div>
    <div className="category-controls"><Choice label="Inspect sensor" value={sensor} onChange={setSensor} numeric options={result.sensors.map((name, index) => [index, name])} /><Choice label={'Stored direct site for s' + sensor} value={direct[sensor]} numeric onChange={value => setDirect(old => old.map((site, index) => index === sensor ? value : site))} options={result.sites.map((name, index) => [index, name])} /></div>
    <div className="category-route-comparison">
      <div><h4>Through the device</h4><ol><li>s{sensor}</li><li>d{result.deviceOf[sensor]}</li><li>{result.sites[result.composed[sensor]]}</li></ol></div>
      <div><h4>Stored shortcut</h4><ol><li>s{sensor}</li><li>{result.sites[direct[sensor]]}</li></ol></div>
    </div>
    <table className="category-table"><caption>All sensors, not just the selected one</caption><thead><tr><th scope="col">Sensor</th><th scope="col">Device</th><th scope="col">Via device</th><th scope="col">Direct</th></tr></thead><tbody>{result.sensors.map((name, index) => <tr key={name} className={sensor === index ? 'category-selected-row' : ''}><th scope="row">{name}</th><td>d{result.deviceOf[index]}</td><td>{result.sites[result.composed[index]]}</td><td>{result.sites[direct[index]]}</td></tr>)}</tbody></table>
    <Result>{result.violations.length ? 'The schema equation fails at ' + result.violations.map(index => 's' + index).join(', ') + '.' : 'Every sensor satisfies the schema equation. Together with the specified identity maps, this is a valid instance of this schema.'}</Result>
    <div className="category-actions"><button onClick={() => setDirect(result.composed.slice())}>Repair direct sites from devices</button><Reset onClick={() => {
        setSensor(1);
        setDirect([0, 1, 1]);
      }} /></div>
  </section>;
}
export function NaturalityLab() {
  const [operation, setOperation] = useState('reverse');
  const [mapping, setMapping] = useState('reverse');
  const [list, setList] = useState('distinct');
  const lists = {
    distinct: [2, 0, 1],
    repeated: [2, 0, 2, 1],
    empty: [],
    singleton: [2]
  };
  const result = naturalitySnapshot(lists[list], operation, mapping);
  return <section className="category-lab" aria-label="Naturality square investigation">
    <h3>Go around the square both ways</h3>
    <p>Horizontal arrows {operation === 'reverse' ? 'reverse the positions' : 'sort using 0 < 1 < 2'}. Vertical arrows apply the same element function to every entry. Compare the two results at the lower-right corner.</p>
    <div className="category-controls"><Choice label="Proposed list transformation" value={operation} onChange={setOperation} options={[['reverse', 'Reverse positions'], ['sort', 'Sort values']]} /><Choice label="Element function" value={mapping} onChange={setMapping} options={functionOptions} /><Choice label="Input list" value={list} onChange={setList} options={[['distinct', '[2, 0, 1]'], ['repeated', '[2, 0, 2, 1]'], ['empty', 'Empty list'], ['singleton', '[2]']]} /></div>
    <div className="category-square">
      <div className="category-corner"><h4>List(A): start</h4><Tokens values={result.original} /></div><span className="category-square-arrow" aria-hidden="true">→</span><div className="category-corner"><h4>List(A): {operation}</h4><Tokens values={result.transformed} /></div>
      <span className="category-square-arrow" aria-hidden="true">↓</span><span /><span className="category-square-arrow" aria-hidden="true">↓</span>
      <div className="category-corner"><h4>List(B): mapped</h4><Tokens values={result.mapped} /></div><span className="category-square-arrow" aria-hidden="true">→</span><div className="category-corner"><h4>List(B): compare</h4><p>Top then right</p><Tokens values={result.topThenRight} /><p>Left then bottom</p><Tokens values={result.leftThenBottom} /></div>
    </div>
    <Result>{result.commutesHere ? 'These routes agree on this input. That alone is not a proof for every map and list.' : 'The routes disagree. This one counterexample refutes naturality across all maps.'} {operation === 'reverse' ? 'The position-by-position proof explains why reversal always works.' : 'Sorting can agree for a particular case and still fail the universal requirement.'}</Result>
    <Reset onClick={() => {
      setOperation('reverse');
      setMapping('reverse');
      setList('distinct');
    }} />
  </section>;
}
export function UniversalProductLab() {
  const [mode, setMode] = useState('complete');
  const [pair, setPair] = useState([0, 0]);
  const result = productSnapshot(mode, ...pair);
  return <section className="category-lab" aria-label="Universal product investigation">
    <h3>Ask for exactly one point with these projections</h3>
    <p>A point must report one a-value and one b-value. Select a cell to ask for that pair. A product must answer every such request with exactly one point.</p>
    <Choice label="Candidate object" value={mode} onChange={setMode} options={[['complete', 'All four pairs once'], ['missing', 'Remove p00'], ['duplicate', 'Add distinct q00 with the same projections']]} />
    <table className="category-product"><caption>Rows are the first projection; columns are the second</caption><thead><tr><td /><th scope="col">b = 0</th><th scope="col">b = 1</th></tr></thead><tbody>{[0, 1].map(row => <tr key={row}><th scope="row">a = {row}</th>{[0, 1].map(column => <td key={column}><button aria-label={'Choose pair (' + row + ', ' + column + ')'} aria-pressed={pair[0] === row && pair[1] === column} onClick={() => setPair([row, column])}>{result.points.filter(point => point.row === row && point.column === column).map(point => <span key={point.id} className="category-product-point">{point.id}</span>)}{result.counts[row][column] === 0 && <span className="category-empty">No point</span>}</button></td>)}</tr>)}</tbody></table>
    <Result>For ({pair.join(', ')}), there {result.mediators.length === 1 ? 'is one mediator' : 'are ' + result.mediators.length + ' mediators'}{result.mediators.length ? ': ' + result.mediators.map(point => point.id).join(', ') : ''}. {result.universal ? 'Every pair has exactly one: this candidate is a product.' : 'This candidate fails the product requirement at (0, 0), even if your selected pair works.'}</Result>
    <p>A mediator from the singleton probe chooses one candidate point. For an arbitrary set Z, repeat that forced choice for every z in Z; that gives the general pairing function.</p>
    <Reset onClick={() => {
      setMode('complete');
      setPair([0, 0]);
    }} />
  </section>;
}
function JointGrid({
  title,
  joint
}) {
  return <div className="category-joint"><h4>{title}</h4><table><caption>First bit = row; second bit = column</caption><thead><tr><td /><th scope="col">0</th><th scope="col">1</th></tr></thead><tbody>{joint.map((row, i) => <tr key={i}><th scope="row">{i}</th>{row.map((p, j) => <td key={j} style={{
            '--category-probability': p
          }}><span>{format(p)}</span></td>)}</tr>)}</tbody></table></div>;
}
export function StochasticCopyLab() {
  const [percent, setPercent] = useState(50);
  const result = stochasticCopySnapshot(percent);
  return <section className="category-lab" aria-label="Stochastic copying investigation">
    <h3>Copy one draw, or draw twice?</h3>
    <p>The same source probability governs both experiments. Each table is an exact theoretical joint distribution calculated from that probability, not a simulated sample.</p>
    <Range label="Chance of bit 1, percent" min={0} max={100} value={percent} onChange={setPercent} />
    <div className="category-two">
      <div><div className="category-wire"><span>One coin</span><strong>↓ copy its result</strong><span>(bit, same bit)</span></div><JointGrid title="One draw copied" joint={result.copied} /></div>
      <div><div className="category-wire"><span>Coin 1 <b>∥</b> Coin 2</span><strong>↓ independent draws</strong><span>(first bit, second bit)</span></div><JointGrid title="Two independent draws" joint={result.independent} /></div>
    </div>
    <Result>Each marginal is [{result.marginal.map(format).join(', ')}]. Disagreement has probability 0 after copying and {format(result.independentDisagreement)} after independent draws. {result.copyCommutes ? 'At this deterministic endpoint the joints coincide.' : 'The identical marginals do not determine the joint distribution.'}</Result>
    <Reset onClick={() => setPercent(50)} />
  </section>;
}
export function TangentCompositionLab() {
  const [input, setInput] = useState(0);
  const [direction, setDirection] = useState(1);
  const [incoming, setIncoming] = useState(1);
  const [choice, setChoice] = useState('shiftedSquare');
  const result = tangentSnapshot(input, direction, choice, incoming);
  const stages = [['Input', result.input, result.tangent, result.inputCotangent], ['After f', result.firstValue, result.intermediateTangent, result.intermediateCotangent], ['After g', result.secondValue, result.outputTangent, result.cotangent]];
  return <section className="category-lab" aria-label="Tangent composition investigation">
    <h3>Carry the point beside its infinitesimal change</h3>
    <div className="category-controls"><Choice label="Polynomial composition" value={choice} onChange={setChoice} options={Object.entries(tangentFunctionChoices).map(([key, value]) => [key, value.first + '; ' + value.second])} /><Range label="Input x" min={-3} max={3} step={0.1} value={input} onChange={setInput} /><Range label="Input tangent v" min={-2} max={2} step={0.25} value={direction} onChange={setDirection} /><Range label="Output cotangent w" min={-2} max={2} step={0.25} value={incoming} onChange={setIncoming} /></div>
    <p>Read the primal and tangent lanes forward. Read the cotangent lane backward, using the derivative values computed at the saved primal points.</p>
    <div className="category-derivative-lanes">{stages.map(([name, point, tangent, cotangent], index) => <div key={name} className="category-derivative-stage"><h4>{name}</h4><p className="category-primal">Point <strong>{format(point)}</strong>{index < 2 && <span aria-hidden="true" className="category-flow-forward" />}</p><p className="category-tangent">Tangent <strong>{format(tangent)}</strong>{index < 2 && <span aria-hidden="true" className="category-flow-forward" />}</p><p className="category-cotangent">{index > 0 && <span aria-hidden="true" className="category-flow-backward" />}Cotangent <strong>{format(cotangent)}</strong></p></div>)}</div>
    <dl className="category-metrics"><div><dt>f′ evaluated at x = {format(input)}</dt><dd>{format(result.firstDerivative)}</dd></div><div><dt>g′ evaluated at f(x) = {format(result.firstValue)}</dt><dd>{format(result.secondDerivative)}</dd></div><div><dt>Composite derivative</dt><dd>{format(result.composedDerivative)}</dd></div><div><dt>Direct derivative of the composite formula</dt><dd>{format(result.directDerivative)}</dd></div></dl>
    <Result>Evaluating g′ at x instead would give the derivative {format(result.wrongDerivative)}. {Math.abs(result.wrongDerivative - result.composedDerivative) < 1e-12 ? 'It happens to agree here; this is not a valid general replacement.' : 'That is the wrong evaluation point for this composition.'} The pairing check is w·output tangent = {format(result.outputPairing)} = input cotangent·v.</Result>
    <Reset onClick={() => {
      setInput(0);
      setDirection(1);
      setIncoming(1);
      setChoice('shiftedSquare');
    }} />
  </section>;
}
export function ImagePreimageLab() {
  const [sourceMask, setSourceMask] = useState(1);
  const [targetMask, setTargetMask] = useState(1);
  const result = adjunctionSnapshot(sourceMask, targetMask);
  return <section className="category-lab" aria-label="Image preimage adjunction investigation">
    <h3>Choose a source selection and a target filter</h3>
    <p>The fixed map sends a and b to 0, c to 1, and d to 2. Selecting both a and b can add no new output: this is where information is lost.</p>
    <div className="category-two"><fieldset><legend>Source subset S</legend>{['a', 'b', 'c', 'd'].map((name, index) => <label key={name} className="category-membership"><input type="checkbox" checked={(sourceMask & 1 << index) !== 0} onChange={() => setSourceMask(old => old ^ 1 << index)} aria-label={'Include source ' + name} /><span>{name}</span><strong>→ {result.mapping[index]}</strong>{result.preimage.includes(index) && <small>passes T</small>}</label>)}</fieldset><fieldset><legend>Target subset T</legend>{[0, 1, 2].map(index => <label key={index} className="category-membership"><input type="checkbox" checked={(targetMask & 1 << index) !== 0} onChange={() => setTargetMask(old => old ^ 1 << index)} aria-label={'Include target ' + index} /><span>{index}</span>{result.image.includes(index) && <small>reached by S</small>}</label>)}</fieldset></div>
    <dl className="category-metrics"><div><dt>Image f(S)</dt><dd>{'{' + result.image.join(', ') + '}'}</dd></div><div><dt>Preimage f⁻¹(T)</dt><dd>{'{' + result.preimage.map(index => 'abcd'[index]).join(', ') + '}'}</dd></div><div><dt>Round trip f⁻¹(f(S))</dt><dd>{'{' + result.saturation.map(index => 'abcd'[index]).join(', ') + '}'}</dd></div></dl>
    <Result>f(S) ⊆ T is {String(result.imageContained)}. S ⊆ f⁻¹(T) is {String(result.sourceContained)}. {result.sourceFailures.length ? 'Source witness ' + 'abcd'[result.sourceFailures[0]] + ' maps to ' + result.mapping[result.sourceFailures[0]] + ', which is outside T, so both containments fail.' : 'Every selected source lands in T, so both containments hold. This includes an empty source selection.'}</Result>
    <Reset onClick={() => {
      setSourceMask(1);
      setTargetMask(1);
    }} />
  </section>;
}
export function HomologyFunctorFigure() {
  return <figure className="category-figure"><div className="category-homology-row">{[false, true].map(filled => <div key={String(filled)}><svg viewBox="0 0 180 146" role="img" aria-label={filled ? 'A filled triangle whose boundary cycle is zero in homology' : 'An unfilled triangle with a nonzero cycle class'}><polygon points="90,18 25,117 155,117" fill={filled ? '#273649' : 'none'} stroke="#eac06e" strokeWidth="3" />{[[90, 18], [25, 117], [155, 117]].map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="7" fill="#eac06e" /><text x={x} y={y + (i === 0 ? 27 : -15)} textAnchor="middle">{i}</text></g>)}</svg><strong>{filled ? 'Face added: [cycle] = 0' : 'Only edges: [cycle] ≠ 0'}</strong></div>)}</div><div className="category-map-equation">Inclusion of spaces <strong>→</strong> Induced map H₁: F₂ → {'{0}'}</div><figcaption>The edge chain is still there. What changed is the set of boundaries we identify with zero. The space inclusion is injective; its induced H₁ map is not.</figcaption><p>C₁ is the edge-chain space; C₀ is the vertex-chain space. Horizontal arrows carry chains into the target. Vertical arrows take their boundaries. Both routes must agree.</p><div className="category-chain-square" role="img" aria-label="Commuting chain square: C1(K) maps by c1 to C1(L); both map down by boundary to C0(K) and C0(L), joined by c0."><strong>C₁(K)</strong><span>c₁ →</span><strong>C₁(L)</strong><span>∂ ↓</span><span /><span>∂ ↓</span><strong>C₀(K)</strong><span>c₀ →</span><strong>C₀(L)</strong></div><p className="category-map-equation">∂ ∘ c₁ = c₀ ∘ ∂</p></figure>;
}
export function ParallelCompositionFigure() {
  return <figure className="category-figure"><div className="category-wire-pair"><div>A <b>─ f →</b> B <b>─ h →</b> C</div><div>X <b>─ g →</b> Y <b>─ k →</b> Z</div></div><figcaption>Two independent lanes. Execute f and g in parallel, then h and k; or compose h∘f and k∘g within their lanes, then pair them. The interchange law equates these typed arrangements. It does not permit crossing a value into the other lane without an additional specified arrow.</figcaption></figure>;
}
export function OptionCompositionFigure() {
  return <figure className="category-figure"><div className="category-option-root">A <strong>─ f →</strong> Option(B)</div><div className="category-option-branches"><div><strong>None</strong><span>stop this pipeline</span><b>↓</b><strong>None</strong></div><div><strong>Some(b)</strong><span>give b to g</span><b>↓</b><strong>g(b) in Option(C)</strong></div></div><figcaption>The failure branch bypasses g. The successful branch passes the payload, not the Some wrapper. A later failure is still possible.</figcaption></figure>;
}
