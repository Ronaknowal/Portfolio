import { useMemo, useState } from 'react';
import { allocationCount, allocationWord, balancedWordCounts, choiceFibers, coefficientStages, enumerateAllocations, inductionCoverage, inductionPresets, overlapContributions, parenthesisPath, rotationOrbits } from '../../data/counting-combinatorics-models.js';
import './counting-combinatorics-labs.css';
function Select({
  label,
  value,
  choices,
  onChange
}) {
  return <label className="counting-control"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{choices.map(([key, text]) => <option value={key} key={key}>{text}</option>)}</select></label>;
}
function Word({
  value
}) {
  return <span className="counting-word">{value.length ? [...value].map((symbol, index) => <span key={index}>{symbol}</span>) : <em>empty</em>}</span>;
}
function Pager({
  index,
  size,
  onChange
}) {
  return <div className="counting-actions"><button type="button" disabled={index <= 0} onClick={() => onChange(index - 1)}>Previous</button><span>{size ? index + 1 : 0} of {size}</span><button type="button" disabled={index >= size - 1} onClick={() => onChange(index + 1)}>Next</button></div>;
}
export function UnequalBranchesFigure() {
  return <figure className="counting-figure"><div className="counting-branches">{[['Ada', ['Bo']], ['Bo', ['Ada', 'Cam']], ['Cam', ['Ada', 'Bo']]].map(([lead, helpers]) => <div key={lead}><strong>{lead} leads</strong><span className="counting-branch-arrow">↓ choose a helper</span><div>{helpers.map(helper => <span className="counting-chip" key={helper}>{helper}</span>)}</div><small>{helpers.length} legal {helpers.length === 1 ? 'assignment' : 'assignments'}</small></div>)}</div><figcaption>The branches are disjoint because the lead differs. Ada has only one eligible helper, so the count is 1 + 2 + 2 = 5, not 3 × 2. Each displayed leaf is an actual allowed assignment.</figcaption></figure>;
}
export function ChoiceIdentityLab() {
  const [labelCount, setLabelCount] = useState(4);
  const [length, setLength] = useState(2);
  const [repeats, setRepeats] = useState(false);
  const [index, setIndex] = useState(0);
  const data = useMemo(() => choiceFibers(labelCount, length, repeats), [labelCount, length, repeats]);
  const selected = data.fibers[Math.min(index, data.fibers.length - 1)];
  function reset() {
    setLabelCount(4);
    setLength(2);
    setRepeats(false);
    setIndex(0);
  }
  return <section className="counting-lab" aria-label="Outcome identity and fibers investigation">
    <h3>What did forgetting order merge?</h3>
    <p>Inspect the number of descriptions for one group. Then allow repetitions: will every group still have the same size?</p>
    <div className="counting-controls"><Select label="Available labels" value={labelCount} choices={[0, 1, 2, 3, 4].map(value => [value, value])} onChange={value => {
        setLabelCount(Number(value));
        setIndex(0);
      }} /><Select label="Positions" value={length} choices={[0, 1, 2, 3].map(value => [value, value])} onChange={value => {
        setLength(Number(value));
        setIndex(0);
      }} /><Select label="Repeated labels" value={String(repeats)} choices={[[false, 'Not allowed'], [true, 'Allowed']]} onChange={value => {
        setRepeats(value === 'true');
        setIndex(0);
      }} /></div>
    <div className="counting-statline" role="status"><span><strong>{data.descriptions.length}</strong> ordered descriptions</span><span><strong>{data.fibers.length}</strong> unordered outcomes</span></div>
    {selected ? <div className="counting-fiber"><div><small>One outcome after order is forgotten</small><Word value={selected.key} /></div><span className="counting-branch-arrow">↑ all descriptions of this same outcome</span><div className="counting-words">{selected.members.map((member, memberIndex) => <Word key={memberIndex} value={member.join('')} />)}</div><p>{selected.members.length} {selected.members.length === 1 ? 'description maps' : 'descriptions map'} here.</p></div> : <p className="counting-result">No legal descriptions exist. There is no nonempty group from which to infer a divisor.</p>}
    <Pager index={Math.min(index, Math.max(0, data.fibers.length - 1))} size={data.fibers.length} onChange={setIndex} />
    <p className="counting-result" role="status">{data.uniform ? `Every group has ${data.sizes[0]} descriptions: ${data.descriptions.length} ÷ ${data.sizes[0]} = ${data.fibers.length}.` : data.fibers.length ? `Group sizes vary: ${data.sizes.join(', ')}. Dividing by one common multiplicity is not justified.` : 'The count is zero.'}</p>
    <div className="counting-actions"><button type="button" onClick={() => {
        setLabelCount(2);
        setLength(2);
        setRepeats(true);
        setIndex(0);
      }}>Show unequal repeat groups</button><button type="button" onClick={reset}>Reset</button></div>
    <p>Transfer: with zero positions, the empty description is one valid outcome, even with no labels. With a positive number of positions and no labels, there are none.</p>
  </section>;
}
export function TaggedSymbolsFigure() {
  return <figure className="counting-figure"><div className="counting-tagged"><div><strong>Tag equal symbols temporarily</strong><Word value={['N₁', 'N₂', 'O₁', 'O₂']} /><p>4! = 24 arrangements of distinct tags</p></div><div className="counting-branch-arrow">erase subscripts ↓</div><div><Word value="NOON" /><p>For this string: choose which N tag occupies its first N slot (2 ways), then which O tag occupies its first O slot (2 ways).</p></div></div><figcaption>Each untagged string has exactly 2! × 2! = 4 tagged descriptions. Therefore there are 24 ÷ 4 = 6 strings. The positions of N and O vary; the number of tag assignments does not.</figcaption></figure>;
}
export function AllocationBarsLab() {
  const [total, setTotal] = useState(5);
  const [rule, setRule] = useState('free');
  const [index, setIndex] = useState(0);
  const capacities = Array(3).fill(rule === 'cap' ? 3 : total);
  const minimums = Array(3).fill(rule === 'positive' ? 1 : 0);
  const allocations = useMemo(() => enumerateAllocations(total, capacities, minimums), [total, rule]);
  const currentIndex = Math.min(index, Math.max(0, allocations.length - 1));
  const values = allocations[currentIndex];
  function transferred(source, destination) {
    if (!values) return -1;
    const next = [...values];
    next[source] -= 1;
    next[destination] += 1;
    return allocations.findIndex(allocation => allocation.every((value, position) => value === next[position]));
  }
  return <section className="counting-lab" aria-label="Stars bars and allocations investigation"><h3>Move a token; read the same allocation as a word</h3><p>Before adding a minimum or a capacity, predict which visible arrangements will disappear. Tokens are identical; the three stations are named.</p>
    <div className="counting-controls"><Select label="Total tokens" value={total} choices={[0, 1, 2, 3, 4, 5, 6, 7, 8].map(value => [value, value])} onChange={value => {
        setTotal(Number(value));
        setIndex(0);
      }} /><Select label="Allocation rule" value={rule} choices={[["free", 'Empty stations allowed'], ['positive', 'At least one at each station'], ['cap', 'At most three at each station']]} onChange={value => {
        setRule(value);
        setIndex(0);
      }} /></div>
    {values ? <><div className="counting-containers">{values.map((value, position) => <div key={position}><strong>Station {'ABC'[position]}</strong><div className="counting-tokens">{Array.from({
              length: value
            }, (_, token) => <span key={token} aria-hidden="true">●</span>)}{value === 0 && <em>empty</em>}</div><span>{value} {value === 1 ? 'token' : 'tokens'}</span></div>)}</div><div className="counting-bar-word" aria-label={`Stars and bars: ${allocationWord(values) || 'empty word'}`}>{values.map((value, position) => <span className="counting-run" key={position}><span>{'★'.repeat(value) || <em>0</em>}</span>{position < 2 && <b>|</b>}</span>)}</div><p className="counting-note">The italic 0 marks an empty run for readability; it is not an extra star or symbol in the mathematical encoding. The actual word is <code>{allocationWord(values)}</code>.</p><div className="counting-actions">{[[0, 1], [1, 0], [1, 2], [2, 1]].map(([source, destination]) => {
          const next = transferred(source, destination);
          return <button type="button" key={`${source}-${destination}`} disabled={next < 0} onClick={() => setIndex(next)}>Move {'ABC'[source]} → {'ABC'[destination]}</button>;
        })}</div></> : <p className="counting-result">No feasible allocation. The requirements cannot fit this total.</p>}
    <p role="status" className="counting-result">Exact count: {allocationCount(total, capacities, minimums).toString()}. {values ? `Current tuple: (${values.join(', ')}).` : 'No tuple is selected.'}</p><Pager index={currentIndex} size={allocations.length} onChange={setIndex} /><button type="button" onClick={() => {
      setTotal(5);
      setRule('free');
      setIndex(0);
    }}>Reset</button><p>Transfer: make the total two and require one at every station. Why must the count be zero before any formula is evaluated?</p>
  </section>;
}
export function DoubleCountingFigure() {
  return <figure className="counting-figure"><div className="counting-double"><div><strong>Committee, then chair</strong><p>Choose {'{A, C, D}'}<br />then mark C as chair.</p><span>C(6, 3) × 3</span></div><div className="counting-paired-object"><span>A</span><strong>C ★</strong><span>D</span><small>the same chaired committee</small></div><div><strong>Chair, then companions</strong><p>Choose C<br />then choose {'{A, D}'} from the other five.</p><span>6 × C(5, 2)</span></div></div><figcaption>Both procedures produce exactly the same object. Removing the chair mark and restoring it reverses either description. The counts agree because the correspondence is one-to-one, not merely because both formulas happen to equal 60.</figcaption></figure>;
}
function defaultMemberships() {
  return Array.from({
    length: 12
  }, (_, index) => [2, 3, 4].map(divisor => (index + 1) % divisor === 0));
}
export function InclusionExclusionLab() {
  const [memberships, setMemberships] = useState(defaultMemberships);
  const [stage, setStage] = useState(1);
  const [selected, setSelected] = useState(11);
  const data = overlapContributions(memberships);
  const current = data.stages[stage - 1];
  return <section className="counting-lab" aria-label="Inclusion exclusion contribution investigation"><h3>Follow one object's overcount</h3><p>Object 12 starts in A, B and C. Inspect its net contribution after adding singles, subtracting pairs, then restoring the triple overlap.</p>
    <div className="counting-actions">{[[1, '1 · Add singles'], [2, '2 · Subtract pairs'], [3, '3 · Add triple']].map(([value, label]) => <button type="button" key={value} aria-pressed={stage === value} onClick={() => setStage(value)}>{label}</button>)}</div>
    <div className="counting-membership-roster">{memberships.map((row, index) => <button type="button" key={index} aria-label={`Inspect object ${index + 1}`} aria-pressed={selected === index} onClick={() => setSelected(index)}><strong>{index + 1}</strong><span>{row.map((present, set) => present ? 'ABC'[set] : '').join('') || 'none'}</span><small>weight {current.weights[index]}</small></button>)}</div>
    <div className="counting-selected-object"><strong>Object {selected + 1}</strong><div className="counting-actions">{[0, 1, 2].map(set => <button type="button" key={set} aria-label={`Object ${selected + 1} in set ${'ABC'[set]}`} aria-pressed={memberships[selected][set]} onClick={() => setMemberships(old => old.map((row, index) => index === selected ? row.map((value, column) => column === set ? !value : value) : row))}>{'ABC'[set]}: {memberships[selected][set] ? 'yes' : 'no'}</button>)}</div><div className="counting-contributions">{data.terms.filter(term => term.sets.length <= stage).map(term => <span key={term.sets.join('')}><small>{term.sets.map(set => 'ABC'[set]).join('∩')}</small><strong>{term.members.includes(selected + 1) ? term.sign > 0 ? '+1' : '−1' : '0'}</strong></span>)}</div></div>
    <p className="counting-result" role="status">Selected object's net weight: {current.weights[selected]}. Signed total at this stage: {current.total}. Actual union size: {data.union.length}.</p><p>Actual union: {'{'}{data.union.join(', ')}{'}'}. At stage three, every member has weight one and every nonmember has weight zero.</p><div className="counting-actions"><button type="button" onClick={() => {
        setMemberships(Array.from({
          length: 12
        }, () => [false, false, false]));
        setStage(3);
      }}>Empty all sets</button><button type="button" onClick={() => {
        setMemberships(defaultMemberships());
        setStage(1);
        setSelected(11);
      }}>Reset</button></div>
  </section>;
}
export function CompressionFigure() {
  return <figure className="counting-figure"><div className="counting-double"><div><strong>Three-bit inputs: 8</strong><div className="counting-small-words">{['000', '001', '010', '011', '100', '101', '110', '111'].map(word => <code key={word}>{word}</code>)}</div></div><div><strong>All shorter outputs: 7</strong><div className="counting-small-words">{['empty', '0', '1', '00', '01', '10', '11'].map(word => <code key={word}>{word}</code>)}</div></div></div><figcaption>An injective map from eight inputs to seven possible outputs cannot exist. Variable output lengths already appear on the right; allowing them does not remove the shortage. Extra headers or side information must be included in the encoding's total description.</figcaption></figure>;
}
export function InductionCoverageLab() {
  const [presetName, setPresetName] = useState('fourSeven');
  const [enabled, setEnabled] = useState([true, true, true, true]);
  const [target, setTarget] = useState(38);
  const data = inductionCoverage(presetName, enabled, target);
  function preset(value) {
    setPresetName(value);
    setEnabled(Array(inductionPresets[value].small).fill(true));
    setTarget(inductionPresets[value].lower + 20);
  }
  return <section className="counting-lab" aria-label="Induction base coverage investigation"><h3>Does this proof chain reach a supported base?</h3><p>Remove one base certificate. Inspect which targets lose support. Their arithmetic representability is checked separately.</p><div className="counting-controls"><Select label="Token values and theorem" value={presetName} choices={[["fourSeven", '4 and 7: every total at least 18'], ['threeFive', '3 and 5: every total at least 8']]} onChange={preset} /><Select label="Target total" value={target} choices={Array.from({
        length: 31
      }, (_, index) => data.lower + index).map(value => [value, value])} onChange={value => setTarget(Number(value))} /></div>
    <div className="counting-base-cases">{data.bases.map(([small, large], index) => <button type="button" key={index} aria-pressed={enabled[index]} aria-label={`Base ${data.lower + index} certificate`} onClick={() => setEnabled(old => old.map((value, position) => position === index ? !value : value))}><strong>{data.lower + index}</strong><span>{small}×{data.small} + {large}×{data.large}</span><small>{enabled[index] ? 'certificate present' : 'certificate removed'}</small></button>)}</div>
    <div className="counting-proof-chain" aria-label={`Subtract ${data.small} from ${target} to base ${data.base}`}>{data.chain.map((value, index) => <span key={value}><strong className={value === data.base ? data.supported ? 'counting-supported' : 'counting-unsupported' : ''}>{value}</strong>{index < data.chain.length - 1 && <small>−{data.small} →</small>}</span>)}</div>
    <p role="status" className="counting-result">{data.supported ? `Supported: ${data.witness[0]}×${data.small} + ${data.witness[1]}×${data.large} = ${target}.` : `Unsupported by the remaining base certificates: the chain reaches missing base ${data.base}.`}</p><p>Independent arithmetic check: {data.actualWitness ? `${data.actualWitness[0]}×${data.small} + ${data.actualWitness[1]}×${data.large} = ${target}, so this target is representable.` : 'no representation exists.'} This check does not replace the proof for every target.</p><button type="button" onClick={() => {
      setPresetName('fourSeven');
      setEnabled([true, true, true, true]);
      setTarget(38);
    }}>Reset</button>
  </section>;
}
function PathPlot({
  data,
  reflected
}) {
  const heights = reflected ? data.reflectedHeights : data.heights;
  const allHeights = [...data.heights, ...(data.reflectedHeights || [])];
  const minimum = Math.min(-1, ...allHeights);
  const maximum = Math.max(2, ...allHeights);
  const x = index => 35 + index / Math.max(1, data.word.length) * 250;
  const y = height => 185 - (height - minimum) / (maximum - minimum) * 155;
  return <svg className="counting-svg counting-path-plot" viewBox="0 0 320 235" role="img" aria-label={`${reflected ? 'Reflected' : 'Original'} path. Prefix heights ${heights.join(', ')}.`}>
    {Array.from({
      length: maximum - minimum + 1
    }, (_, index) => minimum + index).map(height => <g key={height}><line x1="35" y1={y(height)} x2="285" y2={y(height)} stroke={height === 0 ? '#a8bacd' : '#354455'} strokeDasharray={height === 0 ? undefined : '3 4'} /><text x="27" y={y(height) + 4} textAnchor="end">{height}</text></g>)}
    <text x="35" y="16">unmatched opens / height</text>
    {!data.valid && <line x1={x(data.firstBad)} y1="26" x2={x(data.firstBad)} y2="190" stroke="#e4b86b" strokeDasharray="3 3" />}
    <polyline points={heights.map((height, index) => `${x(index)},${y(height)}`).join(' ')} fill="none" stroke={reflected ? '#8ecbb0' : '#e4b86b'} strokeWidth="3" />
    {heights.map((height, index) => <circle key={index} cx={x(index)} cy={y(height)} r="3" fill={reflected ? '#8ecbb0' : '#e4b86b'} />)}
    {heights.map((_, index) => (data.word.length <= 6 || index % 2 === 0) && <text key={index} x={x(index)} y="206" textAnchor="middle">{index}</text>)}
    <text x="160" y="227" textAnchor="middle">prefix length (steps)</text>
  </svg>;
}
export function CatalanPathsLab() {
  const [pairs, setPairs] = useState(3);
  const [mode, setMode] = useState('bad');
  const [index, setIndex] = useState(0);
  const [reflected, setReflected] = useState(false);
  const counts = useMemo(() => balancedWordCounts(pairs), [pairs]);
  const words = mode === 'valid' ? counts.valid : counts.bad;
  const currentIndex = Math.min(index, Math.max(0, words.length - 1));
  const word = words[currentIndex];
  const data = word === undefined ? null : parenthesisPath(word);
  return <section className="counting-lab" aria-label="Balanced paths and reflection investigation"><h3>Give every bad path a reversible description</h3><p>A bad word has equally many opens and closes but crosses below zero. Inspect what reflecting its first offending prefix does to the endpoint. Switch to balanced words to inspect the unique first-return split.</p><div className="counting-controls"><Select label="Parenthesis pairs" value={pairs} choices={[0, 1, 2, 3, 4, 5].map(value => [value, value])} onChange={value => {
        setPairs(Number(value));
        setIndex(0);
        setReflected(false);
      }} /><Select label="Path family" value={mode} choices={[["bad", 'Bad paths: reflection'], ['valid', 'Balanced paths: decomposition']]} onChange={value => {
        setMode(value);
        setIndex(0);
        setReflected(false);
      }} /></div>
    <p className="counting-statline"><span>All: <strong>{counts.total.toString()}</strong></span><span>Bad: <strong>{counts.excluded.toString()}</strong></span><span>Balanced: <strong>{counts.catalan[pairs].toString()}</strong></span></p>
    {data ? <><div className="counting-path-word"><Word value={reflected && data.reflected ? data.reflected.map(step => step === 1 ? '(' : ')').join('') : word} /></div><PathPlot data={data} reflected={reflected && !data.valid} />{data.valid ? <p className="counting-result" role="status">{word.length ? <>First return: step {data.firstReturn}. A = <code>{data.inside || 'empty'}</code>, B = <code>{data.after || 'empty'}</code>. Their pair counts are {data.inside.length / 2} and {data.after.length / 2}.</> : 'One empty balanced word. It is the base object, with no first-return decomposition.'}</p> : <><p className="counting-result" role="status">{reflected ? `Reflected endpoint: +2. The first +1 is at step ${data.firstBad}; reflecting that prefix again restores the original.` : `First height −1: step ${data.firstBad}. Flip exactly these first ${data.firstBad} steps; the remaining steps stay unchanged.`}</p><button type="button" onClick={() => setReflected(!reflected)}>{reflected ? 'Undo at first +1' : 'Reflect at first −1'}</button></>}</> : <p className="counting-result">There are no bad paths with zero pairs. Switch to balanced paths to inspect the empty word.</p>}
    <Pager index={currentIndex} size={words.length} onChange={value => {
      setIndex(value);
      setReflected(false);
    }} /><button type="button" onClick={() => {
      setPairs(3);
      setMode('bad');
      setIndex(0);
      setReflected(false);
    }}>Reset</button><p>Transfer: for four pairs, explain why 70 − 56 = 14. The finite exploration checks examples; the inverse construction in the text proves the formula for arbitrary n.</p>
  </section>;
}
export function CoefficientConstructionLab() {
  const [capacities, setCapacities] = useState([2, 3, 1]);
  const [stage, setStage] = useState(3);
  const [target, setTarget] = useState(3);
  const rows = coefficientStages(capacities);
  const row = rows[stage];
  const prior = stage > 0 ? rows[stage - 1] : null;
  const contributions = prior ? Array.from({
    length: capacities[stage - 1] + 1
  }, (_, extra) => ({
    extra,
    degree: target - extra,
    count: prior[target - extra] || 0n
  })) : [];
  return <section className="counting-lab" aria-label="Generating coefficient construction investigation"><h3>Watch a coefficient collect its contributions</h3><p>Explore how many ways the last station can contribute to total three. A coefficient counts assignments; the exponent records the total number of tokens.</p><div className="counting-controls">{capacities.map((capacity, index) => <Select key={index} label={`Station ${'ABC'[index]} capacity`} value={capacity} choices={[0, 1, 2, 3, 4].map(value => [value, value])} onChange={value => setCapacities(old => old.map((amount, position) => position === index ? Number(value) : amount))} />)}</div>
    <div className="counting-controls"><Select label="Factors included" value={stage} choices={[[0, 'None: constant 1'], [1, 'A'], [2, 'A and B'], [3, 'A, B and C']]} onChange={value => setStage(Number(value))} /><Select label="Target degree" value={target} choices={Array.from({
        length: 14
      }, (_, value) => [value, value])} onChange={value => setTarget(Number(value))} /></div>
    <div className="counting-coefficients" aria-label="Current polynomial coefficients">{row.map((count, degree) => <span key={degree} className={degree === target ? 'counting-selected' : ''}><strong>{count.toString()}</strong><small>x<sup>{degree}</sup></small></span>)}</div>
    {prior ? <div className="counting-coefficient-sources">{contributions.map(({
        extra,
        degree,
        count
      }) => <div key={extra}><span>new station takes <strong>{extra}</strong></span><span>earlier stations total <strong>{degree}</strong></span><strong>{count.toString()} × 1 = {count.toString()}</strong><small>{degree < 0 ? 'negative earlier total: impossible' : degree >= prior.length ? 'beyond earlier capacities: impossible' : 'earlier coefficient × one new choice'}</small></div>)}</div> : <p>The empty product is 1: one way to allocate zero tokens to no stations, zero ways to allocate any positive total.</p>}
    <p className="counting-result" role="status">Coefficient of x^{target}: {(row[target] || 0n).toString()}. {prior ? `Contribution sum: ${contributions.map(entry => entry.count.toString()).join(' + ')}.` : ''}</p><div className="counting-actions"><button type="button" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Remove last factor</button><button type="button" disabled={stage === 3} onClick={() => setStage(stage + 1)}>Include next factor</button><button type="button" onClick={() => {
        setCapacities([2, 3, 1]);
        setStage(3);
        setTarget(3);
      }}>Reset</button></div>
  </section>;
}
export function RotationOrbitsLab() {
  const [length, setLength] = useState(4);
  const [index, setIndex] = useState(0);
  const [shift, setShift] = useState(0);
  const [marked, setMarked] = useState(false);
  const data = useMemo(() => rotationOrbits(length), [length]);
  const group = data.orbits[Math.min(index, data.orbits.length - 1)];
  const word = group.key.slice(shift) + group.key.slice(0, shift);
  const fixing = Array.from({
    length
  }, (_, rotation) => rotation).filter(rotation => word.slice(rotation) + word.slice(0, rotation) === word);
  return <section className="counting-lab" aria-label="Cyclic pattern symmetry investigation"><h3>Does every pattern have the same number of rotations?</h3><p>Find the alternating four-site pattern. Explore how many different images it has and which shifts leave it fixed.</p><div className="counting-controls"><Select label="Ring sites" value={length} choices={[3, 4, 5, 6].map(value => [value, value])} onChange={value => {
        setLength(Number(value));
        setIndex(0);
        setShift(0);
      }} /><Select label="Outcome convention" value={String(marked)} choices={[[false, 'Starting phase ignored'], [true, 'Starting position marked']]} onChange={value => setMarked(value === 'true')} /></div>
    <div className="counting-orbit-view"><svg className="counting-svg" viewBox="0 0 280 260" role="img" aria-label={`Binary ring ${word}. Reference at the top; ${group.members.length} distinct rotated images.`}><circle cx="140" cy="128" r="78" fill="none" stroke="#657b8e" />{[...word].map((bit, site) => {
          const angle = -Math.PI / 2 + site * 2 * Math.PI / length;
          const x = 140 + 78 * Math.cos(angle);
          const y = 128 + 78 * Math.sin(angle);
          return <g key={site}><circle cx={x} cy={y} r="20" fill={bit === '1' ? '#7f6334' : '#152332'} stroke={site === 0 ? '#f1cf8e' : '#89adc7'} strokeWidth={site === 0 ? 3 : 1} /><text x={x} y={y + 5} textAnchor="middle">{bit}</text></g>;
        })}<text x="140" y="17" textAnchor="middle">{marked ? 'marked site' : 'temporary reference'} ↓</text><text x="140" y="245" textAnchor="middle">reflections stay distinct</text></svg><div><strong>Distinct images in this orbit</strong><div className="counting-words">{group.members.map(member => <code key={member} className={member === word ? 'counting-selected' : ''}>{member}</code>)}</div><p>Fixing shifts: {fixing.join(', ')} sites.</p><p>{group.members.length} images × {fixing.length} fixing shifts = {length}.</p><button type="button" onClick={() => setShift((shift + 1) % length)}>Rotate one site</button></div></div>
    <Pager index={index} size={data.orbits.length} onChange={value => {
      setIndex(value);
      setShift(0);
    }} /><div className="counting-fixed-counts">{data.fixed.map((count, rotation) => <span key={rotation}><small>shift {rotation}</small><strong>{count}</strong><small>fixed patterns</small></span>)}</div>
    <p className="counting-result" role="status">{marked ? `${data.words.length} outcomes with a marked start. Rotated words are distinct unless their symbols are unchanged.` : `${data.orbits.length} outcomes when starting phase is ignored: (${data.fixed.join(' + ')}) ÷ ${length} = ${data.orbits.length}.`}</p><div className="counting-actions"><button type="button" onClick={() => {
        setLength(4);
        setIndex(3);
        setShift(0);
        setMarked(false);
      }}>Inspect alternating 0101</button><button type="button" onClick={() => {
        setLength(4);
        setIndex(0);
        setShift(0);
        setMarked(false);
      }}>Reset</button></div>
  </section>;
}
