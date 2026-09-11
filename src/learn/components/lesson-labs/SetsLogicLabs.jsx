import { useId, useMemo, useState } from 'react';
import { diagonalSubset, inspectArgument, inspectQuantifiers, inspectRelation, oddSquareStep, propositionLabels, relationPreset, rosterNames, selectRoster } from '../../data/sets-logic-models.js';
import './sets-logic-labs.css';
const setText = values => values.length ? `{${values.join(', ')}}` : '∅';
const truthText = value => value ? 'true' : 'false';
function Choice({
  label,
  value,
  choices,
  onChange
}) {
  return <label className="sets-logic-control"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>
    {choices.map(([key, text]) => <option value={key} key={key}>{text}</option>)}
  </select></label>;
}
export function SetMembershipFigure() {
  return <figure className="sets-logic-figure">
    <div className="sets-logic-nesting" aria-label="A contains 1 and the singleton set containing 2. The number 2 is inside the singleton, not a direct element of A.">
      <strong>A</strong><span className="sets-logic-brace">{'{'}</span><span>1,</span>
      <span className="sets-logic-inner-set"><span>{'{'}</span><strong>2</strong><span>{'}'}</span><small>one element of A</small></span>
      <span className="sets-logic-brace">{'}'}</span>
    </div>
    <div className="sets-logic-facts"><span>{'{2} ∈ A'}<small>The singleton itself occurs in A.</small></span><span>{'2 ∉ A'}<small>Its inner element does not.</small></span><span>{'{2} ⊄ A'}<small>Its only element, 2, is missing from A.</small></span></div>
    <figcaption>Nesting changes the object. A has two elements: the number 1 and the set {'{2}'}. It does not flatten the inner set.</figcaption>
  </figure>;
}
const setOperations = [['intersection', 'T ∩ B: both'], ['union', 'T ∪ B: either or both'], ['difference', 'T ∖ B: trained only'], ['reverseDifference', 'B ∖ T: badged only'], ['complement', 'U ∖ T: not trained'], ['symmetricDifference', 'T △ B: exactly one']];
const setRules = {
  intersection: 'in T AND in B',
  union: 'in T OR in B',
  difference: 'in T AND not in B',
  reverseDifference: 'in B AND not in T',
  complement: 'in U AND not in T',
  symmetricDifference: 'in exactly one of T and B'
};
export function SetRegionsLab() {
  const [training, setTraining] = useState(3);
  const [badges, setBadges] = useState(6);
  const [operation, setOperation] = useState('intersection');
  const result = selectRoster(training, badges, operation);
  const locations = {};
  for (const region of ['first', 'both', 'second', 'neither']) {
    const members = result.members.filter(member => member.region === region);
    members.forEach((member, index) => {
      locations[member.name] = region === 'neither' ? [320 * (index + 1) / (members.length + 1), 265] : [{
        first: 80,
        both: 160,
        second: 240
      }[region], 123 + (index - (members.length - 1) / 2) * 24];
    });
  }
  return <section className="sets-logic-lab" aria-label="Set membership regions investigation">
    <h3>Select people by a condition</h3><p>Predict the intersection before editing. Bo starts in both sets. Give Ada a badge: which region will change, and why will the intersection grow?</p>
    <Choice label="Selection rule" value={operation} choices={setOperations} onChange={setOperation} />
    <div className="sets-logic-two"><div>
      <svg className="sets-logic-svg" viewBox="0 0 320 285" role="img" aria-label={`Roster universe with training and badge regions. Selected: ${setText(result.selected)}.`}>
        <rect x="2" y="2" width="316" height="281" fill="none" stroke="#5e7084" />
        <text x="14" y="24">U</text><text x="100" y="28">T</text><text x="213" y="28">B</text>
        <circle cx="115" cy="122" r="92" fill="#88b3d0" fillOpacity=".08" stroke="#88b3d0" />
        <circle cx="205" cy="122" r="92" fill="#89c7ab" fillOpacity=".08" stroke="#89c7ab" />
        {result.members.map(member => {
            const [x, y] = locations[member.name];
            return <g key={member.name}><rect x={x - 30} y={y - 17} width="60" height="23" rx="3" fill={member.selected ? '#533f1c' : '#101822'} stroke={member.selected ? '#ecc36f' : 'none'} /><text x={x} y={y} textAnchor="middle" className={member.selected ? 'sets-logic-amber' : ''}>{member.name}</text></g>;
          })}
        <text x="160" y="237" textAnchor="middle">outside both circles</text>
      </svg><p className="sets-logic-note">Position encodes membership; circle area does not encode how many people belong. Amber names satisfy the selected rule.</p>
    </div><div><table className="sets-logic-members"><caption>Edit the actual memberships</caption><thead><tr><th>Person</th><th>In T</th><th>In B</th></tr></thead><tbody>{rosterNames.map((name, index) => <tr key={name}><th scope="row">{name}</th><td><button type="button" aria-label={`${name} trained`} aria-pressed={Boolean(training & 1 << index)} onClick={() => setTraining(training ^ 1 << index)}>{training & 1 << index ? 'Yes' : 'No'}</button></td><td><button type="button" aria-label={`${name} badged`} aria-pressed={Boolean(badges & 1 << index)} onClick={() => setBadges(badges ^ 1 << index)}>{badges & 1 << index ? 'Yes' : 'No'}</button></td></tr>)}</tbody></table>
      <p>T = {setText(result.first)}<br />B = {setText(result.second)}</p><p className="sets-logic-result" role="status" aria-live="polite"><strong>{setRules[operation]}</strong><br />Selected: {setText(result.selected)}</p></div></div>
    <div className="sets-logic-actions"><button type="button" onClick={() => {
        setTraining(0);
        setBadges(0);
      }}>Make both empty</button><button type="button" onClick={() => {
        setTraining(15);
        setBadges(15);
      }}>Put everyone in both</button><button type="button" onClick={() => {
        setTraining(3);
        setBadges(6);
        setOperation('intersection');
      }}>Reset</button></div>
    <p>Transfer: select U ∖ T, then make T empty. Why is the result everyone rather than nobody?</p>
  </section>;
}
export function ImplicationFigure() {
  return <figure className="sets-logic-figure"><div className="sets-logic-solution-sets">
    <div><strong>Original: x = −2</strong><span className="sets-logic-chip">−2</span><small>one permitted value</small></div>
    <div className="sets-logic-arrow"><span>square both sides</span><strong>→</strong><span>every old solution survives</span></div>
    <div><strong>New: x² = 4</strong><span><span className="sets-logic-chip">−2</span> <span className="sets-logic-chip sets-logic-extra">+2</span></span><small>an extra value is admitted</small></div>
  </div><figcaption>The original solution set is contained in the new one. At x=2, the squared equation is true and the original is false: a concrete failure of the reverse arrow. Restricting x≤0 restores equivalence for these two equations.</figcaption></figure>;
}
const argumentPresets = {
  consequent: {
    premises: ['implication', 'q'],
    conclusion: 'p'
  },
  ponens: {
    premises: ['implication', 'p'],
    conclusion: 'q'
  },
  tollens: {
    premises: ['implication', 'notQ'],
    conclusion: 'notP'
  },
  contradiction: {
    premises: ['p', 'notP'],
    conclusion: 'q'
  },
  equivalence: {
    premises: ['implication'],
    conclusion: 'contrapositive'
  }
};
export function TruthArgumentLab() {
  const [premises, setPremises] = useState(argumentPresets.consequent.premises);
  const [conclusion, setConclusion] = useState('p');
  const [preset, setPreset] = useState('consequent');
  const result = inspectArgument(premises, conclusion);
  function load(name) {
    setPreset(name);
    setPremises(argumentPresets[name].premises);
    setConclusion(argumentPresets[name].conclusion);
  }
  return <section className="sets-logic-lab" aria-label="Truth assignments and argument validity investigation">
    <h3>Try to keep the premises true and break the conclusion</h3><p>Read P as “the person is trained” and Q as “the person has a badge.” A countermodel is a possible truth assignment where every premise holds but the conclusion fails. Predict whether a badge alone establishes training.</p>
    <Choice label="Argument to investigate" value={preset} choices={[["consequent", 'Affirming the consequent'], ['ponens', 'Modus ponens'], ['tollens', 'Modus tollens'], ['contradiction', 'Inconsistent premises'], ['equivalence', 'Equivalent contrapositive'], ['custom', 'Custom premises']]} onChange={name => {
      if (name !== 'custom') load(name);else setPreset(name);
    }} />
    <fieldset><legend>Assume these premises together</legend><div className="sets-logic-actions">{['implication', 'p', 'q', 'notP', 'notQ'].map(name => <button type="button" key={name} aria-pressed={premises.includes(name)} onClick={() => {
          setPreset('custom');
          setPremises(premises.includes(name) ? premises.filter(item => item !== name) : [...premises, name]);
        }}>{propositionLabels[name]}</button>)}</div></fieldset>
    <Choice label="Proposed conclusion" value={conclusion} choices={Object.entries(propositionLabels)} onChange={name => {
      setConclusion(name);
      setPreset('custom');
    }} />
    <div className="sets-logic-worlds">{result.worlds.map(world => <article key={`${world.p}-${world.q}`} className={world.counterexample ? 'sets-logic-world sets-logic-countermodel' : world.admitted ? 'sets-logic-world sets-logic-admitted' : 'sets-logic-world'}>
      <strong>P {truthText(world.p)} · Q {truthText(world.q)}</strong><p>{world.admitted ? 'Admitted by every premise' : 'Excluded by a premise'}</p><ul>{premises.map((name, index) => <li key={name}>{propositionLabels[name]}: {truthText(world.premiseValues[index])}</li>)}</ul>{premises.length === 0 && <p>No premises exclude this world.</p>}<p>Conclusion {propositionLabels[conclusion]}: <strong>{truthText(world.conclusion)}</strong></p>{world.counterexample && <strong>Countermodel</strong>}
    </article>)}</div>
    <p className="sets-logic-result" role="status" aria-live="polite">{!result.consistent ? 'No world satisfies all the premises. The conclusion is entailed vacuously; this does not establish a real case with those premises.' : result.valid ? 'Valid argument: every admitted world satisfies the conclusion.' : `${result.counterexamples.length} countermodel${result.counterexamples.length === 1 ? '' : 's'}: the premises do not establish this conclusion.`}</p>
    <button type="button" onClick={() => load('consequent')}>Reset</button><p>This exhausts two Boolean variables exactly. It does not check an arbitrary proof about integers, causal explanations or the truth of the premises in a real system.</p>
  </section>;
}
const jobNames = ['Scan', 'Label', 'Audit'];
const reviewerNames = ['Ada', 'Bo', 'Cam'];
const initialBoard = () => [[true, false, false], [false, true, false], [false, false, true]];
const quantifierModes = [['each', 'Every job has some reviewer'], ['common', 'One reviewer covers every job'], ['notEach', 'Some job has no reviewer'], ['notCommon', 'Every reviewer misses some job']];
const quantifierNotation = {
  each: '∀j ∈ J, ∃r ∈ R: A(j,r)',
  common: '∃r ∈ R, ∀j ∈ J: A(j,r)',
  notEach: '∃j ∈ J, ∀r ∈ R: ¬A(j,r)',
  notCommon: '∀r ∈ R, ∃j ∈ J: ¬A(j,r)'
};
export function QuantifierWitnessLab() {
  const [matrix, setMatrix] = useState(initialBoard);
  const [rowCount, setRowCount] = useState(3);
  const [columnCount, setColumnCount] = useState(3);
  const [mode, setMode] = useState('each');
  const result = inspectQuantifiers(matrix, rowCount, columnCount);
  const truth = {
    each: result.eachHasSomeone,
    common: result.someoneCoversAll,
    notEach: !result.eachHasSomeone,
    notCommon: !result.someoneCoversAll
  }[mode];
  const common = result.commonWitnesses[0];
  function reset() {
    setMatrix(initialBoard());
    setRowCount(3);
    setColumnCount(3);
    setMode('each');
  }
  return <section className="sets-logic-lab" aria-label="Quantifier order and witnesses investigation">
    <h3>Pick separately, or commit to one reviewer first?</h3><p>The diagonal starting board gives each job its own reviewer. Predict what happens when the claim demands one common reviewer. Click a cell to change the actual assignment.</p>
    <Choice label="Quantified claim" value={mode} choices={quantifierModes} onChange={setMode} /><p className="sets-logic-formula">{quantifierNotation[mode]}</p>
    <div className="sets-logic-two"><Choice label="Jobs in the domain" value={rowCount} choices={[0, 1, 2, 3].map(n => [n, String(n)])} onChange={value => setRowCount(Number(value))} /><Choice label="Reviewers in the domain" value={columnCount} choices={[0, 1, 2, 3].map(n => [n, String(n)])} onChange={value => setColumnCount(Number(value))} /></div>
    <table className="sets-logic-board"><caption>A(j,r): this job has this reviewer</caption><thead><tr><th>Job</th>{result.columns.map(column => <th key={column}>{reviewerNames[column]}</th>)}</tr></thead><tbody>{result.rows.map(row => <tr key={row}><th scope="row">{jobNames[row]}</th>{result.columns.map(column => {
            const highlighted = mode === 'each' ? column === result.rowWitnesses[row][0] : mode === 'common' ? column === common : mode === 'notEach' ? row === result.rowFailures[0] : row === result.columnFailures[column];
            return <td key={column}><button type="button" className={highlighted ? 'sets-logic-witness' : ''} aria-pressed={matrix[row][column]} aria-label={`${jobNames[row]} has reviewer ${reviewerNames[column]}`} onClick={() => setMatrix(matrix.map((values, index) => index === row ? values.map((value, j) => j === column ? !value : value) : values))}>{matrix[row][column] ? 'Yes' : 'No'}</button></td>;
          })}</tr>)}</tbody></table>
    {rowCount === 0 && <p>No jobs are selected. A universal claim about all these jobs has no failing job.</p>}{columnCount === 0 && <p>No reviewers are selected. An existential choice of a reviewer is impossible.</p>}
    <p className="sets-logic-result" role="status" aria-live="polite"><strong>This claim is {truthText(truth)}.</strong></p>
    <div className="sets-logic-two"><div><strong>Witnesses available separately</strong><ul>{result.rows.map(row => <li key={row}>{jobNames[row]}: {result.rowWitnesses[row].length ? result.rowWitnesses[row].map(column => reviewerNames[column]).join(', ') : 'no reviewer'}</li>)}</ul>{!rowCount && <p>No row obligations.</p>}</div><div><strong>Reviewers who cover all jobs</strong><p>{setText(result.commonWitnesses.map(column => reviewerNames[column]))}</p><strong>Obstructions to one common reviewer</strong><ul>{result.columns.map(column => <li key={column}>{reviewerNames[column]}: {result.columnFailures[column] === null ? 'no missed job' : `misses ${jobNames[result.columnFailures[column]]}`}</li>)}</ul>{!columnCount && <p>No candidate reviewer exists.</p>}</div></div>
    <div className="sets-logic-actions"><button type="button" onClick={() => setMatrix(matrix.map(row => row.map((value, column) => column === 1 ? true : value)))}>Let Bo review every job</button><button type="button" onClick={() => setMatrix(Array.from({
        length: 3
      }, () => Array(3).fill(false)))}>Clear assignments</button><button type="button" onClick={reset}>Reset</button></div>
    <p>Transfer: select zero jobs and then zero reviewers. Explain why “every job has some reviewer” is true but “one reviewer covers every job” is false when both domains are empty.</p>
  </section>;
}
export function ProofScopeFigure() {
  return <figure className="sets-logic-figure"><div className="sets-logic-proof-scope"><p><strong>Keep n arbitrary, but assume it is an odd integer.</strong></p><ol><li>Definition of odd: there is an integer k with n=2k+1.</li><li>Expand: n²=4k²+4k+1.</li><li>Rewrite: n²=2(2k²+2k)+1.</li><li>The quantity 2k²+2k is an integer, so n² is odd.</li></ol></div><div className="sets-logic-discharge"><span aria-hidden="true">↓</span><p>Close the local assumption:<br /><strong>For every integer n, odd n implies odd n².</strong></p></div><figcaption>k is allowed to depend on the arbitrary n. The argument never assumes that every n is odd, and it never concludes the unconditional statement “every square is odd.”</figcaption></figure>;
}
export function FunctionImageFigure() {
  return <figure className="sets-logic-figure"><svg className="sets-logic-svg" viewBox="0 0 320 245" role="img" aria-label="The square function sends both minus two and plus two to four, although the two singleton input sets are disjoint.">
    <text x="56" y="27" textAnchor="middle">inputs</text><text x="262" y="27" textAnchor="middle">outputs</text>
    <line x1="91" y1="79" x2="232" y2="125" stroke="#eac16f" strokeWidth="2" /><line x1="91" y1="177" x2="232" y2="131" stroke="#89c7ab" strokeWidth="2" />
    <path d="M221 117 L233 125 L219 127 M220 129 L233 131 L224 140" fill="none" stroke="#d5e1e9" />
    <circle cx="60" cy="78" r="30" className="sets-logic-node" /><circle cx="60" cy="177" r="30" className="sets-logic-node" /><circle cx="265" cy="127" r="30" className="sets-logic-node" />
    <text x="60" y="85" textAnchor="middle">−2</text><text x="60" y="184" textAnchor="middle">+2</text><text x="265" y="134" textAnchor="middle">4</text><text x="165" y="233" textAnchor="middle">f(x)=x² merges these inputs</text>
  </svg><figcaption>A={'{−2}'} and B={'{2}'} are disjoint, but f(A)=f(B)={'{4}'}. Following an output back finds a set of inputs; it need not produce one inverse-function value.</figcaption></figure>;
}
function OrderDiagram({
  result,
  labels
}) {
  if (!result.partialOrder) return null;
  const maximum = Math.max(0, ...result.levels);
  const positions = result.levels.map((level, index) => {
    const peers = result.levels.flatMap((candidate, other) => candidate === level ? [other] : []);
    return [(peers.indexOf(index) + 1) * 320 / (peers.length + 1), maximum ? 270 - level * 220 / maximum : 155];
  });
  return <svg className="sets-logic-svg" viewBox="0 0 320 310" role="img" aria-label="Hasse diagram: move upward along cover edges; loops and transitive edges are omitted">{result.covers.map(([first, second]) => <line key={`${first}-${second}`} x1={positions[first][0]} y1={positions[first][1]} x2={positions[second][0]} y2={positions[second][1]} stroke="#9eb5c9" strokeWidth="2" />)}{labels.map((label, index) => <g key={label}><circle cx={positions[index][0]} cy={positions[index][1]} r="19" className="sets-logic-node" /><text x={positions[index][0]} y={positions[index][1] + 6} textAnchor="middle">{label}</text></g>)}<text x="160" y="306" textAnchor="middle">order goes upward</text></svg>;
}
function FailureDiagram({
  property,
  witness,
  labels
}) {
  const marker = useId().replace(/:/g, '');
  if (!witness) return <p>This property has no failing tuple in the displayed finite relation.</p>;
  const unique = [...new Set(witness)];
  const positions = unique.map((_, index) => [unique.length === 1 ? 160 : 55 + index * 210 / (unique.length - 1), 90]);
  const position = index => positions[unique.indexOf(index)];
  let edges;
  if (property === 'reflexive') edges = [[witness[0], witness[0], false]];else if (property === 'symmetric') edges = [[witness[0], witness[1], true], [witness[1], witness[0], false]];else if (property === 'antisymmetric') edges = [[witness[0], witness[1], true], [witness[1], witness[0], true]];else edges = [[witness[0], witness[1], true], [witness[1], witness[2], true], [witness[0], witness[2], false]];
  return <svg className="sets-logic-svg" viewBox="0 0 320 205" role="img" aria-label={`${property} failure at ${witness.map(index => labels[index]).join(', ')}; solid arrows are present, dashed arrows are missing`}><defs><marker id={marker} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#e8bf77" /></marker></defs>{edges.map(([first, second, present], index) => {
      const [x1, y1] = position(first);
      const [x2, y2] = position(second);
      const path = first === second ? `M ${x1 - 14} ${y1 - 15} C ${x1 - 58} ${y1 - 75},${x1 + 58} ${y1 - 75},${x1 + 14} ${y1 - 15}` : index === 2 ? `M ${x1} ${y1 + 22} Q 160 178 ${x2} ${y2 + 22}` : `M ${x1 + (x2 > x1 ? 24 : -24)} ${y1 + (index === 1 && property !== 'transitive' ? 9 : -9)} L ${x2 + (x2 > x1 ? -25 : 25)} ${y2 + (index === 1 && property !== 'transitive' ? 9 : -9)}`;
      return <path key={index} d={path} fill="none" stroke={present ? '#e8bf77' : '#eda5a6'} strokeDasharray={present ? undefined : '5 5'} strokeWidth="2" markerEnd={`url(#${marker})`} />;
    })}{unique.map((index, positionIndex) => <g key={index}><circle cx={positions[positionIndex][0]} cy="90" r="22" className="sets-logic-node" /><text x={positions[positionIndex][0]} y="97" textAnchor="middle">{labels[index]}</text></g>)}<text x="160" y="194" textAnchor="middle">solid: present · dashed: missing</text></svg>;
}
const relationChoices = [['moduloThree', 'Same remainder modulo 3'], ['nearby', 'Distance at most 1'], ['identity', 'Equality only'], ['divisors', 'Divisibility: 1,2,3,6'], ['noLeast', 'Divisibility: 2,3,6'], ['custom', 'Edited relation']];
export function RelationPropertiesLab() {
  const [preset, setPreset] = useState('nearby');
  const [relation, setRelation] = useState(() => relationPreset('nearby'));
  const [property, setProperty] = useState('transitive');
  const result = useMemo(() => inspectRelation(relation.labels.length, relation.pairs), [relation]);
  function load(name) {
    setPreset(name);
    if (name !== 'custom') setRelation(relationPreset(name));
  }
  function toggle(first, second) {
    setPreset('custom');
    setRelation({
      ...relation,
      pairs: result.matrix[first][second] ? relation.pairs.filter(pair => pair[0] !== first || pair[1] !== second) : [...relation.pairs, [first, second]]
    });
  }
  const labelsFor = indices => setText(indices.map(index => relation.labels[index]));
  return <section className="sets-logic-lab" aria-label="Relation properties and grouping investigation"><h3>A plausible grouping still has to satisfy its axioms</h3><p>In the starting relation, 0 is close to 1 and 1 is close to 2. Predict the missing transitivity obligation. Click a pair to change exactly that relationship; adding a reverse pair is a separate edit.</p>
    <Choice label="Relation preset" value={preset} choices={relationChoices} onChange={load} />
    <div className="sets-logic-matrix-scroll" tabIndex="0" role="region" aria-label="Editable relation matrix; scroll horizontally if necessary"><table className="sets-logic-board"><caption>Row x, column y: does x R y hold?</caption><thead><tr><th>x / y</th>{relation.labels.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{relation.labels.map((label, first) => <tr key={label}><th scope="row">{label}</th>{relation.labels.map((other, second) => <td key={other}><button type="button" aria-label={`Relation ${label} to ${other}`} aria-pressed={result.matrix[first][second]} onClick={() => toggle(first, second)}>{result.matrix[first][second] ? '1' : '0'}</button></td>)}</tr>)}</tbody></table></div>
    <p className="sets-logic-note">1 means the pair is present; 0 means absent. This table may pan on a small screen. It contains every pair, including loops.</p>
    <dl className="sets-logic-properties">{Object.entries(result.properties).map(([name, holds]) => <div key={name}><dt>{name}</dt><dd>{holds ? 'holds' : `fails at (${result.failures[name].map(index => relation.labels[index]).join(', ')})`}</dd></div>)}</dl>
    <div className="sets-logic-two"><div><Choice label="Inspect an axiom" value={property} choices={Object.keys(result.properties).map(name => [name, name])} onChange={setProperty} /><FailureDiagram property={property} witness={result.failures[property]} labels={relation.labels} />{result.failures[property] && <p>{property === 'transitive' ? 'The two present steps require a direct pair between their endpoints. That required pair is missing.' : property === 'symmetric' ? 'A present pair requires its reverse. That reverse is missing.' : property === 'reflexive' ? 'This element needs a pair to itself; its loop is missing.' : 'Two distinct elements relate in both directions. Antisymmetry permits that only when the elements are equal.'}</p>}</div>
    <div><strong>Can it define equivalence classes?</strong>{result.equivalence ? <><p>Yes: reflexive, symmetric and transitive all hold.</p><div className="sets-logic-classes">{result.classes.map((members, index) => <div key={index}><span>one class</span><strong>{labelsFor(members)}</strong></div>)}</div></> : <p>No. Listing neighborhoods would not establish a partition into equivalence classes.</p>}
      {result.partialOrder && <><strong>It is also a partial order.</strong><OrderDiagram result={result} labels={relation.labels} /><p>Minimal: {labelsFor(result.minimal)}<br />Least: {labelsFor(result.least)}<br />Maximal: {labelsFor(result.maximal)}<br />Greatest: {labelsFor(result.greatest)}</p><p className="sets-logic-note">Only covers are drawn upward. Reflexive and transitive pairs remain in the matrix. Horizontal spacing has no numerical meaning.</p></>}
    </div></div><button type="button" onClick={() => {
      load('nearby');
      setProperty('transitive');
    }}>Reset</button><p>Transfer: choose equality only. Why can that relation be both symmetric and antisymmetric? Then choose divisibility on 2,3,6 and distinguish a minimal element from a least element.</p>
  </section>;
}
export function PartialOrderFigure() {
  const relation = relationPreset('noLeast');
  const result = inspectRelation(relation.labels.length, relation.pairs);
  return <figure className="sets-logic-figure"><OrderDiagram result={result} labels={relation.labels} /><figcaption>Under divisibility, neither 2 nor 3 has a different element below it, so both are minimal. Neither divides the other, so neither is least. Both divide 6, the greatest element. The lines show covers; each number also relates to itself.</figcaption></figure>;
}
export function OddSquareLab() {
  const [size, setSize] = useState(3);
  const result = oddSquareStep(size);
  const cellWidth = 272 / (size + 1);
  return <section className="sets-logic-lab" aria-label="Induction square border investigation"><h3>One border makes the next square</h3><p>Predict the number of added tiles before moving forward. Count a row of n+1 tiles and a column of n more: the corner is counted once.</p>
    <div className="sets-logic-actions"><button type="button" disabled={size === 0} onClick={() => setSize(size - 1)}>Back</button><strong>n = {size}</strong><button type="button" disabled={size === 7} onClick={() => setSize(size + 1)}>Next square</button><button type="button" onClick={() => setSize(3)}>Reset</button></div>
    <svg className="sets-logic-svg" viewBox="0 0 320 320" role="img" aria-label={`${result.before} old tiles plus ${result.added} border tiles make ${result.after} tiles`}>
      {result.cells.map(cell => <rect key={`${cell.row}-${cell.column}`} x={24 + cell.column * cellWidth} y={18 + cell.row * cellWidth} width={cellWidth} height={cellWidth} fill={cell.added ? '#99713a' : '#2b4c63'} stroke="#c1cdd9" strokeWidth="1" />)}<text x="160" y="312" textAnchor="middle">blue: n² · amber: 2n+1</text>
    </svg><p className="sets-logic-result" role="status" aria-live="polite">{result.before} + {result.added} = {result.after}<br />n² + (2n+1) = (n+1)²</p><p>The display visits only n=0 through 7. The written proof establishes the same algebra for an arbitrary nonnegative integer n and supplies the base case; these eight pictures alone do not.</p>
  </section>;
}
const diagonalStart = () => [[true, false, true, false], [true, false, false, false], [false, true, true, true], [true, true, false, false]];
export function DiagonalSubsetLab() {
  const [matrix, setMatrix] = useState(diagonalStart);
  const [revealed, setRevealed] = useState(false);
  const [selected, setSelected] = useState(0);
  const result = diagonalSubset(matrix);
  const witness = result.differences[selected];
  return <section className="sets-logic-lab" aria-label="Diagonal missing subset investigation"><h3>Construct the subset that defeats this entire proposed list</h3><p>Row i proposes f(i), a subset of U={'{0,1,2,3}'}. The highlighted diagonal asks whether i belongs to its own proposed subset. Predict D by reversing those four answers.</p>
    <table className="sets-logic-board sets-logic-diagonal"><caption>Candidate subsets: click a membership to change it</caption><thead><tr><th>row</th>{[0, 1, 2, 3].map(index => <th key={index}>{index}</th>)}</tr></thead><tbody>{matrix.map((row, first) => <tr key={first}><th scope="row">f({first})</th>{row.map((included, second) => <td key={second}><button type="button" className={first === second ? 'sets-logic-witness' : ''} aria-label={`Element ${second} in subset f(${first})`} aria-pressed={included} onClick={() => setMatrix(matrix.map((values, index) => index === first ? values.map((value, other) => other === second ? !value : value) : values))}>{included ? '1' : '0'}</button></td>)}</tr>)}</tbody></table>
    <div className="sets-logic-actions"><button type="button" onClick={() => setRevealed(true)}>Reveal the diagonal subset</button><button type="button" onClick={() => {
        setMatrix(diagonalStart());
        setRevealed(false);
        setSelected(0);
      }}>Reset</button></div>
    {revealed && <><div className="sets-logic-constructed"><strong>D = {setText(result.members)}</strong><div>{result.subset.map((included, index) => <span key={index}>{index}: {included ? 'in' : 'out'}</span>)}</div></div><Choice label="Compare D with this row" value={selected} choices={[0, 1, 2, 3].map(index => [index, `f(${index})`])} onChange={value => setSelected(Number(value))} /><p className="sets-logic-result" role="status" aria-live="polite">At element {selected}:<br />f({selected}) says {witness.proposed ? 'in' : 'out'}, while D says {witness.constructed ? 'in' : 'out'}.<br />Therefore D ≠ f({selected}).</p></>}
    <p>Changing a diagonal entry changes D in the opposite direction. Changing any other entry cannot remove that row's diagonal disagreement. The general proof uses this same contradiction for an arbitrary index; it does not store an infinite table in the browser.</p>
  </section>;
}
