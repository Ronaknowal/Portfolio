import { useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { CrfBar, CrfPathLedger, number } from './CrfFigures.jsx';
import { chainDistribution, factorPresets, independentDistribution, initialFactors, normalizationBranches, sameWinners, validFactors } from '../../data/crf-models.js';

// Numeric keys prevent formatting-only edits from turning a revealed example into a fresh prediction.
const factorKey = factors => JSON.stringify(factors.map(Number));

function Trellis({ model, selected }) {
  const selectedPath = model.paths.find(path => path.name === selected);
  return <div>
    <svg className="crf-trellis" viewBox="0 0 280 220" role="img" aria-label={`Two-position trellis, selected path ${selected}. Four pair edges connect the two A/B rows.`}>
      <text x="22" y="20">Position 1</text><text x="22" y="205">Position 2</text>
      {model.paths.map(path => <line key={path.name} x1={path.i ? 210 : 70} y1="60" x2={path.j ? 210 : 70} y2="160" style={{ stroke: selected === path.name ? '#e2bb5c' : '#647468', strokeWidth: selected === path.name ? 3 : 1, strokeDasharray: selected === path.name ? undefined : '5 5' }} />)}
      {[0, 1].flatMap(position => ['A', 'B'].map((label, index) => <g key={`${position}${label}`}><circle cx={index ? 210 : 70} cy={position ? 160 : 60} r="23" /><text x={index ? 210 : 70} y={position ? 165 : 65} textAnchor="middle">{label}</text></g>))}
    </svg>
    <p>Selected path <strong>{selected}</strong>: first input {number(selectedPath.factors[0])} × pair {number(selectedPath.factors[1])} × second input {number(selectedPath.factors[2])} = <strong>{number(selectedPath.mass)}</strong>. Gold solid edge is selected; dashed edges are alternatives.</p>
  </div>;
}

export function CrfTrellisLab() {
  const [draft, setDraft] = useState(initialFactors.map(String));
  const [prediction, setPrediction] = useState('');
  const [commitment, setCommitment] = useState(null);
  const [result, setResult] = useState(null);
  const [revealedInputs, setRevealedInputs] = useState(() => new Set([factorKey(initialFactors)]));
  const [selected, setSelected] = useState('AB');
  const [notice, setNotice] = useState('Make a prediction or explore without grading.');
  const valid = validFactors(draft);
  const model = result?.model ?? chainDistribution(initialFactors);
  const key = factorKey(draft);
  const fresh = !revealedInputs.has(key);
  const edit = next => { setDraft(next); setPrediction(''); setCommitment(null); setResult(null); setNotice('Inputs changed. Previous output retired; predict before applying a fresh input.'); };
  function calculate(graded) {
    if (!valid || (graded && (!fresh || commitment?.key !== key))) return;
    const current = chainDistribution(draft);
    const independent = independentDistribution(draft);
    const actual = sameWinners(current, independent) ? 'same' : 'different';
    setResult({ model: current, independent, prediction: graded ? commitment.prediction : null, actual, key });
    setRevealedInputs(previous => new Set([...previous, key]));
    setNotice(graded ? `${commitment.prediction === actual ? 'Prediction matched' : 'Prediction differed'}: the highest-scoring path sets are ${actual}.` : `Exploration: the highest-scoring path sets are ${actual}. No prediction was graded.`);
  }
  return <section className="crf-lab" data-crf-lab="trellis" aria-label="Edit a CRF trellis">
    <h3>Investigate: can a pair preference reverse a local choice?</h3>
    <p>Change dimensionless factors in [0.125, 16]. The independent comparison uses the same four input factors and pair factors 1. Compare the <strong>full sets of highest-scoring paths</strong>, including ties. Numerical ties use relative tolerance 10⁻¹².</p>
    <p>The solved example above is your reference. Change at least one factor for a fresh prediction. Inputs already revealed here remain available through exploration.</p>
    <div className="crf-controls">{Object.entries(factorPresets).map(([name, factors]) => <button type="button" key={name} onClick={() => edit(factors.map(String))}>{name}</button>)}</div>
    <fieldset><legend>Draft input factors</legend><div className="crf-factor-inputs">{['Position 1 · A', 'Position 1 · B', 'Position 2 · A', 'Position 2 · B'].map((name, index) => <label key={name}>{name}<input type="number" min="0.125" max="16" step="any" value={draft[index]} aria-invalid={!validFactors([draft[index]])} onChange={event => edit(draft.map((value, i) => i === index ? event.target.value : value))} />{!validFactors([draft[index]]) && <span className="crf-error">Enter a factor from 0.125 to 16.</span>}</label>)}</div></fieldset>
    <fieldset><legend>Draft pair factors · previous → current</legend><div className="crf-factor-inputs">{['A → A', 'A → B', 'B → A', 'B → B'].map((name, index) => <label key={name}>{name}<input type="number" min="0.125" max="16" step="any" value={draft[index + 4]} aria-invalid={!validFactors([draft[index + 4]])} onChange={event => edit(draft.map((value, i) => i === index + 4 ? event.target.value : value))} />{!validFactors([draft[index + 4]]) && <span className="crf-error">Enter a factor from 0.125 to 16.</span>}</label>)}</div></fieldset>
    <label>Predict the independent and chain winning-path sets<select value={prediction} onChange={event => {setPrediction(event.target.value); setCommitment(null);}}><option value="">Choose a prediction</option><option value="same">Same set</option><option value="different">Different sets</option></select></label>
    <div className="crf-controls"><button type="button" disabled={!valid || !prediction || !fresh} onClick={() => {setCommitment({ key, prediction }); setNotice('Prediction committed for these eight factors. Calculate when ready.');}}>Commit prediction</button><button type="button" disabled={!valid || !fresh || commitment?.key !== key} onClick={() => calculate(true)}>Calculate</button><button type="button" disabled={!valid} onClick={() => calculate(false)}>Explore without a prediction</button><button type="button" onClick={() => {setDraft(initialFactors.map(String)); setPrediction(''); setCommitment(null); setResult(null); setSelected('AB'); setNotice('Reset to the initial factors; prediction and result cleared.');}}>Reset</button></div>
    {!fresh && <p className="lesson-note">These inputs have a revealed solution. Change a factor to predict a new result, or explore without grading.</p>}
    <p role="status">{notice}</p>
    {result && <div className="crf-result"><strong>Applied factors</strong>
      {result && <p>Independent winners: <strong>{result.independent.winners.join(', ')}</strong> (Z = {number(result.independent.partition)}). Chain winners: <strong>{model.winners.join(', ')}</strong> (Z = {number(model.partition)}). {result.prediction && <>Saved prediction: {result.prediction}.</>} {result.actual === 'different' ? 'The pair contributions change which complete paths have the largest product.' : 'The largest-path sets agree; compare their probabilities separately.'}</p>}
      <Trellis model={model} selected={selected} /><CrfPathLedger model={model} selected={selected} onSelect={setSelected} />
      <LessonTable caption="Incoming masses: summation and maximization answer different questions" headers={['Final label', 'Forward sum', 'Largest mass', 'Best predecessor(s)']} rows={['A', 'B'].map((label, index) => [label, number(model.forward[index]), number(model.maximum[index]), model.parents[index]])} />
      <p>Adding forward cells gives Z = {number(model.partition)}. Adding largest incoming masses gives {number(model.maximum[0] + model.maximum[1])}; it discards all other prefixes.</p>
    </div>}
    <details><summary>Independent transfer: reverse the disagreement</summary><p>Construct a case where independent decoding prefers BA but the chain prefers BB. Use your own factors, predict “different,” and calculate. Alternatively, deliberately create a tie: first decide whether the two full maximizing sets should agree. A tie alone does not require them to differ.</p><details><summary>One construction, after trying</summary><p>First inputs [1,3], second [2,1], and BB pair factor 16 with all other pair factors 1: independent BA mass 6; structured BB mass 48.</p></details></details>
  </section>;
}

export function CrfLabelBiasLab() {
  const initial = ['.5', '.01', '1'];
  const [draft, setDraft] = useState(initial);
  const [prediction, setPrediction] = useState('');
  const [commitment, setCommitment] = useState(null);
  const [result, setResult] = useState(null);
  const [revealedInputs, setRevealedInputs] = useState(() => new Set([factorKey(initial)]));
  const [notice, setNotice] = useState('Predict how route A changes under global normalization.');
  const key = factorKey(draft);
  const fresh = !revealedInputs.has(key);
  const valid = validFactors([draft[0]], .05, .95) && validFactors(draft.slice(1), .001, 10);
  const edit = next => {setDraft(next); setPrediction(''); setCommitment(null); setResult(null); setNotice('Draft changed; previous prediction and output retired. Predict before applying a fresh input.');};
  function calculate(graded) {
    if (!valid || (graded && (!fresh || commitment?.key !== key))) return;
    const model = normalizationBranches(...draft);
    setResult({ model, factors: draft.map(Number), prediction: graded ? commitment.prediction : null, key });
    setRevealedInputs(previous => new Set([...previous, key]));
    setNotice(graded ? `${commitment.prediction === model.direction ? 'Prediction matched' : 'Prediction differed'}: route A ${model.direction === 'unchanged' ? 'is unchanged' : model.direction + 's'}.` : `Exploration: route A ${model.direction === 'unchanged' ? 'is unchanged' : model.direction + 's'}. No prediction graded.`);
  }
  const model = result?.model ?? normalizationBranches(...initial);
  const [p, a, b] = result?.factors ?? initial.map(Number);
  return <section className="crf-lab" data-crf-lab="bias" aria-label="Local and global normalization investigation">
    <h3>Investigate: where does the later evidence disappear?</h3>
    <p>Each branch has one exit. Keep the first classifier fixed while changing later compatibility. Predict route A’s probability when moving from local to global normalization.</p>
    <p>The solved .5/.01/1 example above is your reference. Change a factor or choose a new preset before committing. Revealed inputs can always be explored without grading.</p>
    <div className="crf-controls"><button type="button" onClick={() => edit(['.5', '1', '.01'])}>Reverse evidence</button><button type="button" onClick={() => edit(['.5', '1', '1'])}>Equal evidence</button><button type="button" onClick={() => edit(['.3', '1', '1'])}>Equal evidence, changed prior</button></div>
    <div className="crf-factor-inputs">{['First-branch probability p', 'Later A factor', 'Later B factor'].map((label, index) => {
      const minimum = index ? .001 : .05, maximum = index ? 10 : .95;
      return <label key={label}>{label}<input type="number" min={minimum} max={maximum} step="any" value={draft[index]} aria-invalid={!validFactors([draft[index]], minimum, maximum)} onChange={event => edit(draft.map((value, i) => i === index ? event.target.value : value))} />{!validFactors([draft[index]], minimum, maximum) && <span className="crf-error">Use {minimum} to {maximum}.</span>}</label>;
    })}</div>
    <label>Predict route A’s change<select value={prediction} onChange={event => {setPrediction(event.target.value); setCommitment(null);}}><option value="">Choose a prediction</option><option value="decrease">Decrease</option><option value="unchanged">Unchanged</option><option value="increase">Increase</option></select></label>
    <div className="crf-controls"><button type="button" disabled={!valid || !prediction || !fresh} onClick={() => {setCommitment({ key, prediction }); setNotice('Prediction committed for this prior and these two factors.');}}>Commit prediction</button><button type="button" disabled={!valid || !fresh || commitment?.key !== key} onClick={() => calculate(true)}>Apply</button><button type="button" disabled={!valid} onClick={() => calculate(false)}>Explore without a prediction</button><button type="button" onClick={() => {setDraft(initial); setPrediction(''); setCommitment(null); setResult(null); setNotice('Reset: prediction and applied result cleared.');}}>Reset</button></div>
    {!fresh && <p className="lesson-note">These inputs have a revealed solution. Change a factor to predict a new result, or explore without grading.</p>}
    <p role="status">{notice}</p>
    {result && <div className="crf-result"><strong>Applied routes</strong>
      {result?.prediction && <p>Saved prediction: {result.prediction}.</p>}
      <div className="crf-two-panel"><div><h4>Private denominators</h4><div className="crf-route-formula">A: {number(p)} × ({number(a)} / {number(a)}) = {number(model.local[0])}</div><div className="crf-route-formula">B: {number(1 - p)} × ({number(b)} / {number(b)}) = {number(model.local[1])}</div><CrfBar label="Local route A" value={model.local[0]} /><CrfBar label="Local route B" value={model.local[1]} /></div><div><h4>One shared denominator</h4><div className="crf-route-formula">A: {number(p)} × {number(a)} = {number(model.masses[0])}</div><div className="crf-route-formula">B: {number(1 - p)} × {number(b)} = {number(model.masses[1])}</div><p>Z = {number(model.partition)}</p><CrfBar label="Global route A" value={model.global[0]} /><CrfBar label="Global route B" value={model.global[1]} /></div></div>
      <p>Change in route A: <strong>{(model.global[0] - model.local[0]).toPrecision(6)}</strong>. Probability labels are rounded to six decimal places; changes within 10⁻¹⁰ count as unchanged.</p>
      <p>All bars use probability 0–1. At a private one-exit denominator, its compatibility cancels. At the shared denominator, the two complete path masses compete. Equal compatibility preserves the prior.</p>
    </div>}
    <details><summary>Independent transfer: restore a 50–50 outcome</summary><p>Set p = 0.25. Find valid compatibility factors giving global route A probability 0.5, while its local probability stays 0.25. Apply your own values and inspect the two results.</p><details><summary>Reveal the relation</summary><p>0.25a = 0.75b, so a = 3b; a = 3 and b = 1 is one solution.</p></details></details>
  </section>;
}
