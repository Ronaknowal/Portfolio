import { useMemo, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { CrfBar, CrfPathLedger, number } from './CrfFigures.jsx';
import { chainDistribution, factorPresets, independentDistribution, initialFactors, normalizationBranches, sameWinners, validFactors } from '../../data/crf-models.js';

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
  const initial = initialFactors.map(String);
  const [draft, setDraft] = useState(initial);
  
  
  
  
  const [selected, setSelected] = useState('AB');
  
  const valid = validFactors(draft);
  const result = useMemo(() => valid ? { model: chainDistribution(draft), independent: independentDistribution(draft) } : null, [draft, valid]);
  const model = result?.model;

  const edit = setDraft;
  
  return <section className="crf-lab" data-live-exploration="crf" data-crf-lab="trellis" aria-label="Edit a CRF trellis">
    <h3>Investigate: can a pair preference reverse a local choice?</h3>
    <p>Change dimensionless factors in [0.125, 16]. The independent comparison uses the same four input factors and pair factors 1. Compare the <strong>full sets of highest-scoring paths</strong>, including ties. Numerical ties use relative tolerance 10⁻¹².</p>
    <p>Change the factors and compare the two mechanisms live. Presets provide useful starting points for your own edits.</p>
    <div className="crf-controls">{Object.entries(factorPresets).map(([name, factors]) => <button type="button" key={name} onClick={() => edit(factors.map(String))}>{name}</button>)}</div>
    <fieldset><legend>Draft input factors</legend><div className="crf-factor-inputs">{['Position 1 · A', 'Position 1 · B', 'Position 2 · A', 'Position 2 · B'].map((name, index) => <label key={name}>{name}<input type="number" min="0.125" max="16" step="any" value={draft[index]} aria-invalid={!validFactors([draft[index]])} onChange={event => edit(draft.map((value, i) => i === index ? event.target.value : value))} />{!validFactors([draft[index]]) && <span className="crf-error">Enter a factor from 0.125 to 16.</span>}</label>)}</div></fieldset>
    <fieldset><legend>Draft pair factors · previous → current</legend><div className="crf-factor-inputs">{['A → A', 'A → B', 'B → A', 'B → B'].map((name, index) => <label key={name}>{name}<input type="number" min="0.125" max="16" step="any" value={draft[index + 4]} aria-invalid={!validFactors([draft[index + 4]])} onChange={event => edit(draft.map((value, i) => i === index + 4 ? event.target.value : value))} />{!validFactors([draft[index + 4]]) && <span className="crf-error">Enter a factor from 0.125 to 16.</span>}</label>)}</div></fieldset>
    
    <button type="button" onClick={() => { setDraft(initial); }}>Reset</button>
    
    <p role="status">{valid ? "The trellis and probabilities update with every valid factor edit." : "Correct the marked inputs; no calculation is shown for invalid factors."}</p>
    {result && <div className="crf-result"><strong>Applied factors</strong>
      <p>Independent winners: <strong>{result.independent.winners.join(', ')}</strong> (Z = {number(result.independent.partition)}). Chain winners: <strong>{model.winners.join(', ')}</strong> (Z = {number(model.partition)}). {sameWinners(model, result.independent) ? 'The largest-path sets agree; compare their probabilities separately.' : 'The pair contributions change which complete paths have the largest product.'}</p>
      <Trellis model={model} selected={selected} /><CrfPathLedger model={model} selected={selected} onSelect={setSelected} />
      <LessonTable caption="Incoming masses: summation and maximization answer different questions" headers={['Final label', 'Forward sum', 'Largest mass', 'Best predecessor(s)']} rows={['A', 'B'].map((label, index) => [label, number(model.forward[index]), number(model.maximum[index]), model.parents[index]])} />
      <p>Adding forward cells gives Z = {number(model.partition)}. Adding largest incoming masses gives {number(model.maximum[0] + model.maximum[1])}; it discards all other prefixes.</p>
    </div>}
    <details><summary>Independent transfer: reverse the disagreement</summary><p>Construct a case where independent decoding prefers BA but the chain prefers BB. Use your own factors and compare the winning paths. Alternatively, deliberately create a tie: first decide whether the two full maximizing sets should agree. A tie alone does not require them to differ.</p><details><summary>One construction, after trying</summary><p>First inputs [1,3], second [2,1], and BB pair factor 16 with all other pair factors 1: independent BA mass 6; structured BB mass 48.</p></details></details>
  </section>;
}

export function CrfLabelBiasLab() {
  const initial = ['.5', '.01', '1'];
  const [draft, setDraft] = useState(initial);
  
  
  
  
  

  const valid = validFactors([draft[0]], .05, .95) && validFactors(draft.slice(1), .001, 10);
  const edit = setDraft;
  
  const result = useMemo(() => valid ? { model: normalizationBranches(...draft), factors: draft.map(Number) } : null, [draft, valid]);
  const model = result?.model;
  const [p, a, b] = result?.factors ?? initial.map(Number);
  return <section className="crf-lab" data-live-exploration="crf" data-crf-lab="bias" aria-label="Local and global normalization investigation">
    <h3>Investigate: where does the later evidence disappear?</h3>
    <p>Each branch has one exit. Keep the first classifier fixed while changing later compatibility. Observe route A’s probability when moving from local to global normalization.</p>
    <p>Change the factors and compare the two mechanisms live. Presets provide useful starting points for your own edits.</p>
    <div className="crf-controls"><button type="button" onClick={() => edit(['.5', '1', '.01'])}>Reverse evidence</button><button type="button" onClick={() => edit(['.5', '1', '1'])}>Equal evidence</button><button type="button" onClick={() => edit(['.3', '1', '1'])}>Equal evidence, changed prior</button></div>
    <div className="crf-factor-inputs">{['First-branch probability p', 'Later A factor', 'Later B factor'].map((label, index) => {
      const minimum = index ? .001 : .05, maximum = index ? 10 : .95;
      return <label key={label}>{label}<input type="number" min={minimum} max={maximum} step="any" value={draft[index]} aria-invalid={!validFactors([draft[index]], minimum, maximum)} onChange={event => edit(draft.map((value, i) => i === index ? event.target.value : value))} />{!validFactors([draft[index]], minimum, maximum) && <span className="crf-error">Use {minimum} to {maximum}.</span>}</label>;
    })}</div>
    
    <button type="button" onClick={() => { setDraft(initial); }}>Reset</button>
    
    <p role="status">{valid ? "The trellis and probabilities update with every valid factor edit." : "Correct the marked inputs; no calculation is shown for invalid factors."}</p>
    {result && <div className="crf-result"><strong>Applied routes</strong>
      <div className="crf-two-panel"><div><h4>Private denominators</h4><div className="crf-route-formula">A: {number(p)} × ({number(a)} / {number(a)}) = {number(model.local[0])}</div><div className="crf-route-formula">B: {number(1 - p)} × ({number(b)} / {number(b)}) = {number(model.local[1])}</div><CrfBar label="Local route A" value={model.local[0]} /><CrfBar label="Local route B" value={model.local[1]} /></div><div><h4>One shared denominator</h4><div className="crf-route-formula">A: {number(p)} × {number(a)} = {number(model.masses[0])}</div><div className="crf-route-formula">B: {number(1 - p)} × {number(b)} = {number(model.masses[1])}</div><p>Z = {number(model.partition)}</p><CrfBar label="Global route A" value={model.global[0]} /><CrfBar label="Global route B" value={model.global[1]} /></div></div>
      <p>Change in route A: <strong>{(model.global[0] - model.local[0]).toPrecision(6)}</strong>. Probability labels are rounded to six decimal places; changes within 10⁻¹⁰ count as unchanged.</p>
      <p>All bars use probability 0–1. At a private one-exit denominator, its compatibility cancels. At the shared denominator, the two complete path masses compete. Equal compatibility preserves the prior.</p>
    </div>}
    <details><summary>Independent transfer: restore a 50–50 outcome</summary><p>Set p = 0.25. Find valid compatibility factors giving global route A probability 0.5, while its local probability stays 0.25. Apply your own values and inspect the two results.</p><details><summary>Reveal the relation</summary><p>0.25a = 0.75b, so a = 3b; a = 3 and b = 1 is one solution.</p></details></details>
  </section>;
}
