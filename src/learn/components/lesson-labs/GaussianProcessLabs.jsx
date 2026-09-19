import { useId, useRef, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { gaussianProcessData as data } from '../../data/gaussian-process-data.js';
import {
  compareTarget, createConditioner, directConditional, forecastFromPrefix,
  initialObservations, measurementGains, scoreForecast, spatialGrid, spatialKernel,
} from '../../data/gaussian-process-model.js';
import { ForecastResult, GPBands, GPChart, gpNumber, ProbeResults } from './GaussianProcessFigures.jsx';
import './gaussian-process.css';

function NumericField({ label, value, onChange, min, max, step = 'any' }) {
  return <label>{label}<input type="number" value={value} min={min} max={max} step={step} onChange={event => onChange(event.target.value)} /></label>;
}

function requireNumber(value, label, min, max) {
  const parsed = value === '' ? NaN : Number(value);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) throw new Error(`${label}: enter a number from ${min} to ${max}.`);
  return parsed;
}

function ObservationEditor({ rows, onChange }) {
  const nextId = () => Math.max(0, ...rows.map(row => row.id)) + 1;
  return <div><p>Measured readings (0–12). Duplicate positions mean independent noisy readings.</p>
    {rows.map((row, i) => <div className="gp-observation-row" key={row.id}>
      <NumericField label={`Reading ${i + 1} position`} value={row.x} min={-1} max={4} onChange={x => onChange(rows.map(item => item.id === row.id ? { ...item, x } : item))} />
      <NumericField label={`Reading ${i + 1} value`} value={row.y} min={-4} max={4} onChange={y => onChange(rows.map(item => item.id === row.id ? { ...item, y } : item))} />
      <button type="button" onClick={() => onChange(rows.filter(item => item.id !== row.id))}>Remove reading {i + 1}</button>
    </div>)}
    <div className="gp-actions"><button type="button" disabled={rows.length >= 12} onClick={() => onChange([...rows, { id: nextId(), x: 1, y: 0 }])}>Add reading</button><button type="button" disabled={!rows.length} onClick={() => onChange([])}>Remove all readings</button></div>
  </div>;
}

function parsedObservations(rows) {
  return rows.map((row, i) => ({ ...row, x: requireNumber(row.x, `Reading ${i + 1} position`, -1, 4), y: requireNumber(row.y, `Reading ${i + 1} value`, -4, 4) }));
}

function Prediction({ question, options, choice, setChoice, reason, setReason, onReveal, revealed, action = 'Commit prediction and reveal' }) {
  const group = useId();
  return <div className="gp-commit"><fieldset><legend>{question}</legend>{options.map(([value, label]) => <label key={value} className="gp-choice"><input type="radio" name={group} value={value} checked={choice === value} onChange={() => setChoice(value)} />{label}</label>)}</fieldset>
    <label>Reason before the result<textarea rows={2} value={reason} onChange={event => setReason(event.target.value)} placeholder="Use the covariance or evaluation mechanism." /></label>
    <div className="gp-actions"><button type="button" disabled={!choice || !reason.trim() || revealed} onClick={onReveal}>{action}</button></div>
  </div>;
}

const changes = [['mean', 'Mean only'], ['variance', 'Latent variance only'], ['both', 'Both'], ['neither', 'Neither']];
const defaultSpatial = () => ({ observations: initialObservations.map(row => ({ ...row })), target: 1, length: 1, noise: 0.25, kind: 'rbf' });
const defaultDirect = () => ({ rho: 0.5, value: 2, noise: 0.25 });

export function GaussianConditioningLab({ direct = false }) {
  const [settings, setSettings] = useState(direct ? defaultDirect : defaultSpatial);
  const [baseline, setBaseline] = useState(direct ? defaultDirect : defaultSpatial);
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const conditioner = useRef(createConditioner());
  const clear = () => { setChoice(''); setReason(''); setResult(null); setError(''); };
  const edit = next => { setSettings(next); clear(); };
  const calculate = state => {
    if (direct) return directConditional({ rho: requireNumber(state.rho, 'Covariance', -0.95, 0.95), value: requireNumber(state.value, 'Reading', -4, 4), noise: requireNumber(state.noise, 'Noise variance', 0.01, 2) });
    const rows = parsedObservations(state.observations);
    const target = requireNumber(state.target, 'Target', -1, 4);
    const length = requireNumber(state.length, 'Length', 0.1, 4);
    const noise = requireNumber(state.noise, 'Noise variance', 0.01, 2);
    const posterior = conditioner.current.predict({ x: rows.map(row => row.x), y: rows.map(row => row.y), targets: [target, ...spatialGrid], noise, kernel: spatialKernel(state.kind, length), kernelKey: `${state.kind}:${length}` });
    return { mean: posterior.mean[0], variance: posterior.variance[0], observationVariance: posterior.variance[0] + noise, posterior, target, rows, noise };
  };
  const reveal = () => {
    try {
      const before = calculate(baseline);
      const after = calculate(settings);
      setResult({ before, after, expected: compareTarget(before, after) });
      setError('');
    } catch (failure) { setResult(null); setError(failure.message); }
  };
  const reset = () => {
    setSettings(direct ? defaultDirect() : defaultSpatial());
    setBaseline(direct ? defaultDirect() : defaultSpatial());
    clear();
  };
  const baselineText = direct
    ? `ρ = ${baseline.rho}, observed value = ${baseline.value}, noise variance = ${baseline.noise}`
    : `${baseline.observations.map(row => `(${row.x}, ${row.y})`).join(', ') || 'no readings'}; target ${baseline.target}; ${baseline.kind}; length ${baseline.length}; noise variance ${baseline.noise}`;
  return <section className="gp-lab" data-gp-lab={direct ? 'direct' : 'conditioning'} aria-label={direct ? 'One-reading conditioning investigation' : 'Editable Gaussian conditioning investigation'}>
    <h3>{direct ? 'One reading: change the connection' : 'Move the measurements; predict the information'}</h3>
    <p>Compare your edited state with this saved baseline: <strong>{baselineText}</strong>. Initially the two states match. Change an input, then predict; an unchanged experiment is a valid null case.</p>
    {direct ? <div className="gp-controls">
      <NumericField label="Covariance ρ" value={settings.rho} min={-0.95} max={0.95} onChange={rho => edit({ ...settings, rho })} />
      <NumericField label="Observed value" value={settings.value} min={-4} max={4} onChange={value => edit({ ...settings, value })} />
      <NumericField label="Noise variance" value={settings.noise} min={0.01} max={2} onChange={noise => edit({ ...settings, noise })} />
    </div> : <>
      <ObservationEditor rows={settings.observations} onChange={observations => edit({ ...settings, observations })} />
      <div className="gp-controls">
        <NumericField label="Target position" value={settings.target} min={-1} max={4} onChange={target => edit({ ...settings, target })} />
        <NumericField label="RBF length" value={settings.length} min={0.1} max={4} onChange={length => edit({ ...settings, length })} />
        <NumericField label="Reading noise variance" value={settings.noise} min={0.01} max={2} onChange={noise => edit({ ...settings, noise })} />
        <label>Covariance model<select value={settings.kind} onChange={event => edit({ ...settings, kind: event.target.value })}><option value="rbf">RBF spatial connection</option><option value="independent">Independent values at distinct positions</option></select></label>
      </div>
    </>}
    <div className="gp-actions"><button type="button" onClick={() => { try { calculate(settings); setBaseline(structuredClone(settings)); clear(); } catch (failure) { setError(failure.message); } }}>Save this state as baseline</button><button type="button" onClick={reset}>Reset investigation</button></div>
    <Prediction question="Compared with the saved baseline, what changes at the target?" options={changes} choice={choice} setChoice={value => { setChoice(value); setResult(null); }} reason={reason} setReason={value => { setReason(value); setResult(null); }} onReveal={reveal} revealed={Boolean(result)} />
    {error && <p role="alert" className="gp-error">{error}</p>}
    {result && <div data-gp-result="conditioning"><div className="gp-feedback" role="status"><strong>{result.expected === choice ? 'Prediction matches.' : 'Reconsider your prediction.'}</strong> The calculated change is: {changes.find(row => row[0] === result.expected)[1].toLowerCase()}. Comparisons treat absolute differences ≤ 10⁻⁹ as numerical equality.</div>
      <LessonTable caption="Target calculation, before and after (dimensionless)" headers={['quantity', 'saved baseline', 'edited state']} rows={[
        ['Mean', gpNumber(result.before.mean), gpNumber(result.after.mean)], ['Latent variance', gpNumber(result.before.variance), gpNumber(result.after.variance)], ['Future observation variance', gpNumber(result.before.observationVariance), gpNumber(result.after.observationVariance)],
      ]} />
      {direct ? <><p>Mean = ρ × reading / (1 + noise); latent variance = 1 − ρ² / (1 + noise). At ρ = 0 the reading contributes nothing to either update.</p>
        <GPBands label="Conditional target intervals" x={[0, 1]} mean={[result.after.mean, result.after.mean]} latentSD={Array(2).fill(Math.sqrt(result.after.variance))} observationSD={Array(2).fill(Math.sqrt(result.after.observationVariance))} xLabel="one target, shown across a strip" />
      </> : <>
        <GPBands label="Edited GP posterior and its two uncertainty bands" x={spatialGrid} mean={result.after.posterior.mean.slice(1)} latentSD={result.after.posterior.latentSD.slice(1)} observationSD={result.after.posterior.observationSD.slice(1)} points={[...result.after.rows, { x: result.after.target, y: result.after.mean, target: true }]} />
        <p>Under the independent-value kernel, joining the finite grid is only a plotting convention; no continuity is asserted. Values at positions not sampled on that grid still use the exact kernel in the target calculation.</p>
        <details><summary>Inspect the target covariance row</summary><LessonTable caption="Each reading's connection to the target" headers={['reading (x, y)', 'cross-covariance', 'solved weight', 'mean contribution']} rows={result.after.rows.map((row, i) => [String([row.x, row.y]), gpNumber(result.after.posterior.cross[0][i]), gpNumber(result.after.posterior.weights[i]), gpNumber(result.after.posterior.cross[0][i] * result.after.posterior.weights[i])])} /></details>
        <p>With fixed positions, kernel and noise, values change the weights and mean but not the covariance subtraction. Try a symmetric change of length: the midpoint mean can stay zero while uncertainty changes.</p>
      </>}
      <p>Both bands are pointwise, conditional on this model. The dashed outer limits include a new independent measurement error.</p>
    </div>}
  </section>;
}

export function GaussianForecastLab() {
  const [mode, setMode] = useState('reported');
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [developmentVisible, setDevelopmentVisible] = useState(false);
  const [testChoice, setTestChoice] = useState('');
  const [testReason, setTestReason] = useState('');
  const [testVisible, setTestVisible] = useState(false);
  const [settings, setSettings] = useState({ family: 'trend_periodic', cutoff: 72, horizon: 24, estimate: '' });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const conditioner = useRef(createConditioner());
  const developmentValues = Object.values(data.real.development).flatMap(forecast => forecast.mean.flatMap((mean, i) => [mean - 1.96 * forecast.observation_sd[i], mean + 1.96 * forecast.observation_sd[i], forecast.actual[i]]));
  const developmentDomain = [Math.floor(Math.min(...developmentValues)), Math.ceil(Math.max(...developmentValues))];
  const clear = () => { setChoice(''); setReason(''); setDevelopmentVisible(false); setTestChoice(''); setTestReason(''); setTestVisible(false); setResult(null); setError(''); };
  const edit = next => { setSettings(next); clear(); };
  const revealExploration = () => {
    try {
      const cutoff = requireNumber(settings.cutoff, 'Cutoff', 72, 96);
      const horizon = requireNumber(settings.horizon, 'Horizon', 1, 24);
      const estimate = requireNumber(settings.estimate, 'First forecast-month estimate', 250, 500);
      const forecast = forecastFromPrefix(data.observations, data.real.development, { family: settings.family, cutoff, horizon }, conditioner.current);
      setResult({ forecast, scores: scoreForecast(forecast.rows, forecast.mean, forecast.observationSD), estimate });
      setError('');
    } catch (failure) { setResult(null); setError(failure.message); }
  };
  return <section className="gp-lab" data-gp-lab="forecast" aria-label="Historical CO2 forecast investigation"><h3>A forecast is a decision made before future readings</h3>
    <label>Experiment mode<select value={mode} onChange={event => { setMode(event.target.value); clear(); }}><option value="reported">Reported train/development/test protocol</option><option value="explore">Explore a new forecast with frozen kernel settings</option></select></label>
    {mode === 'reported' ? <>
      <p>Observe 1990–95 first. The annual period in the composite family is fixed at one year. Predict which family better continues both trend and seasonality into 1996–97.</p>
      <GPChart label="Only the available 1990–95 monthly CO2 observations" series={[{ values: data.observations.slice(0, 72).map(row => [1990 + row.x, row.co2]) }]} xLabel="calendar year" yLabel="CO₂ (ppm)" xFormat={v => Number(v.toFixed(1))} />
      <Prediction question="Which family do you expect to give lower development MAE?" options={[['rbf', 'RBF only'], ['trend_periodic', 'Trend + periodic + RBF'], ['tie', 'Equal errors']]} choice={choice} setChoice={value => { setChoice(value); setDevelopmentVisible(false); setTestVisible(false); setTestChoice(''); setTestReason(''); }} reason={reason} setReason={value => { setReason(value); setDevelopmentVisible(false); setTestVisible(false); setTestChoice(''); setTestReason(''); }} onReveal={() => setDevelopmentVisible(true)} revealed={developmentVisible} />
      {developmentVisible && <div data-gp-result="development"><p role="status" className="gp-feedback">{choice === 'trend_periodic' ? 'Prediction matches.' : 'The recorded result differs.'} Composite development MAE: {gpNumber(data.real.development.trend_periodic.mae)} ppm; RBF-only: {gpNumber(data.real.development.rbf.mae)} ppm. Freeze the composite family for the final refit.</p>
        <p>The two forecast panels share calendar and vertical scales.</p><div className="gp-panels">{Object.entries(data.real.development).map(([family, forecast]) => <ForecastResult key={family} forecast={forecast} cutoff={72} title={`${family} · development`} yDomain={developmentDomain} />)}</div>
        <Prediction question="If test MAE beats the baseline, does that by itself validate the 95% predictive intervals?" options={[['yes', 'Yes, lower point error establishes interval performance'], ['no', 'No, the interval claim needs its own assessment']]} choice={testChoice} setChoice={value => { setTestChoice(value); setTestVisible(false); }} reason={testReason} setReason={value => { setTestReason(value); setTestVisible(false); }} onReveal={() => setTestVisible(true)} revealed={testVisible} action="Commit interval prediction and reveal test" />
        {testVisible && <div data-gp-result="test"><p role="status" className="gp-feedback">{testChoice === 'no' ? 'Prediction matches.' : 'Point error and interval behavior answer different questions.'} Test MAE is {gpNumber(data.real.final_test.mae)} ppm versus {gpNumber(data.real.seasonal_naive.mae)} ppm for seasonal-naive, yet only {data.real.final_test.covered}/24 observations fall inside the intervals.</p><ForecastResult forecast={data.real.final_test} cutoff={96} title="Final test after the frozen-family refit" /></div>}
      </div>}
    </> : <>
      <p><strong>Kernel settings from 1990–95; condition on the selected prefix.</strong> No hyperparameters are refitted here. Centering uses only that prefix. This exploration does not reset the already-used historical test.</p>
      <div className="gp-controls"><label>Kernel family<select value={settings.family} onChange={event => edit({ ...settings, family: event.target.value })}><option value="rbf">RBF only</option><option value="trend_periodic">Trend + periodic + RBF</option></select></label>
        <NumericField label="Conditioning cutoff (months from January 1990)" value={settings.cutoff} min={72} max={96} step={1} onChange={cutoff => edit({ ...settings, cutoff })} />
        <NumericField label="Forecast horizon (months)" value={settings.horizon} min={1} max={24} step={1} onChange={horizon => edit({ ...settings, horizon })} />
        <NumericField label="Your first forecast-month estimate (ppm)" value={settings.estimate} min={250} max={500} onChange={estimate => edit({ ...settings, estimate })} />
      </div>
      <Prediction question="With the same prefix and kernel, what happens to the first month's prediction if we request more future months?" options={[['unchanged', 'It remains unchanged'], ['changes', 'It changes because the horizon is longer']]} choice={choice} setChoice={value => { setChoice(value); setResult(null); }} reason={reason} setReason={value => { setReason(value); setResult(null); }} onReveal={revealExploration} revealed={Boolean(result)} />
      {error && <p className="gp-error" role="alert">{error}</p>}
      {result && <div data-gp-result="exploration"><p role="status" className="gp-feedback">{choice === 'unchanged' ? 'Prediction matches.' : 'The first target has not changed.'} The same observed prefix and kernel give the same marginal prediction at that target. Additional requested targets do not become new observations.</p>
        <p>Your first-month estimate: {result.estimate} ppm. Model: {gpNumber(result.forecast.mean[0])} ppm; actual: {result.forecast.rows[0].co2} ppm. This estimate is an ungraded comparison.</p>
        <p>Prefix mean: {gpNumber(result.forecast.center)} ppm. MAE {gpNumber(result.scores.mae)} ppm; interval count {result.scores.covered}/{result.forecast.horizon}.</p>
        <ForecastResult forecast={result.forecast} cutoff={result.forecast.cutoff} title="Exploratory forecast under frozen kernel settings" showPrefix />
        <p>Now shorten the horizon to six with the same prefix and family. Predict again and compare the first six table rows. Then change the cutoff: that introduces genuinely new observations.</p>
      </div>}
    </>}
    <div className="gp-actions"><button type="button" onClick={() => { setMode('reported'); setSettings({ family: 'trend_periodic', cutoff: 72, horizon: 24, estimate: '' }); clear(); }}>Reset forecast investigation</button></div>
  </section>;
}

export function GaussianProbeLab() {
  const defaults = () => ({ observations: initialObservations.map(row => ({ ...row })), candidates: [{ id: 1, x: 1, noise: 0.25 }, { id: 2, x: 4, noise: 0.25 }], target: 1, length: 1, noise: 0.25, kind: 'rbf' });
  const [settings, setSettings] = useState(defaults);
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const clear = () => { setChoice(''); setReason(''); setResult(null); setError(''); };
  const edit = next => { setSettings(next); clear(); };
  const reveal = () => {
    try {
      const rows = measurementGains({ observations: parsedObservations(settings.observations), candidates: settings.candidates.map((row, i) => ({ ...row, x: requireNumber(row.x, `Candidate ${i + 1} position`, -1, 4), noise: requireNumber(row.noise, `Candidate ${i + 1} noise`, 0.01, 2) })), target: requireNumber(settings.target, 'Target', -1, 4), length: requireNumber(settings.length, 'Length', 0.1, 4), noise: requireNumber(settings.noise, 'Observed noise', 0.01, 2), kind: settings.kind });
      const maximum = Math.max(...rows.map(row => row.reduction));
      const winners = rows.filter(row => maximum - row.reduction <= 1e-9).map(row => String(row.id));
      setResult({ rows, winners }); setError('');
    } catch (failure) { setResult(null); setError(failure.message); }
  };
  return <section className="gp-lab" data-gp-lab="probe" aria-label="Next probe investigation"><h3>Buy information about a particular target</h3><p>Choose the reading that most reduces the latent variance at your target. Candidate measurement values are still unknown; the gain does not depend on their future observed values.</p>
    <ObservationEditor rows={settings.observations} onChange={observations => edit({ ...settings, observations })} />
    <div className="gp-controls"><NumericField label="Target to clarify" value={settings.target} min={-1} max={4} onChange={target => edit({ ...settings, target })} /><NumericField label="Probe RBF length" value={settings.length} min={0.1} max={4} onChange={length => edit({ ...settings, length })} /><NumericField label="Existing reading noise variance" value={settings.noise} min={0.01} max={2} onChange={noise => edit({ ...settings, noise })} /><label>Probe covariance model<select value={settings.kind} onChange={event => edit({ ...settings, kind: event.target.value })}><option value="rbf">RBF</option><option value="independent">Independent at distinct positions</option></select></label></div>
    {settings.candidates.map((candidate, i) => <div className="gp-observation-row" key={candidate.id}><NumericField label={`Candidate ${i + 1} position`} value={candidate.x} min={-1} max={4} onChange={x => edit({ ...settings, candidates: settings.candidates.map(row => row.id === candidate.id ? { ...row, x } : row) })} /><NumericField label={`Candidate ${i + 1} noise variance`} value={candidate.noise} min={0.01} max={2} onChange={noise => edit({ ...settings, candidates: settings.candidates.map(row => row.id === candidate.id ? { ...row, noise } : row) })} /><button type="button" disabled={settings.candidates.length <= 2} onClick={() => edit({ ...settings, candidates: settings.candidates.filter(row => row.id !== candidate.id) })}>Remove candidate {i + 1}</button></div>)}
    <div className="gp-actions"><button type="button" disabled={settings.candidates.length >= 5} onClick={() => edit({ ...settings, candidates: [...settings.candidates, { id: Math.max(...settings.candidates.map(row => row.id)) + 1, x: 0, noise: 0.25 }] })}>Add candidate</button><button type="button" onClick={() => { setSettings(defaults()); clear(); }}>Reset probe investigation</button></div>
    <Prediction question="Which candidate gives the largest reduction? Any tied maximum is accepted (tolerance 10⁻⁹)." options={settings.candidates.map((row, i) => [String(row.id), `Candidate ${i + 1} at ${row.x}`])} choice={choice} setChoice={value => { setChoice(value); setResult(null); }} reason={reason} setReason={value => { setReason(value); setResult(null); }} onReveal={reveal} revealed={Boolean(result)} />
    {error && <p role="alert" className="gp-error">{error}</p>}
    {result && <div data-gp-result="probe"><p role="status" className="gp-feedback">{result.winners.includes(choice) ? 'Your choice is a maximum.' : 'Another candidate reduces target variance more.'} {result.winners.length > 1 && 'The maximum is tied.'}</p><ProbeResults rows={result.rows} /><p>The subtraction starts from the current posterior, before the new reading. Zero covariance gives exactly zero gain. Move the target toward 4, or increase one candidate’s noise, then make a new prediction.</p></div>}
  </section>;
}
