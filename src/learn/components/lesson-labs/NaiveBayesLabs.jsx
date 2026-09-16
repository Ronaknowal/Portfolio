import { useState } from 'react';
import { Investigation, Stepper } from './LessonInvestigation';
import { LessonTable } from './LessonElements';
import { naiveBayesVocabulary, naiveBayesDocuments, tokenEvidenceState, presenceEvidenceState, gaussianObservationState, gaussianGeometryState, copiedAlarmState, reliabilityState, formatNaiveBayesNumber as number } from '../../data/naive-bayes-models.js';
import './naive-bayes-labs.css';
const classNames = ['Ham', 'Spam'];
const classColors = ['#a4c9b2', '#e5b95b'];
const probability = value => value === null ? 'undefined' : number(value * 100, 2) + '%';
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="nb-range"><span>{label} <strong>{number(value)}</strong></span>
    <input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function PlotFrame({
  children,
  label,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  xTicks,
  yTicks,
  height = 310
}) {
  const left = 49;
  const right = 311;
  const top = 23;
  const bottom = height - 55;
  const x = value => left + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * (right - left);
  const y = value => bottom - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * (bottom - top);
  return <svg className="nb-plot" viewBox={'0 -12 330 ' + (height + 38)} role="img" aria-label={label}>
    <path d={'M' + left + ',' + top + 'V' + bottom + 'H' + right} className="nb-axis" />
    {xTicks.map(value => <g key={value}><path className="nb-grid" d={'M' + x(value) + ',' + top + 'V' + bottom} />
      <text x={x(value)} y={bottom + 25} textAnchor={value === xDomain[0] ? 'start' : value === xDomain[1] ? 'end' : 'middle'}>{number(value, 2)}</text></g>)}
    {yTicks.map(value => <g key={value}><path className="nb-grid" d={'M' + left + ',' + y(value) + 'H' + right} />
      <text x={left - 9} y={y(value) + 5} textAnchor="end">{number(value, 2)}</text></g>)}
    {children({
      x,
      y,
      left,
      right,
      top,
      bottom
    })}
    <text x={(left + right) / 2} y={height + 12} textAnchor="middle">{xLabel}</text>
    <text x={left} y={15} className="nb-axis-title">{yLabel}</text>
  </svg>;
}
function pathOf(points, x, y) {
  return points.map((point, index) => (index ? 'L' : 'M') + x(point[0]) + ',' + y(point[1])).join(' ');
}
export function NaiveBayesModelFigure() {
  return <figure className="nb-inline">
    <div className="nb-model-fork">
      <div><strong>Class Y</strong><span>Which kind of case?</span></div>
      <span className="nb-model-connector" aria-hidden="true">↓</span>
      <div className="nb-feature-branches"><span>Feature X₁</span><span>Feature X₂</span><span>Feature X₃</span></div>
    </div>
    <figcaption>The model gives each feature a distribution for each class. With the class held fixed, its joint rule multiplies those feature likelihoods. This is a modeling assumption; the arrows do not establish real-world causation.</figcaption>
  </figure>;
}
export function NaiveBayesCorpusFigure() {
  return <figure className="nb-inline">
    <div className="nb-corpus">{naiveBayesDocuments.map((document, i) => <div className="nb-document" key={i}>
      <strong>{classNames[document.label]} {i + 1}</strong>
      <div className="nb-token-strip">{document.text.split(' ').map((word, j) => <span key={j} data-word={word}>{word}</span>)}</div>
      <span className="nb-inline-arrow" aria-hidden="true">→</span>
      <span className="nb-count-vector">[{document.counts.join(', ')}]</span>
    </div>)}</div>
    <figcaption>Column order: free, money, win, meeting, agenda. Each repeated token adds to the same column. The two spam documents contain six tokens in total; the ham document contains two.</figcaption>
  </figure>;
}
export function TokenEvidenceLab() {
  const [draft, setDraft] = useState('free');
  const [alphaDraft, setAlphaDraft] = useState('1');
  const [state, setState] = useState(() => tokenEvidenceState());
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const frame = state.frames[step];
  const reset = () => {
    setDraft('free');
    setAlphaDraft('1');
    setState(tokenEvidenceState());
    setStep(0);
    setError('');
  };
  const apply = event => {
    event.preventDefault();
    try {
      if (!alphaDraft.trim()) throw new Error('Enter an alpha value before applying.');
      const next = tokenEvidenceState(draft, Number(alphaDraft));
      setState(next);
      setStep(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  };
  const finiteTerms = state.frames.map(item => item.logOdds).filter(Number.isFinite);
  const extent = Math.max(2, ...finiteTerms.map(Math.abs));
  const position = frame.logOdds === null ? null : Math.max(-1, Math.min(1, frame.logOdds / extent));
  return <Investigation id="token-evidence" kicker="FOLLOW A MESSAGE" title="Spend one token of evidence at a time">
    <p>Predict which way adding <strong>meeting</strong> after <strong>free</strong> will move the odds. The five-word vocabulary and three training documents stay fixed.</p>
    <form className="nb-form" onSubmit={apply}>
      <label>Message <input aria-label="Message" value={draft} onChange={event => setDraft(event.target.value)} maxLength={301} /></label>
      <label>Alpha <input aria-label="Alpha" value={alphaDraft} onChange={event => setAlphaDraft(event.target.value)} inputMode="decimal" /></label>
      <button type="submit">Apply message and alpha</button>
    </form>
    {error && <p role="alert" className="nb-error">{error} The active message is unchanged.</p>}
    <p className="lesson-note">Edits take effect when applied. Lowercase ASCII word tokens; punctuation is a separator. At most 30 tokens. Alpha is 0 or 10⁻⁸–10.</p>
    <div className="nb-token-strip" aria-label="Submitted known tokens">{state.knownTokens.length ? state.knownTokens.map((token, i) => <span key={i} className={i < step ? 'nb-observed' : 'nb-future'} aria-label={token + (i < step ? ', observed' : ', not yet observed')}>{token}</span>) : <span>No vocabulary tokens</span>}</div>
    {state.unknownTokens.length > 0 && <p>Ignored by this fixed vocabulary: <strong>{state.unknownTokens.join(', ')}</strong>. Smoothing does not create new columns.</p>}
    <Stepper step={step} count={state.frames.length} setStep={setStep} onReset={reset} />
    <div className="nb-evidence-readout" aria-live="polite">
      <strong>{frame.label}</strong>
      <p>{step} of {state.knownTokens.length} vocabulary tokens counted. Active alpha: {number(state.parameters.alpha)}.</p>
      <div className="nb-evidence-axis"><span>Ham ←</span><span>0 log odds</span><span>→ Spam</span></div>
      <div className="nb-evidence-track"><span className="nb-evidence-zero" />
        {position !== null && <span className="nb-evidence-bar" style={{
          left: (position < 0 ? 50 + position * 50 : 50) + '%',
          width: Math.abs(position) * 50 + '%',
          background: position < 0 ? classColors[0] : classColors[1]
        }} />}
      </div>
      <p><strong>Log odds: {number(frame.logOdds)}.</strong> {frame.logOdds === null ? 'No bar is drawn because both class weights are zero.' : Number.isFinite(frame.logOdds) ? 'Bar endpoints are ±' + number(extent) + ' natural-log units for this submitted message.' : 'Infinite odds are shown at the edge; they are not a finite bar length.'}</p>
      {step > 0 && <p>This <strong>{frame.token}</strong> token contributes {number(frame.contribution)} log units:
        log θ(spam) − log θ(ham) = {number(frame.terms[1])} − ({number(frame.terms[0])}).</p>}
      {frame.probabilities ? <p>Model P(spam): <strong>{probability(frame.probabilities[1])}</strong>.
        Larger score chooses <strong>{classNames[frame.decision]}</strong>; exact ties choose Ham here.</p> : <p className="nb-error">Both classes assign zero likelihood to this observed prefix. The posterior is undefined; choosing a class would hide a failed model.</p>}
    </div>
    <LessonTable caption="Active fitted token probabilities" headers={['Word', 'Ham count / θ', 'Spam count / θ']} rows={naiveBayesVocabulary.map((word, j) => [word, state.parameters.wordCounts[0][j] + ' / ' + number(state.parameters.probabilities[0][j]), state.parameters.wordCounts[1][j] + ' / ' + number(state.parameters.probabilities[1][j])])} />
    <details><summary>Check the class scores at this step</summary>
      <LessonTable caption="Scores are normalized across classes" headers={['Class', 'Unnormalized log score', 'Probability']} rows={classNames.map((label, c) => [label, number(frame.scores[c]), frame.probabilities ? probability(frame.probabilities[c]) : 'undefined'])} />
    </details>
    <p><strong>Transfer:</strong> try <em>free meeting</em> with alpha 0, then alpha 1. Explain why stable arithmetic cannot repair a likelihood that the model made exactly zero.</p>
  </Investigation>;
}
export function PresenceEvidenceLab() {
  const [counts, setCounts] = useState([1, 0, 0, 0, 0]);
  const state = presenceEvidenceState(counts);
  const change = (index, amount) => setCounts(previous => previous.map((count, j) => j === index ? Math.max(0, Math.min(5, count + amount)) : count));
  return <Investigation id="presence-evidence" kicker="CHANGE THE OBSERVATION MODEL" title="A missing word can contribute evidence">
    <p>Predict whether changing free from one occurrence to two will change both models. These controls describe a completely observed message; zero means the word is absent.</p>
    <div className="nb-presence-slots">{naiveBayesVocabulary.map((word, j) => <div key={word} className={counts[j] ? 'nb-slot-present' : ''}>
      <strong>{word}</strong><div className="nb-counter"><button aria-label={'Remove one ' + word} disabled={!counts[j]} onClick={() => change(j, -1)}>−</button>
        <span>{counts[j]}</span><button aria-label={'Add one ' + word} disabled={counts[j] === 5} onClick={() => change(j, 1)}>+</button></div>
      <span>{state.binary[j] ? 'Present: b = 1' : 'Absent: b = 0'}</span>
      <small>Spam factor: {number(state.binary[j] ? state.parameters[1][j] : 1 - state.parameters[1][j])}</small>
    </div>)}</div>
    <button onClick={() => setCounts([1, 0, 0, 0, 0])}>Reset presence comparison</button>
    <div className="nb-result-pair" aria-live="polite">
      <p><strong>Count model</strong><span>P(spam) {probability(state.multinomial.probabilities[1])}</span>Repeats count; zero contributes no token term.</p>
      <p><strong>Bernoulli model</strong><span>P(spam) {probability(state.bernoulli.probabilities[1])}</span>Each presence or absence contributes one factor.</p>
    </div>
    <LessonTable caption="Bernoulli log factors for this observed message" headers={['Word / b', 'Ham log factor', 'Spam log factor']} rows={naiveBayesVocabulary.map((word, j) => [word + ' / ' + state.binary[j], number(state.terms[0][j]), number(state.terms[1][j])])} />
    <p><strong>Transfer:</strong> remove every word. Explain why the count model returns its prior but the Bernoulli model can still update it.</p>
  </Investigation>;
}
export function GaussianObservationLab() {
  const [reading, setReading] = useState(0);
  const [width, setWidth] = useState(0.4);
  const state = gaussianObservationState(reading, width);
  return <Investigation id="gaussian-observation" kicker="HEIGHT IS NOT PROBABILITY" title="Which spread explains this reading?">
    <p>Two equally common classes have the same mean, 0 mV, but standard deviations 1 and 2 mV. Predict which class gains probability as a reading moves far into either tail.</p>
    <div className="nb-controls"><Range label="Reading in mV" value={reading} onChange={setReading} min={-5} max={5} step={0.1} />
      <Range label="Interval width in mV" value={width} onChange={setWidth} min={0.1} max={1} step={0.1} /></div>
    <figure className="nb-figure"><PlotFrame label="Normal density curves with observed reading and interval" xLabel="Reading (mV)" yLabel="Density (1/mV)" xDomain={[-6, 6]} yDomain={[0, 0.44]} xTicks={[-6, -3, 0, 3, 6]} yTicks={[0, 0.2, 0.4]}>
      {({
          x,
          y,
          top,
          bottom
        }) => <>
        <rect x={x(reading - width / 2)} y={top} width={x(reading + width / 2) - x(reading - width / 2)} height={bottom - top} fill="#e5b95b" opacity=".12" />
        {state.curves.map((curve, c) => <path key={c} d={pathOf(curve, x, y)} fill="none" stroke={classColors[c]} strokeWidth="3" />)}
        <path d={'M' + x(reading) + ',' + top + 'V' + bottom} stroke="#e5dfd3" strokeDasharray="4 4" />
        {state.densities.map((density, c) => <circle key={c} cx={x(reading)} cy={y(density)} r="5" fill={classColors[c]} />)}
      </>}
    </PlotFrame><figcaption>Green: standard deviation 1 mV. Amber: 2 mV. Shading marks a common interval; the curve's area within it is a probability. Lines are analytic Gaussian densities.</figcaption></figure>
    <LessonTable caption="Same reading, different likelihood densities" headers={['Class spread', 'Density', 'Interval probability']} rows={state.variances.map((variance, c) => [Math.sqrt(variance) + ' mV', number(state.densities[c]) + ' /mV', probability(state.masses[c])])} />
    <p aria-live="polite">Model P(wide class | this reading): <strong>{probability(state.probabilities[1])}</strong>.
      Equal-density crossings are ±{number(state.crossing)} mV. The interval areas use 64-piece Simpson integration; the posterior uses density at the specified reading.</p>
    <button onClick={() => {
      setReading(0);
      setWidth(0.4);
    }}>Reset Gaussian observation</button>
    <p><strong>Transfer:</strong> change only the interval width. Why do the interval probabilities change while the point-reading posterior does not?</p>
  </Investigation>;
}
export function GaussianGeometryLab() {
  const [mode, setMode] = useState('unequal');
  const [point, setPoint] = useState([0, 0]);
  const state = gaussianGeometryState(mode, point);
  return <Investigation id="gaussian-geometry" kicker="DERIVE THE BOUNDARY" title="The boundary is where two computed scores tie">
    <p>Class 0 is centered at (−2,−2), class 1 at (2,2), with equal priors. Both features use the same declared measurement unit. Predict what changes when class 1's variance grows from 0.5 to 2 units².</p>
    <label className="nb-select">Class variances <select aria-label="Class variances" value={mode} onChange={event => setMode(event.target.value)}>
      <option value="equal">Equal: 0.5 and 0.5</option><option value="unequal">Unequal: 0.5 and 2</option></select></label>
    <div className="nb-controls">{point.map((value, j) => <Range key={j} label={'Probe ' + (j ? 'y' : 'x')} value={value} onChange={next => setPoint(previous => previous.map((entry, index) => index === j ? next : entry))} min={-8} max={6} step={0.25} />)}</div>
    <figure className="nb-figure"><PlotFrame label="Computed Gaussian decision regions, analytic boundary and movable probe" xLabel="Feature 1 (units)" yLabel="Feature 2 (units)" xDomain={[-8, 6]} yDomain={[-8, 6]} xTicks={[-8, -4, 0, 4, 6]} yTicks={[-8, -4, 0, 4, 6]} height={340}>
      {({
          x,
          y
        }) => <>
        {state.grid.map(cell => <rect key={cell.x + ',' + cell.y} x={x(cell.x - 0.25)} y={y(cell.y + 0.25)} width={x(0.5) - x(0)} height={y(0) - y(0.5)} fill={classColors[cell.label]} opacity=".13" />)}
        <path d={pathOf(state.boundary, x, y)} fill="none" stroke="#e5dfd3" strokeWidth="2.6" />
        {state.means.map((mean, c) => <g key={c}><circle cx={x(mean[0])} cy={y(mean[1])} r="6" fill={classColors[c]} />
          <text x={x(mean[0]) + 10} y={y(mean[1]) - 8}>μ{c}</text></g>)}
        <path d={'M' + (x(point[0]) - 7) + ',' + y(point[1]) + 'h14M' + x(point[0]) + ',' + (y(point[1]) - 7) + 'v14'} stroke="#f0bd70" strokeWidth="3" />
      </>}
    </PlotFrame><figcaption>Green cells choose class 0; amber cells choose class 1. Each cell evaluates its center, so the cell boundary is approximate. The white equal-score line/circle is calculated analytically. The cross is your probe, not a fitted sample.</figcaption></figure>
    <p aria-live="polite">Probe ({number(point[0])}, {number(point[1])}): log score 0 = {number(state.scores[0])}, log score 1 = {number(state.scores[1])}.
      P(class 1) = <strong>{probability(state.probabilities[1])}</strong>. {mode === 'equal' ? 'The quadratic terms cancel: the boundary is x + y = 0.' : 'The narrow class wins inside a circle centered at (−10/3, −10/3), radius about 4.0088 units. The wider class wins outside it.'}</p>
    <button onClick={() => {
      setMode('unequal');
      setPoint([0, 0]);
    }}>Reset Gaussian geometry</button>
    <p><strong>Transfer:</strong> compare (−2,−2) with (−8,−8). Why can a point far beyond the narrow class's center favor the wider class again?</p>
  </Investigation>;
}
export function CopiedAlarmLab() {
  const [copies, setCopies] = useState(3);
  const [positive, setPositive] = useState(true);
  const state = copiedAlarmState(copies, positive);
  return <Investigation id="copied-alarm" kicker="TEST THE ASSUMPTION" title="Five copies are still one measurement">
    <p>Fault prevalence is 20%. One alarm is positive in 80% of fault cases and 40% of normal cases. Its recorded copies are exact duplicates. Predict whether adding a copy can change the true posterior.</p>
    <div className="nb-controls"><Range label="Number of recorded copies" value={copies} onChange={setCopies} min={1} max={5} />
      <label className="nb-select">Observed alarm <select aria-label="Observed alarm" value={String(positive)} onChange={event => setPositive(event.target.value === 'true')}><option value="true">Positive</option><option value="false">Negative</option></select></label></div>
    <figure className="nb-inline"><svg className="nb-copy-diagram" viewBox="0 0 330 170" role="img" aria-label={'One alarm copied to ' + copies + ' recorded columns'}>
      <rect x="85" y="9" width="160" height="38" rx="4" className="nb-node" /><text x="165" y="34" textAnchor="middle">One alarm {positive ? '+' : '−'}</text>
      {Array.from({
          length: copies
        }, (_, i) => {
          const center = 165 + (i - (copies - 1) / 2) * 61;
          return <g key={i}><path d={'M165,47L' + center + ',103'} className="nb-connector" />
          <rect x={center - 23} y="104" width="46" height="42" rx="4" className="nb-node" />
          <text x={center} y="132" textAnchor="middle">X{i + 1}</text></g>;
        })}
    </svg><figcaption>Every branch copies the same observed value. Conditional on the original alarm, the copies add no new information.</figcaption></figure>
    <div className="nb-result-pair" aria-live="polite"><p><strong>Actual joint law</strong><span>P(fault) {probability(state.truth.probabilities[1])}</span>
      Uses the alarm likelihood once. Decision: {state.truth.decision ? 'fault' : 'normal'}.</p>
      <p><strong>Independent-copy model</strong><span>P(fault) {probability(state.naive.probabilities[1])}</span>
        Raises its likelihood to power {copies}. Decision: {state.naive.decision ? 'fault' : 'normal'}.</p></div>
    <LessonTable caption="Evaluate that classifier under the actual population" headers={['Alarm', 'Mass', 'True P(fault)', 'Model P(fault)']} rows={state.rows.map(row => [row.positive ? 'Positive' : 'Negative', probability(row.total), probability(row.actual), probability(row.reported)])} />
    <p>With equal error costs and ties choosing normal, actual population accuracy is <strong>{probability(state.accuracy)}</strong>;
      expected Brier loss is {number(state.brier, 6)}. These are exact finite-law calculations up to floating-point rounding, not a sampled benchmark.</p>
    <button onClick={() => {
      setCopies(3);
      setPositive(true);
    }}>Reset copied alarm</button>
    <p><strong>Transfer:</strong> at two copies the reported positive posterior is one half. At three copies the decision changes. Identify which statistical assumption changed and which observation did not.</p>
  </Investigation>;
}
export function ComplementPoolingFigure() {
  return <figure className="nb-inline">
    <div className="nb-complement-rows"><p><strong>Class 0</strong> [8, 2, 0]</p><p><strong>Class 1</strong> [1, 7, 2]</p><p><strong>Class 2</strong> [0, 2, 8]</p></div>
    <div className="nb-pooling"><span>For class 0, pool classes 1 and 2</span><strong>[1, 7, 2] + [0, 2, 8] = [1, 9, 10]</strong>
      <span>Add one in every column and divide by 23</span><strong>Complement θ = [2/23, 10/23, 11/23]</strong></div>
    <figcaption>The row for class 0 describes its complement. A document that fits this complement poorly gets a larger class-0 score after negating its log likelihood.</figcaption>
  </figure>;
}
export function CalibrationOwnershipFigure() {
  return <figure className="nb-inline">
    <ol className="nb-fit-lanes">
      <li><strong>Base-fit cases</strong><span>Fit feature mapping and class parameters.</span><b>Freeze base model ↓</b></li>
      <li><strong>Separate calibration cases</strong><span>Get unseen base predictions; fit their relationship to labels.</span><b>Freeze probability mapping ↓</b></li>
      <li><strong>Untouched test cases</strong><span>Report predictions, outcomes, bin counts and costs.</span><b>Evaluate; do not fit here</b></li>
    </ol><figcaption>This simple three-way protocol makes ownership visible. Cross-validation can reuse development data through held-out predictions; the final test still evaluates the complete choice.</figcaption>
  </figure>;
}
export function ReliabilityLab() {
  const [bins, setBins] = useState(4);
  const [compression, setCompression] = useState(false);
  const state = reliabilityState(bins, compression);
  return <Investigation id="reliability-bins" kicker="COMPARE PROBABILITIES WITH OUTCOMES" title="A reliability curve is made of finite groups">
    <p>These twelve invented cases have recorded predictions and binary outcomes. Predict whether changing the number of bins can change the curve without changing a single prediction.</p>
    <Range label="Number of reliability bins" value={bins} onChange={setBins} min={2} max={6} />
    <label className="nb-checkbox"><input type="checkbox" checked={compression} onChange={event => setCompression(event.target.checked)} />
      Compare the predeclared mapping q = 0.15 + 0.7p</label>
    <p className="lesson-note">The mapping is an illustrative fixed alternative, not a fitted calibrator. No dataset or model is trained in this lab.</p>
    <figure className="nb-figure"><PlotFrame label="Mean prediction and observed fraction in each nonempty reliability bin" xLabel="Mean predicted probability" yLabel="Observed positive fraction" xDomain={[0, 1]} yDomain={[0, 1]} xTicks={[0, 0.5, 1]} yTicks={[0, 0.5, 1]}>
      {({
          x,
          y
        }) => <>
        <path d={'M' + x(0) + ',' + y(0) + 'L' + x(1) + ',' + y(1)} stroke="#c2bbab" strokeDasharray="5 4" />
        {state.bins.filter(bin => bin.count).map(bin => <g key={bin.index}>
          <circle cx={x(bin.meanPrediction)} cy={y(bin.observedFraction)} r="7" fill="#e5b95b" />
          <text className="nb-point-label" x={x(bin.meanPrediction) + (bin.meanPrediction > 0.8 ? -12 : 12)} y={y(bin.observedFraction) + (bin.observedFraction > 0.9 ? 23 : bin.observedFraction < 0.1 ? -12 : 5)} textAnchor={bin.meanPrediction > 0.8 ? 'end' : 'start'}>n={bin.count}</text>
        </g>)}
      </>}
    </PlotFrame><figcaption>Each amber point is a group average, with its case count. The diagonal marks equality of average prediction and observed fraction. Empty bins do not create points.</figcaption></figure>
    <LessonTable caption="Trace every bin to its cases" headers={['Probability bin', 'Case IDs', 'Mean p / observed fraction']} rows={state.bins.map(bin => [number(bin.lower, 2) + ' ≤ p ' + (bin.upper === 1 ? '≤ ' : '< ') + number(bin.upper, 2), bin.cases.map(item => item.id).join(', ') || 'empty', bin.count ? number(bin.meanPrediction) + ' / ' + number(bin.observedFraction) : 'not defined'])} />
    <p aria-live="polite">Brier loss: <strong>{number(state.brier, 6)}</strong>. Log loss: <strong>{number(state.logLoss, 6)}</strong>.
      These use all twelve cases and do not depend on bin count. Lower scores are preferable on this fixed outcome set; they do not alone isolate calibration.</p>
    <details><summary>Inspect all twelve predictions and outcomes</summary><LessonTable caption="Cases behind the reliability view" headers={['Case', 'Predicted p', 'Outcome']} rows={state.cases.map(item => [item.id, number(item.probability), item.outcome])} /></details>
    <button onClick={() => {
      setBins(4);
      setCompression(false);
    }}>Reset reliability view</button>
    <p><strong>Transfer:</strong> find a bin containing one case. Why is its observed fraction of 0 or 1 weak evidence about long-run reliability?</p>
  </Investigation>;
}
