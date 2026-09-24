import { useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, neuralColors, formatNeural as fmt } from './NeuralLessonElements.jsx';
import { mean, regressionPenalty, regressionSlope, fittedConstant, focalTerm, confusionAt, tripletGeometry, candidateCompetition } from '../../data/loss-functions-models.js';
import measurements from '../../data/loss-functions-measurements.js';
import './loss-functions-labs.css';
const regressionNames = [['mse', 'Mean squared error'], ['mae', 'Mean absolute error'], ['huber', 'Huber, δ = 1']];
export function LossUpdateFigure() {
  return <figure className="loss-update-figure"><figcaption>One error, three different quantities</figcaption><div className="neural-flow">
    <div><strong>Prediction → residual</strong><br />x = 1, w = 3, b = 0 → prediction = 3. Target = 5 → residual = −2.</div>
    <div><strong>Penalty → slope</strong><br />Squared error = 4. Its prediction derivative = −4: raising the prediction locally reduces the penalty.</div>
    <div><strong>Two parameter updates → new prediction</strong><br />At rate 0.1, w becomes 3.4 and b becomes 0.4. Prediction = 3.8; squared error = 1.44.</div>
  </div><NeuralPlot title="Loss height and tangent at prediction 3" xLabel="prediction, target units" yLabel="squared error" xDomain={[1, 7]} yDomain={[-4, 16]} series={[{
      label: 'Squared penalty',
      values: Array.from({
        length: 61
      }, (_, i) => [1 + i / 10, (1 + i / 10 - 5) ** 2])
    }, {
      label: 'Local tangent, slope −4',
      dashed: true,
      values: [[2, 8], [5, -4]]
    }]} points={[{
      x: 3,
      y: 4,
      label: 'Prediction 3, loss 4'
    }, {
      x: 3.8,
      y: 1.44,
      color: neuralColors[1],
      label: 'After updating both parameters: prediction 3.8, loss 1.44'
    }]} /><p>The tangent can cross below zero; the squared penalty itself never does. Exact constructed calculation.</p></figure>;
}
export function RegressionInfluenceLab() {
  const initial = [0, 0, 0, 0, 0, 0, 10];
  const [values, setValues] = useState(initial),
    [method, setMethod] = useState('mse'),
    [index, setIndex] = useState(6),
    [baseline, setBaseline] = useState(initial);
  const optimum = fittedConstant(values, method),
    prior = fittedConstant(baseline, method);
  const low = Math.min(...values) - 1,
    high = Math.max(...values) + 1;
  const curve = Array.from({
    length: 81
  }, (_, i) => {
    const c = low + (high - low) * i / 80;
    return [c, mean(values.map(value => regressionPenalty(c - value, method)))];
  });
  const update = value => setValues(values.map((v, i) => i === index ? value : v));
  return <NeuralLab title="Which observation moves the fitted answer?" id="loss-regression"><p>Seven measurements share one constant answer. Edit an actual observation; the fit, loss curve and influence ledger update together. Click a measurement to select it, then drag its slider or type a value.</p>
    <div className="loss-observations">{values.map((value, i) => <button key={i} aria-pressed={index === i} onClick={() => setIndex(i)}>Observation {i + 1}<strong>{fmt(value)}</strong></button>)}</div>
    <div className="neural-controls"><NeuralNumber label={`Observation ${index + 1} value`} value={values[index]} onChange={update} min={-100} max={100} step="any" /><NeuralSelect label="Fitting objective" value={method} onChange={setMethod} options={regressionNames} /></div>
    <div className="neural-buttons"><button onClick={() => setBaseline([...values])}>Save measurements as baseline</button><button onClick={() => setValues(Array(7).fill(3))}>All measurements are 3</button><button onClick={() => {
        setValues(initial);
        setBaseline(initial);
        setMethod('mse');
        setIndex(6);
      }}>Reset measurements</button></div>
    <p className="neural-result" data-result="regression">Current fit {fmt(optimum)}; saved-data fit under the same objective {fmt(prior)}; change {fmt(optimum - prior)}. {Math.abs(optimum - prior) < 1e-9 ? 'This edit leaves the optimum unchanged.' : 'The changed observations move the optimum.'}</p>
    <figure className="loss-observation-locations"><figcaption>Where the seven measurements sit</figcaption><p>Each row is one observation on the same fixed −100 to 100 scale. The amber vertical line is the fitted constant; editing an observation moves its dot.</p><div className="loss-location-scale"><span>−100</span><span>0</span><span>100</span></div>{values.map((value, i) => <div className="loss-location-row" key={i}><span>{i + 1}</span><div className="loss-location-track"><i className="loss-location-fit" style={{
            left: `${(optimum + 100) / 2}%`
          }} /><i className="loss-location-dot" style={{
            left: `${(value + 100) / 2}%`
          }} /></div><strong>{fmt(value)}</strong></div>)}</figure>
    <NeuralPlot title="The fitted answer sits at the minimum" xLabel="candidate constant" yLabel={method === 'mse' ? 'mean squared penalty' : method === 'mae' ? 'mean absolute penalty' : 'mean Huber penalty'} xDomain={[low, high]} yDomain={[0, Math.max(.01, ...curve.map(row => row[1])) * 1.05]} series={[{
      label: 'Mean penalty of the current seven observations',
      values: curve
    }]} points={[{
      x: optimum,
      y: mean(values.map(v => regressionPenalty(optimum - v, method))),
      color: neuralColors[1],
      selected: true,
      label: `Optimum ${fmt(optimum)}`
    }]} />
    <NeuralTable caption="Why this constant is stationary" headers={['Observation', 'Value', 'Residual at fit', 'Per-observation derivative']} rows={values.map((v, i) => [i + 1, fmt(v), fmt(optimum - v), fmt(regressionSlope(optimum - v, method))])} />
    <p>Derivatives here are before division by seven. At an MAE tie we display the zero subgradient; a median is optimal when the sum can include zero, even if these selected subgradients do not sum to zero. With exactly seven observations, the median value is unique. Huber clips each large residual’s derivative at ±1; its units and curvature differ from squared error.</p>
  </NeuralLab>;
}
export function FocalContributionLab() {
  const [count, setCount] = useState(1000),
    [negative, setNegative] = useState(.01),
    [positive, setPositive] = useState(.1),
    [gamma, setGamma] = useState(2),
    [weighted, setWeighted] = useState(false);
  const n = focalTerm(negative, 0, gamma, weighted ? .25 : null),
    p = focalTerm(positive, 1, gamma, weighted ? .25 : null);
  const ce = count * negative + positive - 1,
    contributions = [count * n.slope, p.slope],
    total = contributions[0] + contributions[1],
    scale = Math.max(...contributions.map(Math.abs), .001);
  return <NeuralLab title="Focal loss changes who moves a shared bias" id="loss-focal"><p>All examples share one additive logit bias. These are constructed probabilities, held fixed while comparing derivatives. A negative sum tells gradient descent to increase that bias.</p>
    <div className="neural-controls"><NeuralNumber label="Number of negatives" value={count} onChange={setCount} min={1} max={2000} step={1} integer /><NeuralNumber label="Negative examples: probability of class 1" value={negative} onChange={setNegative} min={.001} max={.4} /><NeuralNumber label="Positive example: probability of class 1" value={positive} onChange={setPositive} min={.01} max={.99} /><NeuralNumber label="Focusing gamma" value={gamma} onChange={setGamma} min={0} max={5} /></div>
    <label className="loss-check"><input type="checkbox" checked={weighted} onChange={e => setWeighted(e.target.checked)} /> Use α = 0.25 for positives and 0.75 for negatives</label>
    <NeuralPlot title="Signed contributions, not just loss weights" xLabel="group: 0 negatives; 1 positive" yLabel="summed bias derivative" xDomain={[-.2, 1.2]} yDomain={[-scale * 1.2, scale * 1.2]} series={contributions.map((value, i) => ({
      label: i ? 'One positive' : `${count} negatives`,
      values: [[i, 0], [i, value]]
    }))} points={contributions.map((value, i) => ({
      x: i,
      y: value,
      color: neuralColors[i],
      selected: true,
      label: `${i ? 'Positive' : 'Negatives'}: ${fmt(value)}`
    }))} />
    <NeuralTable caption="Loss and derivative are different quantities" headers={['Group', 'Summed focal loss', 'Summed derivative']} rows={[[`${count} negatives`, fmt(count * n.loss), fmt(contributions[0])], ['One positive', fmt(p.loss), fmt(p.slope)], ['Total', fmt(count * n.loss + p.loss), fmt(total)]]} />
    <p className="neural-result" data-result="focal">Current derivative sum {fmt(total)} → {Math.abs(total) < 1e-10 ? 'no first-order bias update' : total < 0 ? 'gradient descent raises the bias' : 'gradient descent lowers the bias'}. Unweighted BCE on these same probabilities gives {fmt(ce)}. {gamma === 0 && !weighted ? 'At γ = 0 the full loss and gradient match BCE.' : 'The focusing factor itself contributes to the derivative.'}</p>
    <p>For a correctly classified example with pₜ = 0.9 and γ = 2, the loss ratio is 0.01 but the derivative ratio is about 0.028965. Multiplying a BCE derivative by the loss weight alone gives the wrong gradient.</p>
    <button onClick={() => {
      setCount(1000);
      setNegative(.01);
      setPositive(.1);
      setGamma(2);
      setWeighted(false);
    }}>Reset focal comparison</button>
  </NeuralLab>;
}
export function LossDecisionLab() {
  const [runIndex, setRunIndex] = useState(0),
    [threshold, setThreshold] = useState(.5),
    [selected, setSelected] = useState(0);
  const run = measurements.records[runIndex];
  const specimens = measurements.validation_source_ids.map(id => measurements.specimens.find(row => row.source_id === id));
  const labels = specimens.map(row => row.digit === 9 ? 1 : 0),
    counts = confusionAt(run.validation_probabilities, labels, threshold),
    specimen = specimens[selected];
  const below = run.validation_probabilities.filter(p => p < threshold);
  const above = run.validation_probabilities.filter(p => p >= threshold);
  const unchangedLower = below.length ? Math.max(...below) : 0;
  const unchangedUpper = above.length ? Math.min(...above) : 1;
  const selectedCell = labels[selected] ? run.validation_probabilities[selected] >= threshold ? 'tp' : 'fn' : run.validation_probabilities[selected] >= threshold ? 'fp' : 'tn';
  return <NeuralLab title="Same probabilities, different decisions" id="loss-decisions"><p>Inspect recorded validation predictions from nine matched CPU fits. Moving the threshold makes decisions; it does not retrain the model. Points in row 1 are actual nines; row 0 contains other digits.</p>
    <div className="neural-controls"><NeuralSelect label="Recorded run" value={runIndex} onChange={value => {
        setRunIndex(Number(value));
        setSelected(0);
      }} options={measurements.records.map((record, i) => [i, `Seed ${record.seed} · ${record.objective}`])} /><NeuralNumber label="Decision threshold" value={threshold} onChange={setThreshold} min={0} max={1} /></div>
    <NeuralPlot title="A vertical threshold crosses fixed probability marks" xLabel="probability of nine" yLabel="observed binary label" xDomain={[0, 1]} yDomain={[-.15, 1.15]} series={[{
      label: 'Class 1 when p ≥ threshold',
      values: [[threshold, -.1], [threshold, 1.1]],
      dashed: true
    }]} points={run.validation_probabilities.map((probability, i) => ({
      id: i,
      x: probability,
      y: labels[i],
      color: neuralColors[labels[i]],
      selected: selected === i,
      label: `Source ${specimens[i].source_id}; observed ${specimens[i].digit}; p ${fmt(probability)}`
    }))} onSelect={setSelected} />
    <div className="loss-confusion" aria-label="Confusion matrix">{[['tn', 'True negative'], ['fp', 'False positive'], ['fn', 'False negative'], ['tp', 'True positive']].map(([key, label]) => <div key={key} className={selectedCell === key ? 'loss-confusion-selected' : ''}>{label}<strong>{counts[key]}</strong>{selectedCell === key && <span>Selected specimen is here</span>}</div>)}</div>
    <p className="neural-result" data-result="decisions">Threshold {fmt(threshold)}: TP {counts.tp}, FP {counts.fp}, FN {counts.fn}, TN {counts.tn}. {counts.tp === 12 && counts.fp <= 2 ? 'This operating point retains all 12 positives with at most 2 false positives.' : 'Compare the missed positives with the false-alarm cost.'}</p>
    <p>The same decisions hold for thresholds in {below.length ? '(' : '['}{fmt(unchangedLower, 10)}, {fmt(unchangedUpper, 10)}]. These are adjacent recorded probability boundaries, rounded for display. The lower boundary is excluded when an observation lies there; the upper boundary is included because p ≥ threshold counts as positive. Moving inside this interval changes the control but crosses no specimen.</p>
    <NeuralTable caption="Fixed-probability measures: unchanged by threshold" headers={['Measure', 'Value']} rows={[['Average precision', fmt(run.average_precision, 6)], ['Brier score', fmt(run.brier, 6)], ['Unweighted log loss, nats', fmt(run.unweighted_log_loss, 6)]]} />
    <NeuralSelect label="Inspect validation specimen" value={selected} onChange={v => setSelected(Number(v))} options={specimens.map((row, i) => [i, `Source ${row.source_id}: digit ${row.digit}, p ${fmt(run.validation_probabilities[i], 4)}`])} />
    <div className="neural-two"><div className="neural-pixels" role="img" aria-label={`Actual 8 by 8 scan, source ${specimen.source_id}, digit ${specimen.digit}`}>{Array.from({
          length: 64
        }, (_, i) => <span key={i} style={{
          '--pixel': `rgb(${specimen[`pixel_${i}`] / 16 * 255},${specimen[`pixel_${i}`] / 16 * 255},${specimen[`pixel_${i}`] / 16 * 255})`
        }} />)}</div><p>Source {specimen.source_id}; observed digit {specimen.digit}. Stored p(nine) = {fmt(run.validation_probabilities[selected], 8)}. Current decision: {run.validation_probabilities[selected] >= threshold ? 'nine' : 'not nine'}. A specimen’s pixels and probability retain their original row identity.</p></div>
    <button onClick={() => {
      setRunIndex(0);
      setThreshold(.5);
      setSelected(0);
    }}>Reset threshold investigation</button>
  </NeuralLab>;
}
const startingPoints = [[0, 0], [1, 0], [.5, 0], [1.2, 0], [2, 0]];
function NegativeMarker({
  x,
  y,
  kind,
  selected
}) {
  const style = {
    fill: neuralColors[{
      hard: 3,
      'semi-hard': 1,
      easy: 2
    }[kind]],
    stroke: selected ? '#fff' : '#53675f',
    strokeWidth: selected ? 3 : 1
  };
  return kind === 'hard' ? <polygon points={`${x},${y - 8} ${x - 7},${y + 6} ${x + 7},${y + 6}`} {...style} /> : kind === 'semi-hard' ? <rect x={x - 6} y={y - 6} width="12" height="12" {...style} /> : <polygon points={`${x},${y - 8} ${x + 8},${y} ${x},${y + 8} ${x - 8},${y}`} {...style} />;
}
export function TripletGeometryLab() {
  const [points, setPoints] = useState(startingPoints),
    [selected, setSelected] = useState(3),
    [margin, setMargin] = useState(1),
    [squared, setSquared] = useState(true),
    [pairDistance, setPairDistance] = useState(.2);
  const names = ['Anchor', 'Positive', 'Negative 0', 'Negative 1', 'Negative 2'],
    result = tripletGeometry(points, margin, squared);
  const edit = (axis, value) => setPoints(points.map((point, i) => i === selected ? point.map((v, j) => j === axis ? value : v) : point));
  return <NeuralLab title="Move a candidate; change which triplet teaches" id="loss-triplet"><p>The coordinate board uses the same scale on both axes. Edit the selected point with either coordinate slider. The nearest strictly semi-hard candidate is selected; boundary ties are excluded, and equal eligible distances use row order.</p>
    <div className="neural-controls"><NeuralSelect label="Point to move" value={selected} onChange={value => setSelected(Number(value))} options={names.map((name, i) => [i, name])} /><NeuralSelect label="Distance convention" value={squared ? 'squared' : 'distance'} onChange={v => setSquared(v === 'squared')} options={[['squared', 'Squared distance D²'], ['distance', 'Euclidean distance D']]} /><NeuralNumber label={`${names[selected]} x`} value={points[selected][0]} onChange={v => edit(0, v)} min={-3} max={3} /><NeuralNumber label={`${names[selected]} y`} value={points[selected][1]} onChange={v => edit(1, v)} min={-3} max={3} /><NeuralNumber label={`Triplet margin (${squared ? 'squared coordinate' : 'coordinate'} units)`} value={margin} onChange={setMargin} min={0} max={4} /></div>
    <figure className="loss-coordinate-board"><figcaption>Anchor, positive and three candidate negatives</figcaption><svg className="neural-chart" viewBox="0 0 320 320" role="img" aria-label="Equal-scale coordinate board, both coordinates from minus three to three; exact coordinates in the table below."><line x1="30" y1="160" x2="290" y2="160" stroke="#7a8b82" /><line x1="160" y1="30" x2="160" y2="290" stroke="#7a8b82" />{[-3, -2, -1, 1, 2, 3].map(v => <g key={v}><text x={160 + v * 40} y="179" textAnchor="middle">{v}</text><text x="147" y={164 - v * 40} textAnchor="end">{v}</text></g>)}{points.map(([x, y], i) => <g key={i}><line x1={160 + points[0][0] * 40} y1={160 - points[0][1] * 40} x2={160 + x * 40} y2={160 - y * 40} stroke="#53675f" strokeDasharray="3 4" />{i < 2 ? <circle cx={160 + x * 40} cy={160 - y * 40} r={i === 0 ? 8 : 5} fill={neuralColors[i]} stroke="#fff" /> : <NegativeMarker x={160 + x * 40} y={160 - y * 40} kind={result.candidates[i - 2].kind} selected={result.selected === i - 2} />}</g>)}</svg><p>Circles: anchor/positive. Triangle: hard negative; square: strictly semi-hard; diamond: easy. A white border marks the mined candidate. Boundary ties follow the exact table below. Coincident points remain coincident; use the named rows below to inspect them.</p></figure>
    <NeuralTable caption="Coordinates and the exact eligibility test" headers={['Entity', '(x, y)', 'Distance from anchor', 'Constraint']} rows={points.map((point, i) => [names[i], point.map(v => fmt(v)).join(', '), i === 0 ? '0' : fmt(i === 1 ? result.positive : result.candidates[i - 2].distance), i < 2 ? 'Reference' : `${result.candidates[i - 2].kind}; loss ${fmt(result.candidates[i - 2].loss)}`])} />
    <p className="neural-result" data-result="triplet">Positive distance {fmt(result.positive)}; eligible interval ({fmt(result.positive)}, {fmt(result.positive + margin)}), endpoints excluded. {result.selected === null ? 'No eligible candidate: skip this anchor.' : `Selected negative ${result.selected}, loss ${fmt(result.candidates[result.selected].loss)}.`}</p>
    <div className="neural-buttons"><button onClick={() => setPoints(points.map(([x, y]) => [x, Math.min(3, y + .5)]))} disabled={points.some(point => point[1] > 2.5)}>Translate every point up 0.5</button><button onClick={() => setPoints(Array.from({
        length: 5
      }, () => [0, 0]))}>Inspect complete collapse</button><button onClick={() => {
        setPoints(startingPoints);
        setMargin(1);
        setSelected(3);
        setSquared(true);
        setPairDistance(.2);
      }}>Reset geometry</button></div>
    {points.every(point => point[0] === points[0][0] && point[1] === points[0][1]) && squared && <p>Collapsed squared triplets have loss {fmt(margin)} and zero coordinate gradients. A positive penalty does not guarantee motion.</p>}
    <h4>A pair asks a different question</h4><NeuralNumber label="Pair distance D" value={pairDistance} onChange={setPairDistance} min={0} max={2} /><p>With pair margin 1 and same = 1: matching loss D² = {fmt(pairDistance ** 2)}; nonmatching loss max(0, 1−D)² = {fmt(Math.max(0, 1 - pairDistance) ** 2)}. At D = 0, the norm has no unique derivative direction; this scalar display does not invent one.</p>
  </NeuralLab>;
}
export function InfoNceLab() {
  const [scores, setScores] = useState([.8, .2, -.1]),
    [temperature, setTemperature] = useState(.2);
  const result = candidateCompetition(scores, temperature);
  return <NeuralLab title="Temperature sharpens a competition" id="loss-infonce"><p>Key 0 is the designated positive. Similarities below are direct mathematical inputs, not outputs from a trained encoder. Try making a negative more similar than the positive, then lower temperature.</p>
    <div className="neural-controls">{scores.map((score, i) => <NeuralNumber key={i} label={`Key ${i} similarity${i === 0 ? ' (positive)' : ''}`} value={score} onChange={value => setScores(scores.map((v, j) => j === i ? value : v))} min={-1} max={1} />)}<NeuralNumber label="Temperature" value={temperature} onChange={setTemperature} min={.05} max={2} /></div>
    <div className="loss-probability-shares">{result.probabilities.map((probability, i) => <div key={i}><strong>Key {i}{i === 0 ? ' · positive' : ''}</strong><div className="loss-probability-track"><span style={{
            width: `${probability * 100}%`,
            background: neuralColors[i]
          }} /></div><span>{fmt(probability * 100, 3)}%</span></div>)}</div>
    <NeuralTable caption="Similarity → logit → probability" headers={['Key', 'Similarity', 'Scaled logit', 'Probability']} rows={scores.map((score, i) => [i, fmt(score), fmt(result.logits[i]), fmt(result.probabilities[i])])} />
    <p className="neural-result" data-result="infonce">Loss = −log p(key 0) = {fmt(result.loss, 9)} nats. Uniform baseline log 3 = {fmt(Math.log(3), 6)}. {scores.every(s => s === scores[0]) ? 'All scores tie, so temperature cannot change their equal shares.' : scores[0] < Math.max(...scores) ? 'A negative wins: sharpening can make the correct candidate even less probable.' : 'The positive leads or ties: inspect whether a tied competitor prevents its share reaching one.'}</p>
    <div className="neural-buttons"><button onClick={() => setScores([.4, .4, .4])}>Equal similarities</button><button onClick={() => setScores([.8, .8, -.1])}>Duplicate the positive score</button><button onClick={() => {
        setScores([.8, .2, -.1]);
        setTemperature(.2);
      }}>Reset competition</button></div>
    <figure><figcaption>Candidate masks are part of the objective</figcaption><div className="neural-two"><div><strong>One-way paired B × B</strong><p>Each query row chooses its matching key column. Diagonal = positive; other columns = designated negatives. All B columns participate.</p></div><div><strong>Two-view SimCLR, 2B × 2B</strong><p>Each view row excludes its own column, includes the other view of the same item as positive, and treats other items’ views as negatives. Both positive directions participate.</p></div></div><NeuralTable caption="B = 2: one-way paired candidates" headers={['Query', 'Key A', 'Key B']} rows={[['Query A', 'positive', 'negative'], ['Query B', 'negative', 'positive']]} /><NeuralTable caption="B = 2: SimCLR candidates by view identity" headers={['Query', 'A₁', 'B₁', 'A₂', 'B₂']} rows={[['A₁', 'self: excluded', 'negative', 'positive', 'negative'], ['B₁', 'negative', 'self: excluded', 'negative', 'positive'], ['A₂', 'positive', 'negative', 'self: excluded', 'negative'], ['B₂', 'negative', 'positive', 'negative', 'self: excluded']]} /></figure>
  </NeuralLab>;
}
export function LossScalingFigure() {
  return <figure><figcaption>Four times the matrix side, sixteen times the score storage</figcaption><div className="loss-matrix-area"><div className="loss-matrix-small" /><div className="loss-matrix-large" /></div><NeuralTable caption="Exact float32 score-matrix arithmetic" headers={['Batch B', 'Score cells B²', 'Four-byte storage']} rows={[[1024, '1,048,576', '4 MiB'], [4096, '16,777,216', '64 MiB']]} /><p>Square sides and areas share a scale. This counts scores only, excluding encoder activations and gradients. It is not measured runtime.</p><NeuralTable caption="A reduction changes whose error counts more" headers={['Sequence', 'Per-token losses', 'Sum', 'Per-sequence mean']} rows={[['Short', '[4]', 4, 4], ['Long', '[1, 1, 1]', 3, 1]]} /><p>Summing every token gives 7. A token mean gives 7 ÷ 4 = 1.75: the long sequence owns three quarters of the weight. Averaging the two sequence means gives (4 + 1) ÷ 2 = 2.5: each sequence owns half the weight. These different objectives need explicit valid-token masks and denominators; padding must not silently count as evidence.</p></figure>;
}
