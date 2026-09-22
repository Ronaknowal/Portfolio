import { useId, useState } from 'react';
import { abstentionDecision, allocateInspections, binaryDecision, contingentInspection, finiteTailRisk, formatDecisionNumber as number, informationValue, inspectionProbabilities, mixedStateRisk, provisioningRisk, signalRuleRisks, utilityLottery } from '../../data/decision-theory-models.js';
import './decision-theory-labs.css';
function Range({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01
}) {
  const id = useId();
  return <label className="decision-control" htmlFor={id}><span>{label} <output>{number(value)}</output></span>
    <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Investigation({
  title,
  children,
  reset,
  prompt
}) {
  return <section data-live-exploration className="decision-investigation" aria-label={title}>
    <div className="decision-investigation-heading"><h3>{title}</h3><button onClick={reset}>Reset</button></div>
    <p className="decision-guidance"> {prompt}</p>{children}
  </section>;
}
function Plot({
  label,
  xLabel,
  yLabel,
  xMax = 1,
  yMax,
  series,
  markers = [],
  children
}) {
  const x = value => 43 + 240 * value / xMax;
  const y = value => 210 - 170 * value / yMax;
  return <svg className="decision-plot" viewBox="0 0 310 258" role="img" aria-label={label}>
    {[0, 0.5, 1].map(fraction => <g key={fraction}>
      <line x1="43" x2="283" y1={y(fraction * yMax)} y2={y(fraction * yMax)} className="decision-grid" />
      <text x="37" y={y(fraction * yMax) + 4} textAnchor="end">{number(fraction * yMax, 2)}</text>
      <text x={x(fraction * xMax)} y="229" textAnchor="middle">{number(fraction * xMax, 2)}</text>
    </g>)}
    <text x="163" y="249" textAnchor="middle">{xLabel}</text>
    <text x="43" y="21">{yLabel}</text>
    {series.map(({
      points,
      color,
      dashed = false
    }, index) => <polyline key={index} fill="none" stroke={color} strokeWidth="2.4" strokeDasharray={dashed ? '5 4' : undefined} points={points.map(([horizontal, vertical]) => `${x(horizontal)},${y(vertical)}`).join(' ')} />)}
    {markers.map(({
      point,
      color
    }, index) => <circle key={index} cx={x(point[0])} cy={y(point[1])} r="4" fill={color} stroke="var(--color-bg)" strokeWidth="1.5" />)}
    {children}
  </svg>;
}
const gold = '#e8b44a';
const blue = '#75b7ed';
const mint = '#78cfab';
export function ConditionalLossLab() {
  const [p, setP] = useState(0.08);
  const [losses, setLosses] = useState([[0, 80], [10, 10]]);
  const result = binaryDecision(p, losses);
  const names = ['Release', 'Quarantine'];
  return <Investigation title="Read the lower loss envelope" prompt="Raise the faulty-quarantine cell. Which line tilts, and does the crossing move?" reset={() => {
    setP(0.08);
    setLosses([[0, 80], [10, 10]]);
  }}>
    <div className="decision-matrix" role="group" aria-label="Editable loss matrix">
      <span /> <strong>Sound</strong><strong>Faulty</strong>
      {losses.map((row, action) => <div className="decision-matrix-row" key={action}><strong>{names[action]}</strong>{row.map((loss, state) => <label key={state}><span className="decision-sr">{names[action]} when {state ? 'faulty' : 'sound'} loss</span><input type="number" min="0" max="100" step="1" value={loss} onChange={event => {
            const value = Number(event.target.value);
            if (event.target.value !== '' && Number.isFinite(value) && value >= 0 && value <= 100) setLosses(losses.map((old, index) => index === action ? old.map((cell, column) => column === state ? value : cell) : old));
          }} /></label>)}</div>)}
    </div>
    <Range label="Probability faulty" value={p} onChange={setP} step={0.005} />
    <Plot label="Two conditional risk lines with the current probability marked" xLabel="Probability faulty" yLabel="Expected loss / item" yMax={Math.max(1, ...losses.flat())} series={losses.map((row, index) => ({
      points: [[0, row[0]], [1, row[1]]],
      color: [gold, blue][index]
    }))} markers={result.risks.map((risk, index) => ({
      point: [p, risk],
      color: [gold, blue][index]
    }))} />
    <p className="decision-legend"><span style={{
        color: gold
      }}>Gold: release</span><span style={{
        color: blue
      }}>Blue: quarantine</span></p>
    <div className="decision-contributions">{names.map((name, index) => <p key={name}><strong>{name}</strong>: {number(result.contributions[index][0])} sound + {number(result.contributions[index][1])} faulty = <strong>{number(result.risks[index])}</strong></p>)}</div>
    <p className="decision-result" aria-live="polite">Best action{result.actions.length > 1 ? 's (tie)' : ''}: {result.actions.map(index => names[index]).join(' or ')}. {result.identical ? 'The two loss rows are identical.' : result.crossing !== null && result.crossing >= 0 && result.crossing <= 1 ? `The lines meet at p=${number(result.crossing, 5)}.` : 'No unique crossing lies in [0, 1].'}</p>
    <p className="lesson-note">Each line joins its two actual loss cells. Readouts are rounded; calculations use the input numbers. The bounded editor accepts losses from 0 to 100.</p>
  </Investigation>;
}
export function ProcedureRiskLab() {
  const [prior, setPrior] = useState(0.3);
  const [selected, setSelected] = useState(2);
  const result = signalRuleRisks(prior);
  const names = ['Always low', 'Always high', 'Follow signal', 'Invert signal'];
  return <Investigation title="Hold the state fixed, then average the states" prompt="If high states become rare, does following the same 80%-accurate signal still give the best rule?" reset={() => {
    setPrior(0.3);
    setSelected(2);
  }}>
    <Range label="Prior probability of high state" value={prior} onChange={setPrior} />
    <label className="decision-select">Rule fixed before the signal <select value={selected} onChange={event => setSelected(Number(event.target.value))}>{names.map((name, index) => <option key={name} value={index}>{name}</option>)}</select></label>
    <div className="decision-state-columns">{['Low state held fixed', 'High state held fixed'].map((name, state) => <div key={name}>
      <h4>{name}</h4><div className="decision-signal-tiles">{[0, 1].map(signal => <div key={signal} className={result.rules[selected][signal] === state ? 'decision-correct' : 'decision-error'}>
        <strong>X={signal}</strong><span>chance {signal === state ? '.8' : '.2'}</span><span>announce {result.rules[selected][signal] ? 'high' : 'low'}</span><span>{result.rules[selected][signal] === state ? 'loss 0' : 'loss 1'}</span>
      </div>)}</div><p>R = <strong>{number(result.ruleRisks[selected][state])}</strong></p><p>Prior weight: {number(result.prior[state])}</p>
    </div>)}</div>
    <p className="decision-result" aria-live="polite">Bayes risk of {names[selected].toLowerCase()}: {number(result.prior[0])} × {number(result.ruleRisks[selected][0])} + {number(result.prior[1])} × {number(result.ruleRisks[selected][1])} = <strong>{number(result.bayesRisks[selected])}</strong>. Best rule{result.actions.length > 1 ? 's' : ''}: {result.actions.map(index => names[index]).join(' or ')}.</p>
    <p>Posterior high: after X=0, {number(result.signals[0].posterior)}; after X=1, {number(result.signals[1].posterior)}. These conditional probabilities average a different set of cases from a fixed-state error rate.</p>
  </Investigation>;
}
export function ProvisioningLab() {
  const [quantity, setQuantity] = useState(2);
  const [under, setUnder] = useState(4);
  const result = provisioningRisk(quantity, undefined, undefined, under, 1);
  const samples = Array.from({
    length: 41
  }, (_, index) => index / 4);
  const curve = samples.map(q => [q, provisioningRisk(q, undefined, undefined, under, 1).risk]);
  return <Investigation title="See the shortage behind the weighted average" prompt="Move the order from 2 to 10. Which scenario gets cheaper, and which scenarios now carry surplus?" reset={() => {
    setQuantity(2);
    setUnder(4);
  }}>
    <Range label="Order quantity" value={quantity} onChange={setQuantity} min={0} max={10} step={0.25} />
    <Range label="Underage cost per unit (overage stays 1)" value={under} onChange={setUnder} min={1} max={6} step={0.5} />
    <div className="decision-demand">{result.terms.map(term => <div key={term.value}><div className="decision-demand-label"><strong>Demand {term.value}</strong><span>mass {number(term.mass)}</span></div>
      <div className="decision-demand-track"><span style={{
            width: `${term.value * 10}%`
          }} /><i style={{
            left: `${quantity * 10}%`
          }} /></div>
      <p>{term.shortage ? `${number(term.shortage)} short × ${under}` : `${number(term.surplus)} surplus × 1`}; weighted cost <strong>{number(term.contribution)}</strong></p>
    </div>)}</div>
    <Plot label="Computed asymmetric expected cost across order quantities" xLabel="Order quantity" yLabel="Expected cost" xMax={10} yMax={Math.max(...curve.map(point => point[1]), 1)} series={[{
      points: curve,
      color: gold
    }]} markers={[{
      point: [quantity, result.risk],
      color: blue
    }]} />
    <p className="decision-result" aria-live="polite">Expected cost {number(result.risk)}. Target quantile level {number(result.quantileLevel)}. Gold demand bars show required quantity; the vertical marker is your order.</p>
  </Investigation>;
}
export function FallbackCapacityLab() {
  const [p, setP] = useState(0.3);
  const [fallback, setFallback] = useState(0.6);
  const [capacity, setCapacity] = useState(2);
  const abstention = abstentionDecision(p, fallback);
  const allocation = allocateInspections(undefined, capacity);
  const names = ['Predict sound', 'Predict faulty', 'Use fallback'];
  const strips = Array.from({
    length: 101
  }, (_, index) => abstentionDecision(index / 100, fallback).actions);
  return <Investigation title="Add a real fallback, then share limited slots" prompt="The upper example permits three independent actions. The lower one couples six items. Which item loses a slot when capacity falls from 3 to 2?" reset={() => {
    setP(0.3);
    setFallback(0.6);
    setCapacity(2);
  }}>
    <h4>A. A separate classification loss: FN 8, FP 2, correct 0</h4>
    <Range label="Probability faulty in fallback example" value={p} onChange={setP} step={0.005} />
    <Range label="Fixed fallback loss" value={fallback} onChange={setFallback} min={0} max={2} step={0.1} />
    <div className="decision-region" aria-label="Optimal action across probability from zero to one">{strips.map((actions, index) => <span key={index} style={{
        background: actions.length > 1 ? '#eeeeee' : [gold, blue, mint][actions[0]]
      }} />)}<i style={{
        left: `${p * 100}%`
      }} /></div>
    <p className="decision-legend"><span style={{
        color: gold
      }}>Sound</span><span style={{
        color: blue
      }}>Faulty</span><span style={{
        color: mint
      }}>Fallback</span><span>left p=0 → right p=1</span></p>
    <p className="lesson-note">This strip samples 101 probabilities at intervals of .01. Narrow ties and boundaries, such as .075, can fall between samples. The selected-point risks below are calculated directly at your chosen probability.</p>
    <p aria-live="polite">Risks {abstention.risks.map(value => number(value)).join(' / ')}. Best: <strong>{abstention.actions.map(index => names[index]).join(' or ')}</strong>.</p>
    <h4>B. Original release/quarantine losses: damage 80, handling 10</h4>
    <Range label="Quarantine slots for all six items" value={capacity} onChange={setCapacity} min={0} max={6} step={1} />
    <div className="decision-items">{inspectionProbabilities.map((probability, index) => <div key={index} className={allocation.selected.includes(index) ? 'decision-item-selected' : ''}>
      <strong>Item {index + 1}</strong><span>p={number(probability)}</span><span>saving {number(allocation.benefits[index])}</span><b>{allocation.selected.includes(index) ? 'Quarantine' : 'Release'}</b><span>cost {number(allocation.contributions[index])}</span>
    </div>)}</div>
    <p className="decision-result" aria-live="polite">Used {allocation.selected.length} of {capacity} slots. Total expected loss <strong>{number(allocation.risk)}</strong>. Only positive savings use slots; equal savings favor the earlier item.</p>
  </Investigation>;
}
export function InformationValueLab() {
  const [p, setP] = useState(0.08);
  const [sensitivity, setSensitivity] = useState(0.8);
  const [falsePositive, setFalsePositive] = useState(0.1);
  const [price, setPrice] = useState(2);
  const [early, setEarly] = useState(false);
  const result = informationValue(p, sensitivity, falsePositive, price);
  return <Investigation title="Put the information on the correct side of the decision" prompt="An informative signal arrives after you must act. Can its posterior now change the action for this item?" reset={() => {
    setP(0.08);
    setSensitivity(0.8);
    setFalsePositive(0.1);
    setPrice(2);
    setEarly(false);
  }}>
    <div className="decision-controls"><Range label="Prior faulty probability" value={p} onChange={setP} step={0.01} /><Range label="Sensitivity P(+ | faulty)" value={sensitivity} onChange={setSensitivity} /><Range label="False-positive P(+ | sound)" value={falsePositive} onChange={setFalsePositive} /><Range label="Test price" value={price} onChange={setPrice} min={0} max={8} step={0.1} /></div>
    <label className="decision-checkbox"><input type="checkbox" checked={early} onChange={event => setEarly(event.target.checked)} /> Commit the action before the result arrives</label>
    <div className="decision-tree" aria-label="Signal branches and their contingent decisions">
      <div className="decision-tree-root">{early ? 'Choose once → test result' : 'Observe test → choose within each branch'}</div>
      <div className="decision-branches">{result.branches.map((branch, index) => <div key={index} className={branch.mass === 0 ? 'decision-impossible' : ''}>
        <h4>{index ? 'Positive' : 'Negative'} result</h4><p>Branch mass <strong>{number(branch.mass)}</strong></p><p>Sound joint {number(branch.joint[0])}<br />Faulty joint {number(branch.joint[1])}</p>
        <p>Posterior faulty: <strong>{branch.posterior === null ? 'undefined: impossible' : number(branch.posterior)}</strong></p>
        <p>{branch.mass === 0 ? 'No action needed on this null branch.' : early ? `Already committed: ${binaryDecision(p).actions.map(action => ['release', 'quarantine'][action]).join(' or ')}.` : `Choose ${branch.actions.map(action => ['release', 'quarantine'][action]).join(' or ')}.`}</p>
        <p>Contribution before price: <strong>{number(early ? branch.weightedRisks[binaryDecision(p).actions[0]] : branch.minimum)}</strong></p>
      </div>)}</div>
    </div>
    <p className="decision-result" aria-live="polite">No-test risk {number(result.baseline)}. With the chosen timing and price: <strong>{number((early ? result.baseline : result.afterSignal) + price)}</strong>. Gross information value {number(early ? 0 : result.value)}; perfect-state value {number(result.perfectValue)}.</p>
    <p className="lesson-note">Each branch contribution already includes its probability. Average chance branches; minimize only when a decision may actually use that information. Signal precision and costs are synthetic inputs.</p>
  </Investigation>;
}
export function MinimaxMixLab() {
  const [weight, setWeight] = useState(0.4);
  const result = mixedStateRisk(weight);
  return <Investigation title="Balance the two state risks" prompt="Rule C has risk 2.5 in either state. Where does the A–B segment first fit inside a smaller square?" reset={() => setWeight(0.4)}>
    <Range label="Probability of choosing rule A" value={weight} onChange={setWeight} />
    <Plot label="Risk plane: rule A at (0,6), B at (4,0), C at (2.5,2.5), current mixture on segment" xLabel="Risk if state 0" yLabel="Risk if state 1" xMax={6} yMax={6} series={[{
      points: [[0, 6], [4, 0]],
      color: gold
    }, {
      points: [[0, result.worst], [result.worst, result.worst], [result.worst, 0]],
      color: mint,
      dashed: true
    }]} markers={[{
      point: [0, 6],
      color: gold
    }, {
      point: [4, 0],
      color: gold
    }, {
      point: [2.5, 2.5],
      color: blue
    }, {
      point: result.risk,
      color: mint
    }]}>
      <text x="50" y="37">A</text><text x="208" y="205">B</text><text x="150" y="131">C</text>
    </Plot>
    <p className="decision-legend"><span style={{
        color: gold
      }}>A–B mixtures</span><span style={{
        color: blue
      }}>Rule C</span><span style={{
        color: mint
      }}>Current mixture / worst-risk square</span></p>
    <p className="decision-result" aria-live="polite">State risks ({number(result.risk[0])}, {number(result.risk[1])}); worst <strong>{number(result.worst)}</strong>. At weight .4 the worst risk is 2.4.</p>
    <p>The state weights (.6, .4) give A and B average risk 2.4 and C risk 2.5. Any mixture has an average of at least 2.4, and its worst risk is at least its average. This certifies the global finite minimax value, independently of the drawing.</p>
  </Investigation>;
}
export function UtilityTailLab() {
  const [sure, setSure] = useState(95);
  const [alpha, setAlpha] = useState(0.9);
  const utility = utilityLottery(50, 150, 0.5, sure);
  const tail = finiteTailRisk(undefined, undefined, alpha);
  const curve = Array.from({
    length: 81
  }, (_, index) => {
    const value = index * 2;
    return [value, Math.sqrt(value)];
  });
  return <Investigation title="Change the consequence criterion explicitly" prompt="The lottery's mean is 100. Is a certain 95 preferred under square-root utility? Then fill exactly the worst tenth of the separate loss distribution." reset={() => {
    setSure(95);
    setAlpha(0.9);
  }}>
    <h4>A. Declared utility u(w)=√w of nonnegative final outcomes</h4>
    <Range label="Sure final outcome" value={sure} onChange={setSure} min={50} max={150} step={1} />
    <Plot label="Square-root utility curve and lottery chord; sure outcome lies on the curve" xLabel="Final outcome w" yLabel="Utility √w" xMax={160} yMax={13} series={[{
      points: curve,
      color: gold
    }, {
      points: [[50, Math.sqrt(50)], [150, Math.sqrt(150)]],
      color: blue
    }]} markers={[{
      point: [100, utility.expectedUtility],
      color: blue
    }, {
      point: [sure, utility.sureUtility],
      color: mint
    }]} />
    <p aria-live="polite">Lottery: mean {number(utility.mean)}, expected utility {number(utility.expectedUtility)}, certainty equivalent {number(utility.certaintyEquivalent)}. Sure {sure}: utility {number(utility.sureUtility)}. Prefer {utility.sureUtility > utility.expectedUtility ? 'the sure outcome' : utility.sureUtility < utility.expectedUtility ? 'the lottery' : 'either (tie)'}.</p>
    <p className="decision-legend"><span style={{
        color: gold
      }}>Utility curve</span><span style={{
        color: blue
      }}>Lottery chord / mean utility</span><span style={{
        color: mint
      }}>Sure outcome</span></p>
    <h4>B. A different question: the upper tail of loss</h4>
    <Range label="Tail level alpha" value={alpha} onChange={setAlpha} min={0.5} max={0.99} step={0.01} />
    <div className="decision-tail" aria-label={`Loss atoms with exactly ${number(1 - alpha)} mass in the upper tail`}>{tail.atoms.map(atom => {
        const used = tail.tail.find(entry => entry.index === atom.index).usedMass;
        return <div key={atom.index} style={{
          flexGrow: atom.mass
        }} aria-label={`Loss ${atom.value}, mass ${atom.mass}, selected mass ${used}`}><span className="decision-tail-fill" style={{
            width: `${100 * used / atom.mass}%`
          }} /></div>;
      })}</div>
    <p className="lesson-note">Left to right: loss 0 (mass .8), loss 10 (mass .15), loss 100 (mass .05). Widths are probability masses. Hatching selects the worst tail, including a partial atom when needed; the ledger below supplies exact readouts.</p>
    <div className="decision-tail-ledger">{tail.tail.map(atom => <p key={atom.index}>Loss {atom.value}: use {number(atom.usedMass, 5)} of mass {number(atom.mass)} → tail weight {number(atom.usedMass / tail.tailMass, 5)}</p>)}</div>
    <p className="decision-result" aria-live="polite">VaR {number(tail.valueAtRisk)}; CVaR <strong>{number(tail.cvar)}</strong>; mean {number(tail.mean)}. The selected mass totals {number(tail.tailMass)}.</p>
  </Investigation>;
}
export function ContingentInspectionLab() {
  const [index, setIndex] = useState(3);
  const [price, setPrice] = useState(1);
  const [capacity, setCapacity] = useState(2);
  const result = contingentInspection(index, price, capacity);
  const candidates = inspectionProbabilities.map((_, testIndex) => contingentInspection(testIndex, price, capacity));
  const best = Math.min(result.baseline.risk, ...candidates.map(candidate => candidate.total));
  return <Investigation title="Test first; reallocate after the result" prompt="The most uncertain item and the highest-risk item need not have the most valuable test. Which result could change the two-slot allocation?" reset={() => {
    setIndex(3);
    setPrice(1);
    setCapacity(2);
  }}>
    <label className="decision-select">Item receiving a perfect test <select value={index} onChange={event => setIndex(Number(event.target.value))}>{inspectionProbabilities.map((p, item) => <option key={item} value={item}>Item {item + 1}, prior {p}</option>)}</select></label>
    <Range label="Price of the one test" value={price} onChange={setPrice} min={0} max={15} step={0.1} />
    <Range label="Available quarantine slots" value={capacity} onChange={setCapacity} min={0} max={6} step={1} />
    <div className="decision-branches">{result.branches.map(branch => <div key={branch.state}><h4>Item {index + 1} is {branch.state ? 'faulty' : 'sound'}</h4>
      <p>Probability {number(branch.mass)}</p><div className="decision-mini-items">{inspectionProbabilities.map((_, item) => <span key={item} className={branch.selected.includes(item) ? 'decision-item-selected' : ''}>{item + 1}<small>{branch.selected.includes(item) ? 'hold' : 'release'}</small></span>)}</div>
      <p>Conditional expected loss <strong>{number(branch.risk)}</strong></p><p>Weighted contribution {number(branch.mass * branch.risk)}</p>
    </div>)}</div>
    <p className="decision-result" aria-live="polite">Selected test: {number(result.afterSignal)} expected operational loss + {number(price)} price = <strong>{number(result.total)}</strong>. No test: {number(result.baseline.risk)}.</p>
    <div className="decision-candidate-costs" aria-label="All possible test costs">{candidates.map((candidate, item) => <p key={item}>Test {item + 1}: <strong>{number(candidate.total)}</strong>{candidate.total === best ? ' · optimal' : ''}</p>)}<p>No test: <strong>{number(result.baseline.risk)}</strong>{result.baseline.risk === best ? ' · optimal' : ''}</p></div>
    <p className="lesson-note">Only this capstone assumes independent faults, so learning one item's state leaves the others' probabilities unchanged. A test consumes money but no quarantine slot. Actions are recomputed within each branch.</p>
  </Investigation>;
}
