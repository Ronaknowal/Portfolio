import { memoryRead } from '../../data/recurrent-attention-models.js';
import './recurrent-attention-intuition.css';

const amber = '#e2b55a', white = '#e6e6e6', muted = '#ababab';
const rounded = number => number.toFixed(6);

function MemoryRoute({ revisitable }) {
  const tokens = ['past', 'w', 'a', 'l', 'k', 'end'];
  return <div className="attention-route">
    <h3>{revisitable ? 'Keep the notes available' : 'Carry one final summary'}</h3>
    <svg viewBox="0 0 340 280" role="img" aria-label={revisitable
      ? 'Each source position has an encoder state. The final state initializes the writer; all saved states can also supply a read at each writing step.'
      : 'The encoder reads past, w, a, l, k, end in order. Only its final state initializes a recurrent writer; earlier states are not available for a separate read.'}>
      <text x="12" y="18">source, read left to right</text>
      {tokens.map((token, index) => {
        const x = 12 + index * 54;
        return <g key={token}>
          <rect x={x} y="30" width="45" height="30" rx="3" fill="#1c1c1c" stroke="#666" />
          <text x={x + 22.5} y="50" textAnchor="middle">{token}</text>
          <path d={`M${x + 22.5} 62v16m-4-4 4 4 4-4`} stroke={muted} fill="none" />
          <rect x={x + 7} y="82" width="31" height="28" rx="2" fill={revisitable || index === 5 ? '#302617' : '#1b1b1b'} stroke={revisitable || index === 5 ? amber : '#666'} />
          <text x={x + 22.5} y="102" textAnchor="middle">h{index}</text>
          {index < 5 && <path d={`M${x + 39} 96h21m-4-4 4 4-4 4`} stroke={muted} fill="none" />}
        </g>;
      })}
      {revisitable ? <>
        {[34.5, 88.5, 142.5, 196.5, 250.5, 304.5].map(x => <path key={x} d={`M${x} 113Q${x} 149 170 177`} stroke={amber} strokeOpacity=".7" fill="none" />)}
        <text x="170" y="158" textAnchor="middle" className="attention-svg-label">writer chooses this read</text>
      </> : <text x="170" y="153" textAnchor="middle">earlier notes no longer accessible</text>}
      <path d="M306 112v65H36v23m-4-4 4 4 4-4" stroke={white} strokeDasharray="4 4" fill="none" />
      <text x="39" y="193" className="attention-svg-label">initialize once</text>
      {[36, 170, 304].map((x, index) => <g key={x}>
        <circle cx={x} cy="218" r="17" fill="#171717" stroke={white} />
        <text x={x} y="224" textAnchor="middle">s{index}</text>
        {index < 2 && <path d={`M${x + 20} 218h93m-5-4 5 4-5 4`} stroke={white} fill="none" />}
      </g>)}
      {revisitable && <path d="M170 177v21m-4-4 4 4 4-4" stroke={amber} fill="none" />}
      <text x="170" y="262" textAnchor="middle">writer state changes at each step</text>
    </svg>
    <p>{revisitable
      ? 'One read is drawn. The current writer state determines its shares. At the next step, the writer asks again with its new state and can read a different mixture of the same notes.'
      : 'Every later source-dependent decision relies on what reached the writer through that initial summary.'}</p>
  </div>;
}

export function AttentionRecallContrast() {
  return <figure data-figure="summary-versus-notes" className="attention-intuition">
    <div className="attention-intuition-pair"><MemoryRoute /><MemoryRoute revisitable /></div>
    <figcaption>Two information paths for the same task. h denotes a reader state and s a writer state. Dashed white = one-time initialization; amber = an additional read. This is a wiring illustration, not measured word alignment. The full decoder loop comes in section 3.</figcaption>
  </figure>;
}

export function AttentionLookupBridge() {
  const records = [['12:00', 18], ['14:00', 24], ['16:00', 21]];
  const reads = [
    { name: 'Exact lookup: 14:00', shares: [0, 1, 0], formula: '0 × 18 + 1 × 24 + 0 × 21 = 24°C' },
    { name: 'Average all records', shares: [1 / 3, 1 / 3, 1 / 3], formula: '(18 + 24 + 21) / 3 = 21°C' },
    { name: 'A specified soft mixture', shares: [.25, .75, 0], formula: '0.25 × 18 + 0.75 × 24 = 22.5°C' },
  ];
  return <figure data-figure="lookup-to-mixture" className="attention-intuition attention-lookup">
    <p><strong>Query: “temperature at 14:00?”</strong> Compare the times; return the temperature.</p>
    <div className="attention-records">{records.map(([time, temperature]) => <div key={time}><span>key · time</span><strong>{time}</strong><span className="attention-record-link">↓ associated value</span><strong>{temperature}°C</strong></div>)}</div>
    {reads.map(read => <div className="attention-lookup-read" key={read.name}>
      <strong>{read.name}</strong>
      <div className="attention-mixture-strip" aria-label={`Shares for 12:00, 14:00, 16:00: ${read.shares.join(', ')}`}>{read.shares.map((share, index) => share > 0 && <span key={index} style={{ flex: share }}><span>{records[index][0]}</span><span>{Math.round(share * 1000) / 10}%</span></span>)}</div>
      <p>{read.formula}</p>
    </div>)}
    <figcaption>Each strip is one whole read; segment length encodes its share. The last shares are assigned for illustration, not derived from the query or trained on weather. Attention will learn a rule that computes shares from the query and keys.</figcaption>
  </figure>;
}

export function AttentionSoftmaxSteps() {
  const read = memoryRead([1, 0]);
  const masses = read.scores.map(score => Math.exp(score - Math.max(...read.scores)));
  const total = masses.reduce((sum, value) => sum + value, 0);
  return <figure data-figure="softmax-intermediates" className="attention-intuition">
    <div className="attention-softmax-lanes">{read.scores.map((score, index) => <div key={index}>
      <h3>Memory {['A', 'B', 'C'][index]}</h3>
      <dl><dt>Score</dt><dd>{score}</dd><dt>Subtract maximum (1)</dt><dd>{score - 1}</dd><dt>Exponentiate</dt><dd>{rounded(masses[index])}</dd></dl>
      <div className="attention-mass-track"><span style={{ width: `${100 * masses[index]}%` }} /></div>
    </div>)}</div>
    <div className="attention-denominator"><span aria-hidden="true">↳</span> Shared total: 1 + 0.367879 + 0.135335 = <strong>{rounded(total)}</strong> <span aria-hidden="true">↲</span></div>
    <div className="attention-softmax-lanes">{read.attention.map((share, index) => <div key={index}>
      <p>{['A', 'B', 'C'][index]}: {rounded(masses[index])} ÷ {rounded(total)}</p>
      <strong>{rounded(share)} of the read</strong>
      <div className="attention-mass-track"><span style={{ width: `${share * 100}%` }} /></div>
    </div>)}</div>
    <figcaption>Exact constructed calculation, rounded for display. Upper bars: exponentiated mass on a 0–1 scale for this example. Lower bars: normalized shares on a 0–1 scale. All three lower shares sum to 1; their common denominator couples them.</figcaption>
  </figure>;
}

export function AttentionAdditiveSteps() {
  const projectedQuery = [.5, -.5], projectedMemory = [1, .5];
  const sum = projectedQuery.map((value, index) => value + projectedMemory[index]);
  const curved = sum.map(Math.tanh);
  return <figure data-figure="additive-comparison" className="attention-intuition">
    <div className="attention-projection-inputs">
      <div><strong>Decoder query</strong><span>96 coordinates</span><span>↓ learned Wq projection</span><code>(0.5, −0.5)</code></div>
      <div><strong>One source memory</strong><span>192 coordinates</span><span>↓ learned Wh projection</span><code>(1, 0.5)</code></div>
    </div>
    <div className="attention-comparison-spine">
      <p><strong>1 · Add in the same two-coordinate space</strong><code>(0.5 + 1, −0.5 + 0.5) = (1.5, 0)</code></p>
      <p><strong>2 · Bend each coordinate with tanh</strong><code>({curved[0].toFixed(6)}, {curved[1]})</code></p>
      <p><strong>3 · Combine with scoring weights va = (2, −1)</strong><code>2 × {curved[0].toFixed(6)} − 1 × 0 = {(2 * curved[0]).toFixed(6)}</code><span>One score for this memory; repeat for the other positions.</span></p>
    </div>
    <p className="attention-value-bypass">The original 192-coordinate memory bypasses this scorer and supplies the value to the later weighted read.</p>
    <figcaption>Constructed projection outputs with comparison width 2, chosen to expose the arithmetic. The 96- and 192-coordinate inputs and their matrices are not specified here. A real model learns those matrices; the scorer's two comparison coordinates need not have named meanings.</figcaption>
  </figure>;
}

export function AttentionPaddingShares() {
  return <figure data-figure="padding-denominator" className="attention-intuition">
    <div className="attention-intuition-pair">{[false, true].map(buggy => <div key={String(buggy)}>
      <h3>{buggy ? 'Bug: allow storage to compete' : 'Correct: exclude storage first'}</h3>
      <p>All allowed scores equal 0.</p>
      <div className="attention-padding-shares">{[2, 4, 0].map((value, index) => <div key={index} className={index === 2 ? 'attention-padding-slot' : ''} style={{ flex: index === 2 && !buggy ? 0 : 1 }}>
        <strong>{index === 2 ? 'PAD' : `value ${value}`}</strong>
        <span>{index === 2 && !buggy ? 'excluded' : buggy ? '⅓' : '½'}</span>
      </div>)}</div>
      <p className="attention-denominator">{buggy ? '(2 + 4 + 0) / 3 = 2' : '(2 + 4) / 2 = 3'}</p>
    </div>)}</div>
    <figcaption>The real values never move. The denominator changes from 2 to 3 when padding is admitted. PAD's zero value contributes nothing, yet its share reduces both real contributions. Dashed outline identifies storage, including when excluded.</figcaption>
  </figure>;
}

export function AttentionLearningDirection() {
  const context = memoryRead([1, 0]).context;
  const margin = context[1] - context[0];
  const x = value => 42 + (value + 2) * 87;
  return <figure data-figure="learning-direction" className="attention-intuition attention-learning-direction">
    <p><strong>More class-2 support →</strong> Value coordinate 2 minus coordinate 1</p>
    <svg viewBox="0 0 430 190" role="img" aria-label={`Value A has margin minus 2. The current context has margin ${rounded(margin)}. Values B and C both have margin 2. Moving share from A toward B or C increases class-2 support.`}>
      <line x1="42" y1="96" x2="393" y2="96" stroke={muted} />
      {[-2, -1, 0, 1, 2].map(value => <g key={value}><path d={`M${x(value)} 90v12`} stroke={muted} /><text x={x(value)} y="122" textAnchor="middle">{value}</text></g>)}
      <circle cx={x(-2)} cy="96" r="6" fill={white} />
      <text x={x(-2)} y="69" textAnchor="middle">A</text>
      <circle cx={x(2)} cy="96" r="6" fill={amber} />
      <text x={x(2)} y="69" textAnchor="middle">B, C</text>
      <path d={`M${x(margin)} 88l8 8-8 8-8-8z`} fill="#111" stroke={amber} strokeWidth="2" />
      <text x={x(margin)} y="156" textAnchor="middle">context {margin.toFixed(3)}</text>
      <path d={`M${x(margin) + 7} 36H${x(2) - 8}m-7-5 7 5-7 5`} fill="none" stroke={amber} strokeWidth="2" />
    </svg>
    <figcaption>For this two-class output head, increasing the margin increases the probability of class 2. B and C have different vectors but the same margin (+2); either can improve on the current mixture (−0.661). The derivative quantifies the local effect of changing each score.</figcaption>
  </figure>;
}
