import { entranceMessages, weightedReadExample, silentStepExamples, sameMeanPaths } from '../../data/long-context-intuition.js';
import { formatNeural as format } from './NeuralLessonElements.jsx';
import './long-context-labs.css';

function IntuitionFigure({ title, children }) {
  return <figure className="long-figure long-intuition"><figcaption>{title}</figcaption>{children}</figure>;
}

export function EntranceMessageFigure() {
  return <IntuitionFigure title="The message was read earlier. Is it still available now?">
    <ol className="long-message-timeline">{entranceMessages.map(message => <li key={message.position} className={message.available ? 'is-visible' : ''}>
      <span className="long-message-position">{message.position}</span>
      <div><strong>{message.text}</strong><span>{message.status}</span></div>
    </li>)}</ol>
    <p><strong>Last two messages only:</strong> the current read cannot directly consult the entrance note. Nothing in “bring your badge” supplies its missing detail.</p>
    <p><strong>A memory path:</strong> keep the earlier record, carry its influence in a summary, or reread an available input bank. These are the three mechanisms we will build.</p>
    <small>Constructed access example. Message rows stand in for input positions; no trained model answer is being shown.</small>
  </IntuitionFigure>;
}

export function WeightedReadSharesFigure() {
  const totalSupport = weightedReadExample.reduce((sum, row) => sum + row.support, 0);
  const output = weightedReadExample.reduce((sum, row) => sum + row.contribution, 0);
  return <IntuitionFigure title="Six equal shares distribute one read among three donors">
    <div className="long-support-shares" aria-label="Donor A receives two of six shares, B one, and C three">{weightedReadExample.flatMap(row => Array.from({ length: row.support }, (_, share) => <span key={`${row.donor}-${share}`} className={`long-donor-${row.donor.toLowerCase()}`} aria-hidden="true">{row.donor}</span>))}</div>
    <dl className="long-contribution-list">{weightedReadExample.map(row => <div key={row.donor}>
      <dt>{row.donor} · value {row.value}</dt>
      <dd>{row.support}/{totalSupport} of the read × {row.value} = <strong>{row.support * row.value}/{totalSupport}</strong></dd>
    </div>)}</dl>
    <p className="long-result">Add the contributions: (4 + 4 + 24)/6 = <strong>32/6 ≈ {format(output, 3)}</strong>.</p>
    <p>Each square is an equal share of influence, not an extra observation. C receives half the read and contributes 4 to the answer.</p>
  </IntuitionFigure>;
}

export function SilentStepGateFigure() {
  return <IntuitionFigure title="No new signal: which control can preserve the old state?">
    <p>Every row starts at <strong>h = 0.6</strong> with new input <strong>x = 0</strong>. Fixed base a = 0.8.</p>
    <div className="long-silent-steps">{silentStepExamples.map(example => <section key={example.label}>
      <h4>{example.label}</h4>
      <p>Input gate i = {example.input}; recurrence gate r = {format(example.recurrence, 3)}.<br />Retention = {format(example.decay, 2)}; injected signal = 0.</p>
      <div className="long-static-bar" role="img" aria-label={`State after the step ${format(example.state, 2)}, common scale zero to 0.6`}><span style={{ width: `${example.state / .6 * 100}%` }} /></div>
      <div className="long-static-bar-scale"><span>0</span><span>0.6</span></div>
      <p>{format(example.decay, 2)} × 0.6 + 0 = <strong>{format(example.state, 2)}</strong></p>
    </section>)}</div>
    <p>The second row still loses 0.12. Closing a path already carrying zero cannot stop decay on the other path. The third row is the exact r = 0 mathematical limit; finite sigmoid gates approach it.</p>
    <small>Static comparison on a shared linear scale. Change the gates yourself in the live investigation below.</small>
  </IntuitionFigure>;
}

function ConstructedPath({ path }) {
  const projectX = x => 36 + 200 * x;
  const projectY = y => 224 - 200 * y;
  return <section className="long-intuition-path"><h4>{path.label}</h4>
    <svg viewBox="0 0 270 270" role="img" aria-label={`${path.label}. Five ordered points. Mean x and y both 0.5.`}>
      <line x1="36" y1="24" x2="36" y2="224" stroke="#777" />
      <line x1="36" y1="224" x2="236" y2="224" stroke="#777" />
      <line x1="36" y1="124" x2="236" y2="124" stroke="#555" strokeDasharray="3 5" />
      <line x1="136" y1="24" x2="136" y2="224" stroke="#555" strokeDasharray="3 5" />
      <text x="36" y="249" textAnchor="middle">0</text><text x="236" y="249" textAnchor="middle">1</text>
      <text x="21" y="228" textAnchor="end">0</text><text x="21" y="30" textAnchor="end">1</text>
      <text x="136" y="264" textAnchor="middle">x coordinate</text><text x="7" y="14">y</text>
      <polyline points={path.points.map(point => `${projectX(point.x)},${projectY(point.y)}`).join(' ')} fill="none" stroke="#e2b55a" strokeWidth="2.5" />
      {path.points.map((point, index) => <g key={index}>
        {index === 4 ? <rect x={projectX(point.x) - 4} y={projectY(point.y) - 4} width="8" height="8" fill="#eee" /> : <circle cx={projectX(point.x)} cy={projectY(point.y)} r={index === 0 ? 5 : 3} fill={index === 0 ? '#090909' : '#e2b55a'} stroke={index === 0 ? '#eee' : '#e2b55a'} strokeWidth="2" />}
        <text x={projectX(point.x)} y={projectY(point.y) - 13} textAnchor="middle">{index + 1}</text>
      </g>)}
      <path d="M130 118l12 12m0-12l-12 12" stroke="#eee" strokeWidth="2" />
    </svg>
    <p>Mean y = ({path.heights.join(' + ')})/5 = <strong>{path.mean.y}</strong>.</p>
  </section>;
}

export function PathMeanCollisionFigure() {
  return <IntuitionFigure title="Two shapes become indistinguishable if only their center survives">
    <div className="long-path-comparison">{sameMeanPaths.map(path => <ConstructedPath key={path.label} path={path} />)}</div>
    <p>Follow points <strong>1 → 5</strong>. Open circle = start; square = end; × = mean. Both horizontal means are (0 + .25 + .5 + .75 + 1)/5 = .5.</p>
    <p className="long-result">A later classifier receives <strong>(.5, .5)</strong> for either path. It cannot recover which shape produced those same two numbers.</p>
    <small>Constructed paths with equal x/y scales, not UCI observations. A learned latent read need not equal this uniform mean.</small>
  </IntuitionFigure>;
}
