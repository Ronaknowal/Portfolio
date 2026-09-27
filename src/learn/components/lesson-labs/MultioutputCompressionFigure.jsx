import './multioutput-compression.css';

export default function MultioutputCompressionFigure() {
  const horizontal = Math.SQRT1_2;
  const x = value => 165 + 110 * value;
  const y = value => 175 - 110 * value;
  return <figure className="mo-compression-intuition" data-figure="rare-label-projection">
    <p><strong>A small discarded direction can contain every rare positive</strong></p>
    <svg viewBox="0 0 330 245" role="img" aria-label="Four locations in centered label coordinates: ninety-nine negatives and one rare positive at each common-label state. Rank-one horizontal projection places both rare positives on the zero rare-coordinate line.">
      <path d={`M40 ${y(0)}H291M165 45V197`} stroke="#727272" fill="none" />
      <text x="22" y="24">centered rare coordinate</text>
      <text x="145" y={y(1) + 5} textAnchor="end">1</text>
      <text x="150" y={y(0) + 18} textAnchor="end">0</text>
      {[-horizontal, horizontal].map((value, index) => <g key={value}>
        <path d={`M${x(value)} ${y(.99)}V${y(0)}`} stroke="#e8b44a" strokeWidth="2" strokeDasharray="5 5" />
        <circle cx={x(value)} cy={y(.99)} r="6" fill="#e8b44a" />
        <circle cx={x(value)} cy={y(-.01)} r="6" fill="#ddd" />
        <path d={`M${x(value) - 8} ${y(0) - 8}l16 16m-16 0l16 -16`} stroke="#e8b44a" strokeWidth="2" />
        <text x={x(value)} y="215" textAnchor="middle">{index ? '+0.707' : '−0.707'}</text>
      </g>)}
      <text x="165" y="238" textAnchor="middle">common-label coordinate</text>
    </svg>
    <figcaption>Calculated positions from the program's 200-row fixture. At each common state, 99 rows have rare label 0 (white dot) and one has rare label 1 (amber dot). Marker size does not encode count. The amber crosses show the rank-one reconstruction: centered rare coordinate 0. Adding its mean .01 makes every rare score .01, below a .5 threshold. The common coordinate varies by ±√.5; axes use the same physical unit scale.</figcaption>
  </figure>;
}
