import './depthwise-intuition.css';

export function DilationIntervalFigure() {
  const offsetX = v => 30+(v+5)*28;
  return <figure className="depthwise-intuition" data-figure="dilation-interval-join"><svg viewBox="0 0 340 266" role="img" aria-label="Existing reachable integers are minus one, zero, one. Shifting them by minus three, zero, three makes all integers minus four to four. Shifting by minus four, zero, four leaves holes at minus two and two.">
    {[3,4].map((d,row)=><g key={d}>
      <text x="15" y={22+row*126}>S=1, next dilation d={d}</text>
      {[-d,0,d].map((center,i)=><g key={center}><path d={`M${offsetX(center-1)} ${46+row*126+i*13}H${offsetX(center+1)}`} stroke="#e8b44a" strokeWidth="3"/>{[-1,0,1].map(v=><circle key={v} cx={offsetX(center+v)} cy={46+row*126+i*13} r="4" fill="#e8b44a"/>)}</g>)}
      {Array.from({length:11},(_,i)=>i-5).map(v=><text key={v} x={offsetX(v)} y={100+row*126} textAnchor="middle">{v}</text>)}
      {row===1&&[-2,2].map(v=><path key={v} d={`M${offsetX(v)-4} 202l8 8m0-8l-8 8`} stroke="#ddd" strokeWidth="2"/>)}
    </g>)}
    <text x="170" y="254" textAnchor="middle">original-input integer offset</text>
  </svg><figcaption>Each amber three-dot segment is a shifted copy of the previous reachable set. At d=3 the copies touch as consecutive integers: −2 is followed by −1, and 1 by 2. At d=4, −2 and 2 are missing (crosses). Compare the right end of the left copy, −d+S, with the left end of the middle copy, −S: no missing integer requires −S≤−d+S+1, or d≤2S+1.</figcaption></figure>;
}
