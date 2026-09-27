import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralPlot, NeuralTable, formatNeural } from './NeuralLessonElements.jsx';
import './neural-lesson-neutral.css';
import './xlstm-memory-labs.css';
export const f = formatNeural;
export const vec = x => '[' + x.map(v => f(v, 6)).join(', ') + ']';
export function XFigure({
  title,
  width = 640,
  height,
  children,
  description
}) {
  return <figure className="xl-figure"><figcaption>{title}</figcaption><div className="xl-scroll" role="region" aria-label={title} tabIndex={0}><svg viewBox={`0 0 ${width} ${height}`} style={{
        width: '100%',
        minWidth: width
      }} role="img" aria-label={description || title}>{children}</svg></div>{description && <p className="xl-legend">{description}</p>}</figure>;
}
export function XArrow({
  x1,
  y1,
  x2,
  y2,
  color = '#e6bb60',
  dashed = false
}) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  return <g fill={color} stroke={color}><line x1={x1} y1={y1} x2={x2} y2={y2} strokeDasharray={dashed ? '5 4' : undefined} /><path d={`M${x2},${y2}L${x2 - 7 * Math.cos(a - .5)},${y2 - 7 * Math.sin(a - .5)}L${x2 - 7 * Math.cos(a + .5)},${y2 - 7 * Math.sin(a + .5)}Z`} /></g>;
}
export function XNode({
  x,
  y,
  width = 135,
  height = 44,
  lines,
  color = '#555'
}) {
  return <g><rect x={x} y={y} width={width} height={height} fill="#191919" stroke={color} rx="3" />{lines.map((t, i) => <text key={i} x={x + width / 2} y={y + height / 2 + (i - (lines.length - 1) / 2) * 18 + 5} textAnchor="middle">{t}</text>)}</g>;
}
export function XVector({
  label,
  value,
  onChange,
  min = -4,
  max = 4
}) {
  return <fieldset className="xl-vector"><legend>{label}</legend><div className="neural-controls">{value.map((v, i) => <NeuralNumber key={i} label={`${label} coordinate ${i + 1}`} value={v} min={min} max={max} range={false} onChange={n => onChange(value.map((old, j) => j === i ? n : old))} />)}</div></fieldset>;
}
export function XValues({
  caption,
  rows
}) {
  return <NeuralTable caption={caption} headers={['Quantity', 'Current value']} rows={rows} />;
}
export function XPlot(props) {
 const series=props.series.map(row=>({...row,values:row.values||row.points})),all=series.flatMap(row=>row.values),bounds=k=>{const low=Math.min(...all.map(p=>p[k])),high=Math.max(...all.map(p=>p[k])),pad=low===high?.1:(high-low)*.04;return[low-pad,high+pad];},[xmin,xmax]=props.xDomain||bounds(0),[ymin,ymax]=props.yDomain||bounds(1),height=props.height||270,x=value=>105+(value-xmin)/(xmax-xmin)*295,y=value=>height-55-(value-ymin)/(ymax-ymin)*(height-90),tick=value=>{const text=f(value,2);return text.length>12?Number(value).toExponential(3):text;},points=[...(props.points||[]),...series.filter(row=>row.values.length===1).map(row=>({x:row.values[0][0],y:row.values[0][1],label:row.label,color:row.color,selected:true}))];
 const exactRows=[...series.flatMap(row=>row.values.map(([a,b])=>[row.label,f(a,8),f(b,8)])),...(props.points||[]).map(point=>[point.label,f(point.x,8),f(point.y,8)])];
 return <><XFigure title={props.title} width={460} height={height} description={'Horizontal: '+props.xLabel+'. Vertical: '+props.yLabel+'. Exact coordinates are available below.'}>
 <line x1="105" x2="400" y1={height-55} y2={height-55} stroke="#888"/><line x1="105" x2="105" y1="35" y2={height-55} stroke="#888"/>
 {[0,.5,1].map(t=><g key={t}><text x={105+295*t} y={height-29} textAnchor="middle">{tick(xmin+(xmax-xmin)*t)}</text><text x="95" y={y(ymin+(ymax-ymin)*t)+5} textAnchor="end">{tick(ymin+(ymax-ymin)*t)}</text></g>)}
 {ymin<0&&ymax>0&&<line x1="105" x2="400" y1={y(0)} y2={y(0)} stroke="#555" strokeDasharray="4 3"/>}
 {series.map((row,i)=><polyline key={row.label} points={row.values.map(([a,b])=>x(a)+','+y(b)).join(' ')} fill="none" stroke={row.color||['#e6bb60','#87bdf1','#e9979f'][i%3]} strokeWidth="2" strokeDasharray={row.dashed?'5 4':undefined}/>)}
 {points.map((point,i)=><circle key={i} cx={x(point.x)} cy={y(point.y)} r={point.selected?6:4} fill={point.color||'#e6bb60'} stroke={point.selected?'white':undefined}><title>{point.label}</title></circle>)}
 </XFigure><ul className="xl-plot-legend">{series.map((row,i)=><li key={row.label}><span style={{borderColor:row.color||['#e6bb60','#87bdf1','#e9979f'][i%3],borderStyle:row.dashed?'dashed':'solid'}}/>{row.label}</li>)}</ul><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact plotted values</h4><NeuralTable caption={props.title} headers={['Series',props.xLabel,props.yLabel]} rows={exactRows}/></section></>;
}
export function XMatrix({
  title,
  values,
  rowLabel = 'key',
  columnLabel = 'value',
  selected = null
}) {
  const maximum = Math.max(1e-12, ...values.flat().map(Math.abs)),
    cellWidth = values[0].length > 4 ? 55 : 105,
    cellHeight = 44,
    width = Math.max(340, 95 + values[0].length * cellWidth),
    height = 75 + values.length * cellHeight;
  return <><XFigure title={title} width={width} height={height} description={`Rows are ${rowLabel} coordinates; columns are ${columnLabel} coordinates. Positive is gold, negative is rose; exact signed values are printed.`}>{values[0].map((_, j) => <text key={j} x={80 + j * cellWidth + cellWidth / 2} y="23" textAnchor="middle">{columnLabel}{j + 1}</text>)}{values.map((row, i) => <g key={i}><text x="5" y={61 + i * cellHeight}>{rowLabel}{i + 1}</text>{row.map((v, j) => <g key={j}><rect x={80 + j * cellWidth} y={33 + i * cellHeight} width={cellWidth - 2} height={cellHeight - 2} fill={v < 0 ? '#843b45' : '#8e6b23'} fillOpacity={.12 + .68 * Math.abs(v) / maximum} stroke={selected && selected[0] === i && selected[1] === j ? 'white' : '#555'} strokeWidth={selected && selected[0] === i && selected[1] === j ? 3 : 1} /><text className={cellWidth < 80 ? 'xl-small' : undefined} x={80 + j * cellWidth + cellWidth / 2} y={60 + i * cellHeight} textAnchor="middle">{f(v, cellWidth < 80 ? 3 : 5)}</text></g>)}</g>)}</XFigure><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Complete matrix values</h4><NeuralTable caption={title} headers={[rowLabel, ...values[0].map((_, i) => columnLabel + (i + 1))]} rows={values.map((r, i) => [i + 1, ...r.map(x => f(x, 10))])} /></section></>;
}
export function XLedger({
  rows,
  trace,
  step,
  output,
  scaled = true
}) {
  const current = trace[step],
    weights = current.weights,
    contributions = current.contributions,
    max = Math.max(1e-12, ...weights, ...contributions.map(Math.abs)),
    x = i => 85 + i * 90,
    zero = 140;
  return <><XFigure title={`Evidence surviving at step ${step + 1}: ${scaled ? 'current stored scale' : 'raw scale'}`} width={Math.max(480, 120 + weights.length * 90)} height={350} description="Upper bars show positive mass. Lower signed bars retain each candidate's contribution; a negative candidate does not become negative weight."><text x="10" y="25">mass</text><text x="10" y="190">content</text>{weights.map((w, i) => <g key={i}><rect x={x(i)} y={zero - w / max * 95} width="35" height={w / max * 95} fill="#e6bb60" /><text x={x(i) + 17} y="162" textAnchor="middle">t{i + 1}</text><text x={x(i) + 17} y="183" textAnchor="middle">{f(w, 4)}</text><line x1={x(i) - 5} x2={x(i) + 40} y1="245" y2="245" stroke="#777" /><rect x={x(i)} y={245 - Math.max(0, contributions[i]) / max * 48} width="35" height={Math.abs(contributions[i]) / max * 48} fill={contributions[i] < 0 ? '#e9979f' : '#87bdf1'} /><text x={x(i) + 17} y="309" textAnchor="middle">z={f(rows[i].value, 3)}</text><text x={x(i) + 17} y="333" textAnchor="middle">wz={f(contributions[i], 4)}</text></g>)}</XFigure><XFigure title="Keep content, evidence mass and exposed output distinct" width={620} height={155}><XNode x={5} y={15} width={170} lines={['content c', f(current.cell, 7)]} /><XNode x={5} y={91} width={170} lines={['positive mass n', f(current.normalizer, 7)]} /><XArrow x1={175} y1={37} x2={225} y2={70} /><XArrow x1={175} y1={113} x2={225} y2={70} /><XNode x={225} y={48} width={160} lines={['ratio c/n', f(current.estimate, 7)]} /><XArrow x1={385} y1={70} x2={440} y2={70} /><XNode x={440} y={48} width={175} lines={[`× output gate ${f(output)}`, f(current.hidden, 7)]} /></XFigure></>;
}
export function XAddressPlane({
  keys,
  query
}) {
  const bound = Math.max(1, ...keys.flat().map(Math.abs), ...query.map(Math.abs)),
    x = v => 180 + v / bound * 135,
    y = v => 175 - v / bound * 135;
  return <XFigure title="Address alignment: keys and current query share one coordinate plane" width={360} height={355} description={`Equal axes span ±${f(bound)}. Key arrows are gold; the query is blue and dashed.`}><line x1="35" y1="175" x2="325" y2="175" stroke="#777" /><line x1="180" y1="30" x2="180" y2="320" stroke="#777" />{keys.map((k, i) => <g key={i}><XArrow x1={180} y1={175} x2={x(k[0])} y2={y(k[1])} /><text x={Math.max(40, Math.min(300, x(k[0]) + 8))} y={Math.max(22, Math.min(315, y(k[1]) - 8))}>k{i + 1}</text></g>)}<XArrow x1={180} y1={175} x2={x(query[0])} y2={y(query[1])} color="#87bdf1" dashed /><text x="10" y="20">y</text><text x="310" y="347">x</text><text x="35" y="344">−{f(bound, 2)}</text><text x="295" y="344">{f(bound, 2)}</text></XFigure>;
}
export function XCausalGrid({
  rows,
  selected = rows.length - 1,
  coefficients
}) {
  const side = 34,
    width = Math.max(400, 100 + rows.length * side),
    height = 100 + rows.length * side,
    max = Math.max(1e-12, ...coefficients.flat().map(Math.abs));
  return <XFigure title="Causal coefficient matrix: current query time down, written record across" width={width} height={height} description="Future cells are crossed out. Signed entries are address coefficients, not probabilities.">{rows.map((_, j) => <text key={j} x={85 + j * side} y="25" textAnchor="middle">{j + 1}</text>)}{rows.map((_, i) => <g key={i}><text x="35" y={60 + i * side}>{i + 1}</text>{rows.map((__, j) => <g key={j}><rect x={68 + j * side} y={37 + i * side} width={side - 2} height={side - 2} fill={j > i ? '#141414' : coefficients[i][j] < 0 ? '#843b45' : '#8e6b23'} fillOpacity={j > i ? 1 : .12 + .75 * Math.abs(coefficients[i][j]) / max} stroke={i === selected ? 'white' : '#555'} />{j > i ? <path d={`M${73 + j * side},${42 + i * side}l22,22m0,-22l-22,22`} stroke="#555" /> : <text className="xl-small" x={84 + j * side} y={58 + i * side} textAnchor="middle">{f(coefficients[i][j], 2)}</text>}</g>)}</g>)}</XFigure>;
}
export function XChunkJoin({
  result
}) {
  return <><XFigure title="Old state and local writes join before a single denominator" height={260}><XNode x={10} y={20} width={215} lines={['incoming numerator', vec(result.incomingNumerator)]} /><XNode x={10} y={107} width={215} lines={['local numerator', vec(result.localNumerator)]} /><XArrow x1={225} y1={42} x2={280} y2={84} /><XArrow x1={225} y1={129} x2={280} y2={84} /><XNode x={280} y={62} width={150} lines={['add numerators']} /><XArrow x1={430} y1={84} x2={485} y2={84} /><XNode x={485} y={62} width={145} lines={['divide once', 'current read below']} /><text x="10" y="190">signed mass: {f(result.incomingMass, 5)} + {f(result.localMass, 5)} = {f(result.mass, 5)}</text><XArrow x1={370} y1={190} x2={555} y2={106} /><text x="10" y="220">denominator max(|combined mass|,1) = {f(result.denominator, 7)}</text><text x="10" y="248">read = {vec(result.read)}</text></XFigure></>;
}
export function XImage({
  title,
  pixels,
  selectedRow = -1
}) {
  return <figure className="xl-image"><svg viewBox="0 0 80 80" role="img" aria-label={title}>{pixels.flat().map((v, i) => {
        const c = v / 16 * 255;
        return <rect key={i} x={i % 8 * 10} y={Math.floor(i / 8) * 10} width="10" height="10" fill={`rgb(${c},${c},${c})`} />;
      })}{selectedRow >= 0 && <rect x=".5" y={selectedRow * 10 + .5} width="79" height="9" fill="none" stroke="#e6bb60" strokeWidth="1.5" />}</svg><figcaption>{title}</figcaption></figure>;
}
export function XSignedWrites({
  contributions,
  denominator
}) {
  const extent = Math.max(1e-12, ...contributions.flat().map(Math.abs)),
    height = 100 + contributions.length * 44;
  return <XFigure title="Signed write contributions add before the one denominator" width={640} height={height} description="Gold extends right for positive contributions; rose extends left for negative contributions. Both value coordinates share the same scale.">{[0, 1].map(j => <g key={j}><text x={190 + j * 300} y="25" textAnchor="middle">value coordinate {j + 1}</text><line x1={190 + j * 300} y1="38" x2={190 + j * 300} y2={height - 50} stroke="#aaa" />{contributions.map((row, i) => <g key={i}><text x={5 + j * 300} y={64 + i * 44}>write {i + 1}</text><rect x={190 + j * 300 + Math.min(0, row[j]) / extent * 105} y={46 + i * 44} width={Math.abs(row[j]) / extent * 105} height="21" fill={row[j] < 0 ? '#e9979f' : '#e6bb60'} /><text x={190 + j * 300} y={85 + i * 44} textAnchor="middle">{f(row[j], 6)}</text></g>)}</g>)}<text x="10" y={height - 13}>Column sums ÷ {f(denominator, 8)} = {vec([0, 1].map(j => contributions.reduce((sum, row) => sum + row[j], 0) / denominator))}</text></XFigure>;
}
