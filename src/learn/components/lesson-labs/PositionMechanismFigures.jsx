import { useState } from 'react';
import { NeuralSelect, NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { alibiCompetition } from '../../data/positional-encoding-models.js';
import { rotaryConventionTrace } from '../../data/position-convention-models.js';
const textVector = row => '[' + row.map(x => f(x, 5)).join(', ') + ']';
function ScrollFigure({
  title,
  width,
  height,
  description,
  children
}) {
  return <figure className="position-figure"><figcaption>{title}</figcaption><div className="position-mechanism-scroll" role="region" aria-label={title} tabIndex={0}><svg width={width} style={{
        minWidth: width,
        width: '100%',
        height: 'auto'
      }} viewBox={'0 0 ' + width + ' ' + height} role="img" aria-label={description}>{children}</svg></div><p>{description}</p></figure>;
}
export function AlibiDecompositionFigure({
  scores,
  keys,
  query,
  slope,
  result
}) {
  const reference = alibiCompetition(scores, keys, query, 0);
  const bound = Math.max(1, ...scores.map(Math.abs), ...result.bias.map(Math.abs), ...result.final.filter(Number.isFinite).map(Math.abs));
  const centers = [160, 315, 470, 640],
    height = 95 + keys.length * 52;
  return <ScrollFigure title="Content plus distance bias, then one softmax" width={730} height={height} description={'The first three columns share a signed logit scale from −' + f(bound) + ' to ' + f(bound) + '. The last uses probability 0–1. Thin white markers retain the same content with slope zero; they are a comparison, not extra attention mass.'}>
    {['Content', 'Distance bias', 'Final logit', 'Weight 0–1'].map((label, i) => <text key={label} x={centers[i]} y="25" textAnchor="middle" fill="#eee" fontSize="14">{label}</text>)}
    <text x="239" y="25" fill="#e6b854">+</text><text x="395" y="25" fill="#e6b854">=</text><text x="555" y="25" textAnchor="middle" fill="#eee" fontSize="11">softmax →</text>
    {keys.map((key, i) => {
      const y = 70 + i * 52;
      const values = [scores[i], result.bias[i], result.final[i]];
      return <g key={i}>
      <text x="7" y={y + 4} fill="#eee" fontSize="12">{'r' + i + ' @ ' + key}</text>
      {values.map((value, j) => <g key={j}><line x1={centers[j]} y1={y - 17} x2={centers[j]} y2={y + 17} stroke="#777" />{Number.isFinite(value) ? <><rect x={centers[j] + Math.min(0, value) / bound * 59} y={y - 7} width={Math.abs(value) / bound * 59} height="14" fill={j === 1 ? '#ba9fd1' : '#e6b854'} /><text x={centers[j]} y={y + 29} textAnchor="middle" fill="#ddd" fontSize="11">{f(value, 3)}</text></> : <text x={centers[j]} y={y + 5} textAnchor="middle" fill="#aaa" fontSize="12">masked</text>}</g>)}
      {result.legal[i] && <line x1={470 + scores[i] / bound * 59} x2={470 + scores[i] / bound * 59} y1={y - 12} y2={y + 12} stroke="#eee" strokeWidth="2" />}
      <rect x="596" y={y - 7} width="90" height="14" fill="#292929" />{result.weights && <><rect x="596" y={y - 7} width={90 * result.weights[i]} height="14" fill="#e6b854" /><line x1={596 + 90 * reference.weights[i]} x2={596 + 90 * reference.weights[i]} y1={y - 12} y2={y + 12} stroke="#eee" strokeWidth="2" /><text x="641" y={y + 29} fill="#ddd" fontSize="11" textAnchor="middle">{f(result.weights[i], 5)}</text></>}
    </g>;
    })}
  </ScrollFigure>;
}
export function RotaryConventionFigure({
  query,
  keyVector,
  queryId,
  keyId,
  base
}) {
  const [pairs, setPairs] = useState(1);
  const {
    permutation,
    packed,
    halfRotated,
    restored,
    adjacent,
    contributions
  } = rotaryConventionTrace(query, keyVector, queryId, keyId, base, pairs);
  return <>
    <ScrollFigure title="Move the coordinate basis along with the rotary convention" width={570} height={235} description="Adjacent order [q0,q1,q2,q3] becomes [q0,q2,q1,q3]. The half-split routine pairs slots (0,2) and (1,3). Applying the inverse permutation returns the same rotated vector; projection outputs must use the corresponding basis.">
      <text x="80" y="22" fill="#eee" fontSize="13">adjacent order</text><text x="380" y="22" fill="#eee" fontSize="13">half-split storage</text>
      {query.map((v, i) => <g key={i}><rect x="45" y={40 + 42 * i} width="150" height="30" fill="#222" stroke={i < 2 ? '#e6b854' : '#ba9fd1'} /><text x="120" y={60 + 42 * i} textAnchor="middle" fill="#eee" fontSize="13">{'q' + i + ' = ' + f(v, 4)}</text><path d={'M195 ' + (55 + 42 * i) + ' L365 ' + (55 + 42 * permutation.indexOf(i))} fill="none" stroke={i < 2 ? '#e6b854' : '#ba9fd1'} /><rect x="365" y={40 + 42 * i} width="155" height="30" fill="#222" stroke={permutation[i] < 2 ? '#e6b854' : '#ba9fd1'} /><text x="442" y={60 + 42 * i} textAnchor="middle" fill="#eee" fontSize="13">{'slot' + i + ': q' + permutation[i] + ' = ' + f(packed[i], 4)}</text></g>)}
    </ScrollFigure>
    <NeuralTable caption="Current query checks the coordinate mapping" headers={['Route', 'Rotated output']} rows={[['Adjacent pairs', textVector(adjacent)], ['Half-split output, its own storage order', textVector(halfRotated)], ['Inverse permutation to adjacent order', textVector(restored)]]} />
    <NeuralSelect label="Pairs receiving rotary position in the partial head" value={pairs} onChange={v => setPairs(Number(v))} options={[[0, '0: all content remains unrotated'], [1, '1: first pair rotates'], [2, '2: both pairs rotate']]} />
    <ScrollFigure title="A partial rotary head adds a positional part and a content part" width={570} height={195} description="This comparison uses the current query/key vectors and logical IDs. Each pair contributes its own dot product. The head still has four coordinates, so its final dot product is divided by √4.">
      {[0, 1].map(pair => <g key={pair}><text x="15" y={49 + 75 * pair} fill="#eee" fontSize="13">{'pair ' + pair}</text><rect x="85" y={25 + 75 * pair} width="280" height="45" fill="#222" stroke={pair < pairs ? '#e6b854' : '#ba9fd1'} /><text x="225" y={52 + 75 * pair} fill="#eee" textAnchor="middle" fontSize="13">{(pair < pairs ? 'rotate q and k, then dot: ' : 'unchanged content dot: ') + f(contributions[pair], 5)}</text><path d={'M365 ' + (47 + 75 * pair) + ' L430 86'} stroke="#ddd" fill="none" /></g>)}
      <circle cx="446" cy="86" r="17" fill="#222" stroke="#e6b854" /><text x="446" y="92" fill="#eee" textAnchor="middle">+</text><path d="M463 86 H505" stroke="#ddd" /><text x="530" y="83" fill="#eee" textAnchor="middle" fontSize="12">÷ 2</text><text x="530" y="104" fill="#eee" textAnchor="middle" fontSize="12">{f(contributions.reduce((a, b) => a + b, 0) / 2, 5)}</text>
    </ScrollFigure>
    <NeuralTable caption="Exact contributions in the selected partial head" headers={['Pair', 'Treatment', 'Dot contribution']} rows={contributions.map((value, i) => [i, i < pairs ? 'rotary' : 'unrotated content', f(value, 8)])} />
  </>;
}
