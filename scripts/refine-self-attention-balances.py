from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
p=root/'src/learn/components/lesson-labs/SelfAttentionLabs.jsx'
text=p.read_text(encoding='utf-8')
anchor='export function SelfAttentionMixtureLab()'
assert anchor in text
helper='''function ContributionBalances({ first, second }) {
  const scale = Math.max(1e-12, ...first.map(Math.abs), ...second.map(Math.abs));
  return <div className="attention-balances">{[first, second].map((values, index) => <figure key={index}><figcaption>Row {index ? 'B' : 'A'} contributions · shared scale ±{f(scale)}</figcaption>{values.map((value, donor) => <div className="attention-balance-row" key={donor}><span>{donor + 1}</span><div><i style={{ left: value < 0 ? `${50 + 50 * value / scale}%` : '50%', width: `${50 * Math.abs(value) / scale}%`, background: index ? '#83b4e8' : '#e0b660' }} /></div><output>{f(value)}</output></div>)}<p>Sum = {f(values.reduce((a, b) => a + b, 0))}</p></figure>)}</div>;
}
'''
text=text.replace(anchor,helper+anchor).replace('<><NeuralTable caption="Two weighted balances', '<><ContributionBalances first={result.contributionsA} second={result.contributionsB} /><NeuralTable caption="Two weighted balances')
p.write_text(text,encoding='utf-8',newline='\n')
p=root/'src/learn/components/lesson-labs/SelfAttentionFigures.jsx'
text=p.read_text(encoding='utf-8').replace('Aquerycompareswiththreekeys, eachmatchnormalizestoaweight. Separately, threevaluesmultiplybythoseweightsandconvergeatAoutput.', "A query compares with three keys; each match normalizes to a weight. Separately, three values multiply by those weights and converge at the output for A.").replace('class4 hand','class 4 hand')
p.write_text(text,encoding='utf-8',newline='\n')
p=root/'src/learn/components/lesson-labs/self-attention-labs.css'
with p.open('a',encoding='utf-8') as stream:
 stream.write('''
.attention-balances { display: flex; flex-wrap: wrap; gap: 1rem; }
.attention-balances > figure { flex: 1 1 240px; margin: 1rem 0; min-width: 0; }
.attention-balance-row { display: grid; grid-template-columns: 1.5rem minmax(60px,1fr) 4.5rem; gap: .5rem; align-items: center; }
.attention-balance-row > div { position: relative; height: .8rem; background: linear-gradient(90deg,#252525 49.8%,#aaa 49.8%,#aaa 50.2%,#252525 50.2%); }
.attention-balance-row i { position: absolute; height: 100%; }
.attention-balance-row output { font-variant-numeric: tabular-nums; font-size: .85rem; }
''')
for topic in ('rwkv-linear-attention-models','self-attention-multi-head-attention'):
 p=root/f'docs/teaching/deep-learning-completion/{topic}/implementation-scope.json'
 record=json.loads(p.read_text());record['runtimeFiles'].append('src/learn/components/lesson-labs/neural-number-control.css')
 p.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
