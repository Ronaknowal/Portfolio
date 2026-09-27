import fs from 'node:fs';

const file = 'src/learn/components/lesson-labs/RwkvMemoryLabs.jsx';
let source = fs.readFileSync(file, 'utf8');
function replace(before, after) {
  if (!source.includes(before)) throw new Error(`Missing unique edit: ${before}`);
  source = source.replace(before, after);
}
replace("const [input, setInput] = useState(summaryDefaults), [selected, setSelected] = useState(0);", "const [input, setInput] = useState(summaryDefaults), [selected, setSelected] = useState(3), [baseline, setBaseline] = useState(summaryDefaults);");
replace("const rows = kernelSummary(input), row = rows[selected];", "const rows = kernelSummary(input), row = rows[selected], pinned = kernelSummary(baseline);");
replace("setInput(summaryDefaults()); setSelected(0);", "setInput(summaryDefaults()); setSelected(3); setBaseline(summaryDefaults());");
replace('<NeuralTable caption="The entire current causal computation"', '<MemoryContributionRibbon title="Current numerator contribution from each record" values={row.weights.map((weight, i) => weight === null ? 0 : weight * input.values[i])} /><div className="neural-buttons"><button onClick={() => setBaseline(structuredClone(input))}>Pin current summary baseline</button></div><p>Pinned final answer: {f(pinned.at(-1).output)} from {baseline.values.length} records; current final answer: {f(rows.at(-1).output)}. The pinned input is unchanged by later edits.</p><NeuralTable caption="The entire current causal computation"');
replace('<button onClick={() => { setInput(defaults()); setSelected(2); }}>Reset weighted memory</button>', '<button disabled={input.values.length >= 12} onClick={() => setInput({ ...input, values: [...input.values, 2], keys: [...input.keys, 0] })}>Append token</button><button disabled={input.values.length <= 1} onClick={() => { setInput({ ...input, values: input.values.slice(0, -1), keys: input.keys.slice(0, -1) }); setSelected(0); }}>Remove last token</button><button onClick={() => { setInput(defaults()); setSelected(2); }}>Reset weighted memory</button>');
replace('<NeuralTable caption="Answer and saved state have separate meanings"', '<div className="rwkv-panels"><MemoryContributionRibbon title={`Read token ${selected + 1}: normalized value contributions`} values={rows[selected].weights.map((weight, i) => weight * input.values[i])} /><MemoryContributionRibbon title={`Stored history through token ${selected + 1}: scaled signed writes`} values={input.values.slice(0, selected + 1).map((value, i) => value * Math.exp(input.keys[i] + input.offset + (selected - i) * Math.log(input.retention) - rows[selected].p))} /></div><p>Left is the current answer mixture, including u. Right sums to saved a and uses ordinary keys plus decay, with no u. Both views update from the same edited records.</p><NeuralTable caption="Answer and saved state have separate meanings"');
replace('return <figure className="rwkv-figure"><figcaption>{kind === \'correction\'', 'return <figure className="rwkv-figure"><MemoryAddressCompass keys={keys.slice(0, 2)} memory={rows.at(-1).delta} /><figcaption>{kind === \'correction\'');
fs.writeFileSync(file, source);

const manuscriptPath = 'docs/teaching/drafts/rwkv-linear-attention-models/lesson.md';
let manuscript = fs.readFileSync(manuscriptPath, 'utf8');
manuscript = manuscript.replace('includes a constant-value case', 'includes a constant-value case')
  .replace('begins with an visible sequence', 'begins with a visible sequence')
  .replace('choose a retrieval query and Show', 'choose a retrieval query and show')
  .replace('Include both checked fixtures and reveal the resulting matrix after the learner has interpreted the signs.', 'Show both checked fixtures and every resulting matrix immediately as the removal direction changes.')
  .replace('then[2,7]', 'then [2,7]').replace('then[5,7]', 'then [5,7]').replace('key by[.6,.8]', 'key by [.6,.8]')
  .replace('with unit norm, and k̃=βk', 'with unit norm, and k̃=βk');
fs.writeFileSync(manuscriptPath, manuscript);
console.log('Updated local visual and live-state teaching support.');
