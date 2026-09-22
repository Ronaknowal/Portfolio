import { useState } from 'react';
import { packedSetView, parityTrace, sparseBitTrace, toggleMember, twoSingletonPartition, wordBits, wordInterpretation } from '../../data/bitwise-foundations-models.js';
import './bitwise-foundations-labs.css';
const showSet = members => `{${members.join(', ')}}`;
function BitRow({
  value,
  width,
  label,
  onToggle,
  emphasis = [],
  annotations
}) {
  const cells = wordBits(value, width);
  return <div className="bitwise-row">
    <span className="bitwise-row__label">{label}</span>
    <div className="bitwise-cells" style={{
      '--bit-columns': width
    }}>
      {cells.map((cell, index) => {
        const cellClass = `bitwise-cell${cell.bit ? ' is-one' : ''}${emphasis.includes(cell.position) ? ' is-emphasized' : ''}`;
        const content = <><small>{annotations ? annotations[index] : `b${cell.position}`}</small><strong>{cell.bit}</strong></>;
        return onToggle ? <button type="button" className={cellClass} key={cell.position} aria-label={`${label}, bit ${cell.position}`} aria-pressed={cell.bit === 1} onClick={() => onToggle(cell.position)}>{content}</button> : <span className={cellClass} key={cell.position}>{content}</span>;
      })}
    </div>
    <span className="bitwise-row__value">{value}</span>
  </div>;
}
function StepButtons({
  step,
  count,
  setStep,
  label = 'Step'
}) {
  return <div className="bitwise-buttons">
    <button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Previous</button>
    <span>{label} {step} of {count - 1}</span>
    <button type="button" disabled={step === count - 1} onClick={() => setStep(step + 1)}>Next</button>
    <button type="button" onClick={() => setStep(0)}>Restart trace</button>
  </div>;
}
export function PackedSetLab() {
  const [width, setWidth] = useState(6);
  const [left, setLeft] = useState(37);
  const [right, setRight] = useState(22);
  const [operation, setOperation] = useState('intersection');
  const [notice, setNotice] = useState('Toggle any labelled bit to add or remove that member.');
  const view = packedSetView(left, right, width, operation);
  const operationLabels = {
    intersection: 'A ∩ B',
    union: 'A ∪ B',
    difference: 'A ∖ B',
    symmetric: 'A △ B',
    complement: 'Universe ∖ A'
  };
  function reset() {
    setWidth(6);
    setLeft(37);
    setRight(22);
    setOperation('intersection');
    setNotice('Reset to A = {0, 2, 5}, B = {1, 2, 4}, with six possible members.');
  }
  return <section className="bitwise-lab" id="packed-set-lab" aria-labelledby="packed-set-title">
    <p className="bitwise-kicker">ONE COLUMN · ONE MEMBERSHIP DECISION</p>
    <h3 id="packed-set-title">Can one integer answer a set query?</h3>
    <p>Inspect the shared members before choosing intersection. A lit cell means present; its label fixes which member it represents. Repeatedly adding a member would leave that bit at 1.</p>
    <div className="bitwise-controls">
      <label>Universe width<select aria-label="Universe width" value={width} onChange={event => {
          const nextWidth = Number(event.target.value);
          setWidth(nextWidth);
          setLeft(left % 2 ** nextWidth);
          setRight(right % 2 ** nextWidth);
          setNotice(`Universe is now 0 through ${nextWidth - 1}. Higher positions were discarded; expanding does not restore them.`);
        }}><option value="4">4 members</option><option value="6">6 members</option><option value="8">8 members</option></select></label>
      <label>Set query<select aria-label="Set query" value={operation} onChange={event => setOperation(event.target.value)}>
        <option value="intersection">AND · both</option><option value="union">OR · either</option>
        <option value="difference">A minus B</option><option value="symmetric">XOR · exactly one</option>
        <option value="complement">Complement A</option>
      </select></label>
    </div>
    <BitRow value={left} width={width} label="A" onToggle={position => setLeft(toggleMember(left, width, position))} />
    <p className="bitwise-members">A = {showSet(view.leftMembers)}</p>
    <BitRow value={right} width={width} label="B" onToggle={position => setRight(toggleMember(right, width, position))} />
    <p className="bitwise-members">B = {showSet(view.rightMembers)}</p>
    <div className="bitwise-result"><BitRow value={view.result} width={width} label="Result" />
      <p aria-live="polite">{operationLabels[operation]} = <strong>{showSet(view.resultMembers)}</strong> · mask {view.result}</p>
    </div>
    <p className="bitwise-note">Columns run from high bit to bit 0. The decimal mask is the sum of the occupied powers of two; its numerical size is not the number of members. Complement ignores B and depends on the displayed universe.</p>
    <p role="status">{notice}</p><button type="button" onClick={reset}>Reset sets</button>
  </section>;
}
export function WordInterpretationLab() {
  const [width, setWidth] = useState(8);
  const [unsigned, setUnsigned] = useState(246);
  const [shift, setShift] = useState(2);
  const view = wordInterpretation(unsigned, width, shift);
  const rightLabels = view.rightOrigins.map(cell => cell.source === null ? 'fill' : `←b${cell.source}`);
  const leftLabels = view.leftOrigins.map(cell => cell.source === null ? 'fill' : `←b${cell.source}`);
  const weightSum = signed => view.cells.filter(cell => cell.bit).map(cell => signed ? cell.signedWeight : cell.weight).join(' + ') || '0';
  return <section className="bitwise-lab" id="word-interpretation-lab" aria-labelledby="word-interpretation-title">
    <p className="bitwise-kicker">SAME PATTERN · TWO VALUE CONTRACTS</p>
    <h3 id="word-interpretation-title">What does the highest bit mean?</h3>
    <p>Toggle the highest bit. Explore how its contribution changes between unsigned and signed interpretation. Then follow each source position through a shift; “fill” means no source bit moved there.</p>
    <div className="bitwise-controls">
      <label>Word width<select aria-label="Word width" value={width} onChange={event => {
          const nextWidth = Number(event.target.value);
          setWidth(nextWidth);
          setUnsigned(unsigned % 2 ** nextWidth);
          setShift(Math.min(shift, nextWidth));
        }}><option value="4">4 bits</option><option value="8">8 bits</option></select></label>
      <label>Shift positions<select aria-label="Shift positions" value={shift} onChange={event => setShift(Number(event.target.value))}>
        {Array.from({
            length: width + 1
          }, (_, value) => <option key={value} value={value}>{value}</option>)}
      </select></label>
    </div>
    <BitRow value={unsigned} width={width} label="Pattern" onToggle={position => setUnsigned(toggleMember(unsigned, width, position))} />
    <div className="bitwise-interpretations">
      <div><h4>Unsigned weights</h4><p>{weightSum(false)} = <strong>{unsigned}</strong></p></div>
      <div><h4>Signed: highest weight is negative</h4><p>{weightSum(true)} = <strong>{view.signed}</strong></p></div>
    </div>
    <h4>Right shift by {shift}</h4>
    <p className="bitwise-note">Each result column names its source bit. Positions above b{width - 1} are filled; the lowest {shift} original position{shift === 1 ? '' : 's'} fall out.</p>
    <BitRow value={view.logical} width={width} label="Logical" annotations={rightLabels} />
    <p>Fill with 0 → unsigned result <strong>{view.logical}</strong>.</p>
    <BitRow value={view.arithmeticWord} width={width} label="Arithmetic" annotations={rightLabels} />
    <p>Fill with the original sign bit ({view.cells[0].bit}) → signed result <strong>{view.arithmetic}</strong>. Its stored word has unsigned value {view.arithmeticWord}.</p>
    <h4>Left shift by {shift}</h4>
    <BitRow value={view.leftWord} width={width} label="Bounded" annotations={leftLabels} />
    <p aria-live="polite">Discarding overflow gives {view.leftWord}. Python’s unbounded nonnegative integer shift gives {view.unboundedLeft}; masking it to {width} bits gives the bounded result.</p>
    <button type="button" onClick={() => {
      setWidth(8);
      setUnsigned(246);
      setShift(2);
    }}>Reset word</button>
    <p className="bitwise-note">Narrowing the width discards higher bits; widening zero-extends the retained unsigned pattern. This explicit word model is not a claim about every language’s signed overflow or shift rules. Python right shift of a negative integer uses floor division.</p>
  </section>;
}
const parityPresets = {
  single: [12, 5, 12, 9, 5],
  two: [4, 9, 4, 12, 9, 7],
  triple: [6, 6, 6, 2, 2],
  absent: [1, 2, 3],
  zero: [8, 0, 8]
};
export function XorParityLab() {
  const [values, setValues] = useState(parityPresets.single);
  const [draft, setDraft] = useState(parityPresets.single.join(', '));
  const [preset, setPreset] = useState('single');
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const trace = parityTrace(values);
  const state = trace.states[step];
  function install(nextValues, nextPreset = 'custom') {
    setValues([...nextValues]);
    setDraft(nextValues.join(', '));
    setPreset(nextPreset);
    setStep(0);
    setError('');
  }
  function apply(event) {
    event.preventDefault();
    try {
      const tokens = draft.trim().split(',').map(token => token.trim());
      if (tokens.some(token => !/^\d+$/.test(token))) throw new Error('Use comma-separated nonnegative whole numbers.');
      const nextValues = tokens.map(Number);
      parityTrace(nextValues);
      install(nextValues);
    } catch (failure) {
      setError(`${failure.message} The active trace is unchanged.`);
    }
  }
  const promise = trace.oneSingletonPromise ? 'Exactly one singleton; every other value is paired.' : trace.twoSingletonPromise ? 'Two singletons; one XOR gives their combined parity, not either answer.' : 'The paired-plus-one promise fails. The XOR remains defined, but its singleton interpretation does not.';
  return <section className="bitwise-lab" id="xor-parity-lab" aria-labelledby="xor-parity-title">
    <p className="bitwise-kicker">STREAM PREFIX · ODD OR EVEN CONTRIBUTIONS</p>
    <h3 id="xor-parity-title">What survives when matching events cancel?</h3>
    <p>Inspect the final accumulator as you step. Try the triple-occurrence preset: a nonzero result alone does not prove that its value occurred exactly once.</p>
    <label>Parity scenario<select aria-label="Parity scenario" value={preset} onChange={event => install(parityPresets[event.target.value], event.target.value)}>
      <option value="single">One singleton</option><option value="two">Two singletons</option>
      <option value="triple">Odd triple · no singleton</option><option value="absent">XOR result is absent</option>
      <option value="zero">Zero is the singleton</option>{preset === 'custom' && <option value="custom">Custom input</option>}
    </select></label>
    <form onSubmit={apply} className="bitwise-controls">
      <label>Events · 1–12 integers from 0–255<input aria-label="Events · 1–12 integers from 0–255" value={draft} onChange={event => setDraft(event.target.value)} maxLength={72} /></label>
      <button type="submit">Apply events</button>
    </form>
    {error && <p role="alert">{error}</p>}
    <div className="bitwise-stream" aria-label="Event order">{values.map((value, index) => <span key={index} className={index < step ? 'is-consumed' : ''}><small>event {index}</small><strong>{value}</strong><small>{index < step ? 'processed' : 'waiting'}</small></span>)}</div>
    <BitRow value={state.before} width={8} label="Before" />
    <BitRow value={state.value ?? 0} width={8} label={state.value === null ? 'No input' : 'XOR input'} />
    <div className="bitwise-result"><BitRow value={state.accumulator} width={8} label="Parity" /></div>
    <p role="status">{step === 0 ? 'Empty prefix: start at 0.' : `After ${step} event${step === 1 ? '' : 's'}, the prefix XOR is ${state.accumulator}. Each 1 marks a bit supplied an odd number of times so far.`}</p>
    <StepButtons step={step} count={trace.states.length} setStep={setStep} />
    <h4>Diagnostic frequency audit · extra storage</h4>
    <div className="bitwise-frequencies">{trace.frequencies.map(([value, count]) => <span key={value}>{value} × {count}</span>)}</div>
    <p>{promise}</p>
    <p className="bitwise-note">The algorithm needs the current value and one accumulator. This teaching trace and whole-input frequency audit retain extra data; they are not part of its constant-word workspace claim. The browser uses bounded unsigned bytes; the complete Python program also checks signed and zero examples.</p>
    <button type="button" onClick={() => install(parityPresets.single, 'single')}>Reset parity lab</button>
  </section>;
}
export function SparseBitLab() {
  const [value, setValue] = useState(180);
  const [step, setStep] = useState(0);
  const trace = sparseBitTrace(value);
  const state = trace.states[step];
  const removedPositions = state.clearedPosition === null ? [] : [state.clearedPosition];
  return <section className="bitwise-lab" id="sparse-bit-lab" aria-labelledby="sparse-bit-title">
    <p className="bitwise-kicker">BORROW ON SUBTRACTION · REMOVE ONE OCCUPIED PLACE</p>
    <h3 id="sparse-bit-title">Why does x AND (x − 1) remove exactly one bit?</h3>
    <p>Build a byte and watch which 1 disappears at each step. Compare subtraction with the final AND: borrowing also flips lower zeroes, but the AND prevents those new ones from surviving.</p>
    <BitRow value={value} width={8} label="Start" onToggle={position => {
      setValue(toggleMember(value, 8, position));
      setStep(0);
    }} />
    <div className="bitwise-controls"><label>Counting case<select aria-label="Counting case" value={[0, 1, 128, 180, 255].includes(value) ? value : 'custom'} onChange={event => {
          setValue(Number(event.target.value));
          setStep(0);
        }}>
      <option value="180">180 · several sparse ones</option><option value="128">128 · one high bit</option><option value="255">255 · every bit set</option>
      <option value="1">1 · lowest bit</option><option value="0">0 · no iterations</option>{![0, 1, 128, 180, 255].includes(value) && <option value="custom">Custom byte</option>}
    </select></label></div>
    <BitRow value={state.current} width={8} label="x" emphasis={removedPositions} />
    {state.minusOne !== null ? <>
      <BitRow value={state.minusOne} width={8} label="x − 1" emphasis={Array.from({
        length: state.clearedPosition + 1
      }, (_, index) => index)} />
      <div className="bitwise-result"><BitRow value={state.next} width={8} label="AND" emphasis={removedPositions} /></div>
      <p role="status">Remove bit {state.clearedPosition}, whose weight is {2 ** state.clearedPosition}: {state.current} becomes {state.next}. The outlined suffix shows the borrow; the outlined result cell is the one removed.</p>
    </> : <p role="status">Reached 0. Stop before subtracting again: {state.removed} one{state.removed === 1 ? '' : 's'} were removed.</p>}
    <div className="bitwise-count"><strong>{state.removed}</strong><span>removed</span><span>+</span><strong>{trace.population - state.removed}</strong><span>remaining</span><span>=</span><strong>{trace.population}</strong><span>original ones</span></div>
    <StepButtons step={step} count={trace.states.length} setStep={setStep} />
    <p>The original value {value} {trace.isPowerOfTwo ? 'is' : 'is not'} a positive power of two: it has {trace.population} set bit{trace.population === 1 ? '' : 's'}. Zero satisfies the raw AND equality but fails the positive-value guard.</p>
    <button type="button" onClick={() => {
      setValue(180);
      setStep(0);
    }}>Reset counting lab</button>
    <p className="bitwise-note">At most eight removal steps are needed here. A Python helper must reject negative input or explicitly choose a finite word width; infinite sign extension does not have a finite count of set positions.</p>
  </section>;
}
export function SingletonPartitionFigure() {
  const view = twoSingletonPartition([4, 9, 4, 12, 9, 7]);
  return <figure className="bitwise-figure">
    <figcaption>One distinguishing bit separates the two survivors</figcaption>
    <p className="bitwise-partition-total">Total XOR {view.total} = {wordBits(view.total, 4).map(cell => cell.bit).join('')} · lowest set bit has weight {view.separatingBit}</p>
    <div className="bitwise-partition">{view.groups.map((group, index) => <div key={index}>
      <h4>Bit 0 is {index}</h4><p>{group.join(' → ')}</p><span className="bitwise-down" aria-hidden="true">↓</span>
      <p>Matching {index === 0 ? '4s' : '9s'} cancel</p><strong>Survivor {view.answers[index]}</strong>
    </div>)}</div>
    <p className="bitwise-note">The input promise gives exactly two singletons. Equal pairs always enter the same bucket. The result is the unordered pair {'{12, 7}'}, not two positions in the original input.</p>
  </figure>;
}
