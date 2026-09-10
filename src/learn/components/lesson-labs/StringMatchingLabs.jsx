import { useEffect, useRef, useState } from 'react';
import { defaultMatchingPattern, defaultMatchingText, kmpTrace, prefixTrace, rollingTrace, streamTrace, symbols, zFunction } from '../../data/string-matching-models.js';
import './string-matching-labs.css';
function displaySymbol(symbol) {
  if (symbol === ' ') return '␠';
  if (symbol === '\n') return '↵';
  if (symbol === '\t') return '⇥';
  if (/\p{Mark}/u.test(symbol)) return `◌${symbol}`;
  return symbol;
}
function SymbolStrip({
  letters,
  label,
  marked = [],
  active = -1,
  offset = 0,
  total,
  numbers = true
}) {
  const region = useRef(null);
  useEffect(() => {
    const cell = region.current?.querySelector('.string-current');
    if (cell) {
      const parentBox = region.current.getBoundingClientRect();
      const cellBox = cell.getBoundingClientRect();
      if (cellBox.left < parentBox.left || cellBox.right > parentBox.right) {
        region.current.scrollLeft += cellBox.left - parentBox.left - parentBox.width / 2 + cellBox.width / 2;
      }
    }
  }, [active]);
  return <div className="string-sequence">
    <p className="string-row-label">{label}</p>
    <div className="string-scroll" ref={region} tabIndex={0} role="region" aria-label={label}>
      <div className="string-cells" style={{
        gridTemplateColumns: `repeat(${Math.max(total || letters.length + offset, 1)}, 2.4rem)`
      }}>
        {Array.from({
          length: offset
        }, (_, index) => <span className="string-spacer" key={`space-${index}`} />)}
        {letters.length ? letters.map((letter, index) => <span key={index} className={`string-cell ${marked.includes(index) ? 'string-known' : ''} ${active === index ? 'string-current' : ''}`} title={`Index ${index}: U+${letter.codePointAt(0).toString(16).toUpperCase()}`}>
          {numbers && <small>{index}</small>}<strong>{displaySymbol(letter)}</strong>
        </span>) : <span className="string-empty">∅</span>}
      </div>
    </div>
  </div>;
}
function TraceControls({
  index,
  setIndex,
  count,
  nextLabel = 'Next state'
}) {
  return <div className="string-controls">
    <button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</button>
    <button type="button" disabled={index + 1 >= count} onClick={() => setIndex(index + 1)}>{nextLabel}</button>
    <button type="button" disabled={index + 1 >= count} onClick={() => setIndex(count - 1)}>Finish</button>
    <button type="button" onClick={() => setIndex(0)}>Reset</button>
    <span>State {index + 1} / {count}</span>
  </div>;
}
function MatchingAlignment({
  trace,
  current
}) {
  const region = useRef(null);
  const start = current.consumed - current.matched;
  const total = Math.max(trace.letters.length + 1, start + trace.needle.length, 1);
  useEffect(() => {
    const cell = region.current?.querySelector('[data-cursor]');
    if (!cell) return;
    const outer = region.current.getBoundingClientRect();
    const target = cell.getBoundingClientRect();
    if (target.left < outer.left || target.right > outer.right) {
      region.current.scrollLeft += target.left - outer.left - outer.width / 2 + target.width / 2;
    }
  }, [current.consumed, current.matched]);
  return <>
    <p className="string-row-label">Upper row: text indices · lower row: pattern indices</p>
    <p className="string-row-label">{current.consumed} text symbols consumed · candidate start {start} · q={current.matched}</p>
    <div className="string-scroll" ref={region} tabIndex={0} role="region" aria-label="Aligned text and pattern on one shared coordinate axis">
      <div className="string-alignment" style={{
        width: `calc(${total} * 2.4rem + ${total - 1} * 3px)`
      }}>
        <div className="string-cells" style={{
          gridTemplateColumns: `repeat(${total}, 2.4rem)`
        }}>
          {trace.letters.map((letter, index) => <span key={index} data-cursor={index === current.consumed ? true : undefined} className={`string-cell ${index >= start && index < current.consumed ? 'string-known' : ''} ${index === current.consumed ? 'string-current' : ''}`}><small>{index}</small><strong>{displaySymbol(letter)}</strong></span>)}
          <span className={`string-cell ${current.consumed === trace.letters.length ? 'string-current' : ''}`} data-cursor={current.consumed === trace.letters.length ? true : undefined}><small>{trace.letters.length}</small><strong>│</strong><small>end</small></span>
        </div>
        <div className="string-cells" style={{
          gridTemplateColumns: `repeat(${total}, 2.4rem)`,
          marginTop: 10
        }}>
          {Array.from({
            length: start
          }, (_, index) => <span key={`spacer-${index}`} />)}
          {trace.needle.map((letter, index) => <span key={index} className={`string-cell ${index < current.matched ? 'string-known' : ''}`}><small>{index}</small><strong>{displaySymbol(letter)}</strong></span>)}
        </div>
      </div>
    </div>
    {(!trace.letters.length || !trace.needle.length) && <p className="lesson-note">{!trace.letters.length ? 'The text row is empty. ' : ''}{!trace.needle.length ? 'The pattern row is empty.' : ''}</p>}
  </>;
}
function useAppliedTrace(initialInput, build) {
  const [draft, setDraft] = useState(initialInput);
  const [trace, setTrace] = useState(() => build(initialInput));
  const [index, setIndex] = useState(0);
  const [error, setError] = useState('');
  function apply(event) {
    event?.preventDefault();
    try {
      const next = build(draft);
      setTrace(next);
      setIndex(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return {
    draft,
    setDraft,
    trace,
    index,
    setIndex,
    error,
    apply
  };
}
export function OverlapFigure() {
  return <figure className="string-figure" data-figure="overlap">
    <SymbolStrip letters={symbols('ababa')} label="Text · zero-based code-point indices" />
    <SymbolStrip letters={symbols('aba')} label="Match starting at 0" marked={[0, 1, 2]} total={5} />
    <SymbolStrip letters={symbols('aba')} label="Match starting at 2" marked={[0, 1, 2]} offset={2} total={5} />
    <figcaption>The middle a belongs to both occurrences. Reporting a match does not reserve or remove its symbols.</figcaption>
  </figure>;
}
export function BorderEvidenceFigure() {
  return <figure className="string-figure" data-figure="border-evidence">
    <SymbolStrip letters={symbols('ababa')} label="The prefix aba · positions [0,3)" marked={[0, 1, 2]} />
    <SymbolStrip letters={symbols('ababa')} label="The equal suffix aba · positions [2,5)" marked={[2, 3, 4]} />
    <figcaption>Both highlighted regions spell aba. Their length is 3; their starts are 0 and 2. A border is a length of reusable equality, even when the two regions overlap.</figcaption>
  </figure>;
}
export function PalindromicPrefixFigure() {
  return <figure className="string-figure" data-figure="palindromic-prefix">
    <div className="string-mirror"><div><small>prepend reversed remainder</small><strong>c</strong></div><div><small>retain the palindromic prefix</small><strong>a b a</strong></div><div><small>original remainder</small><strong>c</strong></div></div>
    <figcaption>abac → c + aba + c = cabac. The outside c symbols mirror each other; the retained center already mirrors itself. Retaining a shorter palindromic prefix would require more prepended symbols.</figcaption>
  </figure>;
}
export function PrefixBorderLab() {
  const state = useAppliedTrace('ababaca', prefixTrace);
  const current = state.trace.states[state.index];
  const completed = ['equal', 'commit', 'initial'].includes(current.kind);
  const end = current.position + (completed ? 1 : 0);
  const prefix = Array.from({
    length: current.matched
  }, (_, index) => index);
  const suffix = prefix.map(index => end - current.matched + index);
  return <section className="string-lab" data-lab="prefix-border" aria-labelledby="prefix-border-lab">
    <p className="lesson-eyebrow">INVESTIGATE · BORDER LENGTHS</p>
    <h3 id="prefix-border-lab">Which part can extend?</h3>
    <p>Before a fallback, predict the next shorter candidate. Green cells show equal prefix/suffix evidence; the amber outline identifies i. Edits take effect when you apply the pattern.</p>
    <form className="string-form" onSubmit={state.apply}>
      <label>Pattern<input value={state.draft} onChange={event => state.setDraft(event.target.value)} /></label>
      <button type="submit">Apply pattern</button>
    </form>
    {state.error && <p role="alert">{state.error} The previous trace remains.</p>}
    <SymbolStrip letters={state.trace.letters} label={`Prefix candidate · length ${current.matched}`} marked={prefix} active={current.position} />
    <SymbolStrip letters={state.trace.letters} label={`Same symbols as suffix · ending before ${end}`} marked={suffix} active={current.position} />
    <div className="string-scroll" tabIndex={0} role="region" aria-label="Prefix table">
      <table className="string-prefix-table"><caption>π[i] · longest proper border length</caption><thead><tr><th>i</th>{state.trace.letters.map((_, index) => <th key={index}>{index}</th>)}</tr></thead>
        <tbody><tr><th>π[i]</th>{current.table.map((value, index) => <td key={index}>{value === null ? '·' : value}</td>)}</tr></tbody></table>
    </div>
    <p className="string-action" aria-live="polite" data-result="prefix-action">{current.action}</p>
    <TraceControls index={state.index} setIndex={state.setIndex} count={state.trace.states.length} />
    <p className="lesson-note">Try aaaa, then abc. Which one has the most reusable borders? This trace performs real equality checks; the cells are positions, not a timing benchmark.</p>
  </section>;
}
export function KmpAlignmentLab() {
  const state = useAppliedTrace({
    text: defaultMatchingText,
    pattern: defaultMatchingPattern
  }, input => kmpTrace(input.text, input.pattern));
  const current = state.trace.states[state.index];
  return <section className="string-lab" data-lab="kmp-alignment" aria-labelledby="kmp-alignment-lab">
    <p className="lesson-eyebrow">INVESTIGATE · REUSE AFTER A MISMATCH</p>
    <h3 id="kmp-alignment-lab">Move the candidate, keep the unread symbol</h3>
    <p>Predict whether the next step consumes text or only shortens q. Each ribbon shows the state <em>after</em> the action. Green marks retained equality; amber marks the next unread text symbol. Scroll a narrow ribbon to inspect all positions.</p>
    <form className="string-form" onSubmit={state.apply}>
      <label>Text<input value={state.draft.text} onChange={event => state.setDraft({
          ...state.draft,
          text: event.target.value
        })} /></label>
      <label>Pattern<input value={state.draft.pattern} onChange={event => state.setDraft({
          ...state.draft,
          pattern: event.target.value
        })} /></label>
      <button type="submit">Apply search</button>
    </form>
    {state.error && <p role="alert">{state.error} The previous trace remains.</p>}
    <MatchingAlignment trace={state.trace} current={current} />
    <dl className="string-metrics"><div><dt>Comparisons so far</dt><dd data-result="kmp-comparisons">{current.comparisons}</dd></div><div><dt>Reported starts</dt><dd data-result="kmp-matches">{current.matches.join(', ') || 'none'}</dd></div></dl>
    <p className="string-action" aria-live="polite" data-result="kmp-action">{current.action}</p>
    {current.compared && <p className="lesson-note">The completed comparison was text[{current.compared[0]}] with pattern[{current.compared[1]}]. The ribbons now show its consequence.</p>}
    <TraceControls index={state.index} setIndex={state.setIndex} count={state.trace.states.length} />
    <p className="lesson-note">Then search aaaaa for aaa. Predict all starts before finishing. An empty pattern follows the all-boundaries contract; a longer pattern produces no occurrence.</p>
  </section>;
}
export function ChunkMatcherLab() {
  const state = useAppliedTrace({
    chunks: 'xxa||b|aba',
    pattern: 'aba',
    faulty: false
  }, input => streamTrace(input.chunks.split('|'), input.pattern, input.faulty));
  const current = state.trace.states[state.index];
  return <section className="string-lab" data-lab="chunk-matcher" aria-labelledby="chunk-matcher-lab">
    <p className="lesson-eyebrow">INVESTIGATE · A STREAM HAS MEMORY</p>
    <h3 id="chunk-matcher-lab">Feed a chunk without starting over</h3>
    <p>The bars below are delivery boundaries, not separators in the searched text. Predict the first chunk that finishes aba. The optional fault clears q at each delivery; compare the missing reports.</p>
    <form className="string-form" onSubmit={state.apply}>
      <label>Chunks separated by |<input value={state.draft.chunks} onChange={event => state.setDraft({
          ...state.draft,
          chunks: event.target.value
        })} /></label>
      <label>Pattern<input value={state.draft.pattern} onChange={event => state.setDraft({
          ...state.draft,
          pattern: event.target.value
        })} /></label>
      <label className="string-checkbox"><input type="checkbox" checked={state.draft.faulty} onChange={event => state.setDraft({
          ...state.draft,
          faulty: event.target.checked
        })} />Fault: reset q at every seam</label>
      <button type="submit">Apply chunks</button>
    </form>
    {state.error && <p role="alert">{state.error} The previous trace remains.</p>}
    <ol className="string-chunks">{state.trace.chunks.map((chunk, index) => <li key={index} className={index < current.fed ? 'string-fed' : ''}><small>chunk {index + 1} · {index < current.fed ? 'fed' : 'waiting'}</small><strong>{chunk || '∅ empty'}</strong></li>)}</ol>
    <SymbolStrip letters={state.trace.needle} label={`Retained prefix · q=${current.matched}`} marked={Array.from({
      length: current.matched
    }, (_, index) => index)} />
    <dl className="string-metrics"><div><dt>Code points consumed</dt><dd>{current.consumed}</dd></div><div><dt>New starts this delivery</dt><dd>{current.added.join(', ') || 'none'}</dd></div></dl>
    <p aria-live="polite" className="string-action" data-result="stream-action">After {current.fed} chunks: offset={current.consumed}, q={current.matched}; all starts: {current.matches.join(', ') || 'none'}.</p>
    <TraceControls index={state.index} setIndex={state.setIndex} count={state.trace.states.length} nextLabel="Feed next chunk" />
    <p className="lesson-note">An empty chunk consumes nothing. The UI reserves | as its chunk delimiter. This is a simulation of already decoded strings, not a network or UTF-8 decoder.</p>
  </section>;
}
export function RollingFingerprintLab() {
  const state = useAppliedTrace({
    text: 'adbaad',
    pattern: 'ba',
    base: 3,
    modulus: 7
  }, input => rollingTrace(input.text, input.pattern, input.base, input.modulus));
  const row = state.trace.rows[state.index];
  return <section className="string-lab" data-lab="rolling-fingerprint" aria-labelledby="rolling-fingerprint-lab">
    <p className="lesson-eyebrow">INVESTIGATE · CANDIDATE ≠ CERTIFICATE</p>
    <h3 id="rolling-fingerprint-lab">Watch a collision survive the hash and fail the match</h3>
    <p>Predict whether the highlighted window matches ba. The small fixed modulus makes collisions easy to see; these controls do not simulate a random hash family or claim an error rate.</p>
    <form className="string-form" onSubmit={state.apply}>
      <label>Text<input value={state.draft.text} onChange={event => state.setDraft({
          ...state.draft,
          text: event.target.value
        })} /></label>
      <label>Pattern<input value={state.draft.pattern} onChange={event => state.setDraft({
          ...state.draft,
          pattern: event.target.value
        })} /></label>
      <label>Base<select aria-label="Base" value={state.draft.base} onChange={event => state.setDraft({
          ...state.draft,
          base: Number(event.target.value)
        })}><option value={3}>3</option><option value={5}>5</option><option value={31}>31</option></select></label>
      <label>Modulus<select aria-label="Modulus" value={state.draft.modulus} onChange={event => state.setDraft({
          ...state.draft,
          modulus: Number(event.target.value)
        })}><option value={7}>7</option><option value={101}>101</option><option value={1009}>1009</option></select></label>
      <button type="submit">Apply fingerprints</button>
    </form>
    {state.error && <p role="alert">{state.error} The previous trace remains.</p>}
    <SymbolStrip letters={state.trace.letters} label="Text · active window" marked={row ? Array.from({
      length: state.trace.needle.length
    }, (_, index) => row.start + index) : []} active={row?.start} />
    {row ? <>
      <div className="string-verdict"><div><small>Window hash</small><strong>{row.value}</strong></div><span>{row.candidate ? '=' : '≠'}</span><div><small>Pattern hash</small><strong>{state.trace.target}</strong></div><p data-result="rolling-verdict">{row.exact ? 'Exact match · report start ' + row.start : row.candidate ? 'Hash collision · reject after exact comparison' : 'Different hashes · reject without comparing full strings'}</p></div>
      {row.update && <div className="string-arithmetic" aria-label="Calculated rolling update">
        <div><span>1 · remove outgoing weight</span><code>({row.value} − {row.update.outgoing} × {state.trace.power}) mod {state.trace.modulus} = {row.update.remainder}</code></div>
        <div><span>2 · shift and append incoming code point</span><code>({row.update.remainder} × {state.trace.base} + {row.update.incoming}) mod {state.trace.modulus} = {row.update.next}</code></div>
      </div>}
      <p className="string-action" aria-live="polite" data-result="rolling-action">Window start {row.start}; fingerprint {row.value}; exact-verification comparisons {row.checks}. {row.update ? `Next fingerprint: ${row.update.next}.` : 'Last complete window.'}</p>
      <TraceControls index={state.index} setIndex={state.setIndex} count={state.trace.rows.length} nextLabel="Next window" />
    </> : <p className="string-action" data-result="rolling-action">{state.trace.needle.length ? 'Pattern longer than text: no complete windows.' : `Empty pattern: all boundaries ${state.trace.matches.join(', ')}. No rolling arithmetic needed.`}</p>}
    <div className="string-scroll" tabIndex={0} role="region" aria-label="Window audit">
      <table className="string-window-table"><caption>Calculated audit · hash equality and identity checked separately</caption><thead><tr><th>Start</th><th>Window</th><th>Hash</th><th>Decision</th></tr></thead><tbody>{state.trace.rows.map(item => <tr key={item.start} className={item.start === row?.start ? 'string-selected' : ''}><th>{item.start}</th><td>{item.window.map(displaySymbol).join('')}</td><td>{item.value}</td><td>{item.exact ? 'report' : item.candidate ? 'collision' : 'reject hash'}</td></tr>)}</tbody></table>
    </div>
    <p className="lesson-note">Try aaaaa / aaa: every window is a true hit, and every exact verification reads three symbols. More hashing cannot remove that verification cost.</p>
  </section>;
}
export function UnicodeCoordinatesFigure() {
  const text = 'A🙂e\u0301';
  const points = symbols(text);
  const utf16 = Array.from({
    length: text.length
  }, (_, index) => text.charCodeAt(index).toString(16).toUpperCase());
  const utf8 = [...new TextEncoder().encode(text)].map(value => value.toString(16).toUpperCase().padStart(2, '0'));
  return <figure className="string-figure" data-figure="unicode-coordinates">
    <p className="string-row-label">Same raw text · A🙂é</p>
    <div className="string-units"><strong>4 code points</strong><div>{points.map((letter, index) => <span key={index}>{displaySymbol(letter)}<small>U+{letter.codePointAt(0).toString(16).toUpperCase()}</small></span>)}</div></div>
    <div className="string-units"><strong>5 UTF-16 units</strong><div>{utf16.map((unit, index) => <span key={index}>{unit}</span>)}</div></div>
    <div className="string-units"><strong>8 UTF-8 bytes</strong><div>{utf8.map((unit, index) => <span key={index}>{unit}</span>)}</div></div>
    <figcaption>Hexadecimal values are calculated from the displayed text. The emoji spans two UTF-16 units and four UTF-8 bytes. The visible accented letter contains e plus U+0301; the dotted circle makes that combining mark inspectable. NFC changes these four code points into three.</figcaption>
  </figure>;
}
export function PrefixCancellationFigure() {
  return <figure className="string-figure" data-figure="prefix-cancellation">
    <p className="string-row-label">Query [1,3) in xab · B=3, Q=101</p>
    <div className="string-cancellation"><div><strong>H[3]</strong><span>x·B²</span><span>a·B</span><span>b</span></div><div><strong>subtract H[1]·B²</strong><span>x·B²</span><span /><span /></div><div><strong>remaining</strong><span>0</span><span>a·B</span><span>b</span></div></div>
    <p>H[3]=55. The shifted prefix contributes (19×9) mod 101=70. The remaining fingerprint is (55−70) mod 101=86, equal to (97×3+98) mod 101.</p>
    <figcaption>The x contribution cancels only at the same weight B². Values are calculated modular arithmetic; 86 is the fingerprint of ab, not an identity certificate.</figcaption>
  </figure>;
}
export function ZReuseFigure() {
  const text = symbols('aabcaabxaaaz');
  const values = zFunction(text);
  return <figure className="string-figure" data-figure="z-reuse">
    <SymbolStrip letters={text} label="S = aabcaabxaaaz · code-point indices" />
    <div className="string-z-case"><strong>Inside [4,7): i=5</strong><p>Mirror position 1 has Z[1]={values[1]}. There are 2 known cells before R. Since 1&lt;2, the known mismatch also lies inside the box: Z[5]={values[5]}.</p><SymbolStrip letters={text} label="Known box [4,7) · inspect i=5" marked={[4, 5, 6]} active={5} /></div>
    <div className="string-z-case"><strong>At the edge of [8,10): i=9</strong><p>Mirror position 1 again has Z[1]=1, but only one cell remains. Copy 1; compare beyond R. One more a matches, then z differs from b: Z[9]={values[9]}.</p><SymbolStrip letters={text} label="Existing box [8,10), newly verified position 10" marked={[8, 9]} active={10} /></div>
    <figcaption>These are two actual states of the Z computation, using half-open boxes. Green is prior evidence; the outlined cell is the position being inspected. Layout is not a performance measurement.</figcaption>
  </figure>;
}
