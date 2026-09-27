import { useState } from 'react';
import { NeuralLab, NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { Figure, Node, Arrow, Vector, Values, Plot, Stems, Matrix, f, vec } from './HyenaPrimitives.jsx';
import { PathGeometry, ModalRegisters } from './HyenaFigures.jsx';
import { directConvolution, fullConvolution, circularConvolution, fftConvolution, blockedConvolution, gatedConvolution, gatePaths, modalFilter, modalScan, finiteTruncation } from '../../data/hyena-convolution-models.js';
const maxError = (a, b) => Math.max(0, ...a.map((v, i) => Math.abs(v - (b[i] ?? 0))));
export function ConvolutionLab() {
  const [u, setU] = useState([2, -1, 3, 0, 1]),
    [h, setH] = useState([.5, 1, -.25]),
    [method, setMethod] = useState('direct'),
    [selected, setSelected] = useState(2),
    [pinned, setPinned] = useState(null),
    t = Math.min(selected, u.length - 1),
    direct = directConvolution(u, h),
    transformed = fftConvolution(u, h),
    circular = h.length <= u.length ? circularConvolution(u, h) : null,
    output = method === 'fft' ? transformed.output : method === 'circular' && circular ? circular : direct,
    circularActive = method === 'circular' && circular,
    terms = h.map((coefficient, r) => {
      const source = circularActive ? (t - r + u.length) % u.length : t - r,
        valid = source >= 0;
      return {
        r,
        source: valid ? source : null,
        coefficient,
        input: valid ? u[source] : 0,
        product: valid ? coefficient * u[source] : 0,
        wrap: !!circularActive && t - r < 0
      };
    }),
    reset = () => {
      setU([2, -1, 3, 0, 1]);
      setH([.5, 1, -.25]);
      setMethod('direct');
      setSelected(2);
      setPinned(null);
    };
  return <NeuralLab id="hyena-convolution" title="HA · Find the future leak"><p>Edit the signal or filter; every output updates immediately. The intentionally circular route wraps unavailable negative source indices to the end. Correct padding computes the same causal sum as the direct route.</p><Vector label="Signal u" values={u} min={-8} max={8} onChange={setU} /><Vector label="Filter h" values={h} min={-4} max={4} onChange={setH} /><div className="hy-controls"><button disabled={u.length >= 12} onClick={() => setU([...u, 0])}>Add input</button><button disabled={u.length <= 2} onClick={() => setU(u.slice(0, -1))}>Remove input</button><button disabled={h.length >= 12} onClick={() => setH([...h, 0])}>Add filter tap</button><button disabled={h.length <= 1} onClick={() => setH(h.slice(0, -1))}>Remove filter tap</button><button onClick={() => setH([1])}>Identity filter</button><button onClick={() => setH(h.map(() => 0))}>Zero filter</button><button onClick={() => setU(u.map((v, i) => i === u.length - 1 ? Math.min(8, v + 4) : v))}>Increase final input by4 (cap8)</button><button onClick={() => setU(u.map((_, i) => i === 0 ? 1 : 0))}>Unit impulse</button><button onClick={() => setPinned({
        u: [...u],
        h: [...h],
        output: [...output],
        method
      })}>Pin complete calculation</button><button onClick={reset}>Reset convolution</button></div><label className="hy-select">Computation<select value={method} onChange={e => setMethod(e.target.value)}><option value="direct">Direct causal sum</option><option value="fft">Correctly padded FFT</option><option value="circular" disabled={!circular}>Intentionally circular</option></select></label>{!circular && <p>With K&gt;L the circular demonstration is unavailable; no taps are silently trimmed. If circular was selected, the displayed output falls back to the direct causal reference.</p>}<p className="hy-result" aria-live="polite">Output {vec(output)}. Padded FFT length {transformed.size}≥L+K−1={u.length + h.length - 1}; maximum FFT/reference difference {f(maxError(transformed.output, direct), 12)}.</p><NeuralNumber label="Selected output position" value={t} min={0} max={u.length - 1} integer onChange={setSelected} /><Stems title="Current output and a pinned calculation" values={output} selected={t} reference={pinned?.output.length === u.length ? pinned.output : null} />{pinned && <p className="hy-result hy-pinned">Pinned {pinned.method}: u={vec(pinned.u)}, h={vec(pinned.h)}, y={vec(pinned.output)}. {pinned.output.length === u.length ? `Maximum current difference ${f(maxError(output, pinned.output), 8)}.` : 'Different lengths: exact pinned result remains here.'}</p>}<Figure title={`Dependency paths into output ${t}`} height={85 + terms.length * 48} description="Rose arrows are circular wraparound, so they can read an input later than the current output. Gold arrows are ordinary delayed contributions.">{terms.map((row, i) => <g key={row.r}><text x="10" y={35 + i * 48}>{row.source === null ? 'zero padding' : `u[${row.source}]=${f(row.input)}`}</text><Arrow x1={150} y1={30 + i * 48} x2={375} y2={30 + i * 48} color={row.wrap ? '#e9979f' : '#e6bb60'} /><text x="260" y={20 + i * 48} textAnchor="middle">h[{row.r}]={f(row.coefficient)}</text><text x="385" y={35 + i * 48}>{f(row.product, 7)}{row.wrap ? ' (wrapped)' : ''}</text></g>)}<text x="10" y={65 + terms.length * 48}>Sum = {f(output[t], 9)}. One unchanged output alone cannot certify causality.</text></Figure><NeuralTable caption="Selected contribution ledger" headers={['Lag', 'Source index', 'Input', 'Filter', 'Product', 'Wrapped?']} rows={terms.map(r => [r.r, r.source ?? 'outside→zero', f(r.input), f(r.coefficient), f(r.product, 8), r.wrap ? 'yes' : 'no'])} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Actual padded arrays and Fourier bins</h4><Values caption="Time-domain arrays" rows={[["Padded input", vec(transformed.input)], ["Padded filter", vec(transformed.filter)], ["Inverse full linear result", vec(transformed.full)]]} /><NeuralTable caption="Frequency bins are not token positions" headers={['Bin', 'Input real', 'Input imaginary', 'Filter real', 'Filter imaginary', 'Product real', 'Product imaginary']} rows={transformed.product.real.map((v, i) => [i, f(transformed.inputSpectrum.real[i], 8), f(transformed.inputSpectrum.imaginary[i], 8), f(transformed.filterSpectrum.real[i], 8), f(transformed.filterSpectrum.imaginary[i], 8), f(v, 8), f(transformed.product.imaginary[i], 8)])} /></section></NeuralLab>;
}
const gateDefaults = () => ({
  v: [2, -1, 3, 1],
  k: [.5, 1, 0, -1],
  q: [1, -.5, 2, 1],
  h: [1, -.25, .5, 0],
  phi: [1, -.5, 0, 0]
});
export function GateLab() {
  const [state, setState] = useState(gateDefaults),
    [hierarchy, setHierarchy] = useState(false),
    [receiver, setReceiver] = useState(3),
    [sender, setSender] = useState(1),
    [pinned, setPinned] = useState(null),
    n = state.v.length,
    t = Math.min(receiver, n - 1),
    j = Math.min(sender, n - 1),
    r = gatedConvolution(state.v, state.h, state.q, state.k, hierarchy ? state.phi : null),
    paths = hierarchy ? gatePaths(state.h, state.q, state.k, state.phi, t, j) : null,
    edit = (key, value) => setState(s => ({
      ...s,
      [key]: value
    })),
    resize = delta => setState(s => Object.fromEntries(Object.entries(s).map(([key, row]) => [key, delta > 0 ? [...row, ['q', 'k'].includes(key) ? 1 : 0] : row.slice(0, -1)])));
  return <NeuralLab id="hyena-gates" title="HB · Build a selective route"><p>The gates here are independently editable fixed coefficients, so the matrix explains value routing with those gates held fixed. The full learned network also changes its gates when input symbols change; this matrix is not that network’s complete Jacobian.</p><Vector label="Values v" values={state.v} onChange={v => edit('v', v)} /><div className="hy-two"><Vector label="Sending gates k" values={state.k} onChange={v => edit('k', v)} /><Vector label="Receiving gates q" values={state.q} onChange={v => edit('q', v)} /></div><Vector label="Long filter h" values={state.h} min={-2} max={2} onChange={v => edit('h', v)} /><label><input type="checkbox" checked={hierarchy} onChange={e => setHierarchy(e.target.checked)} /> Add preceding filter φ before the sending gate</label>{hierarchy && <Vector label="Preceding filter φ" values={state.phi} min={-2} max={2} onChange={v => edit('phi', v)} />}<div className="hy-controls"><button disabled={n >= 8} onClick={() => resize(1)}>Add position</button><button disabled={n <= 2} onClick={() => resize(-1)}>Remove position</button><button disabled={n < 3} onClick={() => edit('k', state.k.map((v, i) => i === 2 ? 1 : v))}>Open sender2</button><button onClick={() => setState(s => ({
        ...s,
        k: s.k.map(() => 1),
        q: s.q.map(() => 1)
      }))}>All gates1</button><button onClick={() => edit('k', state.k.map(() => 0))}>Close every sender</button><button onClick={() => setPinned({
        state: structuredClone(state),
        hierarchy,
        output: [...r.output]
      })}>Pin full route</button><button onClick={() => {
        setState(gateDefaults());
        setHierarchy(false);
        setReceiver(3);
        setSender(1);
        setPinned(null);
      }}>Reset gates</button></div><p className="hy-result" aria-live="polite">Current output {vec(r.output)}. {hierarchy ? 'The preceding filter adds intermediate paths.' : 'One-filter gate sandwich.'}</p><div className="neural-controls"><NeuralNumber label="Matrix receiver row" value={t} min={0} max={n - 1} integer onChange={setReceiver} /><NeuralNumber label="Matrix sender column" value={j} min={0} max={n - 1} integer onChange={setSender} /></div><Matrix title="Actual signed conditional matrix" values={r.matrix} selected={[t, j]} causal />{hierarchy ? <PathGeometry t={t} j={j} paths={paths} /> : <Figure title="Selected one-filter coefficient" height={120}><Node x={10} y={20} width={160} lines={[`q[${t}]=${f(state.q[t])}`]} /><Arrow x1={170} y1={42} x2={200} y2={42} /><Node x={200} y={20} width={200} lines={[j > t ? 'future: coefficient0' : `h[${t - j}]=${f(state.h[t - j] ?? 0)}`]} /><Arrow x1={400} y1={42} x2={430} y2={42} /><Node x={430} y={20} width={190} lines={[`k[${j}]=${f(state.k[j])}`]} /><text x="15" y="103">Coefficient={f(r.matrix[t][j], 8)}; ×v[{j}] gives {f(r.contributions[t][j], 8)}.</text></Figure>}<Stems title={`Signed sender contributions to output ${t}`} values={r.contributions[t]} selected={j} /><Values caption="Complete intermediate arrays" rows={[["After preceding filter (or identity)", vec(r.first)], ["Sending gate × first stream", vec(r.transmitted)], ["After long filter", vec(r.filtered)], ["Receiving gate × filtered", vec(r.output)]]} />{pinned && <><Stems title="Current versus complete pinned route" values={r.output} reference={pinned.output.length === n ? pinned.output : null} /><details className="hy-pinned"><summary>Pinned settings and exact outputs</summary><Values caption="Pinned route" rows={[...Object.entries(pinned.state).map(([k, v]) => [k, vec(v)]), ['Preceding filter enabled', String(pinned.hierarchy)], ['Output', vec(pinned.output)]]} /></details></>}</NeuralLab>;
}
export function BlockedFftLab() {
  const [u, setU] = useState([2, -1, 3, 0, 1, 2, -.5]),
    [h, setH] = useState([.5, 1, -.25, .125]),
    [size, setSize] = useState(3),
    [selected, setSelected] = useState(0),
    r = blockedConvolution(u, h, size),
    index = Math.min(selected, r.blocks.length - 1),
    block = r.blocks[index],
    direct = directConvolution(u, h),
    reset = u.flatMap((_, i) => i % size === 0 ? directConvolution(u.slice(i, i + size), h) : []),
    same = r.full.slice(Math.floor((h.length - 1) / 2), Math.floor((h.length - 1) / 2) + u.length);
  return <NeuralLab id="hyena-blocked-fft" title="Trace the efficient blocked FFT route"><p>The transformed filter is reused. Each inverse transform contributes its complete valid tail at the block’s original global offset. Try B larger than the signal or smaller than the filter, including an incomplete final block.</p><Vector label="Blocked signal" values={u} min={-4} max={4} onChange={setU} /><Vector label="Blocked filter" values={h} min={-2} max={2} onChange={setH} /><div className="neural-controls"><NeuralNumber label="Input block size B" value={size} min={1} max={16} integer onChange={setSize} /><NeuralNumber label="Inspect input block" value={index} min={0} max={r.blocks.length - 1} integer onChange={setSelected} /></div><div className="hy-controls"><button disabled={u.length >= 12} onClick={() => setU([...u, 1])}>Add signal sample</button><button disabled={u.length <= 2} onClick={() => setU(u.slice(0, -1))}>Remove signal sample</button><button disabled={h.length >= 12} onClick={() => setH([...h, .125])}>Add kernel tap</button><button disabled={h.length <= 1} onClick={() => setH(h.slice(0, -1))}>Remove kernel tap</button><button onClick={() => setH(h.map(() => 0))}>Zero kernel</button><button onClick={() => {
        setU([1, -2, .5, 3, -1, 2, .25, -.5, 1, 2]);
        setH([1, -.5, .25, 0, -.125]);
        setSize(3);
        setSelected(3);
      }}>Ten-sample practice case</button><button onClick={() => {
        setU([2, -1, 3, 0, 1, 2, -.5]);
        setH([.5, 1, -.25, .125]);
        setSize(3);
        setSelected(0);
      }}>Reset blocked route</button></div><p className="hy-result" aria-live="polite">F={r.size}≥B+M−1={size + h.length - 1}. {r.blocks.length} blocks. Full overlap-add/reference difference {f(maxError(r.output, direct), 12)}. Output {vec(r.output)}.</p><Figure title="One filter transform, many bounded block transforms" height={240}><Node x={10} y={20} width={210} lines={['kernel → zero-pad → FFT', 'reused filter spectrum']} height={60} /><Arrow x1={220} y1={50} x2={370} y2={100} /><Node x={10} y={125} width={250} lines={[`block ${index}, global start${block.start}`, `${block.input.length} real values → pad to${r.size}`]} height={60} /><Arrow x1={260} y1={155} x2={310} y2={155} /><Node x={310} y={110} width={310} lines={['FFT → multiply → inverse FFT', `keep ${block.convolved.length} valid terms`, `add at offset ${block.start}, do not overwrite`]} height={90} /><text x="15" y="228">This display traces computation; it does not claim a measured speed advantage.</text></Figure><Figure title="Actual block tails aligned to global output positions" width={Math.max(640, 150 + r.full.length * 64)} height={80 + r.blocks.length * 48} description="Rows start at each block’s original offset; shaded terms extend beyond that input block and must overlap-add.">{r.blocks.map((b, i) => <g key={b.start}><text x="5" y={38 + i * 48}>block {i}</text>{b.convolved.map((v, j) => <g key={j}><rect x={90 + (b.start + j) * 64} y={15 + i * 48} width="61" height="35" fill={j >= b.input.length ? '#493e24' : '#242424'} stroke={i === index ? '#e6bb60' : '#555'} /><text x={120 + (b.start + j) * 64} y={38 + i * 48} textAnchor="middle" className="hy-small">{f(v, 4)}</text></g>)}</g>)}{r.full.map((_, t) => <text key={t} x={120 + t * 64} y={65 + r.blocks.length * 48} textAnchor="middle">{t}</text>)}</Figure><Plot title="Correct output and two alignment/history mistakes" xLabel="Output position" yLabel="Value" series={[{
      label: 'Direct / overlap-add',
      values: direct.map((v, t) => [t, v])
    }, {
      label: 'Reset/crop each block: wrong history',
      values: reset.map((v, t) => [t, v])
    }, {
      label: 'Centered same slice: wrong causal alignment',
      values: same.map((v, t) => [t, v])
    }]} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Selected block: exact arrays and Fourier products</h4><Values caption="Block trace" rows={[["Input zero padding", vec(block.padded)], ["Inverse valid terms", vec(block.convolved)], ["Global array after adding this block", vec(block.after)], ["Complete full convolution", vec(r.full)]]} /><NeuralTable caption="Reused kernel spectrum and selected block spectrum" headers={['Bin', 'Filter real', 'Filter imag', 'Block real', 'Block imag', 'Product real', 'Product imag']} rows={r.filterSpectrum.real.map((v, i) => [i, f(v, 8), f(r.filterSpectrum.imaginary[i], 8), f(block.spectrum.real[i], 8), f(block.spectrum.imaginary[i], 8), f(block.product.real[i], 8), f(block.product.imaginary[i], 8)])} /></section><p>The centered slice shown here uses the same full convolution with offset floor((M−1)/2), matching the library’s centered “same” alignment for this first-input length. The normal causal library route is <code>oaconvolve(u,h,mode="full")[:len(u)]</code>.</p></NeuralLab>;
}
export function StreamingLab() {
  const [u, setU] = useState([2, 0, -1, 3, 1, -.5]),
    [residues, setResidues] = useState([.6, .4]),
    [poles, setPoles] = useState([.5, -.25]),
    [boundary, setBoundary] = useState(3),
    [retained, setRetained] = useState(2),
    [selected, setSelected] = useState(2),
    [reset, setReset] = useState(false),
    [pinned, setPinned] = useState(null),
    split = Math.min(boundary, u.length - 1),
    keep = Math.min(retained, u.length),
    t = Math.min(selected, u.length - 1),
    kernel = modalFilter(residues, poles, u.length),
    full = modalScan(u, residues, poles),
    first = modalScan(u.slice(0, split), residues, poles),
    suffix = modalScan(u.slice(split), residues, poles, {
      initial: reset ? null : first.at(-1).state
    }),
    carried = [...first, ...suffix],
    reference = directConvolution(u, kernel),
    truncation = finiteTruncation(u, kernel, keep);
  return <NeuralLab id="hyena-streaming" title="HD · Preserve history or change it deliberately"><p>Move the chunk boundary while carrying all mode registers. The result should stay fixed to floating-point precision. Resetting a boundary or shortening the filter are different interventions and need not preserve it.</p><Vector label="Streaming inputs" values={u} onChange={setU} /><div className="hy-two"><Vector label="Mode residues R" values={residues} min={-2} max={2} onChange={setResidues} /><Vector label="Mode poles λ" values={poles} min={-.95} max={.95} onChange={setPoles} /></div><div className="neural-controls"><NeuralNumber label="Chunk boundary: next input index" value={split} min={1} max={u.length - 1} integer onChange={setBoundary} /><NeuralNumber label="Retained finite filter taps" value={keep} min={1} max={u.length} integer onChange={setRetained} /><NeuralNumber label="Register inspection input" value={t} min={0} max={u.length - 1} integer onChange={setSelected} /></div><label><input type="checkbox" checked={reset} onChange={e => setReset(e.target.checked)} /> Reset registers at the boundary</label><div className="hy-controls"><button disabled={u.length >= 12} onClick={() => setU([...u, 1])}>Add streaming input</button><button disabled={u.length <= 2} onClick={() => setU(u.slice(0, -1))}>Remove streaming input</button><button disabled={poles.length >= 4} onClick={() => {
        setPoles([...poles, .2]);
        setResidues([...residues, .1]);
      }}>Add mode</button><button disabled={poles.length <= 1} onClick={() => {
        setPoles(poles.slice(0, -1));
        setResidues(residues.slice(0, -1));
      }}>Remove mode</button><button onClick={() => setU(u.map(() => 0))}>Zero inputs</button><button onClick={() => setResidues(residues.map(() => 0))}>Zero residues</button><button onClick={() => setRetained(u.length)}>Keep every finite tap</button><button onClick={() => setPinned({
        u: [...u],
        residues: [...residues],
        poles: [...poles],
        split,
        keep,
        reset,
        output: carried.map(r => r.output)
      })}>Pin streaming calculation</button><button onClick={() => {
        setU([2, 0, -1, 3, 1, -.5]);
        setResidues([.6, .4]);
        setPoles([.5, -.25]);
        setBoundary(3);
        setRetained(2);
        setSelected(2);
        setReset(false);
        setPinned(null);
      }}>Reset streaming</button></div><p className="hy-result" aria-live="polite">Full recurrence/direct difference {f(maxError(full.map(r => r.output), reference), 12)}; chosen boundary/reference difference {f(maxError(carried.map(r => r.output), reference), 9)}. Finite truncation: max error {f(Math.max(...truncation.errors), 9)}≤{f(truncation.bound, 9)}.</p><Figure title="The state crosses the chunk boundary" height={230}><Node x={10} y={20} width={170} lines={[`inputs0…${split - 1}`]} /><Arrow x1={180} y1={42} x2={225} y2={42} /><Node x={225} y={10} width={240} height={35 + 20 * poles.length} lines={['registers at boundary', ...first.at(-1).state.map((v, i) => `s${i}=${f(v, 6)}`)]} /><Arrow x1={465} y1={42} x2={500} y2={42} color={reset ? '#e9979f' : '#87bdf1'} /><Node x={500} y={20} width={125} lines={[`${split}…${u.length - 1}`]} /><text x="15" y="177">{reset ? 'Reset replaces every incoming register by zero.' : 'Carry reuses every register; chunking only regroups the same recurrence.'}</text><text x="15" y="213">A gated streaming block must also retain its separate short-convolution history.</text></Figure><ModalRegisters title={`Chosen path at input ${t}${reset && t >= split ? ' after reset' : ''}`} row={carried[t]} residues={residues} poles={poles} /><Plot title="Current full, chunked and truncated outputs" xLabel="Input position" yLabel="Read value" series={[{
      label: 'Full direct convolution',
      values: reference.map((v, i) => [i, v])
    }, {
      label: reset ? 'Reset chunks' : 'Carried chunks',
      values: carried.map((r, i) => [i, r.output])
    }, {
      label: `Finite truncation to${keep} taps`,
      values: truncation.output.map((v, i) => [i, v])
    }, ...(pinned && pinned.output.length === u.length ? [{
      label: 'Pinned chosen chunks',
      values: pinned.output.map((v, i) => [i, v]),
      dashed: true
    }] : [])]} /><Stems title="Current finite filter and truncation" values={truncation.approximation} reference={kernel} referenceLabel="Full finite kernel" /><Plot title="Per-position finite error against the finite-horizon bound" xLabel="Output position" yLabel="Absolute difference" series={[{
      label: 'Observed truncation error',
      values: truncation.errors.map((v, i) => [i, v])
    }, {
      label: '‖u‖∞ × omitted absolute mass',
      values: truncation.errors.map((_, i) => [i, truncation.bound]),
      dashed: true
    }]} /><Values caption="All current mode outputs" rows={[["Kernel", vec(kernel)], ["Chosen chunk output", vec(carried.map(r => r.output))], ["Finite omitted mass", f(truncation.omitted, 12)]]} />{pinned && <details><summary>Pinned complete settings</summary><Values caption="Pinned streaming calculation" rows={Object.entries(pinned).map(([key, value]) => [key, Array.isArray(value) ? vec(value) : String(value)])} /></details>}</NeuralLab>;
}
