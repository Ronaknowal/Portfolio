import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useRef, useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralLab, NeuralSelect, NeuralTable } from './NeuralLessonElements.jsx';
import { difference, frozenForward, norm, swappedGenerator } from '../../data/spectral-regularization-models.js';
import { SpectralFigure, SpectralPlot, SpectralVector, f, vector } from './SpectralPrimitives.jsx';
const base = '/learn-code/spectral-normalization-gradient-penalty/';
const methods = ['clipping', 'gradient-penalty', 'spectral-normalization'],
  seeds = [11, 29, 47];
const options = methods.flatMap(method => seeds.map(seed => [method + '-' + seed, method + ' · seed ' + seed]));
const initialStudyState = () => ({ imageIndex: 0, role: 'fit', checkpoint: 3, point: [.3, .3], slice: 20, z: [-.7, .4], pinned: [-.7, .4] });
function useResource(filename, active, type = 'json') {
  const [state, setState] = useState({
      data: null,
      error: null,
      key: null
    }),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setState({
      data: null,
      error: null,
      key: filename
    });
    fetch(base + filename, {
      signal: controller.signal
    }).then(response => {
      if (!response.ok) throw Error('unavailable');
      return type === 'json' ? response.json() : response.text();
    }).then(data => {
      if (!controller.signal.aborted) setState({
        data,
        error: null,
        key: filename
      });
    }).catch(() => {
      if (!controller.signal.aborted) setState({
        data: null,
        error: 'The selected saved resource could not be loaded. Retry it or choose another model.',
        key: filename
      });
    });
    return () => controller.abort();
  }, [filename, active, type, attempt]);
  return {
    ...state,
    data: state.key === filename ? state.data : null,
    error: state.key === filename ? state.error : null,
    retry: () => setAttempt(n => n + 1)
  };
}
export function SpectralProgram({
  filename = "critic-regularization-study.py"
}) {
  return <>
    <RemoteCodeBlock source={base + filename} language="python" filename={filename} title={"Read the complete executed " + (filename === 'critic-regularization-study.py' ? 'training' : 'sensitivity') + " program"} />
    <p><a href={base + filename} download>Download this exact program</a> with <a href={base + 'digits-400.csv'} download>the CSV</a>{filename === 'sensitivity-calculations.py' && <> and <a href={base + 'calculated-inputs.json'} download>its required saved-fit results</a></>} and <a href={base + 'data-provenance.md'}>data attribution and reproduction notes</a>. Its native run and saved fits are described above.</p>
  </>;
}
function DataImage({
  row
}) {
  const left = row.pixels.reduce((sum, v, i) => sum + (i % 8 < 4 ? v : 0), 0),
    right = row.pixels.reduce((sum, v, i) => sum + (i % 8 >= 4 ? v : 0), 0);
  return <><SpectralFigure title={'Recorded image: source ' + row.sourceId + ', digit ' + row.digit + ', ' + row.role} width={420} height={245} description="Only this image is an observed digit. Generated two-coordinate profiles do not determine an image, so no generated digit is reconstructed.">{row.pixels.map((value, i) => <rect key={i} x={23 + i % 8 * 25} y={15 + Math.floor(i / 8) * 25} width="25" height="25" fill={'rgb(' + Array(3).fill(Math.round(value / 16 * 255)).join(',') + ')'} />)}<path d="M123 12 V218" stroke="#e6b854" strokeWidth="2" /><path d="M23 218 V223 H121 V218 M125 218 V223 H223 V218" fill="none" stroke="#e6b854" /><text x="72" y="239" textAnchor="middle" fill="#ddd" fontSize="12">left half</text><text x="174" y="239" textAnchor="middle" fill="#ddd" fontSize="12">right half</text><text x="238" y="45" fill="#ddd" fontSize="13">left sum / 512</text><text x="238" y="67" fill="#e6b854" fontSize="14">{left + ' / 512 = ' + f(left / 512, 6)}</text><text x="238" y="106" fill="#ddd" fontSize="13">right sum / 512</text><text x="238" y="128" fill="#e6b854" fontSize="14">{right + ' / 512 = ' + f(right / 512, 6)}</text><text x="238" y="177" fill="#ddd" fontSize="13">each half: 32 pixels</text><text x="238" y="199" fill="#ddd" fontSize="13">each pixel: 0…16</text></SpectralFigure><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect this image’s exact pixel values</h4><NeuralTable caption={'Source ' + row.sourceId + ' pixels'} headers={['row', 0, 1, 2, 3, 4, 5, 6, 7]} rows={Array.from({
        length: 8
      }, (_, i) => [i, ...row.pixels.slice(i * 8, i * 8 + 8)])} /></section></>;
}
function ProfileCloud({
  dataset,
  fit,
  checkpoint,
  role,
  selected
}) {
  const records = dataset.rows.filter(row => role === 'all' || row.role === role);
  const points = fit.history[checkpoint].generated;
  return <SpectralFigure title={'Recorded profiles and actual generated checkpoint ' + fit.history[checkpoint].step} width={370} height={391} description="Both axes run from zero to one with the same scale. Small circles are recorded profiles; amber crosses are all 256 saved generated profiles. Point overlap represents real multiplicity. The larger white ring selects an observed image."><rect x="45" y="30" width="280" height="280" fill="#101010" stroke="#777" />{[0, .25, .5, .75, 1].map(v => <g key={v}><text x={45 + 280 * v} y="330" textAnchor="middle" fill="#ddd" fontSize="12">{v}</text><text x="35" y={314 - 280 * v} textAnchor="end" fill="#ddd" fontSize="12">{v}</text></g>)}{records.map(row => <circle key={row.sourceId} cx={45 + 280 * row.profile[0]} cy={310 - 280 * row.profile[1]} r="2.5" fill={row.role === 'fit' ? '#ddd' : row.role === 'development' ? '#ba9fd1' : '#92c9e6'} opacity=".6" />)}{points.map(([x, y], i) => <path key={i} d={'M' + (43 + 280 * x) + ' ' + (308 - 280 * y) + ' l4 4 m-4 0 l4 -4'} stroke="#e6b854" fill="none" opacity=".55" />)}<circle cx={45 + 280 * selected.profile[0]} cy={310 - 280 * selected.profile[1]} r="7" stroke="#fff" fill="none" /><text x="185" y="355" textAnchor="middle" fill="#ddd" fontSize="13">left normalized ink →</text><text x="5" y="20" fill="#ddd" fontSize="13">right ink ↑</text><text x="185" y="375" textAnchor="middle" fill="#ddd" fontSize="12">{'recorded ' + records.length + '; generated 256; selected image ' + selected.sourceId}</text></SpectralFigure>;
}
function CriticSurface({
  fit, point, setPoint, slice, setSlice
}) {
  const canvas = useRef(null);
  const response = frozenForward(point, fit.critic_layers, 'critic'),
    minimum = Math.min(...fit.grid_score),
    maximum = Math.max(...fit.grid_score),
    maxGradient = fit.grid_max_gradient;
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, 320, 320);
    context.fillStyle = '#141414';
    context.fillRect(0, 0, 320, 320);
    const span = maximum - minimum || 1;
    fit.grid_coordinates.forEach(([x, y], i) => {
      const t = (fit.grid_score[i] - minimum) / span;
      context.fillStyle = 'rgb(' + [Math.round(32 + 190 * t), Math.round(32 + 140 * t), Math.round(32 + 50 * t)].join(',') + ')';
      context.fillRect(35 + x * 250 - 3.1, 285 - y * 250 - 3.1, 6.3, 6.3);
    });
    const arrow = (p, g, stroke) => {
      const scale = maxGradient > 0 ? 18 / maxGradient : 0,
        dx = g[0] * scale,
        dy = -g[1] * scale,
        x = 35 + p[0] * 250,
        y = 285 - p[1] * 250;
      context.strokeStyle = stroke;
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(x + dx, y + dy);
      if (Math.hypot(dx, dy) > 1) {
        const angle = Math.atan2(dy, dx);
        context.moveTo(x + dx - 4 * Math.cos(angle - .45), y + dy - 4 * Math.sin(angle - .45));
        context.lineTo(x + dx, y + dy);
        context.lineTo(x + dx - 4 * Math.cos(angle + .45), y + dy - 4 * Math.sin(angle + .45));
      }
      context.stroke();
    };
    fit.grid_coordinates.forEach((p, i) => {
      if (Math.round(p[0] * 40) % 5 === 0 && Math.round(p[1] * 40) % 5 === 0) arrow(p, fit.grid_gradient[i], '#eee');
    });
    context.lineWidth = 2;
    arrow(point, response.jacobian[0], '#92c9e6');
    context.strokeStyle = '#92c9e6';
    context.beginPath();
    context.arc(35 + point[0] * 250, 285 - point[1] * 250, 5, 0, 2 * Math.PI);
    context.stroke();
    context.fillStyle = '#eee';
    context.font = '12px sans-serif';
    for (const v of [0, .5, 1]) {
      context.textAlign = 'center';
      context.fillText(String(v), 35 + v * 250, 307);
      context.textAlign = 'right';
      context.fillText(String(v), 28, 289 - v * 250);
    }
  }, [fit, point, minimum, maximum, maxGradient, response.jacobian]);
  return <><figure className="spectral-figure"><figcaption>Final frozen critic: score field and input-gradient vectors</figcaption><div className="spectral-scroll" role="region" aria-label="Full 41 by 41 critic field" tabIndex={0}><canvas ref={canvas} width="320" height="320" className="spectral-data-canvas" role="img" aria-label="Every stored grid score is painted. White gradient arrows appear every fifth row and column; blue is the current live probe. Exact values are in the tables below." /></div><p>Horizontal left ink and vertical right ink: 0…1 on equal axes. Score color runs from dark {f(minimum, 6)} to amber {f(maximum, 6)} for this selected critic. White arrows use one shared scale: 18 pixels represents gradient norm {f(maxGradient, 6)}. The blue probe uses that same scale. Critic score units are not calibrated across methods.</p></figure><SpectralVector label="Fresh critic probe" values={point} onChange={setPoint} min={0} max={1} /><NeuralTable caption="Computed from the current complete frozen critic" headers={['Quantity', 'Value']} rows={[['Input', vector(point)], ['Score', f(response.value[0], 8)], ['Input gradient', vector(response.jacobian[0])], ['Gradient norm', f(norm(response.jacobian[0]), 8)], ['Derivative convention', response.atKink ? 'At an activation corner: displayed branch convention; ordinary derivative ambiguous' : 'Ordinary derivative on the current activation region']]} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect all recorded values, one grid column at a time</h4><NeuralNumber label="Grid x-column index" value={slice} min={0} max={40} integer onChange={setSlice} /><NeuralTable caption={'Grid x≈' + f(slice / 40, 3)} headers={['x', 'y', 'score', '∂x', '∂y', 'norm']} rows={fit.grid_coordinates.map((p, i) => ({
        p,
        i
      })).filter(({
        p
      }) => Math.round(p[0] * 40) === slice).map(({
        p,
        i
      }) => [...p.map(v => f(v, 3)), f(fit.grid_score[i], 6), ...fit.grid_gradient[i].map(v => f(v, 6)), f(norm(fit.grid_gradient[i]), 6)])} /></section></>;
}
function FrozenLatent({
  fit,
  comparison, z, setZ, pinned, setPinned
}) {
  const sources = [fit, comparison],
    results = sources.map(source => {
      const baseline = frozenForward(pinned, source.generator_layers).value,
        current = frozenForward(z, source.generator_layers).value,
        changed = frozenForward([...z].reverse(), source.generator_layers).value,
        joint = frozenForward([...z].reverse(), swappedGenerator(source.generator_layers)).value;
      return {
        baseline,
        current,
        changed,
        joint
      };
    });
  return <><h4>Two complete frozen generators respond to the same editable latent input</h4><SpectralVector label="Latent z" values={z} onChange={setZ} /><SpectralFigure title="The selected outputs move; the saved fit metrics do not" width={370} height={370} description="Both axes show normalized ink 0…1. Each line connects a pinned-input output to the current-input output of the same final generator. White marks the selected generator, purple the comparison; a filled dot is current."><rect x="45" y="30" width="280" height="280" fill="#101010" stroke="#777" />{[0, .5, 1].map(v => <g key={v}><text x={45 + v * 280} y="330" textAnchor="middle" fill="#ddd" fontSize="12">{v}</text><text x="35" y={314 - v * 280} textAnchor="end" fill="#ddd" fontSize="12">{v}</text></g>)}{results.map((r, i) => <g key={i}><line x1={45 + r.baseline[0] * 280} y1={310 - r.baseline[1] * 280} x2={45 + r.current[0] * 280} y2={310 - r.current[1] * 280} stroke={i ? '#ba9fd1' : '#fff'} strokeWidth="2" /><circle cx={45 + r.baseline[0] * 280} cy={310 - r.baseline[1] * 280} r="5" fill="none" stroke={i ? '#ba9fd1' : '#fff'} /><circle cx={45 + r.current[0] * 280} cy={310 - r.current[1] * 280} r="4" fill={i ? '#ba9fd1' : '#fff'} /></g>)}<text x="185" y="356" textAnchor="middle" fill="#ddd" fontSize="13">left normalized ink →; right ink ↑</text></SpectralFigure><p>Pinned latent {vector(pinned)}; current latent {vector(z)}. These are final step-600 functions even when the saved-history plot displays an earlier checkpoint.</p><NeuralTable caption="Actual current output and a coordinate-basis null" headers={['Generator', 'Pinned output', 'Current output', 'Input-only swap output', 'Joint input/column-swap output', 'Joint maximum difference']} rows={sources.map((source, i) => [source.method + ' ' + source.seed, vector(results[i].baseline), vector(results[i].current), vector(results[i].changed), vector(results[i].joint), f(difference(results[i].current, results[i].joint), 12)])} /><p>The joint swap changes coordinate naming and the first layer’s matching columns, leaving the represented function unchanged. Input-only swapping changes the input to fixed weights. The full saved parameters produce every output shown here.</p><div className="neural-buttons"><button onClick={() => setPinned([...z])}>Pin current latent</button><button onClick={() => setZ([-.7, 1.1])}>Apply the recorded coordinate edit</button><button onClick={() => setZ(v => [...v].reverse())}>Swap latent coordinates only</button><button onClick={() => {
        setZ([-.7, .4]);
        setPinned([-.7, .4]);
      }}>Reset latent comparison</button></div></>;
}
export function StudyWorkspace({
  dataset,
  fit,
  comparison, view, onViewChange
}) {
  const [localView, setLocalView] = useState(initialStudyState);
  const currentView = view ?? localView, setView = onViewChange ?? setLocalView;
  const { imageIndex, role, checkpoint, point, slice, z, pinned } = currentView;
  const setField = (field, value) => setView(previous => ({ ...previous, [field]: typeof value === 'function' ? value(previous[field]) : value }));
  const setImageIndex = value => setField('imageIndex', value), setRole = value => setField('role', value), setCheckpoint = value => setField('checkpoint', value);
  const row = dataset.rows[imageIndex],
    bound = Math.max(fit.matrix_product_bound, fit.grid_max_gradient, ...fit.assessment_gradient_norm);
  return <>
 <p>Actual measured dataset: 400 images, 394 equal-profile groups, split 241/79/80 into fitting/development/assessment. The source image ID and digit are for interpretation; digit labels were not supplied to either model.</p><div className="neural-controls"><NeuralNumber label="Observed image row" value={imageIndex + 1} min={1} max={400} integer onChange={v => setImageIndex(v - 1)} /><NeuralSelect label="Recorded profile overlay" value={role} onChange={setRole} options={[['fit', 'Fitting records'], ['development', 'Development records'], ['assessment', 'Assessment records'], ['all', 'All roles: white/purple/blue']]} /><NeuralSelect label="Saved generation checkpoint" value={checkpoint} onChange={v => setCheckpoint(Number(v))} options={fit.history.map((h, i) => [i, 'Step ' + h.step])} /></div><DataImage row={row} /><ProfileCloud dataset={dataset} fit={fit} checkpoint={checkpoint} role={role} selected={row} />
 <SpectralPlot title="Measured development discrepancy at four saved checkpoints" xLabel="generator step" yLabel="mean of 64 projected empirical W1 values" xDomain={[1, 600]} yDomain={[0, Math.max(dataset.bootstrap.metrics.development, ...fit.history.map(h => h.development_projected_w1), ...comparison.history.map(h => h.development_projected_w1)) * 1.1]} series={[{
      label: fit.method + ' ' + fit.seed,
      color: '#e6b854',
      values: fit.history.map(h => [h.step, h.development_projected_w1])
    }, {
      label: comparison.method + ' ' + comparison.seed,
      color: '#ba9fd1',
      values: comparison.history.map(h => [h.step, h.development_projected_w1])
    }, {
      label: 'Fit-bootstrap baseline',
      color: '#ddd',
      dashed: true,
      values: [[1, dataset.bootstrap.metrics.development], [600, dataset.bootstrap.metrics.development]]
    }]} points={[...fit.history, ...comparison.history].map((h, i) => ({
      id: i,
      x: h.step,
      y: h.development_projected_w1,
      label: 'Saved step ' + h.step + ': ' + f(h.development_projected_w1, 6)
    }))} /><p>Only steps 1, 100, 300 and 600 were observed. Connecting lines guide the eye; they do not supply intermediate measurements. No curve is recomputed from a single latent edit.</p>
 <NeuralTable caption="Every declared method and seed; final held-out assessment discrepancy" headers={['Method', 'Seed', 'Fit', 'Development', 'Assessment']} rows={[...dataset.fits.map(v => [v.method, v.seed, f(v.metrics.fit, 8), f(v.metrics.development, 8), f(v.metrics.assessment, 8)]), ['Bootstrap fitting profiles', '2026', f(dataset.bootstrap.metrics.fit, 8), f(dataset.bootstrap.metrics.development, 8), f(dataset.bootstrap.metrics.assessment, 8)]]} />
 <SpectralPlot title="Sensitivity measurements and a distinct matrix-product bound" xLabel="assessment record index" yLabel="input-gradient norm / bound, same Euclidean units" xDomain={[1, 80]} yDomain={[0, Math.max(.01, bound) * 1.05]} series={[{
      label: '80 assessment gradient norms',
      color: '#e6b854',
      values: fit.assessment_gradient_norm.map((v, i) => [i + 1, v])
    }, {
      label: 'Maximum on the 41×41 sampled grid',
      color: '#ba9fd1',
      dashed: true,
      values: [[1, fit.grid_max_gradient], [80, fit.grid_max_gradient]]
    }, {
      label: 'Final dense-matrix product bound',
      color: '#ddd',
      dashed: true,
      values: [[1, fit.matrix_product_bound], [80, fit.matrix_product_bound]]
    }]} /><p>Assessment maximum {f(fit.assessment_max_gradient, 8)}, grid maximum {f(fit.grid_max_gradient, 8)}, matrix-product bound {f(fit.matrix_product_bound, 8)}. A sample maximum is an observation. The bound accounts for all three effective dense matrices and the leaky-ReLU bounds; it is separate from generation quality.</p>
 <CriticSurface fit={fit} point={point} setPoint={value => setField('point', value)} slice={slice} setSlice={value => setField('slice', value)} /><FrozenLatent fit={fit} comparison={comparison} z={z} setZ={value => setField('z', value)} pinned={pinned} setPinned={value => setField('pinned', value)} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Read exact weights, measurements and fitting procedure</h4><p><a href={base + 'model-' + fit.method + '-' + fit.seed + '.json'} download>Download the selected complete fit</a>, <a href={base + 'calculated-inputs.json'} download>all nine original results</a>, and <a href={base + 'sensitivity-calculations.py'} download>the independent sensitivity program</a>. The complete program below explains the common stream pairing, distinct gradient-penalty interpolation stream and retained checkpoints.</p></section>
 </>;
}
export function SpectralMeasuredLab() {
  const ref = useRef(null),
    [active, setActive] = useState(false),
    [selected, setSelected] = useState('spectral-normalization-11'),
    [compared, setCompared] = useState('clipping-11'),
    [view, setView] = useState(initialStudyState);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        setActive(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '240px'
    });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const dataset = useResource('dataset.json', active),
    fit = useResource('model-' + selected + '.json', active),
    comparison = useResource('model-' + compared + '.json', active),
    failed = [dataset, fit, comparison].find(r => r.error);
  return <div ref={ref}><NeuralLab id="spectral-measured" title="Inspect recorded ink, actual frozen generators and their critics"><p>Changing a saved fit keeps the current latent and pinned latent inputs, critic probe, image, role and history checkpoint. Each newly selected generator recomputes both latent outputs from its own final weights.</p><div className="neural-controls"><NeuralSelect label="Selected saved fit" value={selected} onChange={setSelected} options={options} /><NeuralSelect label="Comparison saved fit" value={compared} onChange={setCompared} options={options} /></div>{failed ? <p role="alert">{failed.error}<button onClick={failed.retry}>Retry saved resource</button></p> : dataset.data && fit.data && comparison.data ? <StudyWorkspace dataset={dataset.data} fit={fit.data} comparison={comparison.data} view={view} onViewChange={setView} /> : <p role="status">Loading the selected measured dataset and two frozen fits… Edited inputs, pinned latent, image, overlay and checkpoint are preserved.</p>}<div className="neural-buttons"><button onClick={() => setView(initialStudyState())}>Reset measured investigation inputs</button></div></NeuralLab></div>;
}
