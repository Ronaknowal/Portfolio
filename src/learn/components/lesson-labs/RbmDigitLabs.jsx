import useLessonViewport from './useLessonViewport.js';
import { useMemo, useState } from 'react';
import { NeuralLab, NeuralSelect, NeuralTable, NeuralPlot, formatNeural as f } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { completion, hiddenEnumeration, rbmMetrics, exactSample } from '../../data/rbm-models.js';
import summary from '../../data/rbm-study-summary.js';
import { PixelTile, ProbabilityBars, RbmResource, useRbmAsset } from './RbmElements.jsx';
const choices = summary.fits.map(fit => [fit.method + '-' + fit.seed, fit.method.toUpperCase() + ', seed ' + fit.seed]);
const half = () => Array.from({
  length: 64
}, (_, i) => i % 8 < 4);
function ModelChoice({
  value,
  onChange,
  label = 'Retained fitted model'
}) {
  return <NeuralSelect label={label} value={value} onChange={onChange} options={choices} />;
}
export function RbmStudyFigure() {
  const [selected, setSelected] = useState('exact-11');
  const run = summary.fits.find(row => row.method + '-' + row.seed === selected);
  return <figure className="rbm-figure"><figcaption>Nine actual fits, one fixed protocol</figcaption><ModelChoice value={selected} onChange={setSelected} /><div className="rbm-plot-scroll" tabIndex={0} role="region" aria-label="Measured fit and development exact negative log likelihood"><div className="rbm-plot-size"><NeuralPlot title="Recorded exact NLL at five actual post-epoch checkpoints" xLabel="completed epoch" yLabel="nats per binary image" xDomain={[1, 300]} yDomain={[18, 28]} series={['fit', 'development'].map((role, i) => ({
          label: role,
          color: i ? '#bcbcbc' : '#e6b854',
          values: run.history.map(r => [r.epoch, r[role]])
        }))} /></div></div>
    <p>Only epochs 1, 10, 50, 100 and 300 were measured here. Lines join those observations; they are not unrecorded intermediate measurements. The assessment split was scored at the final model.</p>
    <NeuralTable caption="All final assessment outcomes; no selected winner" headers={['Method / seed', 'NLL nats/image', 'Reconstruction MSE', 'Completion MSE', 'Correct missing pixels /2560']} rows={summary.fits.map(row => [row.method + ' / ' + row.seed, f(row.nll, 6), f(row.reconstruction, 6), f(row.completionMse, 6), row.completionCorrect]).concat([['Independent pixels', f(summary.baseline.nll_nats_per_image.assessment, 6), 'not an RBM reconstruction', f(summary.baseline.completion_mse, 6), summary.baseline.completion_correct]])} />
    <p>Completion uses the fixed left-half mask on 80 assessment images. The NLL and completion columns answer different probability questions; edited examples below do not alter these measured results.</p>
  </figure>;
}
export function RbmSampleGallery() {
  const [selected, setSelected] = useState('exact-11'),
    resource = useRbmAsset('model-' + selected + '.json');
  return <NeuralLab id="rbm-samples" title="Independent generation from the exact hidden marginal"><ModelChoice value={selected} onChange={setSelected} /><RbmResource resource={resource}>{data => <><div className="rbm-sample-grid">{data.samples.map((pixels, i) => <div key={i}><PixelTile title={'Saved draw ' + (i + 1)} values={pixels} /><small>h={data.hiddenSamples[i].join('')}</small></div>)}</div><p>All sixteen original saved samples, in original order. Each draws one of the 256 hidden configurations from its exact marginal, then binary visible pixels conditional on that hidden state. None receives a digit input; these are neither reconstructions nor a sixteen-step Gibbs chain.</p></>}</RbmResource></NeuralLab>;
}
export function RbmDigitLab() {
  const [selected, setSelected] = useState('exact-11'),
    images = useRbmAsset('assessment-images.json');
  return <NeuralLab id="rbm-digits" title="Condition on actual digit evidence; keep unknown pixels unknown"><ModelChoice value={selected} onChange={setSelected} /><RbmResource resource={images}>{sources => <LoadedDigitModel selected={selected} sources={sources} />}</RbmResource></NeuralLab>;
}
function LoadedDigitModel({
  selected,
  sources
}) {
  const model = useRbmAsset('model-' + selected + '.json');
  return <RbmResource resource={model}>{data => <DigitWorkspace key={selected} data={data} sources={sources} name={selected} />}</RbmResource>;
}
function DigitWorkspace({
  data,
  sources,
  name
}) {
  const [sourceIndex, setSourceIndex] = useState(0),
    source = sources[sourceIndex];
  const [pixels, setPixels] = useState(sources[0].pixels),
    [mask, setMask] = useState(half),
    [selected, setSelected] = useState(18),
    [seed, setSeed] = useState(71),
    [teachingSection, ready] = useLessonViewport();
  const prepared = useMemo(() => hiddenEnumeration(data.model), [data]);
  const result = useMemo(() => completion(pixels, mask, data.model, prepared), [pixels, mask, data, prepared]);
  const baseline = useMemo(() => completion(source.pixels, half(), data.model, prepared), [source, data, prepared]);
  const fullSource = useMemo(() => rbmMetrics(source.pixels, data.model, prepared), [source, data, prepared]);
  const missing = mask.flatMap((observed, i) => observed ? [] : [i]);
  const mse = missing.length ? missing.reduce((sum, i) => sum + (result.probabilities[i] - source.pixels[i]) ** 2, 0) / missing.length : null;
  const correct = missing.filter(i => Number(result.probabilities[i] >= .5) === source.pixels[i]).length;
  const belief = data.model.b.map((_, i) => prepared.states.reduce((sum, h, j) => sum + h[i] * result.posterior[j], 0));
  const samples = useMemo(() => ready ? exactSample(data.model, seed, 8, pixels, mask) : [], [ready, data, seed, pixels, mask]);
  const reset = () => {
    setPixels(source.pixels);
    setMask(half());
    setSelected(18);
    setSeed(71);
    setShowSamples(false);
  };
  const switchSource = index => {
    setSourceIndex(index);
    setPixels(sources[index].pixels);
    setMask(half());
    setSelected(18);
    setShowSamples(false);
  };
  const selectToggle = () => setPixels(old => old.map((value, i) => i === selected ? 1 - value : value));
  return <><p>Real assessment source {source.id}, original digit label {source.digit}; labels are not RBM inputs. Model {name}. White cells are binary 1; charcoal cells are binary 0. Select any cell or use its numeric index.</p>
    <div className="neural-controls"><NeuralSelect label="Assessment source image" value={sourceIndex} onChange={v => switchSource(Number(v))} options={sources.map((r, i) => [i, 'Source ' + r.id + ', original digit ' + r.digit])} /><NeuralNumber label="Selected pixel index, row-major" value={selected} min={0} max={63} integer range={false} onChange={setSelected} /></div>
    <p>Selected pixel {selected}: row {Math.floor(selected / 8)}, column {selected % 8}; {mask[selected] ? 'observed evidence' : 'missing, placeholder only'}, binary value {pixels[selected]}.</p>
    <div className="neural-buttons"><button onClick={selectToggle}>Toggle selected {mask[selected] ? 'observed pixel' : 'missing placeholder'}</button><button aria-pressed={mask[selected]} onClick={() => setMask(old => old.map((v, i) => i === selected ? !v : v))}>Toggle selected observed / missing</button><button onClick={() => setMask(Array(64).fill(false))}>Hide every pixel</button><button onClick={() => setMask(Array(64).fill(true))}>Observe every pixel</button><button onClick={reset}>Reset current source and half mask</button></div>
    <div className="rbm-pixel-roles"><PixelTile title="1. Current evidence and explicit missing mask" values={pixels} mask={mask} selected={selected} onSelect={setSelected} /><PixelTile title="2. Conditional probabilities; observed values stay fixed" values={result.probabilities} selected={selected} onSelect={setSelected} /><PixelTile title="3. Original withheld truth (never supplied where missing)" values={source.pixels} /><PixelTile title="4. Original source, fixed-half-mask baseline" values={baseline.probabilities} /></div>
    <NeuralTable caption="Current conditional calculation; the original half-mask baseline is pinned" headers={['Quantity', 'Value']} rows={[['Current conditional probability at pixel ' + selected, f(result.probabilities[selected], 8)], ['Pinned baseline at selected pixel', f(baseline.probabilities[selected], 8)], ['Selected probability difference', f(result.probabilities[selected] - baseline.probabilities[selected], 8)], ['Largest change among current missing pixels', missing.length ? f(Math.max(...missing.map(i => Math.abs(result.probabilities[i] - baseline.probabilities[i]))), 8) : 'No missing pixels'], ['Current missing pixel count', missing.length], ['Current-case probability MSE against original withheld truth', mse === null ? 'No hidden pixels to score' : f(mse, 8)], ['Current-case thresholded matches', missing.length ? correct + ' / ' + missing.length : 'No hidden pixels to score'], ['Exact full original-image NLL (immutable source)', f(fullSource.nll, 8) + ' nats']]} />
    <p>Changing an observed bit changes evidence. Changing only a hatched placeholder contributes nothing to the hidden posterior. Hiding all pixels recovers unconditional model marginals; observing all pixels copies the input exactly. The local score uses original withheld pixels as reference and is not the 80-image assessment result after any edit.</p>
    <ProbabilityBars title="Hidden on-probabilities given the current observed evidence" rows={belief.map((current, i) => ({
      label: 'h' + (i + 1),
      current
    }))} />
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect learned feature weights and every current probability</h4><p>All eight filters and all nine models use the same signed scale ±{f(summary.sharedWeightExtent, 4)}. Amber is positive; blue is negative. A pattern is a learned interaction map, not an assigned digit name.</p><div className="rbm-filter-grid">{data.model.b.map((_, j) => <PixelTile key={j} title={'Hidden ' + (j + 1) + ' weights'} values={data.model.w.map(row => row[j])} signed extent={summary.sharedWeightExtent} />)}</div><NeuralTable caption="Current mask, source and conditional probabilities" headers={['Index / row / column', 'Observed?', 'Current binary value', 'Original truth', 'Conditional probability', 'Half-mask baseline']} rows={pixels.map((v, i) => [i + ' / ' + Math.floor(i / 8) + ' / ' + i % 8, mask[i] ? 'yes' : 'no', v, source.pixels[i], f(result.probabilities[i], 8), f(baseline.probabilities[i], 8)])} /></section>
    <section   data-lesson-teaching="" ref={teachingSection} className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Draw eight whole conditional alternatives</h4><NeuralNumber label="Conditional sample stream seed" value={seed} min={0} max={100000} integer range={false} onChange={setSeed} /><div className="rbm-sample-grid">{samples.map((sample, i) => <div key={i}><PixelTile title={'Conditional sample ' + (i + 1)} values={sample.pixels} /><small>h={sample.hidden.join('')}</small></div>)}</div><p>Mulberry32 seed {seed}, categorical hidden posterior then Bernoulli missing pixels. Observed pixels are restored exactly. Samples show joint alternatives; the gray probability image above is their conditional mean.</p></section>
  </>;
}
