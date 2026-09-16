import { useState } from 'react';
import {
  MIXING, SOURCE_FAMILIES, KEEP_LABELS, ENTERED_AMPLITUDE_LIMIT,
  rotationModel, gradeRotation, contributionModel, gradeContribution, rescaleComponent,
} from '../../data/ica-models';
import {
  DataTable, EqualPlot, Investigation, NumberField, Prediction, RangeField, Readout, Reflection, SelectField,
  format, feedbackNumber, feedbackDifference, useIcaInvestigation,
} from './IcaShared';
import './ica-labs.css';

const FAMILY_OPTIONS = [['binary', 'Binary ±1 (κ = −2)'], ['laplace', 'Standardized Laplace (κ = 3)'], ['gaussian', 'Standard Gaussian (κ = 0)']];
const STATE_CLASS = { A: 'ic-state-a', B: 'ic-state-b', C: 'ic-state-c', D: 'ic-state-d' };


/** R1 · Rotate a distribution whose covariance never changes. */
export function IcaRotationLab() {
  const state = useIcaInvestigation({ family: 'binary', angle: 45 }, draft => {
    if (draft.angle === '' || !Number.isFinite(draft.angle)) return 'Enter an angle in degrees.';
    if (draft.angle < 0 || draft.angle > 180) return 'The angle must lie between 0 and 180 degrees.';
    if (!SOURCE_FAMILIES[draft.family]) return 'Choose one of the three source families.';
    return null;
  });
  const active = rotationModel(state.active.family, state.active.angle);
  const proposed = state.committed && !state.error ? rotationModel(state.draft.family, state.draft.angle) : null;
  const family = SOURCE_FAMILIES[state.active.family];
  const curveTop = family.curveMax;
  // The binary support reaches only sqrt(2); the continuous contours reach 3.
  const extent = family.continuous ? 3.4 : 2;
  const width = 300;
  const plotLeft = 40;
  const plotWidth = width - plotLeft - 10;
  const plotHeight = 150;
  const cx = angle => plotLeft + (angle / 180) * plotWidth;
  const cy = value => 14 + plotHeight - (value / curveTop) * plotHeight;
  const evaluate = (inputs, previous, prediction) => {
    if (inputs.family !== previous.family) {
      return {
        kind: 'family',
        nextActive: { family: inputs.family, angle: inputs.angle },
        graded: false,
        message: `The population changed from ${SOURCE_FAMILIES[previous.family].label} to ${SOURCE_FAMILIES[inputs.family].label}, so this is not the same-family comparison the prompt describes; the recorded prediction was not graded. The run restarts at ${format(inputs.angle, 2)}°, with |κ| = ${format(Math.abs(rotationModel(inputs.family, inputs.angle).kurtosis), 6)}. Record a fresh prediction for your next angle.`,
        status: 'New source family applied. Record a fresh prediction for the next angle experiment.',
      };
    }
    const grade = gradeRotation({ family: previous.family, oldAngle: previous.angle, newAngle: inputs.angle, prediction });
    const words = { smaller: 'smaller', same: 'the same', larger: 'larger' };
    const gaussian = previous.family === 'gaussian'
      ? ' Every angle has the same joint Gaussian distribution; this contrast supplies no source direction.'
      : '';
    // Say what actually happened. The fourth moment is unchanged at every angle
    // of a Gaussian pair, and also between two angles that share a |kappa|.
    const movement = inputs.angle === previous.angle
      ? 'You applied the same direction, so the fourth moment did not move either'
      : grade.actual === 'same'
        ? 'Your direction changed while the fourth moment did not'
        : 'Your direction changed the fourth moment';
    return {
      kind: 'angle',
      graded: true,
      correct: grade.correct,
      before: grade.before,
      after: grade.after,
      difference: grade.difference,
      message: `You recorded ${words[prediction]}; the actual change was ${words[grade.actual]}. |κ| went from ${feedbackNumber(grade.before)} at ${feedbackNumber(previous.angle)}° to ${feedbackNumber(grade.after)} at ${feedbackNumber(inputs.angle)}°, a signed difference of ${feedbackDifference(grade.difference)}. ${movement}, while the variance stayed 1 and the covariance stayed the identity.${gaussian}`,
    };
  };
  return <Investigation kind="rotation" title="Investigation R1 · Rotate a distribution, not just its covariance" state={state}>
    <p>Two independent unit-variance sources share the chosen family. The current projection is y₁ = cos θ · s₁ + sin θ · s₂, and its companion is y₂ = −sin θ · s₁ + cos θ · s₂. <strong>This investigation uses the source-aligned whitened frame</strong>: the section 3 diamond is the same binary distribution after a 45° change of axes and a sign convention, so “30°” here always means 30° from source axis 1.</p>
    <div className="ic-controls">
      <SelectField label="Source family (staged)" value={state.draft.family} onChange={value => state.edit({ family: value })} options={FAMILY_OPTIONS} />
      <NumberField label="Proposed angle θ in degrees (staged)" value={state.draft.angle} min={0} max={180} step="any"
        onChange={value => state.edit({ angle: value })} />
      <RangeField label="Proposed angle slider" value={state.draft.angle} min={0} max={180} step={0.5} onChange={value => state.edit({ angle: value })} />
    </div>
    <div className="ic-buttons">
      {[5, 30, 90].map(angle => <button key={angle} type="button" onClick={() => state.edit({ angle })}>{`Draft ${angle}°`}</button>)}
      <button type="button" onClick={() => state.edit({ angle: 45 })}>Draft 45°</button>
    </div>
    <p className="ic-note">Applied population: <strong>{family.label}</strong> at <strong>{format(state.active.angle, 2)}°</strong>. Staged: {SOURCE_FAMILIES[state.draft.family].label} at {state.draft.angle === '' ? '—' : `${format(state.draft.angle, 2)}°`}.</p>
    <Prediction state={state} action="Apply angle" evaluate={evaluate}
      prompt="Compared with the current projection, the new projection's absolute excess kurtosis will be (differences up to 10⁻¹⁰ count as the same):"
      choices={[['smaller', 'smaller'], ['same', 'the same'], ['larger', 'larger']]} />
    <div className="ic-panels">
      <section className="ic-panel">
        <EqualPlot title={`Joint ${family.continuous ? 'density contours' : 'support'} at ${format(state.active.angle, 2)}°`}
          bounds={[-extent, extent, -extent, extent]} ticks={[-Math.floor(extent), 0, Math.floor(extent)]} size={300}
          describe={family.continuous
            ? `Equal-density contours at levels 1, 2 and 3 for the ${family.label} pair, rotated by ${format(state.active.angle, 2)} degrees. They are density level sets, not observations.`
            : `Four equiprobable states, each with probability 0.25, at ${active.support.map(item => `${item.id} (${format(item.projected[0], 4)}, ${format(item.projected[1], 4)})`).join(', ')}.`}>
          {(sx, sy) => <>
            <line className="ic-axis" x1={sx(-extent)} x2={sx(extent)} y1={sy(0)} y2={sy(0)} />
            <line className="ic-axis" x1={sx(0)} x2={sx(0)} y1={sy(-extent)} y2={sy(extent)} />
            <line className="ic-direction" x1={sx(0)} y1={sy(0)} x2={sx(extent * 0.92)} y2={sy(0)} />
            <line className="ic-direction is-companion" x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(extent * 0.92)} />
            {active.contours.map((contour, index) => <polygon key={contour.level} className={`ic-contour${index === 2 ? ' is-outer' : ''}`}
              points={contour.points.map(point => `${sx(point[0])},${sy(point[1])}`).join(' ')} />)}
            {active.support.map(item => <g key={item.id}>
              <circle className={`ic-state-point ${STATE_CLASS[item.id]}`} cx={sx(item.projected[0])} cy={sy(item.projected[1])} r="6" />
              <text className="ic-state-label" x={sx(item.projected[0])} y={sy(item.projected[1]) - 11} textAnchor="middle">{item.id}</text>
            </g>)}
          </>}
        </EqualPlot>
        <p className="ic-note">y₁ runs along the horizontal axis and y₂ along the vertical one, so the distribution turns while the axes stay put. Covariance is exactly the identity at every angle.</p>
      </section>
      <section className="ic-panel">
        <figure className="ic-plot">
          <figcaption>{`|κ| against θ · ${family.label}`}</figcaption>
          <svg viewBox={`0 0 ${width} ${plotHeight + 64}`} style={{ maxWidth: `${width}px` }} role="img"
            aria-label={`Absolute excess kurtosis against angle from 0 to 180 degrees, on a vertical range of 0 to ${format(curveTop, 2)}. ${family.key === 'gaussian' ? 'The curve is flat at zero for every angle.' : `It peaks at ${format(Math.abs(family.kurtosis), 2)} at 0, 90 and 180 degrees and dips to ${format(Math.abs(family.kurtosis) / 2, 2)} at 45 and 135 degrees.`}`}>
            <line className="ic-curve-zero" x1={plotLeft} x2={plotLeft + plotWidth} y1={cy(0)} y2={cy(0)} />
            <polyline className="ic-curve" points={active.curve.map(point => `${cx(point.angle)},${cy(point.magnitude)}`).join(' ')} />
            {[0, 45, 90, 135, 180].map(tick => <g key={tick}>
              <line className="ic-grid" x1={cx(tick)} x2={cx(tick)} y1={cy(0)} y2={cy(0) + 5} />
              <text x={cx(tick)} y={cy(0) + 20} textAnchor="middle">{tick}</text>
            </g>)}
            {[0, curveTop / 2, curveTop].map(value => <text key={value} x={plotLeft - 6} y={cy(value) + 4} textAnchor="end">{format(value, 2)}</text>)}
            <text className="ic-dim" x={cx(90)} y={cy(0) + 42} textAnchor="middle">angle θ, degrees</text>
            {family.key === 'gaussian' && <text className="ic-dim" x={cx(90)} y={cy(0) - 10} textAnchor="middle">flat at zero</text>}
            <circle className="ic-marker-current" cx={cx(active.angle)} cy={cy(active.magnitude)} r="5" />
            {proposed && <circle className="ic-marker-proposed" cx={cx(proposed.angle)} cy={cy(proposed.magnitude)} r="5" />}
          </svg>
        </figure>
        <p className="ic-note">Analytic population curve, not a measured fit. Solid marker: the applied angle. {proposed ? 'Hollow marker: the angle you have committed a prediction against.' : 'The proposed angle is marked only once you record a prediction.'}</p>
      </section>
    </div>
    <div data-ica-readouts>
      <Readout label="Applied angle" value={`${format(state.active.angle, 2)}°`} />
      <Readout label="κ(y₁)" value={format(active.kurtosis, 6)} note={`|κ| = ${format(active.magnitude, 6)}`} />
      <Readout label="κ(y₂)" value={format(active.companionKurtosis, 6)} note="the companion direction has the same value" />
      <Readout label="Variance and covariance" value="Var(y₁) = Var(y₂) = 1, Cov = 0" note="unchanged by every orthogonal rotation" />
    </div>
    {!family.continuous && <DataTable caption="The four transformed states, each with probability 0.25."
      headings={['State', 'source (s₁, s₂)', 'projected (y₁, y₂)']}
      rows={active.support.map(item => [item.id, `(${format(item.source[0])}, ${format(item.source[1])})`, `(${format(item.projected[0], 5)}, ${format(item.projected[1], 5)})`])} />}
    <p className="ic-note">A single fourth moment is a direction-finding contrast, not a test of independence. A non-Gaussian variable can also have excess kurtosis exactly zero, which is why the adjacent section keeps that counterexample.</p>
    <Reflection resetKey={state.resetCount} prompt="Transfer: reproduce practice 2 at 30°, then propose a different angle with the same |κ| and explain the symmetry." />
  </Investigation>;
}

function ContributionDisplay({ model, title }) {
  const magnitude = Math.max(1e-9, ...model.contributions.flatMap(item => item.contribution.map(Math.abs)), ...model.observed.map(Math.abs));
  const bar = value => ({ left: `${50 + (Math.min(value, 0) / magnitude) * 48}%`, width: `${(Math.abs(value) / magnitude) * 48}%` });
  return <section className="ic-panel">
    <h4>{title}</h4>
    <div className="ic-stack">
      {[0, 1].map(sensor => <div className="ic-stack-row" key={sensor}>
        <span>{`Sensor ${sensor + 1}`}</span>
        <div className="ic-bar-axis">
          <i className={model.keptIndices.includes(0) ? '' : 'is-removed'}
            style={{ ...bar(model.contributions[0].contribution[sensor]), background: '#d9a45c' }}
            title={`source 1 contributes ${format(model.contributions[0].contribution[sensor])}`} />
          <i className={`is-source2${model.keptIndices.includes(1) ? '' : ' is-removed'}`}
            style={{ ...bar(model.contributions[1].contribution[sensor]), background: '#82b5a4' }}
            title={`source 2 contributes ${format(model.contributions[1].contribution[sensor])}`} />
          <b style={{ left: `${50 + (model.retained[sensor] / magnitude) * 48}%` }} title={`reconstruction ${format(model.retained[sensor])}`} />
        </div>
      </div>)}
    </div>
    <p className="ic-note">Signed amplitude, zero at the dashed centre line, full half-width ±{format(magnitude)}. A faded bar is a contribution this keep-set removes; the white marker is the reconstruction.</p>
    {[0, 1].map(sensor => <p className="ic-equation" key={sensor}>
      {`Sensor ${sensor + 1}: `}
      {model.keptIndices.length === 0 ? '0' : model.keptIndices.map(j => `${format(model.mixing[sensor][j])}(${format(model.sources[j])})`).join(' + ')}
      {' = '}<strong>{format(model.retained[sensor])}</strong>
      {model.keptIndices.length < 2 && `  ·  removed ${format(model.removed[sensor])}`}
    </p>)}
  </section>;
}

function ScaleMode({ base }) {
  const state = useIcaInvestigation({ component: '1', scale: 2, mode: 'compensate' }, draft => {
    if (draft.scale === '' || !Number.isFinite(draft.scale)) return 'Enter a scale c.';
    if (draft.scale === 0) return 'c = 0 is invalid: the compensating division by zero is undefined, so no division is performed.';
    if (Math.abs(draft.scale) < 0.25 || Math.abs(draft.scale) > 4) return 'Use a scale with magnitude between 0.25 and 4.';
    return null;
  });
  // Nothing about the rescaled state is rendered until a prediction has been
  // recorded and applied; the answer must not sit beside an unset question.
  const revealed = Boolean(state.result);
  const applied = (() => {
    const index = Number(state.active.component) - 1;
    const rescaled = rescaleComponent({ sources: base.sources, index, scale: state.active.scale, compensate: state.active.mode === 'compensate' });
    return { index, rescaled, before: contributionModel({ sources: base.sources, keep: base.keep }), after: contributionModel({ ...rescaled, keep: base.keep }) };
  })();
  const evaluate = (inputs, previousActive, prediction) => {
    const index = Number(inputs.component) - 1;
    const compensate = inputs.mode === 'compensate';
    const rescaled = rescaleComponent({ sources: base.sources, index, scale: inputs.scale, compensate });
    const before = contributionModel({ sources: base.sources, keep: base.keep });
    const after = contributionModel({ ...rescaled, keep: base.keep });
    const unchanged = before.retained.every((value, sensor) => Math.abs(value - after.retained[sensor]) <= 1e-9);
    const actual = unchanged ? 'same' : 'change';
    // The explanation follows the outcome. A source-only rescale usually does
    // change the observation, but not when this keep-set already drops the
    // component or its amplitude is zero, and the sentence has to say which.
    const excluded = !before.keptIndices.includes(index);
    const zeroAmplitude = base.sources[index] === 0;
    let explanation;
    if (compensate && unchanged) {
      explanation = 'Multiplying the source by c and dividing its column by c leaves every product, and so every observation, untouched: that is the scale ambiguity.';
    } else if (!compensate && !unchanged) {
      explanation = `Scaling the source without compensating its column is a different operation: component ${index + 1}'s physical contribution really changed, so the observation changed with it.`;
    } else if (!compensate && excluded) {
      explanation = `Scaling the source alone is a different operation, but this keep-set already drops component ${index + 1}, so nothing it contributes reaches this reconstruction.`;
    } else if (!compensate && zeroAmplitude) {
      explanation = `Scaling the source alone is a different operation, but component ${index + 1}'s amplitude is 0, and c times 0 is still 0, so there was no contribution to change.`;
    } else if (!compensate && inputs.scale === 1) {
      explanation = 'Multiplication by c = 1 leaves the source itself unchanged; no compensating column change is needed.';
    } else {
      explanation = 'Any numerical difference is within the 10⁻⁹ comparison tolerance used here.';
    }
    return {
      graded: true,
      correct: actual === prediction,
      message: `You recorded “${prediction === 'same' ? 'stay the same' : 'change'}”; the observed sensors ${unchanged ? 'stayed the same within tolerance' : 'changed'}. Source ${index + 1} became ${feedbackNumber(rescaled.sources[index])} and its mixing column became (${feedbackNumber(rescaled.mixing[0][index])}, ${feedbackNumber(rescaled.mixing[1][index])}). The reconstruction went from (${feedbackNumber(before.retained[0])}, ${feedbackNumber(before.retained[1])}) to (${feedbackNumber(after.retained[0])}, ${feedbackNumber(after.retained[1])}). ${explanation}${inputs.scale < 0 && compensate && unchanged ? ' A negative c also flips both signs, which is the sign ambiguity.' : ''}`,
    };
  };
  return <Investigation kind="scale" title="Investigation C1b · Rescale a component and compensate its column" state={state}>
    <p>Hold the applied amplitudes ({format(base.sources[0])}, {format(base.sources[1])}) and the keep-set “{KEEP_LABELS[base.keep]}” fixed. Replace source j by c·s<sub>j</sub>, and optionally divide its mixing column by c.</p>
    <div className="ic-controls">
      <SelectField label="Component to rescale (staged)" value={state.draft.component} onChange={value => state.edit({ component: value })}
        options={[['1', 'Component 1'], ['2', 'Component 2']]} />
      <NumberField label="Scale c (staged)" value={state.draft.scale} min={-4} max={4} step="any" onChange={value => state.edit({ scale: value })} />
      <SelectField label="Operation (staged)" value={state.draft.mode} onChange={value => state.edit({ mode: value })}
        options={[['compensate', 'Scale the source and divide its column by c'], ['source', 'Scale the source only']]} />
    </div>
    <div className="ic-buttons">
      {[2, -2, 0.5].map(value => <button key={value} type="button" onClick={() => state.edit({ scale: value })}>{`Draft c = ${format(value)}`}</button>)}
      <button type="button" onClick={() => state.edit({ scale: 0 })}>Draft c = 0</button>
    </div>
    <Prediction state={state} action="Apply scaling" evaluate={evaluate}
      prompt="Compared with the applied reconstruction, the observed sensor amplitudes will (differences up to 10⁻⁹ count as the same):"
      choices={[['same', 'stay the same'], ['change', 'change']]} />
    {revealed
      ? <DataTable stack caption={`Applied state: component ${applied.index + 1} at c = ${format(state.active.scale)}, ${state.active.mode === 'compensate' ? 'with a compensated column' : 'source only'}.`}
        headings={['Quantity', 'before', 'after']}
        rows={[
          ['Source amplitudes', `(${format(applied.before.sources[0])}, ${format(applied.before.sources[1])})`, `(${format(applied.after.sources[0])}, ${format(applied.after.sources[1])})`],
          [`Mixing column ${applied.index + 1}`, `(${format(MIXING[0][applied.index])}, ${format(MIXING[1][applied.index])})`, `(${format(applied.after.mixing[0][applied.index], 6)}, ${format(applied.after.mixing[1][applied.index], 6)})`],
          ['Contribution of that component', `(${format(applied.before.contributions[applied.index].contribution[0])}, ${format(applied.before.contributions[applied.index].contribution[1])})`, `(${format(applied.after.contributions[applied.index].contribution[0], 6)}, ${format(applied.after.contributions[applied.index].contribution[1], 6)})`],
          ['Reconstructed sensors', `(${format(applied.before.retained[0])}, ${format(applied.before.retained[1])})`, `(${format(applied.after.retained[0], 6)}, ${format(applied.after.retained[1], 6)})`],
        ]} />
      : <>
        <DataTable stack caption="The state you are about to rescale. The result of the rescaling appears only after you record a prediction and apply it."
          headings={['Quantity', 'current']}
          rows={[
            ['Source amplitudes', `(${format(applied.before.sources[0])}, ${format(applied.before.sources[1])})`],
            [`Mixing column ${Number(state.draft.component)}`, `(${format(MIXING[0][Number(state.draft.component) - 1])}, ${format(MIXING[1][Number(state.draft.component) - 1])})`],
            ['Reconstructed sensors', `(${format(applied.before.retained[0])}, ${format(applied.before.retained[1])})`],
          ]} />
        <p className="ic-unrevealed" data-ica-scale-unrevealed>No result is shown yet. Record a prediction, then apply the staged scaling.</p>
      </>}
    <p className="ic-note">A unit-variance convention fixes this scale usefully, but component index and polarity remain conventions. The column norm alone is therefore not a universal energy measure across normalization conventions.</p>
  </Investigation>;
}

/** C1 · Edit a contribution, then ask what removing it changes. */
export function IcaContributionLab() {
  const [everApplied, setEverApplied] = useState(false);
  const state = useIcaInvestigation({ s1: 1, s2: -1, keep: 'both' }, (draft, active) => {
    for (const key of ['s1', 's2']) {
      if (draft[key] === '' || !Number.isFinite(draft[key])) return 'Enter both source amplitudes.';
      if (Math.abs(draft[key]) > ENTERED_AMPLITUDE_LIMIT) return `Amplitudes must lie between −${ENTERED_AMPLITUDE_LIMIT} and ${ENTERED_AMPLITUDE_LIMIT}.`;
    }
    if (!KEEP_LABELS[draft.keep]) return 'Choose a keep-set.';
    if (!everApplied && draft.s1 === 1 && draft.s2 === -1) {
      return { notice: 'Change at least one amplitude to start: the worked example above is already solved, so the first independent task needs your own numbers.' };
    }
    if (active) return null;
    return null;
  });
  const applied = contributionModel({ sources: [state.active.s1, state.active.s2], keep: state.active.keep });
  const full = contributionModel({ sources: [state.active.s1, state.active.s2], keep: 'both' });
  const evaluate = (inputs, previousActive, prediction) => {
    const graded = gradeContribution({ sources: [inputs.s1, inputs.s2], keep: inputs.keep, prediction: prediction.sensor1, sensor: 0 });
    const secondary = Number.isFinite(prediction.sensor2)
      ? gradeContribution({ sources: [inputs.s1, inputs.s2], keep: inputs.keep, prediction: prediction.sensor2, sensor: 1 })
      : null;
    const model = graded.model;
    setEverApplied(true);
    return {
      graded: true,
      correct: graded.correct,
      message: `Sensor 1: you recorded ${feedbackNumber(prediction.sensor1)}, the reconstruction is ${feedbackNumber(graded.actual)} — ${graded.correct ? 'a match within tolerance' : `a difference of ${feedbackDifference(graded.difference)}`}.`
        + (secondary ? ` Sensor 2: you recorded ${feedbackNumber(prediction.sensor2)}, the reconstruction is ${feedbackNumber(secondary.actual)} — ${secondary.correct ? 'a match within tolerance' : `a difference of ${feedbackDifference(secondary.difference)}`}.` : '')
        + ` The full observation is (${format(model.observed[0])}, ${format(model.observed[1])}); keeping ${KEEP_LABELS[inputs.keep].toLowerCase()} retains (${format(model.retained[0])}, ${format(model.retained[1])}) and removes (${format(model.removed[0])}, ${format(model.removed[1])}). The altered result is not expected to equal the original sensors.`,
    };
  };
  return <>
    <Investigation kind="contribution" title="Investigation C1 · Edit a contribution, then remove one" state={{ ...state, reset: () => { state.reset(); setEverApplied(false); } }}>
      <p>The mixing matrix stays A = [[{format(MIXING[0][0])}, {format(MIXING[0][1])}], [{format(MIXING[1][0])}, {format(MIXING[1][1])}]]. Enter your own source amplitudes and choose which components to keep. The worked example — amplitudes (1, −1), both kept, observation (1, −1) — is shown below as the applied state, so the first independent task needs at least one changed amplitude.</p>
      <div className="ic-controls">
        <NumberField label="Source 1 amplitude (staged)" value={state.draft.s1} min={-4} max={4} step="any" onChange={value => state.edit({ s1: value })} />
        <NumberField label="Source 2 amplitude (staged)" value={state.draft.s2} min={-4} max={4} step="any" onChange={value => state.edit({ s2: value })} />
        <SelectField label="Keep-set (staged)" value={state.draft.keep} onChange={value => state.edit({ keep: value })}
          options={Object.entries(KEEP_LABELS)} />
      </div>
      <p className="ic-note">Applied: amplitudes ({format(state.active.s1)}, {format(state.active.s2)}), {KEEP_LABELS[state.active.keep].toLowerCase()}. Staged: ({state.draft.s1 === '' ? '—' : format(state.draft.s1)}, {state.draft.s2 === '' ? '—' : format(state.draft.s2)}), {KEEP_LABELS[state.draft.keep].toLowerCase()}.</p>
      <Prediction state={state} action="Reveal reconstruction" evaluate={evaluate}
        prompt="What will sensor 1's reconstructed amplitude be, in arbitrary amplitude units? Answers match within 10⁻⁹."
        numeric={{
          min: -16,
          max: 16,
          fields: [
            { key: 'sensor1', label: 'Predicted sensor 1 amplitude', required: true },
            { key: 'sensor2', label: 'Predicted sensor 2 amplitude (optional transfer)', required: false },
          ],
        }} />
      <div className="ic-panels">
        <ContributionDisplay model={applied} title={`Reconstruction · ${KEEP_LABELS[state.active.keep].toLowerCase()}`} />
        <section className="ic-panel">
          <h4>Mixing columns as sensor patterns</h4>
          <DataTable caption="Column j says where component j appears across the sensors; the amplitude scales that whole pattern."
            headings={['', 'column 1', 'column 2']}
            rows={[
              ['Sensor 1', format(MIXING[0][0]), format(MIXING[0][1])],
              ['Sensor 2', format(MIXING[1][0]), format(MIXING[1][1])],
              ['Amplitude', format(state.active.s1), format(state.active.s2)],
              ['Contribution', `(${format(full.contributions[0].contribution[0])}, ${format(full.contributions[0].contribution[1])})`,
                `(${format(full.contributions[1].contribution[0])}, ${format(full.contributions[1].contribution[1])})`],
            ]} />
          <Readout label="Full observation x" value={`(${format(full.observed[0])}, ${format(full.observed[1])})`} />
          <Readout label="Retained" value={`(${format(applied.retained[0])}, ${format(applied.retained[1])})`} />
          <Readout label="Removed difference" value={`(${format(applied.removed[0])}, ${format(applied.removed[1])})`} />
        </section>
      </div>
      <Reflection resetKey={state.resetCount} prompt="Find source amplitudes for which dropping component 2 makes sensor 1 larger than the full observation, and say why that happens." />
      <p className="ic-note">This is the known two-source hand model. It is not an interface for choosing clinical exclusions, and nothing here identifies a physiological source.</p>
    </Investigation>
    {everApplied
      ? <ScaleMode key={`${state.active.s1}|${state.active.s2}|${state.active.keep}`} base={{ sources: [state.active.s1, state.active.s2], keep: state.active.keep }} />
      : <p className="ic-unrevealed" data-ica-scale-gate>The scale-ambiguity question opens after you complete one exclusion trial above.</p>}
  </>;
}
