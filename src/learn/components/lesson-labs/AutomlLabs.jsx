import { useId, useState } from 'react';
import {
  acquisitionDomain, acquisitionTable, activeRecipe, addOption, commitOperation, countConfigurations,
  declaredCurves, declaredOperations, declaredPareto, declaredResource, declaredSpace, describeOption,
  dimensionLabels, enumerateConfigurations, familyLabels, familyOrder, fixedDimensions, halvingSchedule,
  hyperbandBrackets, improvementDensity, limits, mixtureCurve, mixtureState, paretoAnalysis,
  controlSteps, removeOption, replayComparison, replayPrefix, samplingMeasure,
} from '../../data/automl-models.js';
import { candidates as recordedCandidates, replayOrder, sourceRows } from '../../data/automl-data.js';
import {
  Attempts, Investigation, Legend, NumberField, Plot, Prediction, Stage, Table, TextField,
  exactly, fixed, polyline, round, signed, useInvestigation,
} from './AutomlShared.jsx';
import './automl-labs.css';

/** The six investigations of the AutoML & NAS lesson.
 *
 * Each one follows the same contract: the learner edits the topic's own
 * entities, records a prediction, and only then sees a result computed from the
 * inputs that prediction was recorded against. Nothing is revealed on first
 * paint, a suggested setup fills fields without applying or answering them, and
 * any relevant edit retires the recorded verdict.
 */

const clone = value => JSON.parse(JSON.stringify(value));

/* ======================================================== §2 · I1 · grammar */

const initialChoices = { logistic: { scaling: 0, penalty: 1 }, tree: { depth: 0 }, neighbors: { scaling: 0, count: 0 }, mlp: { scaling: 0, activation: 0, widths: 0 } };
const spaceBaseline = { space: clone(declaredSpace), family: 'logistic', choices: clone(initialChoices) };
const spaceSetups = [
  {
    key: 'depth8', label: 'Add tree depth 8',
    build: state => ({ ...state, space: { ...state.space, tree: { depth: [...state.space.tree.depth, 8] } } }),
  },
  {
    key: 'penalty10', label: 'Add logistic C = 10',
    build: state => ({ ...state, space: { ...state.space, logistic: { ...state.space.logistic, penalty: [...state.space.logistic.penalty, 10] } } }),
  },
  {
    key: 'inactive', label: 'Null: retune the inactive tree depth while logistic is selected',
    build: state => ({ ...state, family: 'logistic', choices: { ...state.choices, tree: { depth: 1 } } }),
  },
  {
    key: 'widths', label: 'Add a (16, 8) width pattern',
    build: state => ({ ...state, space: { ...state.space, mlp: { ...state.space.mlp, widths: [...state.space.mlp.widths, [16, 8]] } } }),
  },
  {
    key: 'dropScaling', label: 'Remove raw preprocessing from the logistic branch',
    build: state => ({ ...state, space: { ...state.space, logistic: { ...state.space.logistic, scaling: ['standard'] } } }),
  },
  {
    key: 'switch', label: 'Null: switch the selected family to the network',
    build: state => ({ ...state, family: 'mlp' }),
  },
];

function OptionEditor({ family, dimension, values, onAdd, onRemove, disabled }) {
  const [draft, setDraft] = useState('');
  const fixedHere = (fixedDimensions[family] ?? []).includes(dimension);
  const parse = text => {
    const trimmed = text.trim();
    if (trimmed === '') return { error: 'Type a value.' };
    if (dimension === 'widths') {
      const parts = trimmed.split(',').map(part => Number(part.trim()));
      if (parts.some(part => !Number.isInteger(part))) return { error: 'Use whole widths, such as 12 or 16,8.' };
      return { value: parts };
    }
    if (dimension === 'scaling') {
      if (!['raw', 'standard'].includes(trimmed)) return { error: 'Type raw or standard.' };
      return { value: trimmed };
    }
    const number = Number(trimmed);
    if (!Number.isFinite(number)) return { error: 'That is not a number.' };
    return { value: number };
  };
  const [problem, setProblem] = useState(null);
  return <div className="am-option-row">
    <span>{dimensionLabels[dimension]}{fixedHere ? ' (fixed by the grammar)' : ''}:</span>
    {values.map((value, index) => (
      <span className={`am-chip${fixedHere ? ' is-inactive' : ''}`} key={describeOption(dimension, value)}>
        {describeOption(dimension, value)}
        {values.length > 1 && !fixedHere && (
          <button type="button" className="is-quiet" disabled={disabled}
            aria-label={`Remove ${dimensionLabels[dimension]} ${describeOption(dimension, value)} from ${familyLabels[family]}`}
            onClick={() => onRemove(index)}> ×</button>
        )}
      </span>
    ))}
    {!fixedHere && <>
      <TextField label={`Add a ${dimensionLabels[dimension]} to ${familyLabels[family]}`}
        value={draft} onChange={setDraft} disabled={disabled}
        placeholder={dimension === 'widths' ? '16,8' : dimension === 'scaling' ? 'standard' : '10'}
        validate={text => (text.trim() === '' ? null : parse(text).error ?? null)} />
      {/* Six of these rows are on screen at once, so the accessible name has to
          say which branch it extends rather than repeating "Add" six times. */}
      <button type="button" disabled={disabled}
        aria-label={`Add the typed ${dimensionLabels[dimension]} to ${familyLabels[family]}`}
        onClick={() => {
          const parsed = parse(draft);
          if (parsed.error) { setProblem(parsed.error); return; }
          try { onAdd(parsed.value); setDraft(''); setProblem(null); } catch (error) { setProblem(error.message); }
        }}>Add</button>
      {problem && <span className="am-field-error">{problem}</span>}
    </>}
  </div>;
}

export function SearchSpaceLab() {
  const state = useInvestigation(spaceBaseline);
  const draft = state.draft;
  const appliedCount = countConfigurations(state.active.space);
  const draftCount = (() => {
    try { return countConfigurations(draft.space); } catch { return null; }
  })();
  const appliedRecipe = activeRecipe(state.active.space, state.active);
  const draftRecipe = activeRecipe(draft.space, draft);
  const answerFor = (proposed, current) => {
    const before = countConfigurations(current.space).total;
    const after = countConfigurations(proposed.space).total;
    return {
      outcome: after > before ? 'grows' : after < before ? 'shrinks' : 'unchanged',
      value: after, before, after,
    };
  };
  const comparison = state.result ? {
    before: countConfigurations(state.previous.space),
    after: appliedCount,
    beforeRecipe: activeRecipe(state.previous.space, state.previous),
    afterRecipe: appliedRecipe,
  } : null;
  const editSpace = update => state.edit(current => ({ space: update(current.space) }));
  return <Investigation
    title="Assemble a valid experiment, then count it"
    question="The grammar below is this lesson's declared search space. Edit it — add a permitted value, remove one, switch the family you are currently describing, or retune a setting on a family you are not describing — then record whether the number of valid configurations will grow, shrink or stay exactly where it is."
    evidence="Constructed search space. Editing it changes a grammar; it does not fit anything and it never attaches an observed score to a new configuration."
    onReset={state.reset}>

    <Stage title="The permitted registry">
      <p className="am-caption">
        The permitted values only. No branch count, product or total appears until a prediction is recorded — working
        out how the branches combine is the task.
      </p>
      <ul className="am-tree">
        {familyOrder.map(family => {
          const branch = draftCount?.branches.find(entry => entry.family === family);
          return <li key={family} className={draft.family === family ? 'is-current' : undefined}>
            <div className="am-branch-head">
              <b>{familyLabels[family]}</b>
              {/* The sum of these four branch counts IS the graded answer, in the
                  notation the answer panel will use. Showing them before the
                  prediction does the arithmetic this investigation exists to
                  teach. Only the permitted values are visible until then. */}
              <span className="am-branch-count">
                {!branch ? 'invalid'
                  : state.result ? `${branch.expression} = ${branch.count}`
                    : `${Object.keys(draft.space[family]).length} dimension${Object.keys(draft.space[family]).length === 1 ? '' : 's'}`}
              </span>
              {/* One of these per family, so the accessible name has to name
                  the family rather than repeating "describe this family". */}
              <button type="button" className={draft.family === family ? 'is-primary' : 'is-quiet'}
                aria-pressed={draft.family === family}
                aria-label={`Describe a ${familyLabels[family]} configuration`}
                onClick={() => state.edit({ family })}>describe this family</button>
            </div>
            <ul>
              {Object.entries(draft.space[family]).map(([dimension, values]) => (
                <li key={dimension}>
                  <OptionEditor family={family} dimension={dimension} values={values}
                    onAdd={value => editSpace(space => addOption(space, family, dimension, value).space)}
                    onRemove={index => editSpace(space => removeOption(space, family, dimension, index).space)} />
                </li>
              ))}
            </ul>
          </li>;
        })}
      </ul>
    </Stage>

    <Stage title="The configuration you are currently describing">
      <div className="am-controls is-wide">
        {draftRecipe.settings.map(setting => (
          <label className="am-field" key={setting.dimension}>
            <span>{setting.label}{setting.fixed && <output>fixed</output>}</span>
            <select value={String(draft.choices[draft.family][setting.dimension] ?? 0)}
              disabled={setting.fixed || draft.space[draft.family][setting.dimension].length < 2}
              onChange={event => state.edit(current => ({
                choices: {
                  ...current.choices,
                  [current.family]: { ...current.choices[current.family], [setting.dimension]: Number(event.target.value) },
                },
              }))}>
              {draft.space[draft.family][setting.dimension].map((value, index) => (
                <option key={describeOption(setting.dimension, value)} value={index}>
                  {describeOption(setting.dimension, value)}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <p className="am-state-strip">
        <span>applied recipe: <b>{appliedRecipe.serialized}</b></span>
        <span className={state.pending ? 'is-draft' : undefined}>draft recipe: <b>{draftRecipe.serialized}</b></span>
      </p>
      <ul className="am-chips">
        {familyOrder.map(family => (
          <li key={family} className={`am-chip ${family === draft.family ? 'is-active' : 'is-inactive'}`}>
            {familyLabels[family]}: {family === draft.family
              ? draftRecipe.settings.map(setting => setting.shown).join(' · ')
              : `retained draft, inactive — ${Object.entries(draft.space[family]).map(([dimension, values]) => describeOption(dimension, values[Math.min(draft.choices[family][dimension] ?? 0, values.length - 1)])).join(' · ')}`}
          </li>
        ))}
      </ul>
      <p className="am-caption">
        Only the highlighted family&rsquo;s settings appear in the active recipe. The others are retained drafts: they
        describe nothing that would be fitted, and moving one of them changes neither the recipe nor the count. Adding a
        value to a branch is a different operation, and it does change the space.
      </p>
    </Stage>

    <div className="am-buttons">
      {spaceSetups.map(setup => (
        <button key={setup.key} type="button"
          onClick={() => state.suggest(setup.build(clone(draft)))}>{setup.label}</button>
      ))}
    </div>

    <Prediction
      prompt={`The applied space holds ${appliedCount.total} valid configurations. Applying the draft, what happens to that number?`}
      options={[['shrinks', 'It shrinks'], ['unchanged', 'Exactly unchanged'], ['grows', 'It grows']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the new total', name: 'the new total', tolerance: 0, digits: 0 }}
      blocked={draftCount ? null : 'The draft grammar is not valid yet; fix the field error above before applying.'}
      committed={shown => `${shown.answer.before} → ${shown.answer.after} configurations`}
      describe={comparison
        ? `The active recipe ${comparison.beforeRecipe.serialized === comparison.afterRecipe.serialized ? 'did not move' : `moved from ${comparison.beforeRecipe.serialized} to ${comparison.afterRecipe.serialized}`}.`
        : undefined} />

    {state.result && comparison && <>
      <Table caption={`Where the ${comparison.after.total} configurations come from. Alternatives add across branches; settings that are simultaneously active inside a branch multiply.`}
        headings={['branch', 'active dimensions', 'expression', 'before', 'now']}
        rows={comparison.after.branches.map((branch, index) => [
          branch.label,
          branch.dimensions.map(entry => `${entry.label} (${entry.size})${entry.fixed ? ' fixed' : ''}`).join(', '),
          branch.expression,
          String(comparison.before.branches[index].count),
          String(branch.count),
        ])}
        rowClass={index => (comparison.before.branches[index].count !== comparison.after.branches[index].count ? 'is-selected' : undefined)} />
      <p className="am-readout" role="status">
        {comparison.after.expression}.{' '}
        {comparison.after.total === comparison.before.total
          ? 'The registry has the same number of permitted configurations. Replacing values can change its recipes without changing that count.'
          : `The change is ${signed(comparison.after.total - comparison.before.total, 0)}, because the edited value pairs with every other simultaneously active setting inside its own branch.`}
        {' '}Multiplying every option in the whole form together instead would give
        {' '}{familyOrder.reduce((product, family) => product * Object.values(comparison.after.branches.find(branch => branch.family === family).dimensions).reduce((inner, entry) => inner * entry.size, 1), 1) === comparison.after.total ? 'the same number here only by coincidence' : `${familyOrder.map(family => comparison.after.branches.find(branch => branch.family === family)).reduce((product, branch) => product * branch.dimensions.reduce((inner, entry) => inner * entry.size, 1), 1)}, counting configurations that do not exist`}.
      </p>
      <details>
        <summary>Enumerate every valid configuration</summary>
        <Table scroll caption={`All ${comparison.after.total} configurations of the applied space, written out.`}
          headings={['#', 'family', 'settings']}
          rows={enumerateConfigurations(state.active.space).map((row, index) => [
            String(index + 1), familyLabels[row.family],
            Object.entries(row.settings).map(([dimension, value]) => `${dimensionLabels[dimension]} ${describeOption(dimension, value)}`).join(', '),
          ])} />
      </details>
      <details>
        <summary>Two sampling rules over the same space</summary>
        <p>
          Random search still needs a distribution, and the two obvious rules disagree. Choosing a family uniformly and
          then a configuration inside it is not the same as choosing uniformly among all configurations.
        </p>
        <svg viewBox="0 0 340 92" role="img"
          aria-label={['family-uniform', 'configuration-uniform'].map(rule => `${rule}: ${samplingMeasure(state.active.space, rule).bands.map(band => `${band.label} ${band.fraction}`).join(', ')}`).join('. ')}>
          {['family-uniform', 'configuration-uniform'].map((rule, row) => {
            const measure = samplingMeasure(state.active.space, rule);
            let cursor = 6;
            return <g key={rule}>
              <text className="is-small" x="6" y={16 + row * 44}>{rule === 'family-uniform' ? 'choose a family, then a configuration' : 'choose a configuration'}</text>
              {measure.bands.map(band => {
                const bandWidth = band.probability * 328;
                const x = cursor;
                cursor += bandWidth;
                return <g key={band.family}>
                  <rect className={`am-node ${band.family === state.active.family ? 'is-commitment' : 'is-selection'}`}
                    x={x} y={22 + row * 44} width={Math.max(bandWidth, 1)} height="20" />
                  {bandWidth > 40 && <text className="is-small" x={x + bandWidth / 2} y={36 + row * 44} textAnchor="middle">{band.fraction}</text>}
                </g>;
              })}
            </g>;
          })}
        </svg>
        <Table caption="Exact sampling probabilities under each rule. Neither is unbiased without naming the reference measure."
          headings={['family', 'configurations', 'family-uniform', 'configuration-uniform']}
          rows={samplingMeasure(state.active.space, 'family-uniform').bands.map((band, index) => [
            band.label, String(band.count), band.fraction,
            samplingMeasure(state.active.space, 'configuration-uniform').bands[index].fraction,
          ])} />
      </details>
      <Attempts entries={state.history} />
    </>}
  </Investigation>;
}

/* ============================================================ §2 · I2 · EI */

const acquisitionBaseline = {
  best: 0.4,
  A: { mean: 0.35, deviation: 0.02 },
  B: { mean: 0.40, deviation: 0.20 },
  C: { mean: 0.50, deviation: 0.00 },
};
const acquisitionSetups = [
  { key: 'default', label: 'Baseline: A(0.35, 0.02), B(0.40, 0.20), C(0.50, 0)', ...acquisitionBaseline },
  { key: 'certainB', label: "Contrast: collapse B's uncertainty to 0", ...acquisitionBaseline, B: { mean: 0.40, deviation: 0 } },
  {
    key: 'offset', label: 'Null: add 0.3 to the incumbent and every mean',
    best: 0.7, A: { mean: 0.65, deviation: 0.02 }, B: { mean: 0.70, deviation: 0.20 }, C: { mean: 0.80, deviation: 0 },
  },
  { key: 'worseB', label: "Contrast: move B's mean up to 0.55", ...acquisitionBaseline, B: { mean: 0.55, deviation: 0.20 } },
  { key: 'tie', label: 'Near tie: give A a deviation of 0.128', ...acquisitionBaseline, A: { mean: 0.35, deviation: 0.128 } },
  { key: 'promisingC', label: 'Make C deterministic and better: mean 0.3', ...acquisitionBaseline, C: { mean: 0.30, deviation: 0 } },
];
const ACQUISITION_IDS = ['A', 'B', 'C'];

export function AcquisitionLab() {
  const state = useInvestigation(acquisitionBaseline);
  const draft = state.draft;
  const asList = record => ACQUISITION_IDS.map(id => ({ id, ...record[id] }));
  const answerFor = proposed => {
    const table = acquisitionTable(proposed.best, asList(proposed));
    return { outcome: table.winner ?? 'tie', value: table.highest };
  };
  const applied = state.result ? acquisitionTable(state.active.best, asList(state.active)) : null;
  const domain = applied ? acquisitionDomain(state.active.best, asList(state.active)) : null;
  const densities = applied ? applied.rows.map(row => improvementDensity(state.active.best, row.mean, row.deviation, domain)) : null;
  const peak = densities ? Math.max(...densities.flatMap(entry => entry.points.map(point => point.density)), 1) : 1;
  const weightPeak = densities ? Math.max(...densities.flatMap(entry => entry.points.map(point => point.weighted)), 1e-9) : 1;
  return <Investigation
    title="Shade the improvement worth buying"
    question="Three constructed surrogate predictions about an unknown loss, and an incumbent best. Record which candidate has the greatest expected improvement before anything is drawn."
    evidence="Constructed Gaussian surrogate predictions in surrogate loss coordinates. These are not fitted classifier probabilities, and they are not observed candidate losses."
    note="Expected improvement is E[max(b − F, 0)]: the average size of the improvement, not the probability that there is one. Those are different quantities and the panels below label them separately."
    onReset={state.reset}>

    <div className="am-controls is-narrow">
      <NumberField label="Incumbent best loss b" value={draft.best} min={limits.incumbent.minimum} max={limits.incumbent.maximum}
        step={String(controlSteps.incumbent)} decimals={3} onChange={best => state.edit({ best })} />
      {ACQUISITION_IDS.map(id => <NumberField key={`${id}-mean`} label={`${id}: predicted mean μ`} value={draft[id].mean}
        min={limits.surrogateMean.minimum} max={limits.surrogateMean.maximum} step={String(controlSteps.surrogateMean)} decimals={3}
        onChange={mean => state.edit({ [id]: { ...draft[id], mean } })} />)}
      {ACQUISITION_IDS.map(id => <NumberField key={`${id}-sd`} label={`${id}: predicted deviation σ`} value={draft[id].deviation}
        min={limits.surrogateDeviation.minimum} max={limits.surrogateDeviation.maximum} step={String(controlSteps.surrogateDeviation)} decimals={3}
        onChange={deviation => state.edit({ [id]: { ...draft[id], deviation } })} />)}
    </div>
    <div className="am-buttons">
      {acquisitionSetups.map(setup => (
        <button key={setup.key} type="button" onClick={() => state.suggest({
          best: setup.best, A: setup.A, B: setup.B, C: setup.C,
        })}>{setup.label}</button>
      ))}
    </div>
    <Table caption="The raw inputs. Nothing derived from them is shown until a prediction is recorded."
      headings={['candidate', 'predicted mean μ', 'predicted deviation σ']}
      rows={ACQUISITION_IDS.map(id => [id, round(draft[id].mean, 3), draft[id].deviation === 0 ? 'exactly 0' : round(draft[id].deviation, 3)])} />

    <Prediction
      prompt={`With the incumbent best loss at ${round(draft.best, 3)}, which candidate has the greatest expected improvement?`}
      options={[...ACQUISITION_IDS.map(id => [id, `${id} (μ = ${round(draft[id].mean, 3)}, σ = ${round(draft[id].deviation, 3)})`]), ['tie', 'They tie']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: its expected improvement', name: 'the greatest EI', tolerance: limits.displayTolerance, digits: 8 }}
      committed={shown => `b = ${round(state.active.best, 3)}, ${ACQUISITION_IDS.map(id => `${id}(${round(state.active[id].mean, 3)}, ${round(state.active[id].deviation, 3)})`).join(', ')}`} />

    {applied && <>
      <div className="am-panels is-stacked">
        {applied.rows.map((row, index) => {
          const density = densities[index];
          const area = density.points.filter(point => point.loss <= state.active.best);
          return <div className="am-panel" key={row.id}>
            <h4>{row.id}: μ = {round(row.mean, 3)}, σ = {row.deviation === 0 ? 'exactly 0' : round(row.deviation, 3)} → EI {fixed(row.value, 8)}</h4>
            <Plot caption={null} width={340} height={116}
              padding={{ left: 40, right: 14, top: 12, bottom: 30 }}
              domain={domain} range={[0, 1]}
              ticks={[domain[0], state.active.best, domain[1]]}
              valueTicks={[0, 1]}
              formatTick={value => round(value, 2)} formatValue={value => (value === 0 ? '0' : 'max')}
              describe={`Candidate ${row.id}. ${row.deterministic ? `A point mass at ${row.mean}` : `A density centred at ${row.mean} with deviation ${row.deviation}`}. The incumbent line sits at ${state.active.best}. The shaded region shows improvement-weighted density, normalized consistently across candidates. Its screen area is proportional to the expected improvement ${row.value}, not numerically equal to it.`}>
              {(scaleX, scaleY) => <>
                {area.length > 1 && (
                  <polygon className="am-area"
                    points={`${scaleX(area[0].loss).toFixed(2)},${scaleY(0).toFixed(2)} ${area.map(point => `${scaleX(point.loss).toFixed(2)},${scaleY(point.weighted / weightPeak).toFixed(2)}`).join(' ')} ${scaleX(area.at(-1).loss).toFixed(2)},${scaleY(0).toFixed(2)}`} />
                )}
                {!row.deterministic && (
                  <polyline className="am-curve is-density"
                    points={polyline(density.points.map(point => [point.loss, point.density / peak]), scaleX, scaleY)} />
                )}
                {!row.deterministic && (
                  <polyline className="am-curve is-weighted"
                    points={polyline(density.points.map(point => [point.loss, point.weighted / weightPeak]), scaleX, scaleY)} />
                )}
                {row.deterministic && (() => {
                  // Beside the marker, on the side away from the incumbent: a
                  // deterministic candidate near or exactly on b is the
                  // informative state, and there a centred or wrong-side label
                  // is crossed by the incumbent's own dashed rule. The side
                  // flips again near either edge so the text cannot leave the
                  // drawing.
                  const markX = scaleX(row.mean);
                  const incumbentX = scaleX(state.active.best);
                  const room = 66;
                  let toRight = markX >= incumbentX;
                  if (toRight && markX + 8 + room > 332) toRight = false;
                  if (!toRight && markX - 8 - room < 8) toRight = true;
                  return <>
                    <line className="am-rule" x1={markX} x2={markX} y1={scaleY(0)} y2={scaleY(0.9)} />
                    <circle className="am-mark" cx={markX} cy={scaleY(0.9)} r="4.5" />
                    <text className="is-small am-halo" y={scaleY(0.9) + 4}
                      x={markX + (toRight ? 8 : -8)}
                      textAnchor={toRight ? 'start' : 'end'}>point mass</text>
                  </>;
                })()}
                <line className="am-rule is-incumbent" x1={scaleX(state.active.best)} x2={scaleX(state.active.best)}
                  y1={scaleY(0)} y2={scaleY(1)} />
                <text className="is-small am-halo" x={scaleX(state.active.best)} y={scaleY(1) - 2} textAnchor="middle">b</text>
              </>}
            </Plot>
            <p>
              {row.deterministic
                ? `A zero deviation is a point mass, not a narrow bell curve. Its improvement is simply max(b − μ, 0) = ${fixed(row.value, 8)}.`
                : `Solid line: the predictive density. Dashed line and shading: that density weighted by (b − f), which is the quantity EI integrates. Shading probability alone would draw the probability of improvement instead, which here is ${fixed(row.probabilityOfImprovement, 6)}.`}
            </p>
          </div>;
        })}
      </div>
      <Legend items={[
        { label: 'predictive density', className: 'am-curve is-density' },
        { label: 'density × improvement', className: 'am-curve is-weighted' },
        { label: 'incumbent best loss b', className: 'am-rule is-incumbent' },
      ]} />
      <p className="am-caption">
        All three panels share one loss axis. Solid densities share one peak normalization across candidates; weighted densities share a separate peak normalization. Compare solid heights with solid heights, or weighted heights with weighted heights, not one kind with the other. The shaded screen area is therefore proportional to EI, not numerically equal to it. The density is not itself a probability.
      </p>
      <Table caption="Expected improvement, and the separate probability of improvement. The acquisition comparison uses unrounded values."
        headings={['candidate', 'μ', 'σ', 'expected improvement', 'probability of improvement']}
        rows={applied.rows.map(row => [
          row.id, round(row.mean, 3), row.deviation === 0 ? 'exactly 0' : round(row.deviation, 3),
          fixed(row.value, 8), fixed(row.probabilityOfImprovement, 6),
        ])}
        rowClass={index => (applied.rows[index].id === applied.winner ? 'is-selected' : undefined)} />
      <p className="am-readout" role="status">
        {applied.winner
          ? <>The acquisition rule picks <b>{applied.winner}</b> at {fixed(applied.highest, 8)}. The lowest predicted mean
            belongs to <b>{applied.lowestMean}</b>{applied.lowestMean === applied.winner ? ', which agrees here.' : ' — a worse predicted mean can still buy the larger expected improvement, because its uncertain lower-loss possibilities are worth something under this model.'}</>
          : <>Candidates {applied.tied.join(' and ')} tie at {fixed(applied.highest, 8)} within {limits.tolerance}. A tie is
            reported as a tie rather than resolved by display rounding.</>}
        {' '}Whichever is chosen, only an actual evaluation determines its real loss.
        {densities.some(entry => entry.truncated) && ' One panel is clipped at the plotted bounds; the tabulated values remain the exact analytic integrals.'}
      </p>
      <Attempts entries={state.history} />
    </>}
  </Investigation>;
}

/* ================================================== §3 · I3 · successive halving */

const curveBaseline = { curves: declaredCurves.map(curve => ({ id: curve.id, losses: [...curve.losses] })), accounting: 'restart' };
const curveSetups = [
  { key: 'default', label: 'Baseline: the nine declared curves', build: () => clone(curveBaseline) },
  {
    key: 'rescue', label: "Contrast: read D's first rung as 0.09",
    build: () => {
      const next = clone(curveBaseline);
      next.curves[3].losses[0] = 0.09;
      return next;
    },
  },
  {
    key: 'offset', label: 'Null: add 0.05 to every value',
    build: () => {
      const next = clone(curveBaseline);
      next.curves.forEach(curve => { curve.losses = curve.losses.map(loss => Number((loss + 0.05).toFixed(4))); });
      return next;
    },
  },
  {
    key: 'nonmonotone', label: 'A curve that gets worse: E rises to 0.3 at nine units',
    build: () => {
      const next = clone(curveBaseline);
      next.curves[4].losses[2] = 0.3;
      return next;
    },
  },
  {
    key: 'tie', label: 'A tie at the first rung: give B the same 0.10 as A',
    build: () => {
      const next = clone(curveBaseline);
      next.curves[1].losses[0] = 0.10;
      return next;
    },
  },
];

export function HalvingLab() {
  const state = useInvestigation(curveBaseline);
  const [showCounterfactual, setShowCounterfactual] = useState(false);
  const draft = state.draft;
  const answerFor = proposed => {
    const schedule = halvingSchedule(proposed.curves);
    return { outcome: schedule.selectedId, value: schedule.work[proposed.accounting] };
  };
  const applied = state.result ? halvingSchedule(state.active.curves) : null;
  const brackets = hyperbandBrackets(9, 3);
  return <Investigation
    title="Follow the survivors, then inspect the evidence you never bought"
    question="Nine candidates, a resource ladder of 1, 3 and 9 units, and a rule that keeps the best third at each rung. Record which candidate the schedule selects, and how much work it costs, before any rung is decided."
    evidence="Constructed loss curves in constructed resource units. Not seconds, not GPU hours, and not a measured training run."
    note="The scheduler only ever ranks the column it has purchased. A value further along a row cannot influence the cut that removed its candidate — which is exactly how a slow starter is lost."
    onReset={() => { state.reset(); setShowCounterfactual(false); }}>

    <div className="am-buttons">
      {curveSetups.map(setup => (
        <button key={setup.key} type="button"
          onClick={() => state.suggest({ ...setup.build(), accounting: draft.accounting })}>{setup.label}</button>
      ))}
    </div>
    <figure className="am-table">
      <figcaption className="am-table-caption">
        The declared curves. Every value is editable in [0, 1]; monotone improvement is not imposed, because a real
        training curve need not improve at every measurement.
      </figcaption>
      <div className="am-table-scroll" role="region" aria-label="Editable loss curves" tabIndex={0}>
        <table role="table">
          <thead role="rowgroup"><tr role="row">
            <th role="columnheader" scope="col">candidate</th>
            {declaredResource.map(unit => <th key={unit} role="columnheader" scope="col">{unit} unit{unit === 1 ? '' : 's'}</th>)}
          </tr></thead>
          <tbody role="rowgroup">
            {draft.curves.map((curve, row) => <tr role="row" key={curve.id}>
              <th role="rowheader" scope="row" data-label="candidate">{curve.id}</th>
              {curve.losses.map((loss, column) => <td role="cell" key={column} data-label={`${declaredResource[column]} units`}>
              <NumberField labelHidden
                label={`${curve.id} at ${declaredResource[column]} unit${declaredResource[column] === 1 ? '' : 's'}`}
                value={loss} min={limits.loss.minimum} max={limits.loss.maximum} step={String(controlSteps.loss)} decimals={4}
                onChange={value => state.edit(current => ({
                  curves: current.curves.map((entry, index) => (index === row
                    ? { ...entry, losses: entry.losses.map((old, position) => (position === column ? value : old)) }
                    : entry)),
                }))} />
              </td>)}
            </tr>)}
          </tbody>
        </table>
      </div>
    </figure>
    <div className="am-controls">
      <label className="am-field">
        <span>Work accounting</span>
        <select value={draft.accounting} onChange={event => state.edit({ accounting: event.target.value })}>
          <option value="restart">each rung restarts training</option>
          <option value="resume">training genuinely resumes from retained state</option>
        </select>
      </label>
    </div>
    <p className="am-caption">
      Ties at a rung take the earlier candidate letter, and that rule is fixed before the task. Switching the accounting
      changes how the work is counted, not which candidate survives.
    </p>

    <Prediction
      prompt="Which candidate does the schedule finally select, and how much work does the accounting you chose report?"
      options={draft.curves.map(curve => [curve.id, `${curve.id} survives to the end`])}
      state={state} answerFor={answerFor}
      numeric={{ label: `Optional: total work in constructed resource units (${draft.accounting === 'restart' ? 'restart' : 'continuation'})`, name: 'the total work', tolerance: 0, digits: 0 }}
      committed={shown => `${state.active.curves.map(curve => `${curve.id}[${curve.losses.join(', ')}]`).join('  ')}`} />

    {applied && <>
      <div className="am-rungs">
        {applied.rungs.map(rung => (
          <div className="am-rung" key={rung.rung}>
            <h5>Rung {rung.rung + 1}: {rung.started.length} candidate{rung.started.length === 1 ? '' : 's'} at {rung.resource} unit{rung.resource === 1 ? '' : 's'} each</h5>
            <ul className="am-cards">
              {rung.survivors.map(id => <li className="am-card is-survivor" key={id}>
                {id} · {round(state.active.curves.find(curve => curve.id === id).losses[rung.rung], 4)} · advances
              </li>)}
              {rung.eliminated.map(entry => <li className="am-card is-eliminated" key={entry.id}>
                {entry.id} · {round(entry.loss, 4)} · out
              </li>)}
            </ul>
            {rung.eliminated.length > 0 && <p className="am-caption">
              {rung.eliminated.map(entry => `${entry.id}: ${entry.reason}`).join('; ')}.
              Only this column was purchased.
            </p>}
          </div>
        ))}
      </div>
      <Plot caption="Purchased evidence, and the evidence that was never bought"
        width={340} height={240} domain={[0, 10]}
        range={[0, Math.max(...state.active.curves.flatMap(curve => curve.losses), 0.1) * 1.06]}
        /* Room on the right for the two end labels to carry their own losses:
           the whole point of the figure is that 0.07 and 0.02 differ, and at
           this scale two bare letters do not make that legible. */
        padding={{ left: 48, right: 56, top: 16, bottom: 36 }}
        ticks={declaredResource} formatTick={value => `${value}u`}
        describe={`Loss against resource. ${applied.traces.map(trace => `${trace.id} purchased ${trace.purchased.map(point => point.join(' at ')).join(', ')}${trace.unpurchased.length > 1 ? `, unpurchased ${trace.unpurchased.slice(1).map(point => point.join(' at ')).join(', ')}` : ''}`).join('. ')}`}>
        {(scaleX, scaleY) => {
          // Six candidates are eliminated at the same rung within a hundredth of
          // each other, so nine end labels cannot be placed without landing on a
          // neighbour or on the dashed continuation that starts at that very
          // point. The rung cards below name all nine with their losses, so the
          // drawing labels only the two the contrast is about — the candidate
          // the schedule selected and the one it never bought — at the far right
          // end of their curves, where nothing continues.
          const marked = [
            { id: applied.selectedId, point: applied.traces.find(trace => trace.id === applied.selectedId).purchased.at(-1) },
            ...(applied.missedBetter
              ? [{ id: applied.counterfactualId, point: applied.traces.find(trace => trace.id === applied.counterfactualId).unpurchased.at(-1) }]
              : []),
          ];
          return <>
            {applied.traces.map(trace => <g key={trace.id}>
              {showCounterfactual && trace.unpurchased.length > 1 && (
                <polyline className="am-curve is-unpurchased" points={polyline(trace.unpurchased, scaleX, scaleY)} />
              )}
              <polyline className={`am-curve ${trace.survived ? 'is-selected' : 'is-purchased'}`}
                points={polyline(trace.purchased, scaleX, scaleY)} />
              {trace.purchased.length === 1 && (
                <circle className="am-mark is-fold" cx={scaleX(trace.purchased[0][0])} cy={scaleY(trace.purchased[0][1])} r="2.6" />
              )}
            </g>)}
            {marked.filter(entry => showCounterfactual || entry.id === applied.selectedId).map(entry => (
              <text key={entry.id} className="is-small am-halo" x={scaleX(entry.point[0]) + 7}
                y={scaleY(entry.point[1]) + 4}>{entry.id} {round(entry.point[1], 4)}</text>
            ))}
          </>;
        }}
      </Plot>
      <div className="am-buttons">
        <button type="button" className={showCounterfactual ? 'is-primary' : 'is-quiet'}
          aria-pressed={showCounterfactual}
          onClick={() => setShowCounterfactual(!showCounterfactual)}>
          {showCounterfactual ? 'Hide the curves that were never bought' : 'Show the curves that were never bought'}
        </button>
      </div>
      <Legend items={[
        { label: 'purchased measurements', className: 'am-curve is-purchased' },
        { label: 'a candidate cut after one measurement', shape: 'dot', className: 'am-mark is-fold' },
        { label: 'the selected candidate', className: 'am-curve is-selected' },
        ...(showCounterfactual
          ? [{ label: 'never bought — counterfactual', className: 'am-curve is-unpurchased' }]
          : []),
      ]} />
      <p className="am-caption">
        {showCounterfactual
          ? `The dashed traces are values the schedule never purchased. They are shown here only so the cost of an early cut is visible; no rung decision could ever consult them.`
          : `The plot shows only the evidence the schedule actually bought, which is all it ever had to decide with. Reveal the rest when you want to see what the early cuts cost.`}
        {' '}Only the selected candidate{showCounterfactual ? ' and the one it never bought are' : ' is'} lettered;
        the rung panels above name all {applied.traces.length} with the loss each was judged on.
      </p>
      <Table caption="Work under each accounting, in constructed resource units."
        headings={['accounting', 'expression', 'units']}
        rows={[
          ['each rung restarts training', applied.work.restartExpression, String(applied.work.restart)],
          ['training genuinely resumes', applied.work.resumeExpression, String(applied.work.resume)],
          ['fit everything at full resource', applied.work.allFullExpression, String(applied.work.allFull)],
        ]}
        rowClass={index => ((index === 0 && state.active.accounting === 'restart') || (index === 1 && state.active.accounting === 'resume') ? 'is-selected' : undefined)} />
      <p className="am-readout" role="status">
        The schedule selects <b>{applied.selectedId}</b>, finishing at {round(applied.selectedFinalLoss, 4)}.
        {applied.missedBetter
          ? <> At full resource <b>{applied.counterfactualId}</b> would have reached {round(applied.counterfactualFinalLoss, 4)} —
            better — but it was eliminated before that evidence was ever purchased. Its dashed trace above is a
            counterfactual, not a measurement the scheduler had.</>
          : <> The selected candidate attains the full-resource minimum here, possibly tied with another candidate, so no strictly better final loss was missed.</>}
        {' '}Continuation is a property of the actual training procedure and its retained state; an estimator option with
        a similar name does not establish it.
      </p>
      <details>
        <summary>Hyperband&rsquo;s second loop: several brackets, not one</summary>
        <p>
          Successive halving fixes one breadth/depth choice. Hyperband runs several brackets that start different numbers
          of candidates at different initial resources, hedging that choice. With R = {brackets.maximumResource} and
          η = {brackets.eta}, the allocation uses floor and ceiling rounding, so the budgets are approximate.
        </p>
        <Table caption={`The three brackets. Their total of ${brackets.totalRestartWork} restart units is not comparable with fitting one fixed set of nine candidates at full resource, because each bracket starts a different set.`}
          headings={['bracket', 'starts', 'ladder (candidates at resource)', 'restart work']}
          rows={brackets.brackets.map(bracket => [
            `s = ${bracket.bracket}`,
            `${bracket.startCount} at ${bracket.startResource}`,
            bracket.stages.map(stage => `${stage.candidates}@${stage.resource}`).join(' → '),
            String(bracket.restartWork),
          ])} />
      </details>
      <Attempts entries={state.history} />
    </>}
  </Investigation>;
}

/* ============================================= §5 · I5 · replay a finite search */

/** Two candidates already revealed, three proposed: the comparison section 5
 * walks through, so the investigation's first question is the real one. */
const replayBaseline = { enabled: recordedCandidates.map((_, index) => index), order: [...replayOrder], budget: 2 };
const replayProposed = { ...replayBaseline, budget: 3 };
const replaySetups = [
  { key: 'default', label: 'Baseline: budget 2 → 3', build: () => ({ ...clone(replayBaseline), budget: 3 }) },
  { key: 'four', label: 'Budget 4: the width-16 network arrives', build: () => ({ ...clone(replayBaseline), budget: 4 }) },
  {
    /** A null by construction: it reverses only the candidates the applied
     * budget has not revealed, and leaves the budget itself alone. Changing the
     * budget as well would change the evidence, which is the opposite of the
     * point being made. */
    key: 'reorderTail', label: 'Null: reorder only the candidates still unrevealed',
    build: current => {
      const next = clone(current);
      const sequence = next.order.filter(index => next.enabled.includes(index));
      const revealed = sequence.slice(0, Math.min(Math.max(next.budget, 1), sequence.length));
      const cut = next.order.indexOf(revealed.at(-1)) + 1;
      next.order = [...next.order.slice(0, cut), ...next.order.slice(cut).reverse()];
      return next;
    },
  },
  {
    key: 'withoutNeighbors', label: 'Disable the k = 3 neighbors candidate',
    build: () => ({ ...clone(replayBaseline), budget: 3, enabled: replayBaseline.enabled.filter(index => index !== 6) }),
  },
  {
    key: 'networksOnly', label: 'Enable only the three networks',
    build: () => ({ ...clone(replayBaseline), budget: 2, enabled: [8, 9, 10] }),
  },
];

export function SearchReplayLab() {
  const state = useInvestigation(replayBaseline, JSON.stringify, replayProposed);
  const draft = state.draft;
  const sequence = draft.order.filter(index => draft.enabled.includes(index));
  const safeBudget = Math.min(Math.max(draft.budget, 1), Math.max(sequence.length, 1));
  const answerFor = (proposed, current) => {
    const currentSequence = current.order.filter(index => current.enabled.includes(index));
    const before = replayPrefix({
      order: current.order, enabled: current.enabled,
      budget: Math.min(Math.max(current.budget, 1), currentSequence.length),
    });
    const after = replayPrefix({
      order: proposed.order, enabled: proposed.enabled,
      budget: Math.min(Math.max(proposed.budget, 1), sequence.length),
    });
    const comparison = replayComparison(before, after);
    return { outcome: comparison.outcome, value: after.fits, comparison };
  };
  const applied = state.result ? replayPrefix({
    order: state.active.order, enabled: state.active.enabled,
    budget: Math.min(Math.max(state.active.budget, 1), state.active.order.filter(index => state.active.enabled.includes(index)).length),
  }) : null;
  const move = (position, direction) => state.edit(current => {
    const order = [...current.order];
    const target = position + direction;
    if (target < 0 || target >= order.length) return {};
    [order[position], order[target]] = [order[target], order[position]];
    return { order };
  });
  return <Investigation
    title="Replay a finite search without borrowing evidence you never bought"
    question="The eleven candidates of section 5 are revealed in a fixed random order. Choose a budget, then record two things at once: whether the best observed score will improve, and whether the recommended configuration will change."
    evidence="Observed data. Every score below was fitted natively and is immutable here. This is a replay of a finished table, not a measured comparison of random search against Bayesian optimization."
    note="Selection takes the greatest mean fold accuracy among the revealed candidates; an exact tie takes the lowest original registry index, whatever order the candidates arrived in. That rule is fixed before the task."
    onReset={state.reset}>

    <div className="am-controls is-narrow">
      <NumberField label="Budget: candidates revealed" value={safeBudget} min={1} max={Math.max(sequence.length, 1)}
        step={String(controlSteps.budget)} decimals={0} onChange={budget => state.edit({ budget })} />
    </div>
    <div className="am-buttons">
      {replaySetups.map(setup => (
        <button key={setup.key} type="button" onClick={() => state.suggest(setup.build(state.active))}>{setup.label}</button>
      ))}
    </div>
    <Table caption={`The reveal order. Candidates beyond the budget are not revealed, so no score of theirs is available. ${sequence.length} candidate${sequence.length === 1 ? ' is' : 's are'} enabled.`}
      headings={['position', 'configuration', 'enabled', 'reorder']}
      rows={draft.order.map((index, position) => [
        String(position + 1),
        recordedCandidates[index].label,
        <label className="am-choice" key="enable">
          <input type="checkbox" checked={draft.enabled.includes(index)}
            aria-label={`Include ${recordedCandidates[index].label} in the replay`}
            onChange={() => state.edit(current => ({
              enabled: current.enabled.includes(index)
                ? (current.enabled.length > 1 ? current.enabled.filter(entry => entry !== index) : current.enabled)
                : [...current.enabled, index],
            }))} />
          <span>{draft.enabled.includes(index) ? 'in' : 'out'}</span>
        </label>,
        <span key="move">
          <button type="button" className="is-quiet" disabled={position === 0}
            aria-label={`Move ${recordedCandidates[index].label} earlier`} onClick={() => move(position, -1)}>↑</button>
          {' '}
          <button type="button" className="is-quiet" disabled={position === draft.order.length - 1}
            aria-label={`Move ${recordedCandidates[index].label} later`} onClick={() => move(position, 1)}>↓</button>
        </span>,
      ])}
      rowClass={index => {
        const position = sequence.indexOf(draft.order[index]);
        if (!draft.enabled.includes(draft.order[index])) return 'is-out';
        return position >= 0 && position < safeBudget ? 'is-selected' : 'is-muted';
      }} />

    <Prediction
      prompt={`Moving from the applied budget to ${safeBudget}: what changes?`}
      options={[
        ['neither', 'Neither: same best score, same recommendation'],
        ['score', 'The best score changes, the recommendation stays'],
        ['recommendation', 'The recommendation changes, the best score stays flat'],
        ['both', 'Both change'],
      ]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: estimator fits paid for at the new budget', name: 'the fit count', tolerance: 0, digits: 0 }}
      committed={shown => `budget ${shown.answer.comparison.before.revealed.length} → ${shown.answer.comparison.after.revealed.length}, ${shown.answer.comparison.after.order.length} candidates enabled`}
      describe={state.result ? `Best score ${fixed(state.result.answer.comparison.before.best, 6)} → ${fixed(state.result.answer.comparison.after.best, 6)}; recommendation ${state.result.answer.comparison.before.recommendedId} → ${state.result.answer.comparison.after.recommendedId}.` : undefined} />

    {applied && <>
      <Plot caption="Best mean fold accuracy so far, against candidates fully evaluated"
        width={340} height={170} domain={[0, Math.max(applied.steps.length, 2)]}
        range={[Math.min(...applied.steps.map(step => step.best)) - 0.004, 1.002]}
        ticks={applied.steps.map(step => step.position)} formatTick={value => String(value)}
        formatValue={value => round(value, 4)}
        describe={`A step chart. ${applied.steps.map(step => `After ${step.position} candidates the best mean fold accuracy is ${step.best.toFixed(6)} and the recommendation is ${step.recommendedId}`).join('. ')}.`}>
        {(scaleX, scaleY) => <>
          <polyline className="am-curve is-step"
            points={applied.steps.flatMap((step, index) => {
              const previous = index === 0 ? step.best : applied.steps[index - 1].best;
              return [
                `${scaleX(step.position - 1).toFixed(2)},${scaleY(previous).toFixed(2)}`,
                `${scaleX(step.position).toFixed(2)},${scaleY(previous).toFixed(2)}`,
                `${scaleX(step.position).toFixed(2)},${scaleY(step.best).toFixed(2)}`,
              ];
            }).join(' ')} />
          {applied.steps.map(step => <g key={step.position}>
            <circle className={`am-mark ${step.recommendationChanged ? '' : 'is-mean'}`}
              cx={scaleX(step.position)} cy={scaleY(step.best)} r="4" />
            {step.tieDecided && (
              <text className="is-small am-halo" y={scaleY(step.best) - 9}
                x={step.position === applied.steps.length ? scaleX(step.position) - 7 : scaleX(step.position)}
                textAnchor={step.position === applied.steps.length ? 'end' : 'middle'}>tie</text>
            )}
          </g>)}
        </>}
      </Plot>
      <Legend items={[
        { label: 'the recommendation changed here', shape: 'dot', className: 'am-mark' },
        { label: 'the recommendation stayed', shape: 'dot', className: 'am-mark is-mean' },
      ]} />
      <Table caption={`The trial ledger. Each revealed candidate costs three estimator fits under this replay protocol, so ${applied.revealed.length} revealed candidates account for ${applied.fits} fits. Building the complete evidence packet actually cost 35, including the two final refits.`}
        headings={['#', 'candidate revealed', 'its mean fold accuracy', 'best so far', 'recommended', 'why']}
        rows={applied.steps.map(step => [
          String(step.position), step.label, fixed(step.score, 6), fixed(step.best, 6), step.recommendedId,
          step.position === 1 ? 'first candidate available'
            : step.scoreImproved && step.recommendationChanged ? 'a better score arrived'
              : step.tieDecided ? 'the score is unchanged, but this candidate ties it and wins the registry tie rule'
                : step.recommendationChanged ? 'the recommendation moved without a better score'
                  : 'no improvement on the incumbent',
        ])}
        rowClass={index => (applied.steps[index].recommendationChanged ? 'is-selected' : undefined)} />
      <p className="am-readout" role="status">
        At budget {applied.revealed.length} the recommendation is <b>{applied.recommendedId}</b> with
        {' '}{fixed(applied.best, 6)}. {applied.steps.some(step => step.tieDecided)
          ? 'Notice a step where the best score is flat and the recommendation still moves: a tie was broken by the registry rule, so a flat maximum does not mean the selected model is unchanged.'
          : 'No tie was broken at this budget.'}
      </p>
      <details>
        <summary>Out-of-fold mistakes of the revealed candidates</summary>
        <Table caption="Which revealed candidates miss which development rows. Aligning error sets is a comparison of stored class predictions; no candidate probability was retained, so nothing here supports a measured ensemble claim."
          headings={['candidate', 'out-of-fold mistakes', 'file lines']}
          rows={applied.revealed.map(index => [
            recordedCandidates[index].label,
            String(recordedCandidates[index].outOfFoldErrorRows.length),
            recordedCandidates[index].outOfFoldErrorRows.length === 0
              ? 'none'
              : recordedCandidates[index].outOfFoldErrorRows.slice(0, 12).map(row => sourceRows[String(row)].line).join(', ')
              + (recordedCandidates[index].outOfFoldErrorRows.length > 12 ? ', …' : ''),
          ])} />
      </details>
      <p className="am-caption">
        The replay owns no inspection outcome. The width-16 network&rsquo;s 205-of-205 result belongs to the procedure
        that was actually selected by the whole search and then refitted; it cannot be attached to a prefix that
        recommends a different configuration, because those predictions were never made.
      </p>
      <Attempts entries={state.history} />
    </>}
  </Investigation>;
}

/* ================================================== §6 · I6 · frontier and cap */

const paretoBaseline = { points: declaredPareto.map(point => ({ ...point })), cap: 5 };
const paretoSetups = [
  { key: 'default', label: 'Baseline: five candidates, 5 ms cap', build: () => clone(paretoBaseline) },
  { key: 'tight', label: 'Contrast: tighten the cap to 3 ms', build: () => ({ ...clone(paretoBaseline), cap: 3 }) },
  { key: 'impossible', label: 'Contrast: a 1 ms cap', build: () => ({ ...clone(paretoBaseline), cap: 1 }) },
  {
    key: 'nullEdit', label: "Null: lower dominated E's accuracy to 0.88",
    build: () => {
      const next = clone(paretoBaseline);
      next.points[4].accuracy = 0.88;
      return next;
    },
  },
  {
    key: 'raiseD', label: "Contrast: raise D's accuracy to 0.97",
    build: () => {
      const next = clone(paretoBaseline);
      next.points[3].accuracy = 0.97;
      return next;
    },
  },
  {
    key: 'identical', label: 'Two identical candidates: make E match A exactly',
    build: () => {
      const next = clone(paretoBaseline);
      next.points[4] = { id: 'E', latency: 2, accuracy: 0.90 };
      return next;
    },
  },
];

export function DeploymentLab() {
  const state = useInvestigation(paretoBaseline);
  const draft = state.draft;
  const answerFor = proposed => {
    const analysis = paretoAnalysis(proposed.points, proposed.cap);
    return { outcome: analysis.selectedId ?? 'none', value: analysis.frontierIds.length };
  };
  const applied = state.result ? paretoAnalysis(state.active.points, state.active.cap) : null;
  const latencies = draft.points.map(point => point.latency);
  return <Investigation
    title="Choose under a hard constraint, not a soft preference"
    question="Five candidates described by inference latency and accuracy, and a latency cap the deployment actually has. Record which candidate you would ship before any frontier is drawn."
    evidence="Hypothetical deployment measurements. These latencies are invented for this task. They carry no product name, no device, and no benchmark: a real latency figure needs hardware, software, input shape, batch size, precision, warm-up and a stated timing boundary."
    note={`A candidate is dominated when another is no worse in both objectives and strictly better in at least one. Two identical pairs therefore do not dominate each other. The shipped candidate is chosen by a rule fixed before the task: ${paretoAnalysis(declaredPareto, 5).selectionRule}.`}
    onReset={state.reset}>

    <div className="am-controls is-narrow">
      <NumberField label="Latency cap (ms)" value={draft.cap} min={limits.cap.minimum} max={limits.cap.maximum}
        step={String(controlSteps.cap)} decimals={2} onChange={cap => state.edit({ cap })} />
      {draft.points.map((point, index) => <NumberField key={`${point.id}-latency`} label={`${point.id}: latency (ms)`}
        value={point.latency} min={limits.latency.minimum} max={limits.latency.maximum} step={String(controlSteps.latency)} decimals={2}
        onChange={latency => state.edit(current => ({
          points: current.points.map((entry, position) => (position === index ? { ...entry, latency } : entry)),
        }))} />)}
      {draft.points.map((point, index) => <NumberField key={`${point.id}-accuracy`} label={`${point.id}: accuracy`}
        value={point.accuracy} min={limits.accuracy.minimum} max={limits.accuracy.maximum} step={String(controlSteps.accuracy)} decimals={4}
        onChange={accuracy => state.edit(current => ({
          points: current.points.map((entry, position) => (position === index ? { ...entry, accuracy } : entry)),
        }))} />)}
    </div>
    <div className="am-buttons">
      {paretoSetups.map(setup => (
        <button key={setup.key} type="button" onClick={() => state.suggest(setup.build())}>{setup.label}</button>
      ))}
      <button type="button" className="is-quiet" disabled={draft.points.length >= limits.points.maximum}
        onClick={() => state.edit(current => ({
          points: [...current.points, {
            id: String.fromCharCode(65 + current.points.length), latency: 6, accuracy: 0.92,
          }],
        }))}>Add a candidate</button>
      <button type="button" className="is-quiet" disabled={draft.points.length <= limits.points.minimum}
        onClick={() => state.edit(current => ({ points: current.points.slice(0, -1) }))}>Remove the last</button>
    </div>
    <Table caption="The raw pairs. No feasibility flag, dominance mark or winner appears until a prediction is recorded."
      headings={['candidate', 'latency (ms)', 'accuracy']}
      rows={draft.points.map(point => [point.id, round(point.latency, 2), round(point.accuracy, 4)])} />

    <Prediction
      prompt={`Under a ${round(draft.cap, 2)} ms cap, which candidate would you ship?`}
      options={[...draft.points.map(point => [point.id, `${point.id} (${round(point.latency, 2)} ms, ${round(point.accuracy, 4)})`]), ['none', 'None is feasible']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: how many candidates are on the frontier', name: 'the frontier size', tolerance: 0, digits: 0 }}
      committed={shown => `cap ${round(state.active.cap, 2)} ms, ${state.active.points.map(point => `${point.id}(${point.latency}, ${point.accuracy})`).join(' ')}`}
      describe={state.result && applied && applied.selectionTied
        ? `${applied.tiedOnAccuracyIds.join(' and ')} tie exactly on accuracy, so accuracy does not settle this one: the rule declared above takes ${applied.tieBrokenBy}.`
        : undefined} />

    {applied && <>
      <Plot caption="Latency against accuracy, with the cap drawn as a boundary rather than a preference"
        width={340} height={210}
        domain={[0, Math.max(...state.active.points.map(point => point.latency), state.active.cap) * 1.15]}
        range={[Math.min(...state.active.points.map(point => point.accuracy)) - 0.02, Math.max(...state.active.points.map(point => point.accuracy)) + 0.02]}
        ticks={(() => {
          const top = Math.max(...state.active.points.map(point => point.latency), state.active.cap) * 1.15;
          const step = [1, 2, 5, 10].find(candidate => top / candidate <= 5) ?? 10;
          const marks = [];
          for (let value = 0; value <= top; value += step) marks.push(value);
          return marks;
        })()}
        formatTick={value => `${round(value, 1)}`}
        formatValue={value => round(value, 3)}
        describe={`A scatter of ${applied.rows.length} candidates. ${applied.rows.map(row => `${row.id} at ${row.latency} milliseconds and accuracy ${row.accuracy}, ${row.dominated ? `dominated by ${row.dominatedBy}` : 'on the frontier'}, ${row.feasible ? 'feasible' : 'over the cap'}`).join('. ')}. The cap is at ${applied.cap} milliseconds. ${applied.reason}.`}>
        {(scaleX, scaleY) => <>
          {/* The cap is inclusive: a candidate at exactly the cap is feasible,
              and the table and the model both say so. The infeasible band
              therefore starts strictly to the RIGHT of the cap line, and the
              line is drawn over it, so a marker sitting on the boundary is not
              visually swallowed by the region that means "rejected". A learner
              reasoning from the drawing must reach the same verdict as the rule. */}
          <rect className="am-node is-untouched" x={scaleX(applied.cap) + 1.5} y="16"
            width={Math.max(scaleX(Math.max(...state.active.points.map(point => point.latency), applied.cap) * 1.15) - scaleX(applied.cap) - 1.5, 0)}
            height={210 - 36 - 16} fillOpacity="0.35" strokeDasharray="4 3" />
          <line className="am-rule" x1={scaleX(applied.cap)} x2={scaleX(applied.cap)} y1="16" y2={210 - 36} />
          {/* One short label, on whichever side has room. What the boundary
              means in words — that the cap itself is feasible — is in the
              legend and the readout, where it reflows. */}
          <text className="is-small am-halo" y="28"
            x={scaleX(applied.cap) + (scaleX(applied.cap) > 250 ? -5 : 5)}
            textAnchor={scaleX(applied.cap) > 250 ? 'end' : 'start'}>cap {round(applied.cap, 2)} ms</text>
          {(() => {
            // Two candidates may hold exactly the same pair — the lab offers a
            // setup that makes them so, and that is when the tie rule matters
            // most. Their marks must stay coincident, because that is the fact;
            // their labels get their own rows instead of printing on top of
            // each other.
            const placed = [];
            applied.rows.forEach(row => {
              const x = scaleX(row.latency);
              const y = scaleY(row.accuracy);
              const key = `${x.toFixed(1)},${y.toFixed(1)}`;
              const already = placed.filter(entry => entry.key === key).length;
              placed.push({ row, x, y, key, labelY: y + 4 + already * 13 });
            });
            return placed.map(entry => <g key={entry.row.id}>
              {entry.row.dominated
                ? <rect className={`am-mark ${entry.row.feasible ? 'is-dominated' : 'is-infeasible'}`}
                  x={entry.x - 4.5} y={entry.y - 4.5} width="9" height="9" />
                : <circle className={`am-mark ${entry.row.feasible ? 'is-frontier' : 'is-infeasible'}`}
                  cx={entry.x} cy={entry.y} r="5" />}
              <text className="is-small am-halo" x={entry.x + 7} y={entry.labelY}>
                {entry.row.id}{entry.row.id === applied.selectedId ? ' ✓' : ''}
              </text>
            </g>);
          })()}
        </>}
      </Plot>
      <Legend items={[
        { label: 'on the frontier, feasible', shape: 'dot', className: 'am-mark is-frontier' },
        { label: 'dominated', shape: 'square', className: 'am-mark is-dominated' },
        { label: 'over the cap', shape: 'dot', className: 'am-mark is-infeasible' },
        { label: 'the infeasible region — it begins past the cap, not at it', shape: 'square', className: 'am-node is-untouched' },
      ]} />
      <p className="am-caption">
        The cap is inclusive: a candidate whose latency is <em>exactly</em> the cap is feasible, which is why the shaded
        region begins to the right of the line rather than at it. A marker sitting on the line is a candidate you may
        ship.
      </p>
      <Table caption="Dominance, feasibility and the choice. These are three separate decisions."
        headings={['candidate', 'latency (ms)', 'accuracy', 'dominated by', 'within the cap', 'shipped']}
        rows={applied.rows.map(row => [
          row.id, round(row.latency, 2), round(row.accuracy, 4),
          row.dominatedBy ?? 'nothing — on the frontier',
          row.feasible ? 'yes' : 'no',
          row.id === applied.selectedId ? 'yes' : 'no',
        ])}
        rowClass={index => (applied.rows[index].id === applied.selectedId ? 'is-selected'
          : applied.rows[index].feasible ? undefined : 'is-out')} />
      <p className="am-readout" role="status">
        The frontier is {applied.frontierIds.join(', ') || 'empty'}.{' '}
        {applied.infeasible
          ? <>No candidate meets the {round(applied.cap, 2)} ms cap. That is the answer: the nearest miss is not a
            permitted choice, and a finite latency penalty added to accuracy would have let one through anyway.</>
          : <>Under the cap the choice is <b>{applied.selectedId}</b> — {applied.reason}.
            {applied.selectionTied
              ? ' Accuracy alone does not settle it here, which is exactly when a declared tie rule earns its keep.'
              : ''} A soft score that subtracts a multiple of latency from accuracy can still prefer an over-budget
            model; filtering encodes the hard bound directly, provided the latency measurement matches the
            requirement.</>}
      </p>
      <Attempts entries={state.history} />
    </>}
  </Investigation>;
}

/* =========================================== §7 · I4 · mix, then commit */

const mixtureBaseline = {
  active: ['zero', 'identity', 'negation'],
  logits: [Math.log(2), 0, 0],
  x: 2, target: 1, step: 0.4,
};
const mixtureSetups = [
  { key: 'default', label: 'Baseline: x = 2, target 1, logits (ln 2, 0, 0)', build: () => clone(mixtureBaseline) },
  { key: 'flip', label: 'Contrast: flip the input to x = −2', build: () => ({ ...clone(mixtureBaseline), x: -2 }) },
  {
    key: 'offset', label: 'Null: add 5 to every active logit',
    build: () => ({ ...clone(mixtureBaseline), logits: mixtureBaseline.logits.map(value => value + 5) }),
  },
  {
    key: 'commit', label: 'The commitment case: identity and negation only, equal logits, target 0',
    build: () => ({ active: ['identity', 'negation'], logits: [Math.log(2), 0, 0], x: 2, target: 0, step: 0.4 }),
  },
  {
    key: 'degenerate', label: 'Every operation outputs zero: x = 0',
    build: () => ({ ...clone(mixtureBaseline), x: 0 }),
  },
];

/** The commitment stage. It is keyed on the applied inputs by its parent, so an
 * edit remounts it and its own prediction is retired with everything else. */
function CommitStage({ applied }) {
  const [choice, setChoice] = useState('');
  const [shown, setShown] = useState(null);
  const committed = shown ? commitOperation(applied, shown) : null;
  return <Stage title="Commit to one operation">
    <p>
      A searched mixture is not a deployable architecture: at some point one operation is chosen and the rest are thrown
      away. Predict what that does to the loss here, before committing.
    </p>
    <div className="am-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>Replacing the mixture with a single operation will make the loss below. Changes within 10⁻¹² count as tied in this numerical check.</p>
        <div className="am-choices">
          {[['lower', 'lower'], ['same', 'the same within 10⁻¹²'], ['higher', 'higher']].map(([value, text]) => (
            <label className="am-choice" key={value}>
              <input type="radio" name="commit-direction" value={value} checked={choice === value}
                disabled={Boolean(shown)} onChange={() => setChoice(value)} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <div className="am-buttons">
        {applied.operations.map(operation => (
          <button key={operation.id} type="button" className="is-primary" disabled={choice === '' || Boolean(shown)}
            onClick={() => setShown(operation.id)}>Commit to {operation.label}</button>
        ))}
        {!shown && <span>Choose a direction first, then commit to one of the operations.</span>}
      </div>
      {committed && <p className={`am-verdict ${((committed.difference > 1e-12 && choice === 'higher') || (committed.difference < -1e-12 && choice === 'lower') || (Math.abs(committed.difference) <= 1e-12 && choice === 'same')) ? '' : 'is-miss'}`} role="status">
        <span className="am-verdict-mark" aria-hidden="true">
          {((committed.difference > 1e-12 && choice === 'higher') || (committed.difference < -1e-12 && choice === 'lower') || (Math.abs(committed.difference) <= 1e-12 && choice === 'same')) ? '=' : '≠'}
        </span>
        {((committed.difference > 1e-12 && choice === 'higher') || (committed.difference < -1e-12 && choice === 'lower') || (Math.abs(committed.difference) <= 1e-12 && choice === 'same'))
          ? 'Your prediction matches. ' : 'Your prediction does not match. '}
        Committing to {shown} gives output {round(committed.output, 6)} and loss {round(committed.loss, 6)}; the mixture&rsquo;s
        loss was {round(committed.mixtureLoss, 6)}, a change of {signed(committed.difference, 6)}.
        {committed.worseThanMixture
          ? ' The committed function is worse than the mixture that was being optimized — discretization itself changed the function, before any retraining.'
          : ' Here the commitment does not cost anything; the contrast appears when the operations disagree and the mixture sits between them.'}
      </p>}
    </div>
    {committed && <Table caption="Every discrete choice, against the mixture. The largest softmax weight is not automatically the best committed operation."
      headings={['operation', 'output', 'loss', 'softmax weight']}
      rows={committed.alternatives.map(entry => {
        const source = applied.operations.find(operation => operation.id === entry.id);
        return [entry.id, round(entry.output, 6), round(entry.loss, 6), round(source.probability, 6)];
      })}
      rowClass={index => (committed.alternatives[index].id === shown ? 'is-selected' : undefined)} />}
    {committed && <p className="am-caption">
      The largest weight belongs to <b>{committed.argmaxId}</b>; the lowest committed loss belongs
      to <b>{committed.bestDiscreteId}</b>. Committing does not retrain anything, so this comparison says nothing about
      how a retrained architecture would perform. That is a separate experiment.
    </p>}
  </Stage>;
}

/** The architecture-step stage, with its own recorded prediction. */
function StepStage({ applied }) {
  const [choice, setChoice] = useState('');
  const [shown, setShown] = useState(false);
  const correct = applied.stepOutcome;
  return <Stage title="Take one architecture step">
    <div className="am-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>
          A gradient step of size {round(applied.step, 3)} on the logits will move the mixed output:
        </p>
        <div className="am-choices">
          {[['toward', 'toward the target'], ['stay', 'by no more than 10⁻¹²'], ['away', 'away from the target']].map(([value, text]) => (
            <label className="am-choice" key={value}>
              <input type="radio" name="step-direction" value={value} checked={choice === value}
                disabled={shown} onChange={() => setChoice(value)} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <div className="am-buttons">
        <button type="button" className="is-primary" disabled={choice === '' || shown}
          onClick={() => setShown(true)}>Take the step</button>
      </div>
      {shown && <p className={`am-verdict ${choice === correct ? '' : 'is-miss'}`} role="status">
        <span className="am-verdict-mark" aria-hidden="true">{choice === correct ? '=' : '≠'}</span>
        {choice === correct
          ? 'Your prediction matches. '
          : `You recorded "${{ toward: 'toward the target', stay: 'by no more than 10⁻¹²', away: 'away from the target' }[choice]}"; the calculation gives "${{ toward: 'toward the target', stay: 'by no more than 10⁻¹²', away: 'away from the target' }[correct]}". `}
        {applied.stepMoved
          ? <>The updated logits are ({applied.updatedLogits.map(value => round(value, 6)).join(', ')}), giving
            output {round(applied.updatedOutput, 12)} and loss {round(applied.updatedLoss, 6)} against the previous
            {' '}{round(applied.loss, 6)}. The weight
            on {applied.operations.find(operation => operation.id === applied.descendId).label} increases, because
            descending on the logits raises whichever operation pulls the mixture toward the target.</>
          : <>The computed output change is within 10⁻¹². The updated logits are
            ({applied.updatedLogits.map(value => round(value, 6)).join(', ')}), the output is
            {round(applied.updatedOutput, 12)} and the loss is {round(applied.updatedLoss, 6)}, because
            {' '}{applied.stepUnchangedReason}.
            {applied.zeroGradient
              ? ' A larger step would not help either, because there is no gradient to descend.'
              : applied.step === 0 ? ' The gradient is nonzero, but the step length is zero.' : ' This is a numerical near-tie, not evidence that a larger step could never change the output.'}</>}
      </p>}
    </div>
  </Stage>;
}

export function MixtureLab() {
  const state = useInvestigation(mixtureBaseline);
  const draft = state.draft;
  const answerFor = proposed => {
    const result = mixtureState(proposed);
    return { outcome: result.zeroGradient ? 'none' : result.descendId, value: result.mixed };
  };
  const applied = state.result ? mixtureState(state.active) : null;
  const curve = applied ? mixtureCurve({ active: state.active.active, logits: state.active.logits, target: state.active.target }) : null;
  const appliedKey = applied ? JSON.stringify(state.active) : '';
  return <Investigation
    title="Mix the operations, then commit to one"
    question="One edge could apply any of three shape-compatible operations. Architecture logits turn into softmax weights, and the edge evaluates their mixture. Record which operation a descending gradient step will favour — and, optionally, the mixed output — before anything is computed."
    evidence="Constructed scalar operations with no trainable parameters, so the mixture calculus is isolated. The logits are architecture variables; they are not class probabilities and the widths below do not encode confidence."
    onReset={state.reset}>

    <div className="am-controls is-narrow">
      <NumberField label="Input x" value={draft.x} min={limits.input.minimum} max={limits.input.maximum}
        step={String(controlSteps.input)} decimals={3} onChange={x => state.edit({ x })} />
      <NumberField label="Target" value={draft.target} min={limits.target.minimum} max={limits.target.maximum}
        step={String(controlSteps.target)} decimals={3} onChange={target => state.edit({ target })} />
      <NumberField label="Step size" value={draft.step} min={limits.stepSize.minimum} max={limits.stepSize.maximum}
        step={String(controlSteps.stepSize)} decimals={3} onChange={step => state.edit({ step })} />
      {declaredOperations.map((operation, index) => (
        <NumberField key={operation.id} label={`logit for ${operation.label}`} value={draft.logits[index]}
          min={limits.logit.minimum} max={limits.logit.maximum} step={String(controlSteps.logit)} decimals={6}
          disabled={!draft.active.includes(operation.id)}
          onChange={value => state.edit(current => ({
            logits: current.logits.map((old, position) => (position === index ? value : old)),
          }))} />
      ))}
    </div>
    <div className="am-buttons" role="group" aria-label="Active operations">
      {declaredOperations.map(operation => (
        <label className="am-choice" key={operation.id}>
          <input type="checkbox" checked={draft.active.includes(operation.id)}
            aria-label={`Keep the ${operation.label} operation active`}
            onChange={() => state.edit(current => {
              const next = current.active.includes(operation.id)
                ? current.active.filter(id => id !== operation.id)
                : [...current.active, operation.id];
              return next.length >= 2 ? { active: next } : {};
            })} />
          <span>{operation.label} — <code>{operation.formula}</code></span>
        </label>
      ))}
    </div>
    <div className="am-buttons">
      {mixtureSetups.map(setup => (
        <button key={setup.key} type="button" onClick={() => state.suggest(setup.build())}>{setup.label}</button>
      ))}
    </div>
    <p className="am-caption">
      An inactive operation leaves the softmax denominator entirely; it is not given a weight of zero while still
      contributing to the normalization. At least two operations stay active.
    </p>

    <Prediction
      prompt={`At x = ${round(draft.x, 3)} with target ${round(draft.target, 3)}, which operation’s logit receives the greatest increase from a descending gradient step (the most negative gradient)?`}
      options={[...declaredOperations.filter(operation => draft.active.includes(operation.id)).map(operation => [operation.id, `${operation.label}, ${operation.formula}`]), ['none', 'None — the gradient is exactly zero']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the mixed output', name: 'the mixed output', tolerance: 1e-6, digits: 12 }}
      committed={shown => `x = ${round(state.active.x, 3)}, target ${round(state.active.target, 3)}, logits (${state.active.active.map(id => round(state.active.logits[declaredOperations.findIndex(operation => operation.id === id)], 6)).join(', ')})`} />

    {applied && <>
      <div className="am-stage">
        <svg viewBox="0 0 340 168" role="img"
          aria-label={`Input ${applied.x} enters ${applied.operations.length} operations. ${applied.operations.map(operation => `${operation.label} outputs ${operation.output} with weight ${operation.probability}`).join('; ')}. Their weighted sum is ${applied.mixed}, the target is ${applied.target}, and the residual is ${applied.residual}.`}>
          <rect className="am-node" x="6" y="62" width="52" height="34" rx="4" />
          <text className="is-small" x="32" y="83" textAnchor="middle">x = {round(applied.x, 3)}</text>
          {applied.operations.map((operation, index) => {
            const y = 16 + index * (136 / applied.operations.length);
            // One encoding, one dimension: the bar's *width* is the softmax
            // weight. Letting its height vary too would put a second, undeclared
            // quantity on screen.
            const height = 18;
            return <g key={operation.id}>
              <line className="am-flow" x1="60" y1="79" x2="96" y2={y + 16} />
              <rect className="am-node is-selection" x="98" y={y} width="92" height="32" rx="4" />
              <text className="is-small" x="144" y={y + 14} textAnchor="middle">{operation.label}</text>
              <text className="is-small" x="144" y={y + 27} textAnchor="middle">→ {round(operation.output, 4)}</text>
              <rect className="am-block" x="196" y={y + 16 - height / 2} width={operation.probability * 58} height={height} />
              <text className="is-small am-halo" x="258" y={y + 20}>p ≈ {operation.probability < 1e-4 ? operation.probability.toExponential(1) : Number(operation.probability.toPrecision(3))}</text>
            </g>;
          })}
          <line className="am-flow is-feedback" x1="196" y1="152" x2="248" y2="152" />
          <text className="is-small" x="196" y="144">weighted sum</text>
          <text className="is-small am-halo" x="254" y="156">ō = {round(applied.mixed, 6)}</text>
        </svg>
        <p className="am-caption">
          Bar width, and only bar width, is the operation&rsquo;s softmax weight in the mixture; every bar has the same
          height. Diagram labels are rounded; the table below retains more precision. It is not a parameter count and it is not a classifier confidence. The target
          is {round(applied.target, 3)} and the residual ō − target is {round(applied.residual, 6)}.
        </p>
      </div>
      <Table caption={`The exact state. Loss is ½(ō − target)² = ${round(applied.loss, 6)}.`}
        headings={['operation', 'logit', 'softmax weight', 'output', '∂L/∂α']}
        rows={applied.operations.map(operation => [
          operation.label, round(operation.logit, 6), round(operation.probability, 6),
          round(operation.output, 6), exactly(operation.gradient, 6),
        ])}
        rowClass={index => (applied.operations[index].id === applied.descendId ? 'is-selected' : undefined)} />
      <p className="am-readout" role="status">
        {applied.zeroGradient
          ? <>The architecture gradient is <b>exactly zero</b> on every active operation, because {applied.zeroGradientReason}.
            No logit is favoured and no architecture step can move the mixture from here.</>
          : <>The most negative gradient belongs to <b>{applied.descendId}</b>, so a descending step raises its logit and
            its weight. Adding the same constant to every active logit leaves the weights, output, loss and gradient
            unchanged — the logits carry one redundant coordinate.</>}
      </p>
      <StepStage key={`step-${appliedKey}`} applied={applied} />
      <CommitStage key={`commit-${appliedKey}`} applied={applied} />
      <details>
        <summary>The exact mixed function over the input range</summary>
        <Plot caption="The constructed mixed function, with the current input marked"
          width={340} height={170} domain={curve.domain}
          range={[Math.min(...curve.points.map(point => point.mixed), applied.target) - 0.5,
            Math.max(...curve.points.map(point => point.mixed), applied.target) + 0.5]}
          describe={`The mixture evaluated across inputs from ${curve.domain[0]} to ${curve.domain[1]}, with the current input ${applied.x} and target ${applied.target} marked.`}>
          {(scaleX, scaleY) => <>
            <polyline className="am-curve is-mixture" points={polyline(curve.points.map(point => [point.x, point.mixed]), scaleX, scaleY)} />
            <line className="am-rule is-incumbent" x1={scaleX(curve.domain[0])} x2={scaleX(curve.domain[1])}
              y1={scaleY(applied.target)} y2={scaleY(applied.target)} />
            <text className="is-small am-halo" x={scaleX(curve.domain[1]) - 4} y={scaleY(applied.target) - 5} textAnchor="end">target</text>
            <circle className="am-mark" cx={scaleX(applied.x)} cy={scaleY(applied.mixed)} r="4.5" />
          </>}
        </Plot>
        <p className="am-caption">
          A constructed function of one scalar input. It is not a fitted banknote decision boundary, and no data from
          section 5 enters it.
        </p>
      </details>
      <Attempts entries={state.history} />
    </>}
  </Investigation>;
}
