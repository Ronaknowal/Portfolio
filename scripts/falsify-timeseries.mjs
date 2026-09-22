// Falsification harness for the forecasting lesson's guards.
//
// A passing verifier proves nothing about a guard that cannot fail. Every inert
// guard found in this repository so far was correct within a domain that
// excluded the defect: a KaTeX check measuring only width when a collapsed SVG
// is also narrow, a leak filter requiring values above 2 where every quantity
// is below 1, a curve sampler that queried paths on a lesson drawing lines, and
// hundreds of assertions comparing a function with itself inside an advertised
// grid. So each guard here is shown to fire, by breaking exactly the thing it
// claims to watch.
//
// The contract this harness keeps:
//
//   * one breakage at a time, applied to a clean tree;
//   * the file is restored byte for byte afterwards and the restoration is
//     verified by SHA-256, whether the case passed or failed;
//   * the verifier must EXIT NON-ZERO **and** its output must name the guard,
//     so a run that fails for an unrelated reason does not count as a firing;
//   * every evidence file and screenshot the run could touch is snapshotted
//     first and restored in a `finally`, so falsifying a guard never rewrites
//     the record it is falsifying -- not even when a case throws. ONLY THIS
//     LESSON'S ARTIFACTS ARE TOUCHED: the globs below are anchored to
//     `timeseries-`, because a sibling harness once came close to deleting 709
//     of another lesson's captures;
//   * the pre-mutation bytes go to a `.orig` sidecar on disk BEFORE the
//     mutation and are removed only after a verified restore, with signal and
//     exception handlers that restore from it, so a kill mid-case cannot leave
//     a breakage applied with nothing on disk saying so;
//   * a lock file refuses a second concurrent instance, and a stale sidecar
//     makes the next run refuse to start rather than mutate a file that is
//     already wrong.
//
// On Windows a forced termination cannot be trapped at all -- `taskkill /F`
// delivers no catchable signal -- so the handlers are a courtesy and the
// SIDECAR plus the refusal-to-start are the actual protection. `--recover`
// is the documented way back.
//
// Browser guards are not falsifiable without a running preview and are deferred
// to phase C. Passing `--browser` adds those cases; without it they are listed
// in the report as deferred rather than silently omitted.
//
// Run: node scripts/falsify-timeseries.mjs
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const python = process.env.PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const MODELS = ['node', ['scripts/verify-timeseries-models.mjs', '--no-evidence']];
const SOURCES = [python, ['scripts/verify-timeseries-sources.py', '--no-evidence']];
const DATA = [python, ['scripts/verify-timeseries-data.py', '--no-evidence']];
const EXAMPLES = [python, ['scripts/verify-timeseries-examples.py', '--no-evidence']];
const RENDER = ['node', ['scripts/verify-timeseries-render.cjs', '--no-evidence']];
const BROWSER = ['node', ['scripts/verify-timeseries-browser.cjs']];

const MODELS_FILE = 'src/learn/data/timeseries-models.js';
const DATA_FILE = 'src/learn/data/timeseries-data.js';
const EXAMPLES_FILE = 'src/learn/data/timeseries-examples.js';
const TOPIC_FILE = 'src/learn/data/topics/time-series-validation-forecasting-baselines.jsx';
const LABS_FILE = 'src/learn/components/lesson-labs/TimeSeriesLabs.jsx';
const FIGURES_FILE = 'src/learn/components/lesson-labs/TimeSeriesFigures.jsx';
const SHARED_FILE = 'src/learn/components/lesson-labs/TimeSeriesShared.jsx';
const CSS_FILE = 'src/learn/components/lesson-labs/timeseries-labs.css';
const EXAMPLES_VERIFIER = 'scripts/verify-timeseries-examples.py';
const MODELS_VERIFIER = 'scripts/verify-timeseries-models.mjs';

const withBrowser = process.argv.includes('--browser');

/** Each case breaks one thing and names the guard that must notice. */
const cases = [
  {
    name: 'the seasonal rule loses its modulo, so it walks past the end of the observed cycle',
    file: MODELS_FILE,
    find: '  const index = historyLength - period + ((horizon - 1) % period);',
    replace: '  const index = historyLength - period + ((horizon) % period);',
    run: MODELS,
    expect: /seasonal at h=\d|the donor index at T=|disagrees with a walk round the cycle/,
    guards: 'the seasonal donor index checked against a route that materialises the final observed cycle and '
      + 'walks round it, over every history length, season length and horizon',
  },
  {
    name: 'the drift slope divides by T instead of T minus one',
    file: MODELS_FILE,
    find: '  const slope = (last - first) / (length - 1);',
    replace: '  const slope = (last - first) / length;',
    run: MODELS,
    expect: /drift at h=|the drift slope the manuscript prints|the packet's drift forecast/,
    guards: 'the drift rule against an independent accumulation route and against the manuscript’s own '
      + 'stated slope of 2.4 over five intervals',
  },
  {
    name: 'a training row is admitted on its target date rather than on its label’s arrival',
    file: MODELS_FILE,
    find: '      eligible: labelArrival <= cutoff,',
    replace: '      eligible: origin + horizon <= cutoff,',
    run: MODELS,
    expect: /the availability-set route gives|the manuscript's eligible set/,
    guards: 'eligibility decided by an availability-SET route that never writes the inequality, swept over '
      + 'every cutoff, horizon, delay and origin the controls admit',
  },
  {
    name: 'the last eligible training origin is off by one, admitting a row whose label has not arrived',
    file: MODELS_FILE,
    find: '  const lastTrainOrigin = issueOrigin - horizon - delay;',
    replace: '  const lastTrainOrigin = issueOrigin - horizon - delay + 1;',
    run: MODELS,
    expect: /the index route and the forward simulation disagree|a fitted window contains an observation later than its origin/,
    guards: 'THE TOPIC’S OWN INVARIANT: every fit of the declared contract audited for an observation '
      + 'dated later than its own origin, and its training rows regenerated by simulating the calendar '
      + 'forward rather than by the module’s own index arithmetic',
  },
  {
    name: 'the information-set audit stops checking feature causality',
    file: MODELS_FILE,
    find: '      if (index > row.origin - delay) {',
    replace: '      if (index > row.origin - delay && false) {',
    run: MODELS,
    expect: /a feature dated after its own row origin must be reported/,
    guards: 'the audit itself, falsified with a deliberately invalid fit, because an audit that always '
      + 'returns clean is not an audit',
  },
  {
    name: 'a horizon with no supplied outcome is scored as if it had one',
    file: MODELS_FILE,
    find: "    const scored = outcome !== null && outcome !== undefined && Number.isFinite(outcome);",
    replace: '    const scored = true;',
    run: MODELS,
    expect: /no outcome means no denominator|only the supplied outcomes are scored|a missing outcome is marked unscored/,
    guards: 'the unscored horizon: a missing outcome contributes no error and lowers the denominator, and a '
      + 'stage with no outcome at all has a null mean rather than zero',
  },
  {
    name: 'the per-horizon winner takes the first method it finds, so an exact tie reports one name',
    file: MODELS_FILE,
    find: '      return { horizon: index + 1, value: best, keys: series.filter(entry => entry.values[index] === best).map(entry => entry.key) };',
    replace: '      return { horizon: index + 1, value: best, keys: [series.find(entry => entry.values[index] === best).key] };',
    run: MODELS,
    expect: /horizon 7 must report BOTH tying methods|a three-way tie must report all three methods/,
    guards: 'every method attaining the minimum, at the real horizon-7 tie and at a constructed three-way tie',
  },
  {
    name: 'the three-way comparison becomes strict, so an exact tie is reported as a difference',
    file: MODELS_FILE,
    find: "  if (Math.abs(difference) <= tolerance) return 'equal';",
    replace: "  if (Math.abs(difference) < tolerance) return 'equal';",
    run: MODELS,
    expect: /the tolerance boundary is inclusive/,
    guards: 'the tie category asserted at its exact boundary, which is the branch investigation 3 lands on '
      + 'at every origin when the season length is one',
  },
  {
    name: 'the calendar-lane inset drops to zero, which used to satisfy its own containment check',
    file: MODELS_FILE,
    find: 'export function issueTimeGeometry({ fixture = ISSUE_TIME_FIXTURE, width = 300, inset = 14, cellGap = 1.4 } = {}) {',
    replace: 'export function issueTimeGeometry({ fixture = ISSUE_TIME_FIXTURE, width = 300, inset = 0, cellGap = 1.4 } = {}) {',
    run: MODELS,
    expect: /the calendar inset is 0, below the 12 units/,
    guards: 'the fixed minimum inset, asserted BEFORE any position is compared against it, so the check '
      + 'cannot be satisfied by the very value it is guarding',
  },
  {
    name: 'a forecast request reads one day past its own origin',
    file: MODELS_FILE,
    find: '  const history = counts.slice(0, origin + 1);',
    replace: '  const history = counts.slice(0, origin + 2);',
    run: MODELS,
    expect: /pooled MAE|editing a future outcome moved a forecast already issued|the manuscript's seasonal MAE at origin/,
    guards: 'the measured baselines recomputed from the served CSV, and the future-edit null that says an '
      + 'outcome after the origin cannot move a forecast issued at it',
  },
  {
    name: 'a measured development count is altered in the generated data module',
    file: DATA_FILE,
    find: '"mae": 1310.3392857142858',
    replace: '"mae": 1311.3392857142858',
    run: MODELS,
    expect: /final naive MAE against the packet|final naive MAE against the manuscript/,
    guards: 'every measured value against the frozen packet AND against the number typed in the manuscript',
  },
  {
    name: 'the same measured value is altered, against the served dataset',
    file: DATA_FILE,
    find: '"mae": 1310.3392857142858',
    replace: '"mae": 1311.3392857142858',
    run: DATA,
    expect: /does not reproduce src\/learn\/data\/timeseries-data\.js/,
    guards: 'byte-identical regeneration of the data module from the served dataset and the declared protocol',
  },
  {
    name: 'a recorded program output is altered',
    file: EXAMPLES_FILE,
    find: 'final naive MAE 1310.34',
    replace: 'final naive MAE 1310.35',
    run: MODELS,
    expect: /the manuscript.s own loop disagrees with the data module/,
    guards: 'the displayed programs’ recorded output against what the browser models compute',
  },
  {
    name: 'the pinned digest of an extracted algorithm no longer matches the frozen program',
    file: EXAMPLES_VERIFIER,
    find: '"extractedSha256": "e71a3de3dacc8ecf23a74440f2b0fc2ae065db8929e6f793a8c7d5cf57af81ed"',
    replace: '"extractedSha256": "0000000000000000000000000000000000000000000000000000000000000000"',
    run: EXAMPLES,
    expect: /the extracted algorithm now hashes/,
    guards: 'the SHA-256 pin on the bytes extracted from the frozen author program',
  },
  {
    name: 'a shell eats an escape and leaves a raw control byte in a source file',
    file: SHARED_FILE,
    find: 'const sign = text =>',
    // The byte is BUILT here rather than written. A literal control byte in
    // this file would be caught by the very guard this case falsifies, which
    // is a pleasing demonstration and a broken harness.
    replace: `const sign = text => /* ${String.fromCharCode(8)} */`,
    run: SOURCES,
    expect: /raw control byte 0x08|a shell ate an escape/,
    guards: 'the raw-control-byte scan over every owned file, its five verifiers and this harness',
  },
  {
    name: 'a required KaTeX sequence disappears from the lesson body',
    file: TOPIC_FILE,
    find: '\\\\bmod',
    replace: 'mod',
    run: SOURCES,
    expect: /the KaTeX sequence \\\\bmod is missing/,
    guards: 'the required-KaTeX-sequence list',
  },
  {
    name: 'a display block is left as one long unwrapped line',
    file: TOPIC_FILE,
    find: "<MathBlock>{'q_T=\\\\frac1{T-m}\\\\sum_{i=m+1}^{T}|y_i-y_{i-m}|.'}</MathBlock>",
    replace: "<MathBlock>{'q_T=\\\\frac1{T-m}\\\\sum_{i=m+1}^{T}|y_i-y_{i-m}| \\\\text{ computed on the training"
      + " history alone and never on the later assessment period}.'}</MathBlock>",
    run: SOURCES,
    expect: /an unwrapped display block of \d+ visible and \d+ raw characters/,
    guards: 'the display-block wrapping scan that keeps formulas inside a 320 px column',
  },
  {
    name: 'a JSX conditional is left with two identical branches',
    file: LABS_FILE,
    find: 'const series = timeSeriesData.series;',
    replace: "const series = timeSeriesData.series;\nconst lost = flag => (flag ? 'same' : 'same');",
    run: SOURCES,
    expect: /a conditional whose branches are identical/,
    guards: 'the identical-branch scan, which is the prose equivalent of an assertion that cannot fail',
  },
  {
    name: 'the lesson body reaches for a member of the global Math, which is shadowed there',
    file: TOPIC_FILE,
    find: 'const toy = toyComparison;',
    replace: 'const toy = toyComparison;\nconst shadowed = Math.round(2.5);',
    run: SOURCES,
    expect: /Math\.round in a module where `Math` is the KaTeX component/,
    guards: 'the shadowed-Math scan',
  },
  {
    name: 'the lesson borrows a sibling lesson’s dataset directory',
    file: TOPIC_FILE,
    find: 'const toy = toyComparison;',
    replace: "const toy = toyComparison;\nconst borrowed = '/learn-assets/pac-learning/banknote-subset.csv';",
    run: SOURCES,
    expect: /belongs to another lesson/,
    guards: 'the own-assets-only scan',
  },
  {
    /* THE MOST IMPORTANT CASE IN THIS FILE. Unscoping the SVG rule puts it back
       in front of KaTeX's radical SVGs, whose height comes from
       `height: inherit`; `auto` leaves them no intrinsic height and every
       square root on the page collapses to nothing. In an earlier lesson that
       turned a radius into the radius SQUARED -- a different quantity, from
       which a learner would derive wrong answers -- with correct DOM, zero
       katex-error nodes and four green offline verifiers. This case proves the
       TEXT-LEVEL guard fires; the browser case below proves the MEASUREMENT
       does. Both exist because either alone has been defeated before. */
    name: 'the SVG layout rule is unscoped, putting it back in front of KaTeX’s radicals',
    file: CSS_FILE,
    find: '.ts-lesson svg.ts-diagram {',
    replace: '.ts-lesson svg {',
    run: SOURCES,
    expect: /a bare `svg` selector/,
    guards: 'the bare-selector scan over the whole stylesheet, which refuses any `svg` token not immediately '
      + 'qualified by the layout class',
  },
  {
    name: 'a figure file opens its own svg tag instead of using the shared wrapper',
    file: FIGURES_FILE,
    /* Real code, not a comment. The scan blanks comments first, because a scan
       that read them would flag the figure file's own explanation of why this
       is forbidden -- a guard nobody could keep green. */
    find: "function Arrow({ x1, y1, x2, y2, kind = '', head = 4.2, inputKind, encoding }) {",
    replace: "function Arrow({ x1, y1, x2, y2, kind = '', head = 4.2, inputKind, encoding }) {\n"
      + '  const stray = <svg viewBox="0 0 10 10" />;',
    run: SOURCES,
    expect: /opens an <svg> tag/,
    guards: 'the rule that only the shared wrapper may open an svg tag, so a new figure cannot omit the '
      + 'class the stylesheet is scoped to',
  },
  {
    /* THE FAILURE THE RENDER VERIFIER WAS ADDED FOR. A sibling lesson in this
       effort did not render at all: its body threw, the error boundary replaced
       the whole page, and three offline verifiers stayed green because the
       model layer, the recorded data and the source text were each perfectly
       correct. Only executing the React tree can see it. */
    name: 'a figure throws during render, so the whole page is replaced by the error boundary',
    file: FIGURES_FILE,
    find: '  const inset = geometry.inset;',
    replace: '  const inset = geometry.missingRows.inset;',
    run: RENDER,
    expect: /the lesson body did not render|FAIL while rendering the lesson body/,
    guards: 'the server-side render, which is the only offline check that executes the React tree rather than '
      + 'reading the model layer, the data or the source text',
  },
  {
    name: 'an investigation renders its reveal panel on first paint',
    file: LABS_FILE,
    find: '    {shown && committedRows && <div className="ts-reveal">',
    replace: '    {committedRows !== undefined && <div className="ts-reveal">',
    run: RENDER,
    expect: /a reveal panel is rendered on first paint|the lesson body did not render/,
    guards: 'the first-paint investigation contract, asserted on the rendered markup rather than on the '
      + 'component source',
  },
  {
    /* THE DEFECT THAT BROKE THE SHARED BUILD DURING PHASE C, reproduced exactly:
       an unescaped apostrophe inside a single-quoted string. A sibling lesson
       shipped one, `npx vite build` failed for every lesson at once, and no
       offline verifier in any of the three saw it -- the model and data
       verifiers import plain JS, and the source scans read text without parsing
       it. A file that does not parse cannot be made to look as though it does,
       so this guard cannot go inert. */
    name: 'a JSX file stops parsing, which no text scan and no JS import can see',
    file: FIGURES_FILE,
    find: "function Arrow({ x1, y1, x2, y2, kind = '', head = 4.2, inputKind, encoding }) {",
    replace: "const broken = 'not this study's development rows';\n"
      + "function Arrow({ x1, y1, x2, y2, kind = '', head = 4.2, inputKind, encoding }) {",
    run: SOURCES,
    expect: /does not parse/,
    guards: 'the parse check over every owned JS and JSX file, including the blueprint, which nothing '
      + 'imports at runtime',
  },
  {
    /* S1's property, offline. The model layer computed `inputKind` per row all
       along and the figure never read it; collapsing the map so that the first
       arrow shares the rest of its lane's style is exactly the defect the
       review found rendered. */
    name: 'the first arrow of a lane stops having its own style, as it did before the review',
    file: MODELS_FILE,
    find: "  observation: 'origin-observation',",
    replace: "  observation: 'prediction-fed',",
    run: MODELS,
    expect: /every lane.s first arrow carries the origin.s own observed count/,
    guards: 'the per-arrow style property: every input kind maps to a drawn style, and every lane\u2019s '
      + 'first arrow shares one because all three carry the same observation',
  },
  {
    /* The floors sit far below the current values on purpose, so they catch a
     * WHOLESALE loss rather than the removal of one call. Breaking a single
     * `record()` therefore does not trip them -- correct behaviour, and the
     * reason this case neuters the counter instead. The limitation is stated in
     * the report rather than hidden by a floor tightened to today's number. */
    name: 'the model suite stops counting what it checked',
    file: MODELS_VERIFIER,
    find: 'const record = name => { counts[name] = (counts[name] ?? 0) + 1; };',
    replace: 'const record = () => {};',
    run: MODELS,
    expect: /the suite has lost coverage|check groups ran/,
    guards: 'the floors under the grouped-check and group counts, which stop a suite that has stopped '
      + 'checking from still printing PASS',
  },
];

/* Browser cases need a production build and a running preview, so they are off
   by default and run with `--browser`. Each one costs a rebuild. */
const browserCases = [
  {
    name: 'investigation 1 renders its reveal before a prediction is committed',
    file: LABS_FILE,
    find: '    {shown && <>\n      <Table caption={`All four rules on this history',
    replace: '    {true && <>\n      <Table caption={`All four rules on this history',
    rebuild: true,
    run: BROWSER,
    /* The message changed when the paired-pin helper replaced the hand-written
       absence assertions, and this pattern was left naming the old one -- so
       the case failed for the right reason and was credited as inert. The
       harness caught it, which is what it is for. */
    expect: /investigation 1, .*is rendered BEFORE it should be/,
    guards: 'the leak property: the graded quantity absent before commitment, pinned by its exact text and '
      + 'by the phrases only the reveal renders',
  },
  {
    name: 'investigation 3 reveals the outcomes at the moment the forecast is issued',
    file: SHARED_FILE,
    find: '  const open = !gated || state.revealed;',
    replace: '  const open = true;',
    rebuild: true,
    run: BROWSER,
    expect: /reveals its graded MAE .* at the moment of issuing|grades the comparison before the outcomes are revealed/,
    guards: 'the SECOND gate: a forecasting investigation must be able to draw a forecast while its outcomes '
      + 'are still concealed',
  },
  {
    /* B1, EXACTLY AS FOUND: an opaque `.ts-cell` rect covering the hatch and
       emitted AFTER it. That was the defect -- 31 hatch lines in the DOM, every
       attribute correct, and a declared stroke that appeared nowhere on the
       canvas while the legend and caption both promised it. SVG has no
       z-index; document order is paint order, so no attribute check can see
       this. Only a check that asks whether a legend's encoding is actually the
       topmost painted thing somewhere on itself can.

       An EARLIER draft of this case inserted a SECOND, covered hatch before the
       band and left the real one in place. The guard was right to stay silent
       -- the picture still contained visible hatching -- and the case was
       reported INERT, which is the harness doing its job: a breakage that does
       not actually break the property proves nothing about the guard. */
    name: 'figure 1 paints an opaque cell over the hatching its legend promises',
    file: FIGURES_FILE,
    find: '            height={geometry.rows.cellHeight} kind="unknown" encoding="unknown" />',
    replace: '            height={geometry.rows.cellHeight} kind="unknown" encoding="unknown" />\n'
      + '          <rect className="ts-cell" x={inset + (lane.originIndex + 1) * geometry.cellWidth}\n'
      + '            y={lane.y} width={(geometry.days.length - lane.originIndex - 1) * geometry.cellWidth}\n'
      + '            height={geometry.rows.cellHeight} />',
    rebuild: true,
    run: BROWSER,
    expect: /a figure legend names an encoding the painted picture does not contain/,
    guards: 'the legend-encoding check: every encoding a legend names must be realized by an element that is '
      + 'the topmost painted thing somewhere on itself, for every figure and at every width',
  },
  {
    /* S1. The styles are PERMUTED, not collapsed. An earlier draft forced every
       arrow to `prediction-fed`, which also emptied two of figure 3's legend
       encodings -- so the legend check fired first and the case was credited to
       the wrong guard, which the harness reported as INERT because the named
       guard was not the one under test. Swapping `prediction` with `newly
       observed outcome` leaves all three styles present in the figure, so the
       legend stays satisfied and ONLY the per-arrow check can see that each
       arrow now contradicts its own `data-input-kind`. */
    name: 'an arrow is drawn in a style that does not match the input it carries',
    file: FIGURES_FILE,
    find: "              const style = invalid ? 'invalid' : arrowStyleFor(row.inputKind);",
    replace: "              const swapped = row.inputKind === 'prediction' ? 'newly observed outcome'\n"
      + "                : row.inputKind === 'newly observed outcome' ? 'prediction' : row.inputKind;\n"
      + "              const style = invalid ? 'invalid' : arrowStyleFor(swapped);",
    rebuild: true,
    run: BROWSER,
    expect: /an arrow is drawn in a style that does not match the input it carries/,
    guards: 'the per-arrow drawn-style check, which compares each arrow\u2019s class against the style its '
      + 'own data-input-kind maps to through the model layer\u2019s own table',
  },
  {
    /* THE PRESENCE HALF of the leak property. An absence assertion alone cannot
       tell "correctly withheld" from "looking for text this page never
       renders", and two pins in earlier revisions of the browser verifier were
       doing exactly the latter -- one appended a word the page never prints,
       one named an aria-label attribute. Both passed every run and proved
       nothing. This breakage makes the donor strip stop rendering the text its
       pin names, so the PRESENCE assertion has to be what fires. */
    name: 'a pinned phrase stops being rendered, so its absence assertion would pass on nothing',
    file: LABS_FILE,
    find: '          label: `h${lead + 1} ← pos ${index + 1}`,',
    replace: '          label: `horizon ${lead + 1}`,',
    rebuild: true,
    run: BROWSER,
    expect: /never appears even after it should, so its absence assertion proves nothing/,
    guards: 'the paired-pin helper, which refuses an absence assertion that has no presence partner',
  },
  {
    /* The one breakage that protects a MATHEMATICAL claim rather than a layout
       one, measured rather than read. */
    name: 'the SVG layout rule is unscoped again, collapsing every KaTeX radical to nothing',
    file: CSS_FILE,
    find: '.ts-lesson svg.ts-diagram {',
    replace: '.ts-lesson svg {',
    rebuild: true,
    run: BROWSER,
    expect: /a square-root radical paints with no height or width/,
    guards: 'the measurement of every .katex .sqrt svg on the rendered page, in BOTH dimensions, which is the '
      + 'only thing standing between a stylesheet and a wrong formula',
  },
  {
    name: 'a stylesheet rule paints a calendar band black, so it vanishes into the page ground',
    file: CSS_FILE,
    /* A BLANKET rule is not enough: every rect on this page already matches a
       more specific rule, so a blanket one never applies and nothing breaks --
       a breakage that breaks nothing proves nothing about the guard. The defect
       this guard exists for is a stylesheet beating the intended paint, so the
       breakage overrides a specific band instead. */
    find: '.ts-lesson svg.ts-diagram .ts-known-band { fill: #1b2a24; stroke: #4e6a5c; stroke-width: 1; }',
    replace: '.ts-lesson svg.ts-diagram .ts-known-band { fill: #000000; stroke: none; }',
    rebuild: true,
    run: BROWSER,
    expect: /black fill with no stroke|attribute overridden/,
    guards: 'the painted-against-declared check, which compares computed styles rather than attributes',
  },
  {
    /* The staircase inset is sized for the three-digit origin label printed
       end-anchored beside each lane. At 16 those labels began about two pixels
       left of the viewBox at every width -- measured on the rendered page, and
       invisible to the shared layout inspector, which is why this lesson
       measures label boxes against SVG boxes itself.
       NOTE: the obvious sibling of this case -- setting figure 1's calendar
       inset to zero -- was tried here and is INERT in the browser, because that
       figure's labels are centred within their own cells and stay inside the
       viewBox even at inset zero. Only the model layer's fixed floor catches
       it, and the offline case above does exactly that. Keeping a browser case
       that cannot fire would have been a guard advertising coverage it does not
       have. */
    name: 'the staircase inset shrinks, so each lane’s origin label is clipped at the viewBox edge',
    file: MODELS_FILE,
    find: 'fixture = STAIRCASE_FIXTURE, mode = \'expanding\', horizon = 1, width = 300, inset = 30,',
    replace: 'fixture = STAIRCASE_FIXTURE, mode = \'expanding\', horizon = 1, width = 300, inset = 16,',
    rebuild: true,
    run: BROWSER,
    expect: /a label is painted past the edge of its own SVG/,
    guards: 'the measurement of every SVG label box against its own SVG box, at all three widths -- a check '
      + 'this lesson adds because the shared inspector did not see a two-pixel clip',
  },
];

/* ONLY THIS LESSON'S ARTIFACTS. Anchored to `timeseries-`, because a sibling
   harness once came close to deleting 709 of another lesson's captures. */
const REPORT = 'docs/teaching/evidence/timeseries-falsification.json';
const evidenceFiles = [
  ...(fs.existsSync('docs/teaching/evidence')
    ? fs.readdirSync('docs/teaching/evidence')
      .filter(name => name.startsWith('timeseries-') && name.endsWith('.json'))
      .map(name => `docs/teaching/evidence/${name}`)
      /* THIS HARNESS'S OWN REPORT IS NOT SNAPSHOTTED. It matched the filter, so
         a run that threw inside the try had its `finally` restore the PREVIOUS
         report -- and the new one is written after the try, so it never ran.
         The record on disk then still read `passed: true, 32 of 32` with
         nothing anywhere saying a later run had started and died. Every one of
         the five verifiers writes a provisional failing record before it
         starts; the harness, whose whole purpose is to show that guards can
         fail, was the one artifact that did the opposite. */
      .filter(name => name !== REPORT)
    : []),
  ...(fs.existsSync('docs/teaching/evidence/screenshots')
    ? fs.readdirSync('docs/teaching/evidence/screenshots')
      .filter(name => name.startsWith('timeseries-'))
      .map(name => `docs/teaching/evidence/screenshots/${name}`)
    : []),
];
const evidenceSnapshot = Object.fromEntries(evidenceFiles.map(file => [file, fs.readFileSync(file)]));

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const results = [];
let fired = 0;
let restoringBuild = null;
let rebuildRetries = 0;
let baselineBuild = null;

/* The provisional record is written once the lock is HELD -- see below. Written
   here instead, a run that refuses to start because another holds the lock
   would destroy the previous run's record on its way out. */

/* ===================================================== surviving a kill */

const LOCK = 'scratch/timeseries-falsification.lock';
const SIDECAR = file => `${file}.timeseries-falsify-orig`;
let activeSidecar = null;

fs.mkdirSync('scratch', { recursive: true });

if (process.argv.includes('--recover')) {
  const pending = [...new Set([...cases, ...browserCases].map(item => item.file))]
    .filter(file => fs.existsSync(SIDECAR(file)));
  if (!pending.length && !fs.existsSync(LOCK)) {
    console.log('Nothing to recover: no sidecar and no lock.');
    process.exit(0);
  }
  for (const file of pending) {
    const sidecar = SIDECAR(file);
    fs.copyFileSync(sidecar, file);
    const restored = digest(file) === crypto.createHash('sha256').update(fs.readFileSync(sidecar)).digest('hex');
    if (!restored) throw new Error(`${file} does not match its sidecar after restore; leaving ${sidecar} in place`);
    fs.rmSync(sidecar, { force: true });
    console.log(`restored ${file} from its sidecar`);
  }
  fs.rmSync(LOCK, { force: true });
  console.log(`recovered ${pending.length} file(s) and cleared the lock. Re-run the verifiers before trusting the tree.`);
  process.exit(0);
}

if (fs.existsSync(LOCK)) {
  throw new Error(`${LOCK} exists: another falsification run is in progress, or one was killed. `
    + 'Two instances mutating the same files is how a breakage was left on disk before. '
    + `Check for stale ${SIDECAR('<file>')} sidecars, run --recover, then remove the lock.`);
}
/* A stale sidecar means a previous run died mid-case. Restoring is not this
   run's job to guess at -- it is reported, loudly, and the run refuses. */
const staleSidecars = [...new Set([...cases, ...browserCases].map(item => item.file))]
  .filter(file => fs.existsSync(SIDECAR(file)));
if (staleSidecars.length) {
  throw new Error(`stale sidecars from an interrupted run: ${staleSidecars.join(', ')}. `
    + 'Each holds the original bytes of the file beside it. Run `node scripts/falsify-timeseries.mjs '
    + '--recover` before running again.');
}
fs.writeFileSync(LOCK, `${process.pid} ${new Date().toISOString()}\n`);

/* THE PROVISIONAL RECORD, now that this run owns the lock and is about to
   mutate files. On disk before the first breakage, so a run that is killed or
   throws leaves a record saying it started and did not finish, rather than
   yesterday's success reading green over a tree nobody has re-checked. Written
   after the lock so a refusal-to-start cannot destroy the previous record. */
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync(REPORT, JSON.stringify({
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-timeseries.mjs',
  mode: withBrowser ? 'offline and browser' : 'offline only',
  status: 'in progress: this record is provisional and is rewritten only after every breakage has been '
    + 'applied, restored and accounted for. If you are reading this, a run started and did not finish, and '
    + 'nothing here describes the current state of the tree.',
  passed: false,
}, null, 2) + '\n');

function restoreFromSidecar(reason) {
  if (activeSidecar) {
    const { file, sidecar } = activeSidecar;
    try {
      fs.copyFileSync(sidecar, file);
      fs.rmSync(sidecar, { force: true });
      process.stderr.write(`\n[falsify] ${reason}: restored ${file} from its sidecar\n`);
    } catch {
      process.stderr.write(`\n[falsify] ${reason}: COULD NOT restore ${file}; its original is at ${sidecar}\n`);
    }
    activeSidecar = null;
  }
  try { fs.rmSync(LOCK, { force: true }); } catch { /* nothing useful to do */ }
}
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) {
  process.on(signal, () => { restoreFromSidecar(signal); process.exit(130); });
}
process.on('uncaughtException', error => {
  restoreFromSidecar('uncaughtException');
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exit(1);
});
process.on('exit', () => { if (activeSidecar || fs.existsSync(LOCK)) restoreFromSidecar('exit'); });

/* A BASELINE BUILD before any breakage, so a tree that was already broken --
   by a concurrent agent editing a shared file, which has happened repeatedly in
   this environment -- is diagnosed as that rather than blamed on the first
   browser case to rebuild. Refusing to start is the honest response: the
   alternative is a report attributing someone else's syntax error to one of
   these guards. */
if (withBrowser) {
  const probe = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-ts'],
    { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
  baselineBuild = { exitCode: probe.status, succeeded: probe.status === 0 };
  if (probe.status !== 0) {
    fs.rmSync(LOCK, { force: true });
    const tail = (probe.stderr ?? '').split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 400);
    throw new Error('the tree does not build BEFORE any breakage is applied, so no browser case can be '
      + 'attributed to its own guard. This is almost always a concurrent edit to a shared file. '
      + `Build output: ${tail}`);
  }
}

const allCases = withBrowser ? [...cases, ...browserCases] : cases;
/* The whole loop sits inside a try/finally. With the evidence restore after the
   loop, a case throwing the restore-mismatch error would exit past it and leave
   the broken page's evidence on disk -- exactly what the snapshot prevents. */
try {
  for (const item of allCases) {
    const original = fs.readFileSync(item.file);
    const before = crypto.createHash('sha256').update(original).digest('hex');
    const text = original.toString('utf8');
    if (!text.includes(item.find)) {
      results.push({
        name: item.name, file: item.file, guards: item.guards, applied: false, fired: false,
        note: `the target text was not found in ${item.file}; this breakage could not be applied`,
      });
      continue;
    }
    const occurrences = text.split(item.find).length - 1;
    /* The sidecar is written and flushed BEFORE the mutation, so a kill at any
       point after this line leaves the original recoverable on disk. */
    const sidecar = SIDECAR(item.file);
    fs.writeFileSync(sidecar, original);
    activeSidecar = { file: item.file, sidecar };
    const mutated = text.replace(item.find, item.replace);
    /* Measured, not assumed. `String.replace` with a string pattern replaces
       the first occurrence only, so reporting a constant 1 would be a claim
       about a substitution nobody counted.

       The subtraction `before - after` is NOT that measurement, though it was
       for one run: several breakages wrap the found text rather than delete it
       (inserting a covering rect after a hatch, say), so the find text is still
       present afterwards and the subtraction reports 0 for a mutation that was
       applied. It read exactly like a stale find string, which is the failure
       it exists to catch, so it has to be able to tell the two apart. What is
       actually being claimed is: the text was there exactly once, and writing
       the replacement changed the file. */
    const occurrencesAfter = mutated.split(item.find).length - 1;
    const applied = mutated !== text;
    if (!applied) {
      throw new Error(`the breakage "${item.name}" left ${item.file} byte-identical, so nothing was broken `
        + 'and any verdict from the verifier that follows would be about the untouched tree.');
    }
    fs.writeFileSync(item.file, mutated, 'utf8');
    let outcome;
    try {
      if (item.rebuild) {
        /* ONE RETRY, because the dominant cause of a build failure here is not
           this breakage. The production build is shared with every other
           lesson, and a concurrent agent saving a half-written file breaks it
           for whoever happens to be building at that moment -- which has now
           killed three runs of this harness. A breakage that genuinely breaks
           the build fails both attempts; a concurrent write almost never does.
           A baseline build before the first browser case (see below) rules out
           a tree that was already broken when the run started, so a failure
           that survives the retry is attributable. */
        let built = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-ts'],
          { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
        if (built.status !== 0) {
          rebuildRetries += 1;
          built = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-ts'],
            { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
        }
        if (built.status !== 0) {
          throw new Error(`the build failed twice while applying "${item.name}". The baseline build before `
            + 'this run succeeded, so either this breakage breaks the build (in which case it is not a '
            + 'browser guard firing and the case needs rewriting) or a concurrent agent is editing a shared '
            + `file. Build output: ${(built.stderr ?? '').split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 400)}`);
        }
      }
      const [command, argv] = item.run;
      const run = spawnSync(command, argv, {
        encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
        env: { ...process.env, DIST_DIR: 'dist-ts' },
      });
      const output = `${run.stdout ?? ''}\n${run.stderr ?? ''}`;
      const failed = run.status !== 0;
      const named = item.expect.test(output);
      outcome = {
        name: item.name,
        file: item.file,
        guards: item.guards,
        verifier: `${command} ${argv.join(' ')}`,
        occurrencesPresent: occurrences,
        occurrencesRemaining: occurrencesAfter,
        mutationApplied: applied,
        exitCode: run.status,
        failedAsIntended: failed,
        namedTheGuard: named,
        fired: failed && named,
        excerpt: (output.match(item.expect) ?? [''])[0].slice(0, 200)
          || output.split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 280),
      };
    } finally {
      fs.writeFileSync(item.file, original);
    }
    const after = digest(item.file);
    outcome.restoredExactly = after === before;
    // The sidecar is removed only once the restore is VERIFIED. If the bytes do
    // not match, it stays on disk and the next run refuses to start until it is
    // dealt with, rather than quietly mutating a file that is already wrong.
    if (outcome.restoredExactly) {
      fs.rmSync(sidecar, { force: true });
      activeSidecar = null;
    } else {
      throw new Error(`${item.file} was not restored byte for byte after "${item.name}"; `
        + `its original bytes are at ${sidecar}`);
    }
    if (outcome.fired) fired += 1;
    results.push(outcome);
  }
} finally {
  /* Leave the tree building the REAL page again, not the last breakage's, and
     put every evidence artifact back the way it was found. */
  if (withBrowser) {
    /* maxBuffer, and then VERIFY. This call carried no `maxBuffer` while the
       per-case build above carried 64 MB, so it inherited Node's 1 MB default.
       A vite build prints a line per emitted asset and overruns that, and the
       rebuild then dies with ENOBUFS -- leaving THE LAST BREAKAGE'S BUILD on
       disk while the sources are correctly restored. Any browser run afterwards
       would test sabotaged bytes and say nothing. A sibling lesson hit exactly
       this.
       Restoring is not enough on its own either: the rebuild is asserted to
       succeed, and the built stylesheet is then checked to contain the scoped
       SVG selector that one of the breakages above removes, so a silently
       sabotaged build cannot be left behind as if it were clean. */
    const restored = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-ts'],
      { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
    restoringBuild = {
      exitCode: restored.status,
      succeeded: restored.status === 0,
      stderrExcerpt: (restored.stderr ?? '').split('\n').filter(Boolean).slice(-2).join(' | ').slice(0, 300),
    };
    if (restored.status === 0) {
      try {
        const cssFile = fs.readdirSync('dist-ts/assets')
          .find(name => name.startsWith('time-series-validation') && name.endsWith('.css'));
        const css = cssFile ? fs.readFileSync(`dist-ts/assets/${cssFile}`, 'utf8') : '';
        restoringBuild.builtStylesheet = cssFile ?? null;
        restoringBuild.scopedSelectorPresent = css.includes('svg.ts-diagram');
      } catch (error) {
        restoringBuild.verificationError = String(error?.message ?? error);
      }
    }
  }
  for (const [file, bytes] of Object.entries(evidenceSnapshot)) fs.writeFileSync(file, bytes);
  fs.rmSync(LOCK, { force: true });
}

const report = {
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-timeseries.mjs',
  harnessSha256: digest('scripts/falsify-timeseries.mjs'),
  mode: withBrowser ? 'offline and browser' : 'offline only',
  breakagesApplied: results.length,
  guardsThatFired: fired,
  baselineBuild,
  restoringBuild,
  rebuildRetries,
  /* Written BEFORE the inert check throws, so a run with an inert guard leaves
     `passed: false` on disk rather than no record at all. */
  passed: fired === results.length && results.length > 0
    && (!withBrowser || Boolean(restoringBuild && restoringBuild.succeeded
      && restoringBuild.scopedSelectorPresent)),
  allRestoredExactly: results.every(result => result.restoredExactly !== false),
  evidenceFilesRestored: evidenceFiles,
  evidenceScope: 'Only docs/teaching/evidence/timeseries-*.json and screenshots/timeseries-* are snapshotted '
    + 'and restored. No other lesson’s artifacts are read or written by this harness.',
  results,
  browserCasesIncluded: withBrowser ? browserCases.length : 0,
  browserCasesDeferred: withBrowser ? [] : browserCases.map(item => item.name),
  method: 'Each breakage is applied alone to a clean tree, the relevant verifier is run, the file is restored '
    + 'and the restoration checked by SHA-256. A case counts as fired only when the verifier exits non-zero '
    + 'AND its output names the guard, so a run that fails for an unrelated reason is not credited.',
  limitations: [
    'The coverage floors in the model suite are set far below its current totals, so they catch a wholesale '
      + 'loss of checking rather than the removal of one assertion. Deleting a single record() call does NOT '
      + 'trip them, and the case above neuters the counter instead. Tightening the floors to today’s '
      + 'numbers would make them fail on every legitimate addition, which is how a floor gets deleted rather '
      + 'than fixed.',
    'Browser guards need a running preview and a production build; without --browser they are listed as '
      + 'deferred rather than silently omitted.',
    'Crash safety is best-effort: on Windows a forced kill delivers no catchable signal, and a power loss '
      + 'cannot be trapped at all. In those cases the `.orig` sidecar beside the mutated file holds its '
      + 'original bytes, and the next run REFUSES TO START until the sidecar is dealt with, rather than '
      + 'mutating a file that is already wrong. `--recover` restores from it and verifies the bytes.',
    'A guard that fires here is a guard that CAN fail. It is not evidence that its domain covers every '
      + 'defect of its kind -- that is what the whole-grid sweeps in the model suite are for.',
  ],
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/timeseries-falsification.json', JSON.stringify(report, null, 2) + '\n');

if (withBrowser && !(restoringBuild && restoringBuild.succeeded && restoringBuild.scopedSelectorPresent)) {
  throw new Error('the restoring rebuild did not produce a verifiably clean build: '
    + JSON.stringify(restoringBuild) + '. The sources are restored, but dist-ts may still hold the last '
    + 'breakage\'s bytes -- rebuild before running any browser check.');
}
const inert = results.filter(result => !result.fired);
if (inert.length) {
  inert.forEach(result => console.log(`INERT ${result.name}: ${result.note ?? JSON.stringify({
    exit: result.exitCode, failed: result.failedAsIntended, named: result.namedTheGuard, excerpt: result.excerpt,
  })}`));
  throw new Error(`${inert.length} of ${results.length} breakages did not make their guard fire and name itself`);
}
console.log(`PASS: ${fired} of ${results.length} breakages made the intended guard fire and name itself; every `
  + `file was restored byte for byte, and ${evidenceFiles.length} evidence files were restored. `
  + (withBrowser
    ? `${browserCases.length} of them are browser guards, each rebuilt and run against the live preview.`
    : `${browserCases.length} browser guards are deferred and listed in the report; pass --browser to run them.`));
