/* Falsification harness for the Calibration & Conformal Prediction lesson.
 *
 * A guard that has never been seen to fail is not evidence. This applies one
 * breakage at a time to a real source file, runs the verifier that is supposed
 * to catch it, records whether it went red AND which assertion it cited, then
 * restores the file and checks the restore is byte-identical by SHA-256.
 *
 * A breakage that leaves a verifier green is reported as a SURVIVOR, which means
 * the guard for it does not work — not that the code is fine.
 *
 * It lives in `scripts/` rather than in `scratch/` on purpose. A harness in a
 * gitignored directory cannot be re-run from a clean checkout, and a check
 * nobody else can run has been reported as passing in this repository before.
 *
 * The run finishes by re-running every verifier on the restored tree, because
 * the source-hygiene checker writes its evidence file before it fails: without
 * that final pass, a falsification run would leave `passed: false` on disk.
 *
 * Run:
 *   node scripts/falsify-calibration.mjs                 (models, data, examples, sources)
 *   INCLUDE_BROWSER=1 node scripts/falsify-calibration.mjs   (adds the browser layer)
 */
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';

const MODELS = 'src/learn/data/calibration-models.js';
const TOPIC = 'src/learn/data/topics/calibration-conformal-prediction.jsx';
const LABS = 'src/learn/components/lesson-labs/CalibrationLabs.jsx';
const CSS = 'src/learn/components/lesson-labs/calibration-labs.css';
const DATA_VERIFIER = 'scripts/verify-calibration-data.py';
const SERVED_PROGRAM = 'public/learn-assets/calibration/reliability_bins.py';
const SERVED_DATA = 'public/learn-assets/calibration/banknote-subset.csv';
const PYTHON = 'scratch/lesson-tools/Scripts/python.exe';
const DIST = process.env.DIST_DIR || 'dist-cal';
const PREVIEW = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4193';
const PLAYWRIGHT = process.env.PLAYWRIGHT_PACKAGE
  || 'C:/Users/ronak/projects/career-ops/node_modules/playwright';
const includeBrowser = process.env.INCLUDE_BROWSER === '1';

const runners = {
  models: () => execFileSync('node', ['scripts/verify-calibration-models.mjs', '--no-evidence'],
    { encoding: 'utf8', stdio: 'pipe' }),
  data: () => execFileSync(PYTHON, [DATA_VERIFIER], { encoding: 'utf8', stdio: 'pipe' }),
  examples: () => execFileSync(PYTHON, ['scripts/verify-calibration-examples.py'],
    { encoding: 'utf8', stdio: 'pipe' }),
  sources: () => execFileSync(PYTHON, ['scripts/verify-calibration-sources.py'],
    { encoding: 'utf8', stdio: 'pipe' }),
  /* The browser layer. None of the runners above can reach a defect in the
     paint or in what is on screen before a prediction is recorded. Each of
     these rebuilds the production bundle over the running preview — vite
     preview reads from disk per request, so no restart is needed — and then
     runs the real browser verifier, every case, not a reduced copy. */
  browser: () => {
    execFileSync('npx', ['vite', 'build', '--outDir', DIST, '--logLevel', 'error'],
      { encoding: 'utf8', stdio: 'pipe', shell: true });
    return execFileSync('node', ['scripts/verify-calibration-browser.cjs'], {
      encoding: 'utf8', stdio: 'pipe',
      env: { ...process.env, DIST_DIR: DIST, LEARNING_BASE_URL: PREVIEW, PLAYWRIGHT_PACKAGE: PLAYWRIGHT },
    });
  },
};

/** Each breakage names the guard it is supposed to trip, so a red result that
 *  cites a different assertion is reported and can be judged. */
const breakages = [
  {
    name: 'the conformal rank computed with ordinary floating point',
    file: MODELS, runner: 'models',
    from: '  const { numerator, denominator } = decimalParts(alpha, \'alpha\');\n'
      + '  const scaled = (n + 1) * (denominator - numerator);\n'
      + '  return Math.floor((scaled + denominator - 1) / denominator);',
    to: '  return Math.ceil((n + 1) * (1 - alpha));',
    expects: 'the integer scan',
  },
  {
    name: 'the rank clipped to n instead of returning infinity',
    file: MODELS, runner: 'models',
    from: '  if (k > scores.length) {\n    return {\n      kind: quantityKinds.calibration.key,\n'
      + '      k, n: scores.length, q: Infinity, finite: false,',
    to: '  if (false) {\n    return {\n      kind: quantityKinds.calibration.key,\n'
      + '      k, n: scores.length, q: Infinity, finite: false,',
    expects: 'unbounded',
  },
  {
    name: 'the set comparison made strict, dropping a score that equals the threshold',
    file: MODELS, runner: 'models',
    from: '  const included = scores.map(score => score <= q);',
    to: '  const included = scores.map(score => score < q);',
    expects: 'weak comparison',
  },
  {
    name: 'a forecast of exactly 1 dropped out of the last bin',
    file: MODELS, runner: 'models',
    from: '  return Math.min(above - 1, edges.length - 2);',
    to: '  return above - 1;',
    expects: 'last bin',
  },
  {
    name: 'an empty bin reporting an observed fraction of zero',
    file: MODELS, runner: 'models',
    from: '        meanP: null, fractionPositive: null, gap: null, members: [],',
    to: '        meanP: 0, fractionPositive: 0, gap: 0, members: [],',
    expects: 'no observed fraction',
  },
  {
    name: 'pool-adjacent-violators merging on ties as well as violations',
    file: MODELS, runner: 'models',
    from: '      if (left.total / left.weight <= right.total / right.weight) break;',
    to: '      if (left.total / left.weight < right.total / right.weight) break;',
    expects: 'violate the required order',
  },
  {
    name: 'a merge taking the average of two block means instead of the combined counts',
    file: MODELS, runner: 'models',
    from: '        start: left.start, end: right.end, total: left.total + right.total, weight: left.weight + right.weight,',
    to: '        start: left.start, end: right.end,\n'
      + '        total: (left.total / left.weight + right.total / right.weight)\n'
      + '          * (left.weight + right.weight) / 2,\n'
      + '        weight: left.weight + right.weight,',
    expects: 'max-min',
  },
  {
    name: 'the grading rule treating a quantity with no value as zero',
    file: MODELS, runner: 'models',
    from: "  if (after === null || after === undefined) return 'undefined';",
    to: '  if (after === null || after === undefined) after = 0;',
    expects: 'has no width to compare',
  },
  {
    name: 'the graded display rounded to six decimals, so a real change prints as no change',
    file: MODELS, runner: 'models',
    from: 'export const gradedDisplayDigits = 12;',
    to: 'export const gradedDisplayDigits = 6;',
    expects: 'the verdict and the printed number disagree',
  },
  {
    name: 'an empty bin given a minimum count-bar height',
    file: MODELS, runner: 'models',
    from: '      height: largest > 0 ? (bin.count / largest) * railHeight : 0,',
    to: '      height: largest > 0 ? Math.max((bin.count / largest) * railHeight, 2) : 2,',
    expects: 'proportional to its count',
  },
  {
    name: 'an interval segment given a minimum drawn length',
    file: MODELS, runner: 'models',
    from: '      x1: x(row.lower), x2: x(row.upper), target: x(row.y),',
    to: '      x1: x(row.lower), x2: Math.max(x(row.upper), x(row.lower) + 3), target: x(row.y),',
    expects: 'no offset',
  },
  {
    name: 'an unbounded threshold drawn as a point at the right-hand edge',
    file: MODELS, runner: 'models',
    from: '    markerX: threshold.finite ? x(threshold.q) : null,',
    to: '    markerX: threshold.finite ? x(threshold.q) : box.width - box.right,',
    expects: 'not be drawn as a point',
  },
  {
    name: 'temperature multiplying the logits instead of dividing them',
    file: MODELS, runner: 'models',
    from: '  const scaled = logits.map(value => value / temperature);',
    to: '  const scaled = logits.map(value => value * temperature);',
    expects: 'softmax at T',
  },
  {
    name: 'the reliability figure claiming an overestimate sits ABOVE the diagonal',
    file: TOPIC, runner: 'models',
    from: 'is <strong>below</strong> the diagonal: the\n      positive probability is overestimated.',
    to: 'is <strong>above</strong> the diagonal: the\n      positive probability is overestimated.',
    expects: 'BELOW the diagonal',
  },
  {
    name: 'an investigation losing its gate, so a graded readout renders on first paint',
    file: LABS, runner: 'models',
    from: '    {revealed && <div className="cal-graded">\n'
      + '      <ReliabilityPlot geometry={draftGeometry} report={draftReport}',
    to: '    {true && <div className="cal-ungated">\n'
      + '      <ReliabilityPlot geometry={draftGeometry} report={draftReport}',
    expects: 'no gated readout',
  },
  {
    name: 'the banknote role boundary moved, so eighty rows change job in silence',
    file: DATA_VERIFIER, runner: 'data',
    from: '    train, probability_calibration = np.arange(240), np.arange(240, 320)',
    to: '    train, probability_calibration = np.arange(240), np.arange(240, 300)',
    expects: 'do not partition',
  },
  {
    name: 'the independent rank scan accepting a rank one place short',
    file: DATA_VERIFIER, runner: 'data',
    from: '        if Fraction(j, n + 1) >= target:',
    to: '        if Fraction(j, n + 1) > target:',
    expects: 'rank',
  },
  {
    name: 'a hand edit to a served program that the page displays',
    file: SERVED_PROGRAM, runner: 'examples',
    from: '    boundary = reliability([0, .5, 1], [0, 1, 1], [0, .5, 1])',
    to: '    boundary = reliability([0, .5, 1], [0, 1, 1], [0, .25, 1])',
    expects: 'differs from a fresh assembly',
  },
  {
    name: 'a single changed byte in the served banknote data',
    file: SERVED_DATA, runner: 'examples',
    from: '479,pool,-1.7781,0.8546,7.1303,0.027572,0',
    to: '479,pool,-1.7782,0.8546,7.1303,0.027572,0',
    expects: 'byte-identical',
  },
  {
    name: 'a JSX conditional whose two branches are the same',
    file: LABS, runner: 'sources',
    from: "const TOLERANCE_NOTE = 'unchanged means within 10⁻¹²';",
    to: "const TOLERANCE_NOTE = true ? 'unchanged means within' : 'unchanged means within';",
    expects: 'identical',
  },
  {
    name: 'a raw control byte where a shell ate an escape',
    file: LABS, runner: 'sources',
    from: 'const TOLERANCE_NOTE =',
    to: 'const TOLERANCE_NOTE\u0008 =',
    expects: 'control byte',
  },
  {
    name: 'a blanket stylesheet fill that would beat every presentation attribute',
    file: CSS, runner: 'sources',
    from: '.cal-lesson svg.cal-plot .cal-axis { fill: none; stroke: #6f7b76; stroke-width: 1.2; }',
    to: '.cal-lesson svg.cal-plot .cal-axis { fill: none; stroke: #6f7b76; stroke-width: 1.2; }\n'
      + '.cal-lesson svg.cal-plot rect { fill: #333; }',
    expects: 'bare element type',
  },
  {
    name: 'the wide-accent display block unwrapped onto one line',
    file: TOPIC, runner: 'sources',
    from: "<MathBlock>{'\\\\begin{gathered}\\\\widehat{\\\\mathrm{ECE}}\\\\\\\\[4pt]=\\\\sum_b\\\\frac{n_b}{n}\\\\left|\\\\bar y_b-\\\\bar p_b\\\\right|.\\\\end{gathered}'}</MathBlock>",
    to: "<MathBlock>{'\\\\widehat{\\\\mathrm{ECE}}=\\\\sum_b\\\\frac{n_b}{n}\\\\left|\\\\bar y_b-\\\\bar p_b\\\\right|.'}</MathBlock>",
    expects: 'wide accent',
  },

  /* ---------------------------------------------------------------------
     Added by the guard audit. Each of these four was applied BEFORE the
     guard was widened and survived; the entries stay so the widening keeps
     having to hold. */
  {
    /* The escape scan allows any alphanumeric after a backslash, because both
       languages define plenty of them — so the eaten backslash it exists to
       catch, `\\alpha` becoming `\alpha`, passed it. */
    name: 'a shell-eaten backslash in a plain math literal',
    file: TOPIC, runner: 'sources',
    from: "<Math>{'\\\\mathbb E[Y\\\\mid P]'}</Math> means",
    to: "<Math>{'\\mathbb E[Y\\\\mid P]'}</Math> means",
    expects: 'odd run',
  },
  {
    /* The same defect in a literal that is CONCATENATED with a value. The
       first version of the eaten-backslash guard matched whole `<Math>{'…'}`
       elements and so saw 136 of 169: these 33 were outside its domain. */
    name: 'a shell-eaten backslash in a concatenated math literal',
    file: TOPIC, runner: 'sources',
    from: "<Math>{'k=\\\\lceil 81\\\\times.9\\\\rceil=' + classification.rank}",
    to: "<Math>{'k=\\lceil 81\\\\times.9\\\\rceil=' + classification.rank}",
    expects: 'odd run',
  },
  {
    /* The subject set of the hygiene checker cut down. Its headline count was
       reported and never floored, so a run over a third of the files printed
       PASS with a smaller number nobody reads. Four of the five verifiers had
       their headline counts unfloored; this breakage is the falsification for
       that whole class. */
    name: 'the hygiene checker file list truncated, so two thirds of the corpus is never read',
    file: 'scripts/verify-calibration-sources.py', runner: 'sources',
    from: '    for relative in FILES:',
    to: '    for relative in FILES[:6]:',
    expects: 'source-hygiene checks ran',
  },
  {
    /* The unscoped `svg` selector put back. KaTeX renders accents, radicals
       and stretchy delimiters as <svg> under the same root and sizes them with
       `height: inherit`; a bare `.cal-lesson svg { height: auto }` has equal
       specificity and loads later, so it wins. A sibling lesson lost every
       radical that way. Here it drew the lesson's one wide accent at 9.078px
       against KaTeX's 4.641px. */
    name: 'the stylesheet rule unscoped from `svg.cal-plot` back to a bare `svg`',
    file: CSS, runner: 'sources',
    from: '.cal-lesson svg.cal-plot {\n  display: block;',
    to: '.cal-lesson svg {\n  display: block;',
    expects: 'without naming this lesson',
  },
  {
    /* The class every scoped rule names, removed from one figure. */
    name: 'an <svg> drawn without the class every scoped rule names',
    file: 'src/learn/components/lesson-labs/CalibrationShared.jsx', runner: 'sources',
    from: 'className="cal-plot"',
    to: 'className="cal-figure-svg"',
    expects: 'without the `cal-plot` class',
  },
  {
    /* A held-out value copied into an investigation fixture — the exact thing
       the isolation check exists for. It survived the original check, whose
       filter (`|value| > 2 && not an integer`) no fixture number could
       satisfy, and which excluded every probability by magnitude.

       Two earlier versions of this breakage put the value into `residuals` and
       into `calibrationScores`, and both were caught first by the control's
       range check and by the fixture precision check — defence in depth, but
       neither exercised the assertion under test. `floatingBoundary` already
       holds a full-precision probability, so a leak there reaches the guard
       itself. */
    name: 'a held-out test probability copied into an investigation fixture',
    file: MODELS, runner: 'models',
    from: '  floatingBoundary: { positiveProbability: 103 / 240 },',
    to: '  floatingBoundary: { positiveProbability: 0.7182613044190139 },',
    expects: 'held-out assessment value',
  },
];

const browserBreakages = [
  {
    /* The original defect class, exactly: an ordinary stylesheet rule beating a
       presentation attribute. No `!important` — the point is that it does not
       need one. The interval segments carry `stroke-width` as an attribute so
       there is something for the paint to be compared against; when the width
       lived only in the stylesheet, this breakage had nothing to contradict and
       went unnoticed. */
    name: 'a stylesheet stroke width overriding every drawn interval segment',
    file: CSS, runner: 'browser',
    from: '.cal-lesson svg.cal-plot .cal-interval { fill: none; stroke: #4a5f54; stroke-linecap: butt; }',
    to: '.cal-lesson svg.cal-plot .cal-interval { fill: none; stroke: #4a5f54; stroke-linecap: butt; }\n'
      + '.cal-lesson svg.cal-plot line.cal-interval { stroke-width: 1; }',
    expects: 'stroke-width',
  },
  {
    /* The verdict comparing two DIFFERENT quantities and naming them both by
       the second one. Found by reading a screenshot: the AUC verdict quoted a
       number that had never been an AUC, because the baseline was read under
       whichever quantity happened to be applied rather than the one being
       asked about. The outcome was still right in that instance, which is why
       nothing else caught it. */
    name: 'the graded baseline read under the applied quantity instead of the one being asked',
    file: LABS, runner: 'browser',
    from: 'const before = cardQuantity(current, next.target);',
    to: 'const before = cardQuantity(current, current.target);',
    expects: 'does not quote the applied state',
  },
  {
    /* A CSS fill beats a fill="none" attribute, which is how a band vanished
       while every model assertion stayed green. */
    name: 'a CSS fill turning the dashed diagonal into a filled shape',
    file: CSS, runner: 'browser',
    from: '.cal-lesson svg.cal-plot .cal-diagonal { fill: none; stroke: #7f8a85; stroke-width: 1.2; stroke-dasharray: 5 4; }',
    to: '.cal-lesson svg.cal-plot .cal-diagonal { fill: #e7b94a; stroke: #7f8a85; stroke-width: 1.2; stroke-dasharray: 5 4; }',
    expects: 'paints a fill',
  },
  {
    /* The dominant defect class: the graded number on screen before the
       prediction is recorded. */
    name: 'the graded ECE printed in the baseline caption, before any commitment',
    file: LABS, runner: 'browser',
    from: '          Applied state: {QUANTITY_LABELS[draft.target]} is {\' \'}',
    to: '          Draft {gradedText(draftReport.ece)}. Applied state: {QUANTITY_LABELS[draft.target]} is {\' \'}',
    expects: 'before a prediction was recorded',
  },
  {
    /* The lesson-root scoping rule. A figure-class selector misses a diagram
       inside an investigation wrapper, which then renders at 16px in a
       300-unit viewBox. */
    /* Narrowed to the investigations, so every inline FIGURE loses the
       lesson-root text rule and falls back to the browser default family and
       size inside a 300-unit viewBox. Aimed at `.cal-figure` first, this
       breakage survived — every SVG in this lesson happens to sit inside a
       `.cal-figure`, so that selector still reached all of them. The hazard is
       real; the arrow had to be pointed at a gap this lesson actually has. */
    name: 'the SVG text rule narrowed so the inline figures stop matching it',
    file: CSS, runner: 'browser',
    from: '.cal-lesson svg.cal-plot text {',
    to: '.cal-investigation svg text {',
    expects: 'declared font',
  },
  {
    /* The same unscoped selector, judged in the browser rather than in the
       source. This is the assertion that measures the accent itself: a KaTeX
       SVG must be exactly as tall as the box KaTeX gave it. */
    name: 'the unscoped `svg` rule resizing the KaTeX accent SVG',
    file: CSS, runner: 'browser',
    from: '.cal-lesson svg.cal-plot {\n  display: block;',
    to: '.cal-lesson svg {\n  display: block;',
    expects: 'KaTeX SVG in',
  },
  {
    /* Added by the guard audit. Every layout assertion in the browser verifier
       is `deepEqual(findings, [])`, which is green both when the page is right
       and when the guard found nothing to look at. Renaming the class the
       table guard queries is the second of those: before the subject census
       was added this breakage left the run fully green while the tables lost
       their scroll container entirely. */
    name: 'the table container class renamed, so the clipped-cell guard has nothing to inspect',
    file: 'src/learn/components/lesson-labs/CalibrationShared.jsx', runner: 'browser',
    from: '<div className="cal-table-scroll" role="region"',
    to: '<div className="cal-table-box" role="region"',
    expects: 'subject set has drifted',
  },
];

if (includeBrowser) breakages.push(...browserBreakages);

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const results = [];
let survivors = 0;

for (const breakage of breakages) {
  const original = fs.readFileSync(breakage.file);
  const before = digest(breakage.file);
  let applied = false;
  try {
    const text = original.toString('utf8');
    const index = text.indexOf(breakage.from);
    if (index === -1) {
      results.push({ name: breakage.name, runner: breakage.runner, status: 'NOT APPLIED',
        detail: 'the target text is no longer in the file' });
      survivors += 1;
      continue;
    }
    const broken = text.slice(0, index) + breakage.to + text.slice(index + breakage.from.length);
    fs.writeFileSync(breakage.file, broken, 'utf8');
    applied = true;
    let status = 'SURVIVED';
    let detail = '';
    try {
      runners[breakage.runner]();
      survivors += 1;
    } catch (error) {
      status = 'CAUGHT';
      const output = `${error.stdout ?? ''}${error.stderr ?? ''}${error.message ?? ''}`;
      const line = output.split('\n').map(entry => entry.trim())
        .find(entry => /AssertionError|FAIL|SystemExit|Error|only/.test(entry) && entry.length > 12)
        ?? output.split('\n').find(entry => entry.trim().length > 0) ?? '';
      detail = line.slice(0, 200);
    }
    const cited = detail.toLowerCase().includes(breakage.expects.toLowerCase());
    results.push({ name: breakage.name, runner: breakage.runner, status, detail, cited });
  } finally {
    if (applied) fs.writeFileSync(breakage.file, original);
    const after = digest(breakage.file);
    if (after !== before) {
      results.push({ name: `${breakage.name} — RESTORE FAILED`, status: 'RESTORE FAILED',
        detail: `${before} → ${after}` });
      survivors += 1;
    }
  }
}

console.log('breakage'.padEnd(74), 'runner'.padEnd(10), 'result');
console.log('-'.repeat(124));
for (const result of results) {
  console.log(result.name.slice(0, 72).padEnd(74), String(result.runner ?? '').padEnd(10),
    `${result.status}${result.status === 'CAUGHT' && !result.cited ? ' (cited a different guard)' : ''}`);
  if (result.detail) console.log(' '.repeat(4), result.detail);
}
console.log('-'.repeat(124));
const caught = results.filter(result => result.status === 'CAUGHT').length;
const citedRight = results.filter(result => result.status === 'CAUGHT' && result.cited).length;
console.log(`${caught} of ${breakages.length} breakages were caught; ${citedRight} cited the guard they were `
  + 'aimed at.');
for (const layer of ['models', 'data', 'examples', 'sources', 'browser']) {
  const inLayer = results.filter(result => result.runner === layer);
  if (!inLayer.length) continue;
  console.log(`  ${layer.padEnd(9)} ${inLayer.filter(entry => entry.status === 'CAUGHT').length} of ${inLayer.length} caught`);
}

/* The source-hygiene checker writes its evidence before it fails, so a run that
   ended on a red source case would leave `passed: false` on disk. Restore the
   record by re-running every verifier on the restored tree. */
const finalRuns = {};
for (const layer of includeBrowser ? ['sources', 'examples', 'data', 'browser'] : ['sources', 'examples', 'data']) {
  try {
    finalRuns[layer] = runners[layer]().trim().split('\n').pop();
  } catch (error) {
    finalRuns[layer] = `FAILED: ${(error.stdout ?? '') + (error.stderr ?? '')}`.slice(0, 200);
    survivors += 1;
  }
}
try {
  finalRuns.models = execFileSync('node', ['scripts/verify-calibration-models.mjs'],
    { encoding: 'utf8', stdio: 'pipe' }).trim().split('\n').pop();
} catch (error) {
  finalRuns.models = `FAILED: ${(error.stdout ?? '') + (error.stderr ?? '')}`.slice(0, 200);
  survivors += 1;
}
console.log('\nRestored tree, every verifier re-run:');
for (const [layer, line] of Object.entries(finalRuns)) console.log(`  ${layer.padEnd(9)} ${line}`);

fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/calibration-falsification.json', JSON.stringify({
  ranAt: new Date().toISOString(),
  harness: 'scripts/falsify-calibration.mjs',
  harnessSha256: crypto.createHash('sha256').update(fs.readFileSync('scripts/falsify-calibration.mjs')).digest('hex'),
  includedBrowserLayer: includeBrowser,
  breakages: breakages.length,
  caught,
  citedExpectedGuard: citedRight,
  survivors,
  results,
  restoredTreeVerifierOutput: finalRuns,
  scope: 'Each breakage is applied alone to a real source file, the verifier that should catch it is run for '
    + 'real, the file is restored and the restore is checked byte-identical by SHA-256. A breakage that leaves '
    + 'its verifier green is a SURVIVOR and means the guard does not work.',
  limitations: [
    'The browser layer runs only with INCLUDE_BROWSER=1 against a running preview; without it those four '
      + 'breakages are not applied at all and are not counted.',
    'A caught breakage that cites a different assertion is reported as such rather than counted as a clean '
      + 'catch; the citation is a substring match on the failure text.',
  ],
  passed: survivors === 0,
}, null, 2) + '\n');
process.exit(survivors === 0 ? 0 : 1);
