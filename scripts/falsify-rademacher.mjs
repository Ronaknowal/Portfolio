// Falsification harness for the Rademacher lesson's verifiers.
//
// A guard that has never failed is a guard nobody has tested. This applies each
// breakage ALONE to a real source file, runs the verifier that is supposed to
// catch it, and requires two things: that the verifier fails, and that its
// output NAMES the guard that fired. A run that merely goes red proves the
// suite is brittle, not that it is watching the right thing.
//
// Every file is restored byte for byte afterwards, and the restoration is
// confirmed by SHA-256 before the harness will report success. A crash restores
// too: the writes happen inside a finally block.
//
// Three defect classes are only observable in a painted page. They live in
// `browserBreakages` and run under `--browser`, which rebuilds the bundle for
// each one, runs the browser verifier against a live preview, and re-captures
// the evidence against the repaired build afterwards. Without that flag they
// are neither run nor counted, and the run says so.
//
// RUN ONE AT A TIME. This mutates tracked source files; two concurrent runs
// will corrupt each other's snapshots. Do not edit any source it touches while
// it is running -- a concurrent edit is either overwritten by the restore or
// reported as `concurrent-edit`, and either way that case's verdict is void.
//
// Run: node scripts/falsify-rademacher.mjs [--fast] [--browser]
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const PYTHON = 'scratch/lesson-tools/Scripts/python.exe';
const MODELS = 'src/learn/data/rademacher-models.js';
const FIGURES = 'src/learn/components/lesson-labs/RademacherFigures.jsx';
const LABS = 'src/learn/components/lesson-labs/RademacherLabs.jsx';
const DATA = 'src/learn/data/rademacher-data.js';
const EXAMPLES = 'src/learn/data/rademacher-examples.js';
const CSS = 'src/learn/components/lesson-labs/rademacher-labs.css';
const SERVED_PROGRAM = 'public/learn-assets/rademacher/complexity_calculations.py';

const modelsVerifier = { command: 'node', args: ['scripts/verify-rademacher-models.mjs', '--no-evidence'] };
/* Every verifier is driven with `--no-evidence`.
 *
 * The harness injects defects on purpose, so any record a driven verifier
 * writes describes a tree the harness broke. The models verifier already had
 * the flag; the independent review found that the four cases driving the
 * SOURCES verifier left `rademacher-sources.json` recording "passed": false,
 * derived from a defect the harness itself put there. All three Python
 * verifiers now accept the flag and it is used here. */
const sourcesVerifier = { command: PYTHON, args: ['scripts/verify-rademacher-sources.py', '--no-evidence'] };
const dataVerifier = { command: PYTHON, args: ['scripts/verify-rademacher-data.py', '--no-evidence'] };
const examplesVerifier = { command: PYTHON, args: ['scripts/verify-rademacher-examples.py', '--no-evidence'] };

/** Each breakage names the defect it simulates, the guard that must catch it,
 *  and a fragment of that guard's own message. */
const breakages = [
  {
    name: 'threshold restrictions lose their two extreme cutoffs',
    defect: 'restricting cutoffs to observed values, which silently drops the all-negative rule',
    file: MODELS,
    find: 'const cuts = [Number.NEGATIVE_INFINITY, ...[...new Set(x)].sort((a, b) => a - b), Number.POSITIVE_INFINITY];',
    replace: 'const cuts = [...[...new Set(x)].sort((a, b) => a - b)];',
    verifier: modelsVerifier,
    expect: 'three ordered inputs give exactly these four rows',
  },
  {
    name: 'sign patterns are enumerated with the first observation varying fastest',
    defect: 'an enumeration order that still averages correctly but misaligns every packet comparison',
    file: MODELS,
    find: 'pattern[index] = (code >> (n - 1 - index)) & 1 ? 1 : -1;',
    replace: 'pattern[index] = (code >> index) & 1 ? 1 : -1;',
    verifier: modelsVerifier,
    expect: 'the same sign patterns, in the same order',
  },
  {
    name: 'the per-pattern maximum is floored at zero',
    defect: 'initialising a supremum at 0, which is wrong for any class with no non-negative option',
    file: MODELS,
    find: 'const maxima = correlations.map(scores => Math.max(...scores));',
    replace: 'const maxima = correlations.map(scores => Math.max(0, ...scores));',
    verifier: modelsVerifier,
    expect: 'singleton: the stated exact value',
  },
  {
    name: 'the absolute-value convention drops its absolute value',
    defect: 'the two conventions collapsing into one, so the lesson could not show what the other measures',
    file: MODELS,
    find: 'const maxima = patterns.map(pattern => Math.max(...rows.map(row => Math.abs(dot(pattern, row)) / n)));',
    replace: 'const maxima = patterns.map(pattern => Math.max(...rows.map(row => dot(pattern, row) / n)));',
    verifier: modelsVerifier,
    expect: 'the absolute convention on the singleton matches the packet',
  },
  {
    name: 'the drawn ramp curve stops being the applied ramp function',
    defect: 'a figure painting one shape while the lab grades against another',
    file: MODELS,
    find: "  return curvePoints(margin => rampLoss([margin], rho)[0], { from, to, samples, width, height, valueRange: [0, 1] });",
    replace: "  return curvePoints(margin => rampLoss([margin], 2 * rho)[0], { from, to, samples, width, height, valueRange: [0, 1] });",
    verifier: modelsVerifier,
    expect: 'the drawn curve equals the applied ramp',
  },
  {
    name: 'the movement verdict compares raw doubles instead of displayed values',
    defect: 'a verdict reading "changed" beside two identical printed numbers',
    file: MODELS,
    find: '  const moved = shownBefore !== shownAfter;',
    replace: '  const moved = before !== after;',
    verifier: modelsVerifier,
    expect: 'the verdict says unchanged exactly when the displayed values are equal',
  },
  {
    name: 'the kernel complexity divides by n squared',
    defect: 'a normalisation slip that leaves the shape of the answer intact',
    file: MODELS,
    find: '  const maxima = quadratics.map(value => (radius * Math.sqrt(Math.max(value, 0))) / n);',
    replace: '  const maxima = quadratics.map(value => (radius * Math.sqrt(Math.max(value, 0))) / (n * n));',
    verifier: modelsVerifier,
    // Caught by the cross-route guard, which fires first and names itself: the
    // Gram path and the vector path must agree.
    expect: 'the kernel route agrees with the vector route',
  },
  {
    name: "Massart's bound loses its factor of two",
    defect: 'a constant that is plausible, monotone in M, and wrong',
    file: MODELS,
    find: '  return (radius * Math.sqrt(2 * Math.log(distinctCount))) / n;',
    replace: '  return (radius * Math.sqrt(Math.log(distinctCount))) / n;',
    verifier: modelsVerifier,
    expect: 'the exponential-moment route agrees',
  },
  {
    name: 'the tie rule sends a zero score to the negative class',
    defect: 'a strict inequality where the lesson states a non-strict one',
    file: MODELS,
    find: 'export const predictSign = score => (score >= 0 ? 1 : -1);',
    replace: 'export const predictSign = score => (score > 0 ? 1 : -1);',
    verifier: modelsVerifier,
    expect: "the zero candidate's fit mistakes, under the score>=0 tie rule",
  },
  {
    name: 'the norm ball is drawn with two different axis scales',
    defect: 'a circle painted as an ellipse, so the supporting point stops being the farthest point along v',
    file: MODELS,
    /* Anchored on the comment that follows it. The same `project` line now
       appears in `hullLayout` as well, and the harness reported the ambiguity
       as `unanchored` rather than silently breaking the wrong function -- which
       is the behaviour a harness has to have to be worth running. */
    find: "  const project = point => ({ x: origin.x + unit * point[0], y: origin.y - unit * point[1] });\n  /* Where the supporting point's label goes.",
    replace: "  const project = point => ({ x: origin.x + unit * point[0], y: origin.y - 1.2 * unit * point[1] });\n  /* Where the supporting point's label goes.",
    verifier: modelsVerifier,
    expect: 'the two axes use one scale',
  },
  {
    name: 'the trivial ceiling is drawn at the end of the bar instead of at 1',
    defect: 'a reference line that makes every vacuous bound look as if it just reaches the ceiling',
    file: MODELS,
    find: '    ceilingX: left + scale,',
    replace: '    ceilingX: left + usable,',
    verifier: modelsVerifier,
    expect: 'a vacuous bound is drawn LONGER than the ceiling',
  },
  {
    /* The independent review's B1. Grading a comparison finer than the page
       prints returns a miss beside two identical numbers; the guard is the
       display-precision sweep over the whole enterable grid. */
    name: 'a graded comparison is tightened below the precision the page prints',
    defect: "the lesson's own named defect class: a verdict that contradicts the numbers printed beside it",
    file: MODELS,
    find: '  if (comparison.outcome === \'unchanged\') return { outcome: equal, comparison };',
    replace: '  if (Math.abs(reference - candidate) <= 1e-12) return { outcome: equal, comparison };',
    verifier: modelsVerifier,
    expect: 'gapOutcome must say "equal" exactly when the operands print the same',
  },
  {
    /* The independent review's B2. A small-multiple figure whose panels
       autoscale independently draws the same length at two sizes. */
    name: 'a small-multiple figure lets its panels autoscale independently',
    defect: 'the same unit vector drawn at half size in one panel of a figure captioned "Same lengths"',
    file: MODELS,
    find: '  const extent = sharedExtent ?? autoExtent;',
    replace: '  const extent = autoExtent;',
    verifier: modelsVerifier,
    expect: 'figure 8 panel 2 uses the shared scale',
  },
  {
    name: 'the margin histogram stops snapping its bin edges to 0 and rho',
    defect: 'a bin straddling the two values the ramp treats differently',
    file: MODELS,
    find: '  [0, rho].forEach(target => {',
    replace: '  [].forEach(target => {',
    verifier: modelsVerifier,
    expect: 'there is a bin edge exactly at 0',
  },
  {
    name: 'the selection rule accepts a candidate carrying its assessment score',
    defect: 'a rule that can read the answer it is supposed to precede',
    file: MODELS,
    find: "    if ('assessment' in candidate) throw new RangeError('selectByValidation: assessment must not be visible here');",
    replace: "    if (false) throw new RangeError('selectByValidation: assessment must not be visible here');",
    verifier: modelsVerifier,
    expect: 'a candidate carrying an assessment field',
  },
  {
    name: 'the frozen representation stops clipping',
    defect: 'a feature map whose rows are no longer bounded, so the sqrt(5) the bound uses is false',
    file: MODELS,
    find: '      Math.min(1, Math.max(-1, (value - mean[index]) / scale[index] / clipStandardDeviations))),',
    replace: '      (value - mean[index]) / scale[index] / clipStandardDeviations),',
    verifier: modelsVerifier,
    expect: 'clipped into [-1, 1]',
  },
  {
    name: 'a fitted coefficient in the generated data module is altered',
    defect: 'a hand edit to a generated file, which is exactly what "do not edit by hand" is there to prevent',
    file: DATA,
    find: '    weights: [-1.6866697123314995,',
    replace: '    weights: [-1.686669712331,',
    verifier: modelsVerifier,
    expect: 'the coefficients',
  },
  {
    name: "a recorded program output digit is altered",
    defect: 'a displayed "expected result" that the program never printed',
    file: EXAMPLES,
    find: '0.6666666666666666',
    // A one-part-in-1e13 edit slips under the verifier's relative tolerance,
    // which is correct: that tolerance exists for float association. The
    // breakage has to be a real difference to be a real defect.
    replace: '0.6000000000000000',
    verifier: modelsVerifier,
    expect: 'the program printed the threshold value this module computes',
  },
  {
    name: 'a raw control byte is injected into a lesson source',
    defect: 'a shell that ate an escape and left the byte it named',
    file: MODELS,
    find: '/* Pure models for the Rademacher-complexity lesson.',
    replace: '/* Pure models\b for the Rademacher-complexity lesson.',
    verifier: sourcesVerifier,
    expect: 'raw control byte 0x08',
  },
  {
    name: 'a backslash-u escape is written in JSX text',
    defect: 'six visible characters where a quotation mark was meant',
    file: LABS,
    find: '<h4 className="rad-question">The same calculation through inner products alone</h4>',
    replace: '<h4 className="rad-question">The same \\u201ccalculation\\u201d through inner products alone</h4>',
    verifier: sourcesVerifier,
    expect: 'in JSX text -- JSX text is not a string literal',
  },
  {
    name: 'a conditional is given two identical branches',
    defect: 'an intended phrase that got lost, leaving an expression that can never render anything',
    file: FIGURES,
    find: "maximiseFirst ? 'row average' : 'ROW AVERAGE']}",
    replace: "maximiseFirst ? 'row average' : 'row average']}",
    verifier: sourcesVerifier,
    expect: 'a conditional whose branches are identical',
  },
  {
    name: 'the SVG text rule is scoped to a figure class instead of the lesson root',
    defect: 'drawings inside investigation wrappers rendering at 16px in a 320-unit viewBox',
    file: CSS,
    find: '.rad-lesson svg.rad-drawing text {',
    replace: '.rad-figure svg.rad-drawing text {',
    verifier: sourcesVerifier,
    expect: 'scoped at the lesson root AND to this lesson\'s own drawings',
  },
  {
    /* The static half of the radical defect: the hygiene checker must refuse a
       bare descendant `svg` selector under the lesson root, which is what it
       previously passed over while reporting the text rule as correctly
       scoped. */
    name: 'a bare descendant svg selector is reintroduced under the lesson root',
    defect: 'the selector that collapses every KaTeX radical and accent on the page',
    file: CSS,
    find: '.rad-lesson svg.rad-drawing .rad-hull {',
    replace: '.rad-lesson svg .rad-hull {',
    verifier: sourcesVerifier,
    expect: 'a bare descendant `svg` selector under the lesson root',
  },
  {
    name: 'the generated data module is edited by hand',
    defect: 'a generated file drifting from the script that generates it',
    file: DATA,
    find: 'export const roles = [',
    replace: 'export const roles = [ // hand edit\n',
    verifier: dataVerifier,
    expect: 'not byte-identical',
    slow: true,
  },
  {
    name: 'the served program copy drifts from the packet program',
    defect: 'a download that is not the program whose output the page displays',
    file: SERVED_PROGRAM,
    find: '"""Exact small Rademacher calculations and bounded Monte Carlo, Python 3.12+.',
    replace: '"""Exact small Rademacher calculations and bounded Monte Carlo, Python 3.12+ (edited).',
    verifier: examplesVerifier,
    expect: 'byte-for-byte the packet program',
    slow: true,
  },
];

/** Breakages that need a painted page. Each is applied to a real source, the
 *  production bundle is REBUILT, and the browser verifier is run against the
 *  running preview. Run with `--browser`, with the preview already serving the
 *  build directory. */
const browserBreakages = [
  {
    name: 'an investigation prints its graded value before a prediction is recorded',
    defect: 'the dominant defect class: showing the answer first',
    file: LABS,
    find: '    {termsShown && bound && <>',
    replace: '    {bound && <>',
    /* Caught by the graded-STRING guard, which fires first and names both the
       investigation and the strings that leaked. The numeric sweep is the
       complementary guard, for a leaked value with no distinctive label. */
    expect: 'margin bound shows a graded quantity before any prediction',
  },
  {
    name: 'a curve is painted through a value label',
    defect: 'a polygon outline crossing text, which the shared straight-line inspector cannot see',
    file: FIGURES,
    find: '          <text className="rad-tiny" x={layout.vertexLabels[index].x} y={layout.vertexLabels[index].y}\n            textAnchor={layout.vertexLabels[index].anchor}>h{index + 1}</text>',
    replace: '          <text className="rad-tiny" x={projected.x + 6} y={projected.y - 5}>h{index + 1}</text>',
    expect: 'a curve runs through a label at 1366px',
  },
  {
    /* The defect the PAC lesson shipped and this one inherited: a bare
       descendant `svg` rule under the lesson root also matches KaTeX's inline
       SVGs, and `height: auto` collapses every radical and stretchy accent.
       Two guards must fire — the hygiene checker statically, and the browser
       measurement on the painted page. This case drives the browser one. */
    name: 'the SVG rule is unscoped again, collapsing every KaTeX radical',
    defect: 'a square root that is not drawn, so the formula states the radius squared rather than the radius',
    file: CSS,
    find: '.rad-lesson svg.rad-drawing {\n  display: block;',
    replace: '.rad-lesson svg {\n  display: block;',
    expect: 'KaTeX SVGs are under 1px tall',
  },
  {
    name: 'a CSS fill overrides a fill="none" attribute',
    defect: 'an outline shape silently becoming a filled blob while every attribute stays correct',
    file: CSS,
    find: '.rad-lesson svg.rad-drawing .rad-ball { fill: none; stroke: #746138; stroke-width: 1.4; }',
    replace: '.rad-lesson svg.rad-drawing .rad-ball { fill: #746138; stroke: #746138; stroke-width: 1.4; }',
    expect: 'a painted value disagrees with its attribute',
  },
];

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

/* Restore on a SIGNAL, not only on a thrown exception.
 *
 * `finally` covers a throw. It does not cover Ctrl-C or an out-of-memory kill,
 * and this machine has killed processes repeatedly for memory and for spend
 * limits. A kill mid-case would otherwise leave a deliberate one-line defect in
 * a tracked source file, and under --browser a broken bundle that a running
 * preview keeps serving, with nothing telling the next reader where to look.
 *
 * `inFlight` holds the one file currently mutated. The handlers put its bytes
 * back and say plainly what happened; the exit handler is the last line of
 * defence for anything that slips past both. */
let inFlight = null;
const putBack = reason => {
  if (!inFlight) return;
  const { file, original } = inFlight;
  inFlight = null;
  try {
    fs.writeFileSync(file, original);
    console.error(`\n${reason}: restored ${file} to its original bytes before exiting.`);
  } catch (error) {
    console.error(`\n${reason}: FAILED to restore ${file} (${error.message}). `
      + `Recover with: git checkout -- "${file}"`);
  }
};
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) {
  process.on(signal, () => { putBack(`Interrupted by ${signal}`); process.exit(130); });
}
process.on('exit', () => putBack('Exiting unexpectedly'));
process.on('uncaughtException', error => { putBack(`Uncaught ${error.message}`); process.exit(1); });

function run({ command, args }) {
  /* 64MB, because the default 1MB can truncate a failing verifier's output --
     and a truncated failure looks exactly like a failure that did not name its
     guard. */
  const result = spawnSync(command, args, { encoding: 'utf8', timeout: 900000, maxBuffer: 64 * 1024 * 1024 });
  return {
    status: result.status,
    output: `${result.stdout ?? ''}${result.stderr ?? ''}`,
    /* "The verifier ran and failed" and "the verifier could not be started"
       are different events and must not be reported as the same one. On a
       loaded machine a spawn can fail outright: `status` is then null and the
       output empty, which the attribution check below would otherwise read as
       a failure that did not name its guard. Two guards were reported
       misattributed that way, and both fired correctly when re-run alone. */
    spawnFailed: Boolean(result.error) || result.status === null,
    spawnError: result.error ? result.error.message : (result.signal ? `killed by ${result.signal}` : null),
  };
}

/**
 * Restore a file and say precisely what happened.
 *
 * There are two very different reasons the bytes on disk can differ from the
 * snapshot afterwards, and reporting both as "restore failed" sent me hunting
 * for a harness bug that did not exist. If the file changed between the
 * breakage being written and the restore, somebody edited it WHILE the harness
 * was running -- which is a real hazard, because the restore then overwrites
 * their edit -- and the run's verdicts for that file are not trustworthy. That
 * is worth a different sentence from "the write did not land".
 */
function restore(file, original, brokenDigest) {
  const beforeRestore = fs.existsSync(file) ? digest(file) : null;
  fs.writeFileSync(file, original);
  const after = digest(file);
  const expected = crypto.createHash('sha256').update(original).digest('hex');
  if (after === expected) {
    return beforeRestore === brokenDigest
      ? { ok: true }
      : { ok: true, concurrentEdit: true };
  }
  return { ok: false };
}

const includeSlow = !process.argv.includes('--fast');
const results = [];
let failures = 0;

// A verifier that is already red proves nothing about a breakage. Baseline first.
const baselines = new Map();
for (const verifier of [modelsVerifier, sourcesVerifier, ...(includeSlow ? [dataVerifier, examplesVerifier] : [])]) {
  const key = verifier.args[0];
  const baseline = run(verifier);
  baselines.set(key, baseline.status);
  if (baseline.status !== 0) {
    console.error(`BASELINE FAILED for ${key}; a breakage cannot be attributed while the suite is red.`);
    console.error(baseline.output.slice(-2000));
    process.exit(1);
  }
}

for (const breakage of breakages) {
  if (breakage.slow && !includeSlow) {
    results.push({ ...breakage, outcome: 'skipped', note: 'slow breakage skipped by --fast' });
    continue;
  }
  const original = fs.readFileSync(breakage.file);
  const originalDigest = crypto.createHash('sha256').update(original).digest('hex');
  const text = original.toString('utf8');
  const occurrences = text.split(breakage.find).length - 1;
  let outcome;
  let detail = '';
  let brokenDigest = null;
  let restoreReport = { ok: true };
  try {
    if (occurrences !== 1) {
      outcome = 'unanchored';
      detail = `the anchor text occurs ${occurrences} times in ${breakage.file}, so this breakage was not applied`;
      failures += 1;
    } else {
      inFlight = { file: breakage.file, original };
      fs.writeFileSync(breakage.file, text.replace(breakage.find, breakage.replace), 'utf8');
      brokenDigest = digest(breakage.file);
      /* A breakage whose `replace` equals its `find` would apply cleanly,
         change nothing, and be reported as CAUGHT by whatever else is red --
         a tautological case the harness would never notice. Both digests are
         already in hand; comparing them is the whole check. */
      if (brokenDigest === originalDigest) {
        throw new Error(`the breakage "${breakage.name}" did not change ${breakage.file}: `
          + 'its replacement text is identical to what it replaces, so this case asserts nothing');
      }
      const result = run(breakage.verifier);
      if (result.status === 0) {
        outcome = 'survived';
        detail = 'the verifier still passed, so the guard for this defect is inert';
        failures += 1;
      } else if (!result.output.includes(breakage.expect)) {
        outcome = 'misattributed';
        detail = `the verifier failed but did not name its guard; expected to see "${breakage.expect}". `
          + `First assertion: ${(result.output.split('\n').find(row => row.includes('Error')) ?? '').trim().slice(0, 200)}`;
        failures += 1;
      } else {
        outcome = 'caught';
        detail = breakage.expect;
      }
    }
  } finally {
    restoreReport = restore(breakage.file, original, brokenDigest);
    inFlight = null;
  }
  if (!restoreReport.ok) {
    outcome = 'restore-failed';
    detail = `${breakage.file} was NOT restored byte for byte`;
    failures += 1;
  } else if (restoreReport.concurrentEdit && brokenDigest !== null) {
    outcome = 'concurrent-edit';
    detail = `${breakage.file} was modified by something else while this breakage was applied; the restore put `
      + 'the snapshot back, so that edit is gone and this verdict is not trustworthy. Re-run with no other work '
      + 'touching these files.';
    failures += 1;
  }
  results.push({
    name: breakage.name,
    defect: breakage.defect,
    file: breakage.file,
    verifier: breakage.verifier.args[0],
    outcome,
    guardNamed: outcome === 'caught' ? breakage.expect : null,
    detail,
    restoredByteForByte: restoreReport.ok,
  });
  const mark = outcome === 'caught' ? 'CAUGHT' : outcome.toUpperCase();
  console.log(`${mark.padEnd(14)} ${breakage.name}`);
  if (outcome !== 'caught') console.log(`               ${detail}`);
}

/* The browser pass. Each breakage needs a production rebuild before the page
 * can show the defect, so these are opt-in: `--browser`, with a preview already
 * serving DIST_DIR. The build is restored and rebuilt afterwards, so the
 * preview is left serving correct bytes. */
const browserResults = [];
if (process.argv.includes('--browser')) {
  const distDir = process.env.DIST_DIR || 'dist-rad';
  const build = () => spawnSync('npx', ['vite', 'build', '--outDir', distDir],
    { encoding: 'utf8', timeout: 900000, shell: true });
  const browserVerifier = { command: 'node', args: ['scripts/verify-rademacher-browser.cjs'] };
  const baseline = run(browserVerifier);
  if (baseline.status !== 0) {
    console.error('BASELINE FAILED for the browser verifier; a breakage cannot be attributed while it is red.');
    console.error(baseline.output.slice(-2000));
    process.exit(1);
  }
  for (const breakage of browserBreakages) {
    const original = fs.readFileSync(breakage.file);
    const text = original.toString('utf8');
    let outcome;
    let detail = '';
    let brokenDigest = null;
    let browserRestore = { ok: true };
    try {
      const occurrences = text.split(breakage.find).length - 1;
      if (occurrences !== 1) {
        outcome = 'unanchored';
        detail = `the anchor text occurs ${occurrences} times in ${breakage.file}`;
        failures += 1;
      } else {
        inFlight = { file: breakage.file, original };
        fs.writeFileSync(breakage.file, text.replace(breakage.find, breakage.replace), 'utf8');
        brokenDigest = digest(breakage.file);
        if (brokenDigest === crypto.createHash('sha256').update(original).digest('hex')) {
          throw new Error(`the breakage "${breakage.name}" did not change ${breakage.file}`);
        }
        build();
        const result = run(browserVerifier);
        if (result.status === 0) {
          outcome = 'survived';
          detail = 'the verifier still passed, so the guard for this defect is inert';
          failures += 1;
        } else if (!result.output.includes(breakage.expect)) {
          outcome = 'misattributed';
          detail = `failed without naming its guard; expected "${breakage.expect}". First assertion: `
            + `${(result.output.split('\n').find(row => row.includes('Error')) ?? '').trim().slice(0, 200)}`;
          failures += 1;
        } else {
          outcome = 'caught';
          detail = breakage.expect;
        }
      }
    } finally {
      browserRestore = restore(breakage.file, original, brokenDigest);
      inFlight = null;
      build();
    }
    if (!browserRestore.ok) {
      outcome = 'restore-failed';
      detail = `${breakage.file} was NOT restored byte for byte`;
      failures += 1;
    } else if (browserRestore.concurrentEdit && brokenDigest !== null) {
      outcome = 'concurrent-edit';
      detail = `${breakage.file} was edited by something else during this breakage; that edit has been overwritten `
        + 'by the restore and this verdict is not trustworthy.';
      failures += 1;
    }
    browserResults.push({
      name: breakage.name, defect: breakage.defect, file: breakage.file,
      verifier: 'scripts/verify-rademacher-browser.cjs', outcome, detail,
      guardNamed: outcome === 'caught' ? breakage.expect : null,
      restoredByteForByte: browserRestore.ok,
    });
    console.log(`${(outcome === 'caught' ? 'CAUGHT' : outcome.toUpperCase()).padEnd(14)} ${breakage.name} [browser]`);
    if (outcome !== 'caught') console.log(`               ${detail}`);
  }
  /* Leave the evidence telling the truth.
   *
   * Each browser breakage RUNS the browser verifier, which writes screenshots
   * and rademacher-browser.json. The last thing written during this section is
   * therefore a capture of a deliberately broken page. Sources and the build
   * are restored above; this restores the evidence too, by re-running the
   * verifier once against the repaired build. Without it the harness would
   * quietly leave behind exactly the stale, wrong-state artefacts the
   * screenshot digests exist to catch. */
  /* Only THIS lesson's captures. The screenshots directory is shared with every
     other lesson in the repository, so a recursive delete here would destroy
     other people's evidence. */
  const shots = 'docs/teaching/evidence/screenshots';
  if (fs.existsSync(shots)) {
    for (const name of fs.readdirSync(shots)) {
      if (name.startsWith('rademacher-') && name.endsWith('.png')) fs.rmSync(`${shots}/${name}`);
    }
  }
  const clean = run(browserVerifier);
  if (clean.status !== 0) {
    console.error('The post-harness clean browser run FAILED; the evidence on disk is not trustworthy.');
    console.error(clean.output.slice(-2000));
    failures += 1;
  } else {
    console.log('RESTORED       evidence re-captured against the repaired build');
  }
}

const caught = results.filter(entry => entry.outcome === 'caught').length;
const skipped = results.filter(entry => entry.outcome === 'skipped').length;

const evidence = {
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-rademacher.mjs',
  harnessSha256: digest('scripts/falsify-rademacher.mjs'),
  mode: includeSlow ? 'full' : 'fast',
  breakagesApplied: results.length - skipped,
  caught,
  skipped,
  /* The headline "N of M" must be recoverable from this file alone. `caught`
     and `breakagesApplied` exclude the browser pass, so a reader taking their
     ratio silently dropped the three browser classes. */
  browserCaught: browserResults.filter(entry => entry.outcome === 'caught').length,
  totalApplied: results.length - skipped + browserResults.length,
  totalCaught: caught + browserResults.filter(entry => entry.outcome === 'caught').length,
  failures,
  results,
  browserBreakagesApplied: browserResults.length,
  browserResults,
  method: 'Each breakage is applied ALONE to a real source file, the verifier that should catch it is run, and the '
    + 'run must both fail and print the guard\'s own message. Every file is restored byte for byte and the '
    + 'restoration is confirmed by SHA-256. A baseline run of each verifier precedes the whole harness, so a '
    + 'breakage cannot be credited to an already-red suite.',
  limitations: [
    'Three defect classes are only observable in a painted page. They are applied under --browser, which rebuilds '
      + 'the bundle for each one and runs the browser verifier against a live preview. Without that flag they are '
      + 'neither run nor counted, and the run says so.',
    'A caught breakage shows the guard fires for that defect. It does not show the guard fires for every defect '
      + 'of that class.',
  ],
  passed: failures === 0,
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/rademacher-falsification.json', `${JSON.stringify(evidence, null, 2)}\n`);

console.log('');
if (!process.argv.includes('--browser')) {
  for (const entry of browserBreakages) {
    console.log(`NOT RUN        ${entry.name} — needs --browser and a live preview`);
  }
}
const browserCaught = browserResults.filter(entry => entry.outcome === 'caught').length;
const applied = results.length - skipped + browserResults.length;
if (failures) {
  console.error(`\nFAIL: ${failures} of ${applied} breakages were not caught and named.`);
  process.exitCode = 1;
} else {
  console.log(`\nPASS: ${caught + browserCaught} of ${applied} breakages applied alone, each caught by its intended `
    + 'verifier with that guard naming itself, every file restored byte for byte'
    + (browserResults.length
      ? `, including ${browserCaught} browser-only classes rebuilt and checked against a live preview.`
      : `. ${browserBreakages.length} browser-only classes were not run: pass --browser with a preview serving DIST_DIR.`));
}
