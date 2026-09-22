// Falsification harness for the PAC/VC lesson's guards.
//
// A passing verifier proves nothing about a guard that cannot fail. Every inert
// guard found in this repository so far was correct within a domain that
// excluded the defect: a leak check matching values surrounded by spaces when
// the DOM renders them glued to their neighbours, a clearance check sampling a
// curve the browser does not paint, and hundreds of assertions comparing a
// function with itself inside an advertised grid. So each guard here is shown
// to fire, by breaking exactly the thing it claims to watch.
//
// The contract this harness keeps:
//
//   * one breakage at a time, applied to a clean tree;
//   * the file is restored byte for byte afterwards and the restoration is
//     verified by SHA-256, whether the case passed or failed;
//   * the verifier must EXIT NON-ZERO **and** its output must name the guard,
//     so a run that fails for an unrelated reason does not count as a firing;
//   * every evidence file the run could touch is snapshotted first and restored
//     in a `finally`, so falsifying a guard never rewrites the record it is
//     falsifying -- not even when a case throws;
//   * the pre-mutation bytes go to a `.orig` sidecar on disk BEFORE the
//     mutation and are removed only after a verified restore, with signal and
//     exception handlers that restore from it, so a kill mid-case cannot leave
//     a breakage applied with nothing on disk saying so;
//   * a lock file refuses a second concurrent instance. Two of them once left
//     `inset = 0` in the model layer and an inert guard reading as correct.
//
// Browser guards are not falsifiable without a running preview and are deferred
// to phase C. Passing `--browser` adds those cases; without it they are listed
// in the report as deferred rather than silently omitted.
//
// Run: node scripts/falsify-pac-guards.mjs
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const python = process.env.PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const MODELS = ['node', ['scripts/verify-pac-models.mjs', '--no-evidence']];
const SOURCES = [python, ['scripts/verify-pac-sources.py', '--no-evidence']];
const DATA = [python, ['scripts/verify-pac-data.py', '--no-evidence']];
const EXAMPLES = [python, ['scripts/verify-pac-examples.py', '--no-evidence']];

const MODELS_FILE = 'src/learn/data/pac-models.js';
const DATA_FILE = 'src/learn/data/pac-data.js';
const EXAMPLES_FILE = 'src/learn/data/pac-examples.js';
const TOPIC_FILE = 'src/learn/data/topics/pac-learning-vc-dimension.jsx';
const LABS_FILE = 'src/learn/components/lesson-labs/PacLabs.jsx';
const SHARED_FILE = 'src/learn/components/lesson-labs/PacShared.jsx';
const EXAMPLES_VERIFIER = 'scripts/verify-pac-examples.py';
const CSS_FILE = 'src/learn/components/lesson-labs/pac-labs.css';
const BROWSER = ['node', ['scripts/verify-pac-browser.cjs']];
const withBrowser = process.argv.includes('--browser');

/** Each case breaks one thing and names the guard that must notice. */
const cases = [
  {
    name: 'the finite-family radius drops the two-sided factor',
    file: MODELS_FILE,
    find: 'return Math.sqrt(Math.log((2 * k) / delta) / (2 * n));',
    replace: 'return Math.sqrt(Math.log(k / delta) / (2 * n));',
    run: MODELS,
    expect: /K=25 radius against the packet|destination note/,
    guards: 'the K = 25, n = 500, delta = .05 radius against the packet and against sqrt(log(1000)/1000)',
  },
  {
    name: 'the interval enumeration admits a pattern with two separate runs',
    file: MODELS_FILE,
    find: "  remember(Array.from({ length: n }, () => 0));",
    replace: "  remember(Array.from({ length: n }, () => 0));\n"
      + "  if (n >= 3) remember(Array.from({ length: n }, (_u, i) => (i === 0 || i === n - 1 ? 1 : 0)));",
    run: MODELS,
    expect: /interval patterns at n=3|the two routes disagree/,
    guards: 'pattern counts by three independent routes',
  },
  {
    name: 'the interval feasibility rule stops noticing a negative between two positives',
    file: MODELS_FILE,
    find: "  const intruder = ordered.find(point => point.label === 0 && point.x > left && point.x < right);",
    replace: "  const intruder = null;",
    run: MODELS,
    expect: /the verdict says true and the enumerated pattern set says false|101 on \.2\/\.5\/\.8/,
    guards: 'the whole-grid feasibility sweep over every ordering and labeling of 2 to 6 points',
  },
  {
    name: 'the unchanged rule becomes strict, so an exactly-tolerant difference is reported as movement',
    file: MODELS_FILE,
    find: "  if (Math.abs(difference) <= tolerance) return { outcome: 'unchanged', difference: 0, rawDifference: difference };",
    replace: "  if (Math.abs(difference) < tolerance) return { outcome: 'unchanged', difference: 0, rawDifference: difference };",
    run: MODELS,
    expect: /unchanged iff the difference is within tolerance|the tolerance boundary is inclusive/,
    guards: 'the unchanged rule asserted as an if and only if, at the exact boundary',
  },
  {
    name: 'the learner predicts 1 rather than 0 at an input it has never seen',
    file: MODELS_FILE,
    find: "  const found = finiteWorldHypotheses.find(rule =>\n"
      + "    [0, 1, 2, 3].every(index => !(mask & (1 << index)) || rule[index] === target[index]));",
    replace: "  const found = [...finiteWorldHypotheses].reverse().find(rule =>\n"
      + "    [0, 1, 2, 3].every(index => !(mask & (1 << index)) || rule[index] === target[index]));",
    run: MODELS,
    expect: /the closed form gives|and it is the first consistent rule/,
    guards: 'the selection rule against its closed form over all 256 target-and-subset combinations',
  },
  {
    name: 'the symmetric difference loses its factor of two',
    file: MODELS_FILE,
    find: "  return (b - a) + (right - left) - 2 * intersection;",
    replace: "  return (b - a) + (right - left) - intersection;",
    run: MODELS,
    expect: /risk at target|the drawn segments add up to the risk/,
    guards: 'symmetric-difference risk recomputed by a union-minus-intersection route over a 2,000-case grid',
  },
  {
    name: 'the explicit VC radius uses a different constant',
    file: MODELS_FILE,
    find: "  return Math.sqrt((32 * (d * Math.log((Math.E * n) / d) + Math.log(8 / delta))) / n);",
    replace: "  return Math.sqrt((8 * (d * Math.log((Math.E * n) / d) + Math.log(8 / delta))) / n);",
    run: MODELS,
    expect: /VC radius at d=|VC radius rebuilt/,
    guards: 'the displayed bound curve against the packet and against a rebuilding of its own expression',
  },
  {
    name: 'a constant-sign rule is drawn as if it had a separating line',
    file: MODELS_FILE,
    find: "    return { kind: 'constant', sign: b >= 0 ? 1 : 0, points: null };",
    replace: "    return { kind: 'line', sign: null, points: [[minX, minY], [maxX, maxY]] };",
    run: MODELS,
    expect: /a drawn endpoint is on the line|a zero normal draws no line/,
    guards: 'the drawn separator lying on the line it claims to draw, and the constant-sign case examined rather than skipped',
  },
  {
    name: 'the candidate bands are drawn at the radius the search did not earn',
    file: MODELS_FILE,
    find: "  const radius = finiteRadius(k, n, delta);",
    replace: "  const radius = finiteRadius(3, n, delta);",
    run: MODELS,
    expect: /its band half-width is the theorem's own radius|the band runs a radius below/,
    guards: 'figure 3 drawing the radius its own K implies',
  },
  {
    name: 'a measured development count is altered in the generated data module',
    file: DATA_FILE,
    find: '"developmentCorrect": 74,',
    replace: '"developmentCorrect": 76,',
    run: MODELS,
    expect: /development count for depth5_tree at n=80|the tree fits every training label/,
    guards: 'every measured count against the frozen packet results',
  },
  {
    name: 'a measured development count is altered, against the served dataset',
    file: DATA_FILE,
    find: '"developmentCorrect": 74,',
    replace: '"developmentCorrect": 76,',
    run: DATA,
    expect: /byte for byte|reproduces src\/learn\/data\/pac-data\.js/,
    guards: 'byte-identical regeneration of the data module from the served dataset',
  },
  {
    name: 'a recorded program output is altered',
    file: EXAMPLES_FILE,
    find: '1 1/2 12.460813 1.0',
    replace: '1 1/3 12.460813 1.0',
    run: MODELS,
    expect: /the program's exact fraction at n=1/,
    guards: "the displayed programs' recorded output against what the browser models compute",
  },
  {
    name: 'the pinned digest of an extracted algorithm no longer matches the packet',
    file: EXAMPLES_VERIFIER,
    find: '"extractedSha256": "20b4f035bd99094d82c6008d7df5c44c9a8476646e1b9e3999278fcbd3fdbe93"',
    replace: '"extractedSha256": "0000000000000000000000000000000000000000000000000000000000000000"',
    run: EXAMPLES,
    expect: /the extracted algorithm now hashes/,
    guards: 'the SHA-256 pin on the bytes extracted from the frozen program',
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
    guards: 'the raw-control-byte scan over every owned file and verifier',
  },
  {
    name: 'a required KaTeX sequence disappears from the lesson body',
    file: TOPIC_FILE,
    find: '\\\\binom ni',
    replace: 'C(n,i)',
    run: SOURCES,
    expect: /the KaTeX sequence \\\\binom is missing/,
    guards: 'the required-KaTeX-sequence list',
  },
  {
    name: 'a JSX conditional is left with two identical branches',
    file: LABS_FILE,
    find: "const sampleKey = state => JSON.stringify(state);",
    replace: "const sampleKey = state => JSON.stringify(state);\nconst lost = flag => (flag ? 'same' : 'same');",
    run: SOURCES,
    expect: /a conditional whose branches are identical/,
    guards: 'the identical-branch scan, which is the prose equivalent of an assertion that cannot fail',
  },
  {
    name: 'the lesson body reaches for a member of the global Math, which is shadowed there',
    file: TOPIC_FILE,
    find: 'const bandFigure = candidateBandGeometry();',
    replace: 'const bandFigure = candidateBandGeometry();\nconst shadowed = Math.sqrt(2);',
    run: SOURCES,
    expect: /Math\.sqrt in a module where `Math` is the KaTeX component/,
    guards: 'the shadowed-Math scan',
  },
  {
    name: 'the lesson borrows a sibling lesson\'s copy of the same dataset',
    file: TOPIC_FILE,
    find: 'const bandFigure = candidateBandGeometry();',
    replace: "const bandFigure = candidateBandGeometry();\nconst borrowed = '/learn-assets/evaluation-metrics/banknote-subset.csv';",
    run: SOURCES,
    expect: /belongs to another lesson/,
    guards: 'the own-dataset-only scan',
  },
  {
    name: 'a display block is left as one long unwrapped line',
    file: TOPIC_FILE,
    find: "<MathBlock>{'\\\\begin{gathered}R(h_S)\\\\le\\\\hat R(h_S)+r_K\\\\\\\\\\\\le\\\\hat R(h^*)+r_K\\\\\\\\\\\\le R(h^*)+2r_K.\\\\end{gathered}'}</MathBlock>",
    replace: "<MathBlock>{'R(h_S)\\\\le\\\\hat R(h_S)+r_K\\\\le\\\\hat R(h^*)+r_K\\\\le R(h^*)+2r_K\\\\text{ on the uniform event}.'}</MathBlock>",
    run: SOURCES,
    expect: /an unwrapped display block of \d+ visible and \d+ raw characters/,
    guards: 'the display-block wrapping scan that keeps formulas inside a 320 px column',
  },
  {
    /* This one exists because the guard it points at was INERT and a concurrent
       run proved it: `inset = 0` sat in the module while every containment
       assertion passed, because they compared positions against the same inset
       they were meant to be checking. The floor is now fixed at 12 units, so
       the breakage has something to hit. */
    name: 'the number line inset drops to zero, which used to satisfy its own containment check',
    file: MODELS_FILE,
    find: 'export function stripGeometry({ experiment, width = 300, inset = 16 }) {',
    replace: 'export function stripGeometry({ experiment, width = 300, inset = 0 }) {',
    run: MODELS,
    expect: /the number line's inset is 0, below the 12 units/,
    guards: 'the fixed minimum inset, checked before any position is compared against it',
  },
  {
    /* The floors sit far below the current values on purpose, so they catch a
     * WHOLESALE loss rather than the removal of one call. Breaking a single
     * `record()` therefore does not trip them -- correct behaviour, and the
     * reason this case neuters the counter instead. The limitation is stated in
     * the report rather than hidden by a floor tightened to today's number. */
    name: 'the model suite stops counting what it checked',
    file: 'scripts/verify-pac-models.mjs',
    find: 'const record = name => { counts[name] = (counts[name] ?? 0) + 1; };',
    replace: 'const record = () => {};',
    run: MODELS,
    expect: /the suite has lost coverage|groups ran/,
    guards: 'the floors under the grouped-check and group counts, which stop a suite that has stopped checking from still printing PASS',
  },
];

/* Browser cases need a production build and a running preview, so they are off
   by default and run with `--browser`. Each one costs a rebuild. */
const browserCases = [
  {
    name: 'investigation 1 renders its reveal before a prediction is committed',
    file: LABS_FILE,
    find: '  const run = shown?.answer.run ?? null;',
    replace: '  const run = shown?.answer.run ?? finiteWorldRun(draft);',
    rebuild: true,
    run: BROWSER,
    /* The marker-class check fires first and the numeric pin would fire next;
       either naming counts, and the pattern accepts all three so the case is not
       tied to which guard happens to be reached first. */
    expect: /shows its reveal before any prediction|shows its graded risk|renders "Rule returned" before/,
    guards: 'the leak property: the graded quantity absent before commitment, pinned numerically and by the '
      + 'phrases only the reveal renders',
  },
  {
    name: 'a stylesheet rule paints a band black, so it vanishes into the page ground',
    file: CSS_FILE,
    /* A BLANKET `svg rect { fill: #000 }` is not enough: every rect on this page
       already matches a more specific rule, so the blanket one never applies and
       nothing breaks -- a breakage that breaks nothing proves nothing about the
       guard. The defect this guard exists for is a stylesheet beating the
       intended paint, so the breakage overrides a specific band instead. */
    find: '.pac-lesson svg .pac-band.is-target { fill: #3d5148; }',
    replace: '.pac-lesson svg .pac-band.is-target { fill: #000000; }',
    rebuild: true,
    run: BROWSER,
    expect: /black fill with no stroke|attribute overridden/,
    guards: 'the painted-against-declared check, which compares computed styles rather than attributes',
  },
  {
    /* The one breakage that protects a MATHEMATICAL claim rather than a layout
       one. Unscoping the SVG rule puts it back in front of KaTeX's radical
       SVGs, whose height comes from `height: inherit`; `auto` leaves them with
       no intrinsic height and every square root on the page collapses. Section
       4 then reads `r_K = ln(2K/delta)/(2n)`, which is the radius SQUARED -- a
       different quantity, from which a learner would derive wrong sample sizes.
       Nothing but a measurement on the rendered page can see it. */
    name: 'the SVG layout rule is unscoped again, collapsing every KaTeX radical to nothing',
    file: CSS_FILE,
    find: '.pac-lesson svg:is(.pac-line, .pac-plot, .pac-panel) {',
    replace: '.pac-lesson svg {',
    rebuild: true,
    run: BROWSER,
    expect: /a square-root radical paints with no height or width/,
    guards: 'the measurement of every .katex .sqrt svg on the rendered page, which is the only thing standing '
      + 'between a stylesheet and a wrong formula',
  },
  {
    name: 'the number line loses its inset, so edge labels fall outside the viewBox',
    file: MODELS_FILE,
    find: 'export function stripGeometry({ experiment, width = 300, inset = 16 }) {',
    replace: 'export function stripGeometry({ experiment, width = 300, inset = 0 }) {',
    rebuild: true,
    run: BROWSER,
    expect: /falls outside the inset line|Figure layout collides/,
    guards: 'the containment assertions on the number line, in the model and in the browser layout inspector',
  },
];

const evidenceFiles = [
  ...(fs.existsSync('docs/teaching/evidence')
    ? fs.readdirSync('docs/teaching/evidence').filter(name => name.startsWith('pac-') && name.endsWith('.json'))
      .map(name => `docs/teaching/evidence/${name}`)
    : []),
  /* The screenshots too. A browser case that fails part way through would
     otherwise leave captures of the BROKEN page in the evidence directory --
     exactly the stale-orphan problem the digests exist to catch, created by the
     harness that checks for it. */
  ...(fs.existsSync('docs/teaching/evidence/screenshots')
    ? fs.readdirSync('docs/teaching/evidence/screenshots').filter(name => name.startsWith('pac-'))
      .map(name => `docs/teaching/evidence/screenshots/${name}`)
    : []),
];
const evidenceSnapshot = Object.fromEntries(evidenceFiles.map(file => [file, fs.readFileSync(file)]));

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const results = [];
let fired = 0;

/* ===================================================== S1 · surviving a kill
 *
 * The original kept the pre-mutation bytes only in this process's heap, so any
 * termination that does not unwind the stack -- Ctrl-C, an OOM kill, a spend
 * limit -- left the breakage applied with nothing on disk saying so. The window
 * is a whole verifier run, and for a browser case a whole production build.
 * One of the breakages restores the CSS rule that collapses every KaTeX
 * radical, so a kill during it silently reinstates wrong mathematics.
 *
 * Three things now stand between a kill and a corrupted tree: a `.orig` sidecar
 * written before each mutation and removed after a verified restore, signal and
 * exception handlers that restore from it, and a lock file so two instances
 * cannot interleave -- which is how `inset = 0` was left behind once already.
 */
const LOCK = 'scratch/pac-falsification.lock';
const SIDECAR = file => `${file}.pac-falsify-orig`;
let activeSidecar = null;

fs.mkdirSync('scratch', { recursive: true });

/* `--recover` is the documented way back from a kill. On Windows a forced
 * termination cannot be trapped at all -- `taskkill /F` delivers no catchable
 * signal -- so the handlers above are a courtesy and the SIDECAR is the actual
 * protection. This restores from it and verifies the bytes rather than leaving
 * a human to guess which file was mid-mutation. */
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
    + `Check for stale ${SIDECAR('<file>')} sidecars, restore any you find, then remove the lock.`);
}
/* A stale sidecar means a previous run died mid-case. Restoring is not this
 * run's job to guess at -- it is reported, loudly, and the run refuses. */
const staleSidecars = [...new Set([...cases, ...browserCases].map(item => item.file))]
  .filter(file => fs.existsSync(SIDECAR(file)));
if (staleSidecars.length) {
  throw new Error(`stale sidecars from an interrupted run: ${staleSidecars.join(', ')}. `
    + 'Each holds the original bytes of the file beside it. Restore them before running again.');
}
fs.writeFileSync(LOCK, `${process.pid} ${new Date().toISOString()}\n`);

function restoreFromSidecar(reason) {
  if (activeSidecar) {
    const { file, sidecar } = activeSidecar;
    try {
      fs.copyFileSync(sidecar, file);
      fs.rmSync(sidecar, { force: true });
      process.stderr.write(`\n[falsify] ${reason}: restored ${file} from its sidecar\n`);
    } catch (error) {
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

const allCases = withBrowser ? [...cases, ...browserCases] : cases;
/* S1: the whole loop sits inside a try/finally. The evidence restore used to
   follow the loop, so a case throwing the restore-mismatch error exited past it
   and left the broken page's evidence and screenshots on disk -- exactly what
   the snapshot exists to prevent. */
try {
for (const item of allCases) {
  const original = fs.readFileSync(item.file);
  const before = crypto.createHash('sha256').update(original).digest('hex');
  const text = original.toString('utf8');
  if (!text.includes(item.find)) {
    results.push({ ...item, applied: false, fired: false,
      note: `the target text was not found in ${item.file}; this breakage could not be applied` });
    continue;
  }
  const occurrences = text.split(item.find).length - 1;
  /* The sidecar is written and flushed BEFORE the mutation, so a kill at any
     point after this line leaves the original recoverable on disk. */
  const sidecar = SIDECAR(item.file);
  fs.writeFileSync(sidecar, original);
  activeSidecar = { file: item.file, sidecar };
  const mutated = text.replace(item.find, item.replace);
  // O9: measured, not assumed. `String.replace` with a string pattern replaces
  // the first occurrence only, so reporting a constant 1 was a claim about a
  // substitution nobody counted.
  const replaced = occurrences - (mutated.split(item.find).length - 1);
  fs.writeFileSync(item.file, mutated, 'utf8');
  let outcome;
  try {
    if (item.rebuild) {
      const built = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-pac'],
        { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
      if (built.status !== 0) throw new Error(`the build failed while applying "${item.name}"`);
    }
    const [command, argv] = item.run;
    const run = spawnSync(command, argv, {
      encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
      env: { ...process.env, DIST_DIR: 'dist-pac' },
    });
    const output = `${run.stdout ?? ''}\n${run.stderr ?? ''}`;
    const failed = run.status !== 0;
    const named = item.expect.test(output);
    outcome = {
      name: item.name,
      file: item.file,
      guards: item.guards,
      verifier: `${command} ${argv.join(' ')}`,
      occurrencesReplaced: replaced,
      occurrencesPresent: occurrences,
      exitCode: run.status,
      failedAsIntended: failed,
      namedTheGuard: named,
      fired: failed && named,
      excerpt: (output.match(item.expect) ?? [''])[0].slice(0, 160)
        || output.split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 240),
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
  }
  if (!outcome.restoredExactly) {
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
    spawnSync('npx', ['vite', 'build', '--outDir', 'dist-pac'], { encoding: 'utf8', shell: true });
  }
  for (const [file, bytes] of Object.entries(evidenceSnapshot)) fs.writeFileSync(file, bytes);
  fs.rmSync(LOCK, { force: true });
}

const report = {
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-pac-guards.mjs',
  harnessSha256: digest('scripts/falsify-pac-guards.mjs'),
  mode: withBrowser ? 'offline and browser' : 'offline only',
  breakagesApplied: results.length,
  guardsThatFired: fired,
  allRestoredExactly: results.every(result => result.restoredExactly !== false),
  evidenceFilesRestored: evidenceFiles,
  results,
  browserCasesIncluded: withBrowser ? browserCases.length : 0,
  browserCasesDeferred: withBrowser ? [] : browserCases.map(item => item.name),
  method: 'Each breakage is applied alone to a clean tree, the relevant verifier is run, the file is restored '
    + 'and the restoration checked by SHA-256. A case counts as fired only when the verifier exits non-zero AND '
    + 'its output names the guard, so a run that fails for an unrelated reason is not credited.',
  limitations: [
    'The coverage floors in the model suite are set far below its current totals, so they catch a wholesale '
      + 'loss of checking rather than the removal of one assertion. Deleting a single record() call does NOT '
      + 'trip them, and the case above neuters the counter instead. Tightening the floors to today\'s numbers '
      + 'would make them fail on every legitimate addition, which is how a floor gets deleted rather than fixed.',
    'Browser guards need a running preview and a production build; without --browser they are listed as '
      + 'deferred rather than silently omitted.',
    'Crash safety is best-effort: SIGKILL and a power loss cannot be trapped. In those cases the `.orig` '
      + 'sidecar beside the mutated file holds its original bytes, and the next run refuses to start until '
      + 'the sidecar is dealt with, rather than mutating a file that is already wrong.',
    'A guard that fires here is a guard that can fail. It is not evidence that its domain covers every defect '
      + 'of its kind -- that is what the whole-grid sweeps in the model suite are for.',
  ],
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync('docs/teaching/evidence/pac-falsification.json', JSON.stringify(report, null, 2) + '\n');

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
