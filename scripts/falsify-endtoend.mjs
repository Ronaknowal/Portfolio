// Falsification harness for the end-to-end lesson's guards.
//
// A passing verifier proves nothing about a guard that cannot fail. Every inert
// guard found in this repository so far read correctly within a domain that
// excluded the defect: a KaTeX check measuring only width when a collapsed SVG
// is also narrow, a leak filter requiring values above 2 where every
// probability is below 1, a curve sampler finding two shapes out of two hundred
// because it queried paths on a lesson that draws lines, and a containment check
// comparing positions against the inset it was meant to be guarding. So each
// guard here is shown to fire, by breaking exactly the thing it claims to watch.
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
//     falsifying -- not even when a case throws. Only this lesson's evidence is
//     touched; a sibling's 709 captures were nearly deleted by a harness that
//     was not this careful;
//   * the pre-mutation bytes go to a `.orig` sidecar on disk BEFORE the
//     mutation and are removed only after a verified restore, with signal and
//     exception handlers that restore from it. On Windows a forced kill delivers
//     no catchable signal at all, so the handlers are a courtesy and the sidecar
//     is the actual protection: the next run REFUSES TO START while one exists;
//   * a lock file refuses a second concurrent instance.
//
// Browser guards are not falsifiable without a running preview and are deferred
// to phase C. Passing `--browser` adds those cases; without it they are listed
// in the report as deferred rather than silently omitted.
//
// Run:  node scripts/falsify-endtoend.mjs
// After a kill:  node scripts/falsify-endtoend.mjs --recover
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const python = process.env.PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const MODELS = ['node', ['scripts/verify-endtoend-models.mjs', '--no-evidence']];
const SOURCES = [python, ['scripts/verify-endtoend-sources.py', '--no-evidence']];
const DATA = [python, ['scripts/verify-endtoend-data.py', '--no-evidence']];
const EXAMPLES = [python, ['scripts/verify-endtoend-examples.py', '--no-evidence']];
/* `--no-evidence`, so a falsification run cannot write the record it is
   falsifying even if it is killed mid-case. The heap snapshot below stays as a
   second line of defence; this is the one that survives a forced kill. */
const BROWSER = ['node', ['scripts/verify-endtoend-browser.cjs', '--no-evidence']];

const MODELS_FILE = 'src/learn/data/endtoend-models.js';
const DATA_FILE = 'src/learn/data/endtoend-data.js';
const EXAMPLES_FILE = 'src/learn/data/endtoend-examples.js';
const TOPIC_FILE = 'src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx';
const SHARED_FILE = 'src/learn/components/lesson-labs/EndToEndShared.jsx';
const FIGURES_FILE = 'src/learn/components/lesson-labs/EndToEndFigures.jsx';
const CSS_FILE = 'src/learn/components/lesson-labs/endtoend-labs.css';
const DATA_VERIFIER = 'scripts/verify-endtoend-data.py';
const withBrowser = process.argv.includes('--browser');

/** Each case breaks one thing and names the guard that must notice. */
const cases = [
  {
    name: 'the slice boundary flips, so a specimen exactly on the cutoff changes sides',
    file: MODELS_FILE,
    find: "  return side === 'lower' ? row.colorIntensity < cutoff : row.colorIntensity >= cutoff;",
    replace: "  return side === 'lower' ? row.colorIntensity <= cutoff : row.colorIntensity > cutoff;",
    run: MODELS,
    expect: /the lower rule disagrees with a bare comparison|a specimen exactly on the cutoff belongs to the upper side|support against the packet/,
    guards: 'the whole-cutoff sweep, which checks both sides of every one of the 1,401 cutoffs the control '
      + 'admits and the one specimen that sits exactly on the default',
  },
  {
    name: 'the drawn cutoff line stops tracking the cutoff it is drawn for',
    file: MODELS_FILE,
    find: '    cutoffY: y(cutoff),',
    replace: '    cutoffY: y(cutoff) + 3,',
    run: MODELS,
    expect: /the drawn line is not at the cutoff|is drawn on the wrong side of the line/,
    guards: 'the drawn-equals-applied check, which compares the line the figure paints with the rule the '
      + 'grading applies at every cutoff',
  },
  {
    name: 'an empty slice is graded as a tie instead of reported as having no comparison',
    file: MODELS_FILE,
    find: "  if (slice.n === 0) {",
    replace: "  if (false) {",
    run: MODELS,
    expect: /an empty slice must have no outcome, not a tie|Cannot read|referenceErrors/,
    guards: 'the empty-slice case, which must report no comparison rather than a zero difference',
  },
  {
    name: 'the selection rule sorts first and takes the top, losing the declared tie order',
    file: MODELS_FILE,
    find: "    if (direction === 'higher' ? difference > 0 : difference < 0) best = entry;",
    replace: "    if (direction === 'higher' ? difference >= 0 : difference <= 0) best = entry;",
    run: MODELS,
    expect: /the rule must pick the FIRST entry achieving the extremum|the module and an explicit scan disagree/,
    guards: 'the tie-rule sweep over every score vector with a repeated extremum, which is the only '
      + 'way to exercise a tie on data where no two candidates tie',
  },
  {
    name: 'a selection the study never tested is offered a held-out estimate anyway',
    file: MODELS_FILE,
    find: '    heldOutAvailable: best === endToEndData.heldOut.selected && !departsFromProtocol,',
    replace: '    heldOutAvailable: true,',
    run: MODELS,
    expect: /held-out availability is wrong|settings offer the held-out report|an available report carries no refusal/,
    guards: 'the refusal property: exactly one of the 48 selection settings may open the held-out report',
  },
  {
    name: 'a score record accepts a role outside the four',
    file: MODELS_FILE,
    find: "  if (!SCORE_ROLES.includes(role)) {",
    replace: "  if (false) {",
    run: MODELS,
    expect: /a role outside the four: should have been refused|score roles/,
    guards: 'the role guard on every score record, which is what stops a number reaching the page unlabelled',
  },
  {
    name: 'a training score can be re-badged as a selection criterion',
    file: MODELS_FILE,
    find: "  if (!record || record.role !== 'validation') {",
    replace: "  if (!record) {",
    run: MODELS,
    expect: /reading a training score as a selection criterion: should have been refused|reading a held-out score/,
    guards: 'the guard that only a validation measurement may be read as a selection criterion',
  },
  {
    name: 'a rule that answers nothing reports a conditional error of zero',
    file: MODELS_FILE,
    find: '    conditionalError: accepted.length === 0 ? null : wrong.length / accepted.length,',
    replace: '    conditionalError: accepted.length === 0 ? 0 : wrong.length / accepted.length,',
    run: MODELS,
    expect: /a rule that answers nothing must report no conditional error|conditionalError/,
    guards: 'the undefined-conditional-error null, swept over every threshold the control admits',
  },
  {
    name: 'editing a case score also changes whether that case was answered correctly',
    file: MODELS_FILE,
    find: '    return { ...item, confidence, edited: override !== undefined && confidence !== item.confidence };',
    replace: '    return { ...item, confidence, correct: confidence >= 0.8, '
      + 'edited: override !== undefined && confidence !== item.confidence };',
    run: MODELS,
    expect: /editing a score must not change whether a case was answered correctly|wrong-among-accepted count/,
    guards: 'the separation between a proposed decision rule and the recorded outcomes it is applied to',
  },
  {
    name: 'balanced accuracy averages a class with no support as if its recall were zero',
    file: MODELS_FILE,
    find: '    if (support === 0) continue;',
    replace: '    if (support === 0) { recalls.push(0); continue; }',
    run: MODELS,
    expect: /balanced accuracy over a single supported class|balanced accuracy by label walk against/,
    guards: 'the definition of balanced accuracy over the classes that have support',
  },
  {
    name: 'the figure inset drops to zero, which used to satisfy its own containment check',
    file: MODELS_FILE,
    find: 'export const MINIMUM_INSET = 12;',
    replace: 'export const MINIMUM_INSET = 0;',
    run: MODELS,
    expect: /the shared minimum inset is 0, below the 12-unit literal/,
    guards: 'the fixed minimum inset, asserted before any position is compared against it',
  },
  {
    name: 'the lane figure gains the arrow the whole section exists to leave out',
    file: MODELS_FILE,
    find: "    { key: 'leak', from: 'report', to: 'selection', kind: 'counterexample',",
    replace: "    { key: 'leak', from: 'test', to: 'selection', kind: 'counterexample',",
    run: MODELS,
    expect: /no arrow runs from the test lane into candidate selection/,
    guards: 'the absent-edge assertions on the information-lane figure',
  },
  {
    name: 'a measured validation score is altered in the generated data module',
    file: DATA_FILE,
    find: '"value": 0.7928571428571428',
    replace: '"value": 0.8028571428571428',
    run: MODELS,
    expect: /against the packet|the manuscript's stated values|the two-feature linear score the manuscript prints/,
    guards: 'every recorded score against the frozen packet',
  },
  {
    name: 'a measured validation score is altered, against the served dataset',
    file: DATA_FILE,
    find: '"value": 0.7928571428571428',
    replace: '"value": 0.8028571428571428',
    run: DATA,
    expect: /byte-identical|is not byte-identical to a fresh regeneration/,
    guards: 'byte-identical regeneration of the data module from the served dataset',
  },
  {
    name: 'the trust-root comparison stops recording which leaves it compared',
    file: DATA_VERIFIER,
    find: '    seen_leaves.add(path)',
    replace: '    pass',
    run: DATA,
    expect: /packet leaves were never compared/,
    guards: 'the measured trust-root coverage, which fails unless every leaf was actually compared',
  },
  {
    name: 'the forest walk drops its single-precision cast, so three specimens route differently',
    file: DATA_VERIFIER,
    find: '        single = np.asarray(sample, dtype=np.float32)',
    replace: '        single = np.asarray(sample, dtype=np.float64)',
    run: DATA,
    expect: /validationProbability|independently computed/,
    guards: 'the independent tree walk, which reproduces the fitted forest exactly only when it compares the '
      + 'way the fitted forest compares',
  },
  {
    name: 'the recorded program output is altered',
    file: EXAMPLES_FILE,
    find: 'linear_three 0.888889 0.888889 0.218008',
    replace: 'linear_three 0.898889 0.888889 0.218008',
    run: EXAMPLES,
    expect: /differs from the manuscript's recorded block|is not byte-identical to a fresh execution/,
    guards: "the displayed program's recorded output against a fresh execution",
  },
  {
    name: 'a held-out quantity is moved into the half the page shows before the gate',
    file: EXAMPLES_FILE,
    find: '"developmentOutput": "split sizes 106 36 36',
    replace: '"developmentOutput": "test 0.966667 0.972222 0.128708\\nsplit sizes 106 36 36',
    run: MODELS,
    expect: /leaks the held-out quantity|development half/,
    guards: 'the leak property on the study\'s own output: no held-out quantity in the half shown before '
      + 'the decision is frozen',
  },
  {
    name: 'the pinned digest of the extracted program no longer matches the manuscript',
    file: 'scripts/verify-endtoend-examples.py',
    find: 'PROGRAM_SHA = "e0dcd56d83d241228d0c92f5308a102fb11ba90fbc8adfec9c4bd201714beecc"',
    replace: 'PROGRAM_SHA = "0000000000000000000000000000000000000000000000000000000000000000"',
    run: EXAMPLES,
    expect: /the extracted program now hashes/,
    guards: 'the SHA-256 pin on the bytes extracted from the frozen manuscript',
  },
  {
    name: 'a shell eats an escape and leaves a raw control byte in a source file',
    file: SHARED_FILE,
    find: 'const minus = text =>',
    // The byte is BUILT here rather than written. A literal control byte in this
    // file would be caught by the very guard this case falsifies, which is a
    // pleasing demonstration and a broken harness.
    replace: `const minus = text => /* ${String.fromCharCode(8)} */`,
    run: SOURCES,
    expect: /raw control byte 0x08|a shell ate an escape/,
    guards: 'the raw-control-byte scan over every owned file and verifier',
  },
  {
    name: 'a required KaTeX sequence disappears from the lesson body',
    file: TOPIC_FILE,
    find: '\\\\begin{pmatrix}',
    replace: 'MATRIX(',
    run: SOURCES,
    expect: /the KaTeX sequence \\\\begin\{pmatrix\} is missing/,
    guards: 'the required-KaTeX-sequence list',
  },
  {
    name: 'the stylesheet reaches KaTeX again with a bare descendant svg selector',
    file: CSS_FILE,
    find: '.endtoend-lesson svg:is(.ete-plot, .ete-strip, .ete-lanes, .ete-rail, .ete-bars) {',
    replace: '.endtoend-lesson svg {',
    run: SOURCES,
    expect: /ends in a bare `svg` compound|the SVG layout rule is scoped by class/,
    guards: 'the selector-level refusal of a bare descendant `svg` rule, which is the worst defect this '
      + 'effort has produced',
  },
  {
    name: 'a figure renders an untagged svg, which would fall out of the layout rule',
    file: SHARED_FILE,
    /* The Canvas site specifically. The shorter anchor matched PlotFrame's svg
       too, and the ambiguity guard above now refuses that rather than letting
       the mutation land on whichever came first. */
    find: '    <svg className={className} viewBox={`0 0 ${geometry.width} ${geometry.height}`} role="img"',
    replace: '    <svg viewBox={`0 0 ${geometry.width} ${geometry.height}`} role="img"',
    run: SOURCES,
    expect: /an <svg> with no ete-plot/,
    guards: 'the requirement that every svg this lesson renders carries a layout class',
  },
  {
    /* The element-level scan above passes an svg whose class comes from a prop,
       so the prop's VALUE is the thing that can silently stop matching the
       stylesheet. That is a different guard and it needs its own breakage. */
    name: 'a figure is given a class the stylesheet\'s layout rule does not match',
    file: FIGURES_FILE,
    find: '<Canvas geometry={geometry} className="ete-lanes"',
    replace: '<Canvas geometry={geometry} className="ete-diagram"',
    run: SOURCES,
    expect: /which is not one of ete-plot|is given className/,
    guards: 'the check that every figure frame is passed one of the layout classes, not merely some class',
  },
  {
    name: 'a bar acquires a minimum length on the chart where that floor actually binds',
    file: MODELS_FILE,
    find: '        barHeight: (height - padding.bottom) - y(value),',
    replace: '        barHeight: Math.max((height - padding.bottom) - y(value), 40),',
    run: MODELS,
    expect: /the bar's height is not its own value's position|bar lengths are proportional/,
    guards: 'the proportionality check across every metric chart, which refuses an unexplained minimum '
      + 'length; the floor is set above the shortest bar on the page so the breakage binds',
  },
  {
    name: 'a score is typed into the lesson body instead of coming from the data module',
    file: TOPIC_FILE,
    find: 'const linearTwo = candidateByKey.linear_two;',
    replace: 'const linearTwo = candidateByKey.linear_two;\nconst typedIn = 0.792857;',
    run: SOURCES,
    expect: /looks like a score typed into a component/,
    guards: 'the scan that refuses a score literal in a component, because such a number carries no role',
  },
  {
    name: 'a held-out quantity is typed into the lesson body',
    file: TOPIC_FILE,
    find: 'const heldOut = endToEndData.heldOut;',
    replace: "const heldOut = endToEndData.heldOut;\nconst quoted = '0.966667';",
    run: SOURCES,
    expect: /is typed into a component|looks like a score typed into a component/,
    guards: 'the scan that refuses a held-out literal in a component, so the gate decides whether it renders',
  },
  {
    name: 'the held-out gate renders its children and hides them instead of withholding them',
    file: SHARED_FILE,
    find: '  if (earned) return children;',
    replace: '  if (true) return children;',
    run: SOURCES,
    expect: /HeldOutOnly returns its children only when earned/,
    guards: 'the requirement that the gate withholds rather than hides, because hidden text is still readable',
  },
  {
    name: 'a JSX conditional is left with two identical branches',
    file: SHARED_FILE,
    find: 'const minus = text =>',
    replace: "const lost = flag => (flag ? 'same' : 'same');\nconst minus = text =>",
    run: SOURCES,
    expect: /a conditional whose branches are identical/,
    guards: 'the identical-branch scan, which is the prose equivalent of an assertion that cannot fail',
  },
  {
    name: 'the lesson body reaches for a member of the global Math, which is shadowed there',
    file: TOPIC_FILE,
    find: 'const linearTwo = candidateByKey.linear_two;',
    replace: 'const linearTwo = candidateByKey.linear_two;\nconst shadowed = Math.sqrt(2);',
    run: SOURCES,
    expect: /Math\.sqrt in a module where `Math` is the KaTeX component/,
    guards: 'the shadowed-Math scan',
  },
  {
    name: "the lesson borrows a sibling lesson's copy of the Wine measurements",
    file: TOPIC_FILE,
    find: 'const heldOut = endToEndData.heldOut;',
    replace: "const heldOut = endToEndData.heldOut;\nconst borrowed = '/learn-assets/pca/wine.csv';",
    run: SOURCES,
    expect: /belongs to another lesson/,
    guards: 'the own-dataset-only scan',
  },
  {
    /* The floors sit below the current values on purpose, so they catch a
     * WHOLESALE loss rather than the removal of one call. Breaking a single
     * `record()` therefore does not trip them -- correct behaviour, and the
     * reason this case neuters the counter instead. The limitation is stated in
     * the report rather than hidden by a floor tightened to today's number. */
    name: 'the model suite stops counting what it checked',
    file: 'scripts/verify-endtoend-models.mjs',
    find: 'const record = name => { counts[name] = (counts[name] ?? 0) + 1; };',
    replace: 'const record = () => {};',
    run: MODELS,
    expect: /the suite has lost coverage|groups ran/,
    guards: 'the floors under the grouped-check and group counts, which stop a suite that has stopped '
      + 'checking from still printing PASS',
  },
  {
    /* S1's repair. The Wilson bounds were in an absence-only list whose subject
       is the program's stdout, which computes no interval at any precision, so
       two of five absence assertions were permanently inert. Every entry now
       carries a presence partner from one list; adding a quantity the program
       never prints must fail on the PRESENCE half. */
    name: 'a quantity the program never prints is added to the leak list, with no presence partner possible',
    file: 'scripts/verify-endtoend-models.mjs',
    find: "nonEmpty(heldOutQuantities, 4, 'the held-out quantities that must not leak into the development output');",
    replace: "heldOutQuantities.push(fixed(wilson.lower, 6));\n"
      + "nonEmpty(heldOutQuantities, 5, 'the held-out quantities that must not leak into the development output');",
    run: MODELS,
    expect: /could never have failed|the held-out half is missing/,
    guards: 'the paired rule applied without exception: every absence assertion in the leak block has a '
      + 'presence partner drawn from the same list, so a pin that can never match fails on the presence half',
  },
  {
    /* S2's repair. The encoding guard used `round()`, which re-snapped every
       float and made the predicate false for every input. */
    name: 'two cutoff settings collapse to the same double',
    file: DATA_VERIFIER,
    find: '    cutoffs = [hundredths / 100 for hundredths in range(0, 1401)]',
    replace: '    cutoffs = [min(hundredths, 1399) / 100 for hundredths in range(0, 1401)]',
    run: DATA,
    expect: /collapse to \d+ distinct doubles|not strictly greater/,
    guards: 'the cutoff encoding checked for distinctness and order rather than through a round() that '
      + 'swallowed the failure it named',
  },
  {
    name: 'the packet row order stops matching the model blocks the positions imply',
    file: DATA_VERIFIER,
    find: '        expected_key = MODEL_KEYS[row_index // len(valid_ids)]',
    replace: '        expected_key = MODEL_KEYS[(row_index // len(valid_ids) + 1) % len(MODEL_KEYS)]',
    run: DATA,
    expect: /validationRows\/\d+\/model/,
    guards: 'the validation-row model key derived from the row position rather than read back out of the '
      + 'row, so the 144 comparisons can fail',
  },
  {
    /* S7(a)-(c)'s repair. Containment was checked against the geometry's own
       inset, so a zeroed inset broke nothing. */
    name: 'the lane figure loses its inset, which used to satisfy its own containment check',
    file: MODELS_FILE,
    find: 'export function informationLaneGeometry({ width = 320, height = 212, inset = 14 } = {}) {',
    replace: 'export function informationLaneGeometry({ width = 320, height = 212, inset = 0 } = {}) {',
    run: MODELS,
    expect: /falls outside the 12-unit frame margin/,
    guards: 'containment asserted against a literal frame margin rather than against the inset the positions '
      + 'are derived from',
  },
  {
    name: 'the acceptance rail loses its inset, which used to satisfy its own containment check',
    file: MODELS_FILE,
    find: '  ledger, width = 320, height = 150, inset = 16,',
    replace: '  ledger, width = 320, height = 150, inset = 0,',
    run: MODELS,
    expect: /falls outside the 12-unit frame margin/,
    guards: 'the tiles on the rail checked against a literal margin rather than against a scale built from '
      + 'the inset, which contained them for any inset whatsoever',
  },
  {
    /* S7(d)'s repair. The segment check read the band the segments are defined
       from, so both disjuncts held by construction. */
    name: 'the threshold rule is drawn straight through the tile number again',
    file: MODELS_FILE,
    find: '{ y1: tileBand.bottom + 2, y2: railY + 3 },',
    replace: '{ y1: tileBand.top - 10, y2: railY + 3 },',
    run: MODELS,
    expect: /reaches the tile number at y=|does not sit in the gap between the two threshold segments/,
    guards: 'the threshold rule checked against the label positions the component actually draws text at, '
      + 'rather than against the band the segments are derived from',
  },
  {
    /* THE ONE BREAKAGE HERE THAT REPRODUCES A DEFECT THAT ACTUALLY SHIPPED.
       An unescaped apostrophe inside a single-quoted string reached this file
       and broke `npx vite build` for the whole repository; a sibling lesson
       found it because it could not regenerate its own evidence. Nothing
       offline saw it, because the model and data verifiers import plain .js and
       the text scans do not parse. */
    name: 'an owned component stops parsing as JSX, which breaks the build for every lesson in the repo',
    file: SHARED_FILE,
    find: "const minus = text =>",
    replace: "const broken = 'an apostrophe that isn't escaped';\nconst minus = text =>",
    run: SOURCES,
    expect: /does not parse as JSX/,
    guards: 'the JSX parse of every owned component, which is the only offline check that sees a syntax '
      + 'error in a file this lesson owns',
  },
  {
    /* The liveness gate, falsified through the source verifier rather than
       here — this harness holds the lock while it runs, so a case that tried to
       test recovery directly would be testing its own run. The source verifier
       performs the refusal instead, and a disabled gate makes it fail. The
       failure mode this guards has occurred twice in one hour. */
    name: 'the harness will recover over a live run again, un-mutating a file it deliberately broke',
    file: 'scripts/falsify-endtoend.mjs',
    /* The anchor includes the line that follows it, because the shorter form
       occurred TWICE in this file -- once at the gate and once in this very
       `find` field -- and `String.replace` takes the first, so the case mutated
       its own definition and left the gate intact. The ambiguity check below
       now refuses any case whose anchor is not unique. */
    find: "  if (live && !process.argv.includes('--force-recover')) {\n    throw new Error(",
    replace: "  if (false) {\n    throw new Error(",
    run: SOURCES,
    expect: /recovered over a LIVE run instead of refusing/,
    guards: 'the heartbeat liveness gate on --recover, exercised by performing the refusal rather than by '
      + 'matching the code that refuses',
  },
  {
    /* The timer version of the heartbeat, restored. It cannot fire against
       `spawnSync`, so the recorded heartbeat never advances. */
    name: 'the heartbeat goes back to a timer, which cannot fire while a child blocks the event loop',
    file: 'scripts/falsify-endtoend.mjs',
    /* The whole wrapper reverts to a bare call — the state the timer version
       was in. Removing only the trailing refresh does NOT disable the advance,
       because the leading one still moves the heartbeat before the probe reads
       it again; a breakage has to remove the property, not one of its halves. */
    find: '  writeLock();\n  try {\n    return spawnSync(command, argv, options);\n  } finally {\n'
      + '    writeLock();\n  }',
    replace: '  return spawnSync(command, argv, options);',
    run: SOURCES,
    expect: /heartbeat did not advance across a blocking child/,
    guards: 'the heartbeat exercised by observing it advance across exactly the kind of call that made the '
      + 'timer version inert',
  },
  {
    /* The one-word change that would silently disarm the gate. */
    name: 'the liveness probe is run through a shell, which rewrites its switch and answers "dead" for a live run',
    file: 'scripts/falsify-endtoend.mjs',
    /* The anchor carries the following line, because the shorter form also
       matches this case's own `find` field — the collision the uniqueness
       guard exists for, which caught it here. */
    find: "  const listed = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/NH'], { encoding: 'utf8' });\n"
      + "  return (listed.stdout ?? '').includes(String(pid));",
    replace: "  const listed = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/NH'], "
      + "{ encoding: 'utf8', shell: true });\n  return (listed.stdout ?? '').includes(String(pid));",
    run: SOURCES,
    /* Either message counts, and both name a real guard: breaking the probe
       makes it report a live process as dead AND disarms the recovery gate that
       reads it, which is the consequence that matters. */
    expect: /reported a RUNNING process .* as not running|recovered over a LIVE run instead of refusing/,
    guards: 'the liveness probe exercised against a real live pid, and the recovery gate that rests on it — '
      + 'a shell here makes the probe answer "dead" for a live run and the gate then permits recovery',
  },
  {
    /* O9's repair. */
    name: 'the data suite stops counting what it checked',
    file: DATA_VERIFIER,
    /* The counter itself is neutered, rather than a local being assigned in
       `main` — which would shadow the module-level `checks` and raise an
       UnboundLocalError instead of reaching the floor. */
    find: '    global checks\n    checks += 1\n    if not condition:',
    replace: '    global checks\n    if not condition:',
    run: DATA,
    expect: /data checks ran; the suite has lost coverage/,
    guards: 'the floor under the counter the data verifier prints',
  },
];

/* Browser cases need a production build and a running preview, so they are off
   by default and run with `--browser`. Each one costs a rebuild. */
const browserCases = [
  {
    /* The one breakage that protects a MATHEMATICAL claim rather than a layout
       one. Unscoping the SVG rule puts it back in front of KaTeX's radical and
       stretchy-delimiter SVGs, whose height comes from `height: inherit`;
       `auto` leaves them with no intrinsic height and the radical in section
       8's Wilson formula collapses. The formula then reads as the square of
       what it says. Nothing but a measurement on the rendered page can see it. */
    name: 'the SVG layout rule is unscoped again, collapsing every KaTeX radical to nothing',
    file: CSS_FILE,
    find: '.endtoend-lesson svg:is(.ete-plot, .ete-strip, .ete-lanes, .ete-rail, .ete-bars) {',
    replace: '.endtoend-lesson svg {',
    rebuild: true,
    run: BROWSER,
    expect: /a KaTeX SVG paints with height|a KaTeX SVG collapsed at/,
    guards: 'the measurement of every .katex svg on the rendered page, in BOTH dimensions, which is the only '
      + 'thing standing between a stylesheet and a wrong formula',
  },
  {
    name: 'the held-out report renders before the decision that earns it is frozen',
    file: SHARED_FILE,
    find: '  if (earned) return children;',
    replace: '  if (true) return children;',
    rebuild: true,
    run: BROWSER,
    expect: /is readable before the decision it reports on has been frozen|a held-out score is printed before/,
    guards: 'the held-out property on the rendered document, pinned numerically against quantities computed '
      + 'in the verifier rather than read off the page',
  },
  {
    name: 'a printed score loses the role attribute the page promises',
    file: SHARED_FILE,
    find: '  return <span className={`ete-score is-${record.role.replace(\'-\', \'\')}`} data-role={record.role}>',
    replace: '  return <span className={`ete-score is-${record.role.replace(\'-\', \'\')}`}>',
    rebuild: true,
    run: BROWSER,
    expect: /carries the role "null"|is not one of the four/,
    guards: 'the assertion that every printed score carries its role in the rendered DOM',
  },
  {
    name: 'a stylesheet paints a figure mark the page ground, so it vanishes',
    /* A BLANKET rule is not enough: every shape on this page already matches a
       more specific rule, so a blanket one never applies and nothing breaks -- a
       breakage that breaks nothing proves nothing about the guard. The defect
       this guard exists for is a stylesheet beating the intended paint, so the
       breakage overrides a specific mark instead. */
    file: CSS_FILE,
    find: '.endtoend-lesson svg .ete-mark.is-right { fill: #3d5148; }',
    replace: '.endtoend-lesson svg .ete-mark.is-right { fill: #000000; }',
    rebuild: true,
    run: BROWSER,
    expect: /black fill with no stroke|painted invisibly/,
    guards: 'the painted-against-declared check, which compares computed styles rather than attributes',
  },
  {
    /* THE CLASS, not the site. This is the blocking defect the independent
       review found, reproduced exactly: the grader repointed at a neighbouring
       quantity while the question's wording stays put. It is the fifth variant
       of this class across the effort, and every previous per-lesson guard was
       written against the last instance's surface. The guard now reads the
       number the PAGE DISPLAYS for the quantity the question names, types it
       in, and requires the verdict to match — so any repointing fails,
       whichever neighbour it picks. */
    name: 'the acceptance grader is repointed at the cost difference, which is not what its question asks for',
    file: 'src/learn/components/lesson-labs/EndToEndLabs.jsx',
    find: '      value: proposed.cost,',
    replace: '      value: comparison.difference,',
    rebuild: true,
    run: BROWSER,
    expect: /typing that exact number is not graded as matching|grades the quantity it names/,
    guards: 'the property that the quantity a question NAMES is the quantity its grader COMPARES, referenced '
      + 'against the value the page itself displays for it',
  },
  {
    name: 'the slice explorer grader is repointed at the reference count instead of the candidate count',
    file: 'src/learn/components/lesson-labs/EndToEndLabs.jsx',
    find: '      value: comparison.candidateErrors,',
    replace: '      value: comparison.referenceErrors,',
    rebuild: true,
    run: BROWSER,
    expect: /typing that exact number is not graded as matching|grades the quantity it names/,
    guards: 'the same property on the other graded question, so the guard is not tied to one investigation',
  },
  {
    name: 'a held-out quantity is printed without its role badge',
    file: 'src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx',
    find: '        accuracy <Count part={heldOut.correct} whole={heldOut.total} role="held-out" /> and balanced',
    replace: '        accuracy {heldOut.correct}/{heldOut.total} and balanced',
    rebuild: true,
    run: BROWSER,
    expect: /outside any role badge/,
    guards: 'the complement of the role check: a sweep for score-shaped numbers that are NOT inside a badge, '
      + 'which is the assertion the lesson\'s headline invariant actually corresponds to',
  },
  {
    name: 'the slice explorer reveals its graded count before a prediction is committed',
    file: 'src/learn/components/lesson-labs/EndToEndLabs.jsx',
    find: '  const comparison = shown?.answer.comparison ?? null;',
    replace: '  const comparison = shown?.answer.comparison ?? sliceComparison(draft);',
    rebuild: true,
    run: BROWSER,
    expect: /shows its graded error count before a prediction is committed/,
    guards: 'the leak property on the investigation, pinned to the exact count-and-denominator text the '
      + 'reveal prints',
  },
];

const evidenceFiles = [
  ...(fs.existsSync('docs/teaching/evidence')
    ? fs.readdirSync('docs/teaching/evidence')
      .filter(name => name.startsWith('endtoend-') && name.endsWith('.json'))
      .map(name => `docs/teaching/evidence/${name}`)
    : []),
  /* The screenshots too, and ONLY this lesson's. A browser case that failed
     part way through would otherwise leave captures of the BROKEN page in the
     evidence directory -- exactly the stale-orphan problem the digests exist to
     catch, created by the harness that checks for it. The prefix filter is not
     decoration: a harness that snapshotted the whole directory once came close
     to deleting 709 of a sibling lesson's captures. */
  ...(fs.existsSync('docs/teaching/evidence/screenshots')
    ? fs.readdirSync('docs/teaching/evidence/screenshots').filter(name => name.startsWith('endtoend-'))
      .map(name => `docs/teaching/evidence/screenshots/${name}`)
    : []),
];
const evidenceSnapshot = Object.fromEntries(evidenceFiles.map(file => [file, fs.readFileSync(file)]));

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const results = [];
let fired = 0;
/* Whether the build directory was put back. Tracked separately from source
   restoration because they are different claims and one used to stand in for
   the other. */
let buildRestored = withBrowser ? false : null;

/* ===================================================== surviving a kill
 *
 * Keeping the pre-mutation bytes only in this process's heap means any
 * termination that does not unwind the stack -- Ctrl-C, an OOM kill, a spend
 * limit -- leaves the breakage applied with nothing on disk saying so. The
 * window is a whole verifier run, and for a browser case a whole production
 * build. One of the breakages restores the CSS rule that collapses every KaTeX
 * radical, so a kill during it would silently reinstate wrong mathematics.
 *
 * Three things stand between a kill and a corrupted tree: a `.orig` sidecar
 * written before each mutation and removed after a verified restore, handlers
 * that restore from it, and a lock file so two instances cannot interleave.
 */
/* Overridable so the liveness gate can be EXERCISED without touching the real
   lock. The source verifier tests the refusal by pointing a child at a throwaway
   lock path; without this it had to skip whenever a lock existed — which is
   precisely when the harness itself runs it, so the check was unreachable from
   inside the harness and its falsification case was inert. */
const LOCK = process.env.ENDTOEND_LOCK || 'scratch/endtoend-falsification.lock';
const SIDECAR = file => `${file}.endtoend-falsify-orig`;
let activeSidecar = null;

/* ================================================ the lock carries a heartbeat
 *
 * A LOCK PLUS A SIDECAR CANNOT DISTINGUISH "CRASHED" FROM "CASE IN PROGRESS",
 * and that ambiguity has already cost a run. Reading exactly these two files,
 * two independent readers reached OPPOSITE wrong conclusions within the same
 * hour: one concluded the run was dead, the other that it had exited. Acting on
 * the second, `--recover` was run against a LIVE run and restored a file the
 * harness had deliberately mutated mid-case, so that case's verifier ran
 * against unmutated source and the whole result had to be discarded.
 *
 * Neither reader was careless. The artifacts simply did not carry the
 * information needed to answer the question, so answering it required the
 * process table -- and the obvious instrument there, `kill -0`, is unreliable
 * from Git Bash against a native Windows pid and reported an exit that had not
 * happened.
 *
 * So the lock now answers it. It carries a heartbeat refreshed while the run is
 * live; `--recover` refuses while that heartbeat is fresh; and a stale
 * heartbeat with a dead process is the unambiguous crash the recovery path
 * exists for.
 */
/* THE HEARTBEAT IS WRITTEN SYNCHRONOUSLY AROUND EVERY BLOCKING CHILD, never by
 * a timer.
 *
 * The first version used `setInterval`, and measured against a live run its
 * `heartbeatAt` sat 3 ms after `startedAt` for the entire run. A timer cannot
 * fire in a process that blocks its event loop with `spawnSync`, and this
 * harness drives every child that way -- so the heartbeat was inert precisely
 * while a case was running, which is essentially the whole lifetime of a run.
 * That is the same shape as the guard whose domain excluded the only situation
 * it existed for, twice over, in the same session.
 *
 * `runWatched` writes the lock at the only two moments this process can write
 * anything: immediately before handing control to a child and immediately after
 * getting it back. The window is therefore one child, not one timer tick, and
 * the staleness threshold has to exceed the longest single child -- a
 * production build followed by a browser run.
 */
const HEARTBEAT_STALE_MS = 600000;

const writeLock = () => fs.writeFileSync(LOCK, `${JSON.stringify({
  pid: process.pid,
  startedAt: lockStartedAt,
  heartbeatAt: new Date().toISOString(),
  note: 'heartbeatAt is refreshed while this run is live. --recover refuses while it is fresh.',
}, null, 2)}\n`);
const lockStartedAt = new Date().toISOString();

/** What the lock says, and whether it describes a run that is still going. */
function readLock() {
  if (!fs.existsSync(LOCK)) return null;
  const raw = fs.readFileSync(LOCK, 'utf8');
  let record;
  try {
    record = JSON.parse(raw);
  } catch {
    /* A lock from before the heartbeat existed. Treated as stale rather than
       as live, because an old-format lock cannot be shown to be alive -- and
       said out loud rather than assumed. */
    return { pid: Number(raw.trim().split(/\s+/)[0]) || null, heartbeatAt: null, ageMs: Infinity,
      fresh: false, format: 'legacy' };
  }
  const ageMs = Date.now() - Date.parse(record.heartbeatAt);
  return { ...record, ageMs, fresh: Number.isFinite(ageMs) && ageMs < HEARTBEAT_STALE_MS, format: 'heartbeat' };
}

/**
 * Whether a pid is in the process table. This decides whether `--recover` is
 * allowed to touch anything, so it is the single most load-bearing line here.
 *
 * `tasklist` rather than `kill -0`: the latter is unreliable from Git Bash
 * against a native Windows pid and reported an exit that had not happened,
 * which led to a `--recover` over a live run.
 *
 * AND DELIBERATELY WITHOUT `shell: true`. Node then hands argv straight to
 * CreateProcess, so the MSYS layer never sees `/FI` and cannot rewrite it into
 * a path. Through a shell the same call returns
 * `ERROR: Invalid argument/option - 'eq'` and this function answers "not
 * running" for a process that is running -- the dangerous direction, because it
 * invites recovery over a live run. Two other instruments in this environment
 * fail the same way and in the same direction. The `shell: true` used elsewhere
 * in this file for `npx` is correct there and would be a silent catastrophe
 * here; `--is-alive` exists so that this is checked rather than trusted.
 */
function processIsAlive(pid) {
  if (!pid) return false;
  if (process.platform !== 'win32') {
    try { process.kill(pid, 0); return true; } catch { return false; }
  }
  const listed = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/NH'], { encoding: 'utf8' });
  return (listed.stdout ?? '').includes(String(pid));
}

/** Every blocking child goes through here, so the lock is refreshed on both
 *  sides of it. Nothing else in this file may call `spawnSync` for a child;
 *  `processIsAlive` is the one exception and runs when no lock is held. */
function runWatched(command, argv, options) {
  writeLock();
  try {
    return spawnSync(command, argv, options);
  } finally {
    writeLock();
  }
}

/** `--is-alive <pid>` so the liveness probe can be tested from outside rather
 *  than trusted. A `shell: true` added to `processIsAlive` makes this answer
 *  `dead` for a live process, and the source verifier fails on it. */
if (process.argv.includes('--is-alive')) {
  const pid = Number(process.argv[process.argv.indexOf('--is-alive') + 1]);
  console.log(processIsAlive(pid) ? 'alive' : 'dead');
  process.exit(0);
}

/** `--heartbeat-probe` makes one short blocking call through `runWatched` and
 *  prints the lock's heartbeat before and after it.
 *
 *  This is how the heartbeat is checked: by observing it advance across exactly
 *  the kind of call that made the timer version inert, rather than by matching
 *  the source that writes it. Matching source text would re-create the class
 *  this whole round has been removing. */
if (process.argv.includes('--heartbeat-probe')) {
  fs.mkdirSync('scratch', { recursive: true });
  writeLock();
  const before = readLock().heartbeatAt;
  runWatched(process.execPath, ['-e', 'const t=Date.now();while(Date.now()-t<60);'], { encoding: 'utf8' });
  const after = readLock().heartbeatAt;
  fs.rmSync(LOCK, { force: true });
  console.log(JSON.stringify({ before, after, advanced: Date.parse(after) > Date.parse(before) }));
  process.exit(0);
}

fs.mkdirSync('scratch', { recursive: true });

/* `--recover` is the documented way back from a kill. On Windows a forced
 * termination cannot be trapped at all -- `taskkill /F` delivers no catchable
 * signal -- so the handlers below are a courtesy and the SIDECAR is the actual
 * protection. This restores from it and verifies the bytes rather than leaving
 * a human to guess which file was mid-mutation. */
if (process.argv.includes('--recover')) {
  const pending = [...new Set([...cases, ...browserCases].map(item => item.file))]
    .filter(file => fs.existsSync(SIDECAR(file)));
  if (!pending.length && !fs.existsSync(LOCK)) {
    console.log('Nothing to recover: no sidecar and no lock.');
    process.exit(0);
  }
  /* THE LIVENESS GATE. Recovery restores a file the harness may have mutated
     one second ago on purpose; doing that to a live run silently invalidates
     its result, which has already happened once. A fresh heartbeat, or a pid
     still in the process table, means this is not a crash. */
  const held = readLock();
  /* THE PROCESS CHECK DECIDES; the heartbeat only corroborates.
     It used to be `fresh || alive`, which was correct only because the window
     was 30 s and everything fell through to the process check almost at once.
     With a window wide enough for a real child, that form would report a DEAD
     process as live for ten minutes -- a worse failure than the one it fixes.
     So the heartbeat now decides only when no pid was recorded at all. */
  const live = held && (held.pid ? processIsAlive(held.pid) : held.fresh);
  if (live && !process.argv.includes('--force-recover')) {
    throw new Error(
      `refusing to recover: a falsification run appears to be LIVE. Its lock names pid ${held.pid}`
      + `${held.heartbeatAt ? `, whose heartbeat is ${Math.round(held.ageMs / 1000)}s old` : ' (legacy lock)'}`
      + `${processIsAlive(held.pid) ? ' and that pid is in the process table' : ''}. `
      + 'Restoring now would un-mutate a file the run deliberately broke, and its result would have to be '
      + 'discarded. Wait for it to finish. If you are certain it is wedged, re-run with --force-recover.');
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
  console.log(`recovered ${pending.length} file(s) and cleared the lock. `
    + 'Re-run the verifiers before trusting the tree.');
  process.exit(0);
}

/* The same distinction at startup. These two situations used to produce one
   message, which is what made the artifacts unreadable. */
const existingLock = readLock();
if (existingLock) {
  const alive = existingLock.fresh || processIsAlive(existingLock.pid);
  throw new Error(alive
    ? `${LOCK} names pid ${existingLock.pid} and that run is LIVE`
      + `${existingLock.heartbeatAt ? ` (heartbeat ${Math.round(existingLock.ageMs / 1000)}s old)` : ''}. `
      + 'Two instances mutating the same files is how a breakage was left on disk before. Wait for it.'
    : `${LOCK} names pid ${existingLock.pid}, which is NOT running`
      + `${existingLock.heartbeatAt ? ` and whose heartbeat is ${Math.round(existingLock.ageMs / 1000)}s old`
        : ' (legacy lock with no heartbeat)'}. That run was killed. `
      + `Run \`node scripts/falsify-endtoend.mjs --recover\` to restore any ${SIDECAR('<file>')} sidecar `
      + 'and clear the lock, then start again.');
}
/* A stale sidecar means a previous run died mid-case. Restoring is not this
 * run's job to guess at -- it is reported, loudly, and the run refuses. */
const staleSidecars = [...new Set([...cases, ...browserCases].map(item => item.file))]
  .filter(file => fs.existsSync(SIDECAR(file)));
if (staleSidecars.length) {
  throw new Error(`stale sidecars from an interrupted run: ${staleSidecars.join(', ')}. `
    + 'Each holds the original bytes of the file beside it. Restore them before running again.');
}
writeLock();

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

const allCases = withBrowser ? [...cases, ...browserCases] : cases;
/* The whole loop sits inside a try/finally. An evidence restore that followed
   the loop would be skipped by a case that throws, leaving the broken page's
   evidence on disk -- exactly what the snapshot exists to prevent. */
try {
  for (const item of allCases) {
    const original = fs.readFileSync(item.file);
    const before = crypto.createHash('sha256').update(original).digest('hex');
    const text = original.toString('utf8');
    if (!text.includes(item.find)) {
      results.push({ name: item.name, file: item.file, guards: item.guards, applied: false, fired: false,
        note: `the target text was not found in ${item.file}; this breakage could not be applied` });
      continue;
    }
    const occurrences = text.split(item.find).length - 1;
    /* AN AMBIGUOUS ANCHOR IS REFUSED, not silently resolved.
       `String.replace` with a string pattern changes the FIRST occurrence only.
       When a case's anchor appears more than once, the mutation may land
       somewhere other than the site the case is about -- and it did: a case
       targeting this file matched its own `find` field before the code it meant
       to break, so it mutated its own definition, left the guard intact, and
       reported INERT for a reason that had nothing to do with the guard. */
    if (occurrences > 1) {
      results.push({ name: item.name, file: item.file, guards: item.guards, applied: false, fired: false,
        occurrencesPresent: occurrences,
        note: `the anchor for this breakage occurs ${occurrences} times in ${item.file}, so a replacement `
          + 'would change the first one rather than necessarily the intended one. Lengthen the anchor until '
          + 'it is unique.' });
      continue;
    }
    /* The sidecar is written and flushed BEFORE the mutation, so a kill at any
       point after this line leaves the original recoverable on disk. */
    const sidecar = SIDECAR(item.file);
    fs.writeFileSync(sidecar, original);
    activeSidecar = { file: item.file, sidecar };
    const mutated = text.replace(item.find, item.replace);
    // Measured, not assumed. `String.replace` with a string pattern replaces the
    // first occurrence only, so reporting a constant 1 would be a claim about a
    // substitution nobody counted.
    const replaced = occurrences - (mutated.split(item.find).length - 1);
    fs.writeFileSync(item.file, mutated, 'utf8');
    let outcome;
    try {
      if (item.rebuild) {
        const built = runWatched('npx', ['vite', 'build', '--outDir', 'dist-e2e'],
          { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
        if (built.status !== 0) throw new Error(`the build failed while applying "${item.name}"`);
      }
      const [command, argv] = item.run;
      const run = runWatched(command, argv, {
        encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
        env: { ...process.env, DIST_DIR: 'dist-e2e' },
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
    // not match it stays on disk, and the next run refuses to start until it is
    // dealt with rather than quietly mutating a file that is already wrong.
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
    /* THE RESTORING REBUILD, through the same call as the case builds and with
       its result CHECKED.
       It used to be a bare `spawnSync` whose status nobody looked at, while the
       case builds used a different call with a different buffer. A build
       directory that outlives the sources it was built from is not a
       theoretical hazard: a stale one caused a browser verifier failure in this
       session that looked exactly like a page regression, and cost a
       misdiagnosis before a clean rebuild settled it. `allRestoredExactly` was
       reporting true throughout, because it describes SOURCES and is blind to
       the build -- which is why it is now named for what it covers. */
    const restored = runWatched('npx', ['vite', 'build', '--outDir', 'dist-e2e'],
      { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
    buildRestored = restored.status === 0;
    if (!buildRestored) {
      process.stderr.write('\n[falsify] THE RESTORING REBUILD FAILED. dist-e2e still contains the last '
        + "breakage's build; delete it and rebuild before running anything against a preview.\n");
    }
  }
  for (const [file, bytes] of Object.entries(evidenceSnapshot)) fs.writeFileSync(file, bytes);
  fs.rmSync(LOCK, { force: true });
}

const report = {
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-endtoend.mjs',
  harnessSha256: digest('scripts/falsify-endtoend.mjs'),
  mode: withBrowser ? 'offline and browser' : 'offline only',
  breakagesApplied: results.length,
  guardsThatFired: fired,
  /* NAMED FOR WHAT IT COVERS. This was `allRestoredExactly`, which reads as a
     claim about everything the run touched and is in fact only about SOURCE
     files -- it reported true while a stale build directory sat on disk. A
     field that overstates its scope is the record-contradicts-tree class, so
     the build has its own field and its own name.
     `!== false` also counted a result with NO `restoredExactly` field -- what
     the not-applied branch pushes -- as restored exactly, reporting success for
     a file nothing touched. */
  allSourcesRestoredExactly: results.every(
    result => result.applied === false || result.restoredExactly === true),
  buildDirectoryRestored: buildRestored,
  breakagesNotApplied: results.filter(result => result.applied === false).length,
  evidenceFilesRestored: evidenceFiles,
  results,
  browserCasesIncluded: withBrowser ? browserCases.length : 0,
  browserCasesDeferred: withBrowser ? [] : browserCases.map(item => item.name),
  method: 'Each breakage is applied alone to a clean tree, the relevant verifier is run, the file is restored '
    + 'and the restoration checked by SHA-256. A case counts as fired only when the verifier exits non-zero '
    + 'AND its output names the guard, so a run that fails for an unrelated reason is not credited.',
  limitations: [
    'The coverage floors in the model suite are set below its current totals, so they catch a wholesale loss '
      + 'of checking rather than the removal of one assertion. Deleting a single record() call does NOT trip '
      + 'them, and the case above neuters the counter instead. Tightening the floors to today\'s numbers '
      + 'would make them fail on every legitimate addition, which is how a floor gets deleted rather than '
      + 'fixed.',
    'Browser guards need a running preview and a production build; without --browser they are listed as '
      + 'deferred rather than silently omitted.',
    'Crash safety is best-effort: SIGKILL and a power loss cannot be trapped, and on Windows a forced '
      + 'termination delivers no catchable signal at all. In those cases the `.orig` sidecar beside the '
      + 'mutated file holds its original bytes, and the next run refuses to start until the sidecar is dealt '
      + 'with, rather than mutating a file that is already wrong. `--recover` restores from it and verifies '
      + 'the bytes.',
    'A guard that fires here is a guard that can fail. It is not evidence that its domain covers every '
      + 'defect of its kind -- that is what the whole-grid sweeps in the model suite are for.',
  ],
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
/* A NARROWER RUN MAY NOT OVERWRITE A BROADER RECORD.
 *
 * This file is rewritten wholesale by every invocation, so it cannot be read as
 * cumulative — and it has now misled a reader twice in one session. An
 * offline-only run replaced a 51-case offline-and-browser record with a 43-case
 * one, and the record then understated what had been proved while
 * `buildDirectoryRestored` reverted to null.
 *
 * The standing advice was "run offline-only before `--browser`, never after",
 * and the advice is correct — but a rule that must be remembered is not a fix.
 * The artifact now enforces it: a run covering fewer cases than the record on
 * disk writes itself beside that record instead of over it, and says so. The
 * same conclusion reached about the lock — put the property in the artifact,
 * not in the reader's discipline.
 */
const reportPath = process.env.ENDTOEND_FALSIFICATION_REPORT
  || 'docs/teaching/evidence/endtoend-falsification.json';
const chooseReportTarget = () => {
  if (!fs.existsSync(reportPath)) return { path: reportPath, narrowed: false };
  let existing;
  try { existing = JSON.parse(fs.readFileSync(reportPath, 'utf8')); } catch { return { path: reportPath, narrowed: false }; }
  const existingBrowser = existing.browserCasesIncluded ?? 0;
  const existingApplied = existing.breakagesApplied ?? 0;
  const narrower = (report.browserCasesIncluded < existingBrowser)
    || (report.browserCasesIncluded === existingBrowser && report.breakagesApplied < existingApplied);
  if (!narrower) return { path: reportPath, narrowed: false };
  return { path: reportPath.replace(/\.json$/, `-${withBrowser ? 'browser' : 'offline'}.json`), narrowed: true };
};
const target = chooseReportTarget();
if (process.argv.includes('--report-target-probe')) {
  console.log(JSON.stringify({ path: target.path, narrowed: target.narrowed }));
  process.exit(0);
}
fs.writeFileSync(target.path, `${JSON.stringify(report, null, 2)}\n`);
if (target.narrowed) {
  process.stderr.write(`\n[falsify] this run covers fewer cases than the record already at ${reportPath}, `
    + `so it was written to ${target.path} instead of replacing it. Re-run with --browser for a record that `
    + 'supersedes it.\n');
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
