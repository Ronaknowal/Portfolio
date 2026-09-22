// Falsification harness for the problem-formulation lesson's guards.
//
// A passing verifier proves nothing about a guard that cannot fail. Every inert
// guard found in this repository so far was correct within a domain that
// excluded the defect: a leak check matching values surrounded by spaces when
// the DOM renders them glued to their neighbours, a clearance check sampling a
// curve the browser does not paint, a containment check comparing positions
// against the very inset it was meant to guard, and hundreds of assertions
// comparing a function with itself inside an advertised grid. So each guard
// here is shown to fire, by breaking exactly the thing it claims to watch.
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
//     falsifying -- not even when a case throws. It touches only this lesson's
//     own evidence: a harness that globbed the directory once came close to
//     deleting hundreds of a sibling's captures;
//   * the pre-mutation bytes go to a `.orig` sidecar on disk BEFORE the
//     mutation and are removed only after a verified restore, with signal and
//     exception handlers that restore from it, so a kill mid-case cannot leave
//     a breakage applied with nothing on disk saying so;
//   * a lock file refuses a second concurrent instance.
//
// Browser guards are not falsifiable without a running preview and are deferred
// to phase C. Passing `--browser` adds those cases; without it they are listed
// in the report as deferred rather than silently omitted.
//
// Run: node scripts/falsify-formulation.mjs
//      node scripts/falsify-formulation.mjs --recover   (after a kill)
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const python = process.env.PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const MODELS = ['node', ['scripts/verify-formulation-models.mjs', '--no-evidence']];
const SOURCES = [python, ['scripts/verify-formulation-sources.py', '--no-evidence']];
const DATA = [python, ['scripts/verify-formulation-data.py', '--no-evidence']];
const EXAMPLES = [python, ['scripts/verify-formulation-examples.py', '--no-evidence']];
const BROWSER = ['node', ['scripts/verify-formulation-browser.cjs']];

const MODELS_FILE = 'src/learn/data/formulation-models.js';
const DATA_FILE = 'src/learn/data/formulation-data.js';
const EXAMPLES_FILE = 'src/learn/data/formulation-examples.js';
const TOPIC_FILE = 'src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx';
const LABS_FILE = 'src/learn/components/lesson-labs/FormulationLabs.jsx';
const FIGURES_FILE = 'src/learn/components/lesson-labs/FormulationFigures.jsx';
const SHARED_FILE = 'src/learn/components/lesson-labs/FormulationShared.jsx';
const CSS_FILE = 'src/learn/components/lesson-labs/formulation-labs.css';
const EXAMPLES_VERIFIER = 'scripts/verify-formulation-examples.py';
const DATA_VERIFIER = 'scripts/verify-formulation-data.py';
const ATTRIBUTION = 'public/learn-assets/problem-formulation/ATTRIBUTION.txt';

const withBrowser = process.argv.includes('--browser');

/** Each case breaks one thing and names the guard that must notice. */
const cases = [
  {
    name: 'the availability predicate is dropped, so a value that had not arrived becomes usable',
    file: MODELS_FILE,
    find: '  const known = record.available <= cutoff;',
    replace: '  const known = true;',
    run: MODELS,
    expect: /the two routes select different records|arrives at \d+, which is later|destination note/,
    guards: 'the whole-grid as-known sweep, and the leakage property asserted on every one of its cases: the '
      + 'selected record never arrives after the cutoff',
  },
  {
    name: 'the age window becomes exclusive, so an event exactly at the boundary is rejected',
    file: MODELS_FILE,
    find: '  const withinAge = record.event >= cutoff - maximumAge;',
    replace: '  const withinAge = record.event > cutoff - maximumAge;',
    run: MODELS,
    expect: /the two routes disagree at cutoff|an arrival exactly at the cutoff is admissible/,
    guards: 'boundary inclusivity stated as an equality, and the two-route sweep',
  },
  {
    name: 'the timeline loses its inset, which used to satisfy its own containment check',
    file: MODELS_FILE,
    find: 'width = 300, laneHeight = 21, labelWidth = 58, inset = 14, topPadding = 26,',
    replace: 'width = 300, laneHeight = 21, labelWidth = 58, inset = 0, topPadding = 26,',
    run: MODELS,
    expect: /the timeline inset is 0, below the 10 units/,
    guards: 'the fixed minimum inset, checked before any position is compared against it',
  },
  {
    name: 'precision is divided by the evaluated set rather than by the capacity asked for',
    file: MODELS_FILE,
    find: '    precision: found / capacity,',
    replace: '    precision: found / order.length,',
    run: MODELS,
    expect: /precision at \d+|precision at 25 is \.48|the drawn lengths order/,
    guards: 'precision recomputed at every capacity the control offers, for all three rankings',
  },
  {
    name: 'the unchanged rule becomes strict, so an exactly-tolerant difference is reported as movement',
    file: MODELS_FILE,
    find: "  if (Math.abs(difference) <= tolerance) return { outcome: 'unchanged', difference: 0, rawDifference: difference };",
    replace: "  if (Math.abs(difference) < tolerance) return { outcome: 'unchanged', difference: 0, rawDifference: difference };",
    run: MODELS,
    expect: /the tolerance boundary is inclusive|unchanged iff the realised difference/,
    guards: 'the unchanged rule asserted as an if and only if, at the exact boundary',
  },
  {
    name: 'the ranking tie rule reverses, so tied cases are ordered by the wrong source row',
    file: MODELS_FILE,
    find: '    .sort((left, right) => (scores[right] - scores[left]) || (ids[left] - ids[right]))',
    replace: '    .sort((left, right) => (scores[right] - scores[left]) || (ids[right] - ids[left]))',
    run: MODELS,
    expect: /the ranking against the packet|the ranking against a repeated-maximum route/,
    guards: 'the ranking against the packet\'s numpy lexsort and against a quadratic repeated-maximum route',
  },
  {
    name: 'average precision weights each step by the recall instead of the precision',
    file: MODELS_FILE,
    find: '    total += (recall - previousRecall) * precision;',
    replace: '    total += (recall - previousRecall) * recall;',
    run: MODELS,
    expect: /average precision by definition, implemented here|the module's average precision|exactly the validation positive fraction/,
    guards: 'average precision implemented from its definition against scikit-learn\'s recorded value, and '
      + 'against the constant baseline\'s closed form',
  },
  {
    name: 'log loss drops the term for the outcomes that did not happen',
    file: MODELS_FILE,
    find: '    total += targets[index] === 1 ? -Math.log(p) : -Math.log(1 - p);',
    replace: '    total += targets[index] === 1 ? -Math.log(p) : 0;',
    run: MODELS,
    expect: /log loss by definition|the module's log loss|the closed form for a constant probability/,
    guards: 'log loss written out against scikit-learn\'s recorded value and against its closed form for a '
      + 'constant probability',
  },
  {
    name: 'a metric bar is drawn from the wrong end of its axis',
    file: MODELS_FILE,
    find: '      endX: scale(value),',
    replace: '      endX: scale(axis.domain[1] - value),',
    run: MODELS,
    expect: /bar ends at its own value|the drawn lengths order differently from the values/,
    guards: 'every bar required to end at its own value on its own scale, and the drawn order required to '
      + 'match the value order',
  },
  {
    name: 'the feedback channel returns the outcome to the prediction rather than to the record store',
    file: MODELS_FILE,
    find: '    toId: first.id,',
    replace: "    toId: 'prediction',",
    run: MODELS,
    expect: /returns to the record store, not to the prediction/,
    guards: 'the decision-flow figure\'s central claim, asserted as data rather than captioned',
  },
  {
    name: 'a partition segment is drawn at a fixed fraction instead of its own row count',
    file: MODELS_FILE,
    find: '      width: (group.rows / scaleTotal) * width,',
    replace: '      width: 0.5 * width,',
    run: MODELS,
    expect: /width encodes its row count|both bars share one scale/,
    guards: 'each partition segment\'s width recomputed from its row count',
  },
  {
    name: 'the cost threshold puts the wrong cost on top',
    file: MODELS_FILE,
    find: '  return falsePositiveCost / (falsePositiveCost + falseNegativeCost);',
    replace: '  return falseNegativeCost / (falsePositiveCost + falseNegativeCost);',
    run: MODELS,
    expect: /the threshold closed form|the threshold found by bisection|practice 6/,
    guards: 'the threshold against its closed form and against a bisection on the loss difference, at all '
      + '144 cost pairs',
  },
  {
    name: 'an exchange no longer requires the dropped case to be selected',
    file: MODELS_FILE,
    find: '  demand(removeAt < capacity, `${removeId} is not currently selected, so it cannot be swapped out`);',
    replace: '  const unusedRemoveCheck = removeAt;',
    run: MODELS,
    expect: /exchanging two unselected opportunities: should have been refused/,
    guards: 'the refusal cases around the exchange control',
  },
  {
    name: 'a measured count is altered in the generated data module',
    file: DATA_FILE,
    find: '      top50Positives: 20,',
    replace: '      top50Positives: 22,',
    run: MODELS,
    expect: /positives in the selected 50|the module carries the count|top-50 count/,
    guards: 'every measured count against the frozen packet results',
  },
  {
    name: 'the same measured count is altered, against the served dataset',
    file: DATA_FILE,
    find: '      top50Positives: 20,',
    replace: '      top50Positives: 22,',
    run: DATA,
    expect: /byte for byte|reproduces src\/learn\/data\/formulation-data\.js/,
    guards: 'byte-identical regeneration of the data module from the served dataset',
  },
  {
    name: 'a trust-root comparison is removed, so a block stops being checked',
    file: DATA_VERIFIER,
    find: '    compare_tree(validation_targets, recorded["validationTargets"], "/validationTargets")',
    replace: '    pass  # comparison removed',
    run: DATA,
    expect: /trust-root leaves were never compared/,
    guards: 'the measured trust-root coverage, which fails when a block stops being compared rather than '
      + 'quietly reporting a smaller number',
  },
  {
    name: 'a recorded program output is altered',
    file: EXAMPLES_FILE,
    find: 'expected: "5 10\\n7 12\\n9 20\\nNone"',
    replace: 'expected: "5 20\\n7 12\\n9 20\\nNone"',
    run: MODELS,
    expect: /agrees with the browser model at cutoff/,
    guards: "the displayed programs' recorded output against what the browser models compute",
  },
  {
    name: 'the pinned digest of an extracted program no longer matches the manuscript',
    file: EXAMPLES_VERIFIER,
    find: '"sha256": "fb9be4a437ef9d61386ae35543ae5509632d15bc8f452fb5968f9bef6bece4f9"',
    replace: '"sha256": "0000000000000000000000000000000000000000000000000000000000000000"',
    run: EXAMPLES,
    expect: /the extracted algorithm now hashes/,
    guards: 'the SHA-256 pin on the bytes extracted from the frozen manuscript',
  },
  {
    name: 'the attribution stops stating the dataset digest',
    file: ATTRIBUTION,
    find: '7e59cf650004d65d1c9d6b08553bad2ee9a9ad70d594f536e3c584ee6ed5df50. The upstream archive',
    replace: '0000000000000000000000000000000000000000000000000000000000000000. The upstream archive',
    run: DATA,
    expect: /the attribution states/,
    guards: 'the attribution file required to carry the licence, the digest, the DOI and the creator',
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
    guards: 'the raw-control-byte scan over every owned file and every verifier',
  },
  {
    name: 'a required KaTeX sequence disappears from the lesson body',
    file: TOPIC_FILE,
    // A sequence that occurs ONCE. `String.replace` takes the first match
    // only, and this body uses tfrac twice, so breaking that one left the
    // sequence still present and the guard correctly silent. A breakage that
    // breaks nothing proves nothing about the guard it points at.
    find: 'c_u\\\\bigl(1-F(q)\\\\bigr)',
    replace: 'c_u(1-F(q))',
    run: SOURCES,
    expect: /the KaTeX sequence \\\\bigl is missing|the KaTeX sequence \\\\bigr is missing/,
    guards: 'the required-KaTeX-sequence list',
  },
  {
    /* An EXPRESSION-valued pair, which is the shape that actually occurred.
       This case used to inject `flag ? 'same' : 'same'` and pass, while the one
       genuine instance in the tree -- `Number.isInteger(value) ?
       sign(String(value)) : sign(String(value))` -- sat unflagged in a file the
       same scan was reading, because the regex matched only quoted literals.
       The guard collected a green tick on a shape it could not catch. */
    name: 'a conditional is left with two identical expression branches',
    file: LABS_FILE,
    find: 'const stateKey = state => JSON.stringify(state);',
    replace: 'const stateKey = state => JSON.stringify(state);\n'
      + 'const lost = flag => (flag ? String(flag).trim() : String(flag).trim());',
    run: SOURCES,
    expect: /a conditional whose branches are identical/,
    guards: 'the identical-branch scan, widened from quoted literals to the construct',
  },
  {
    name: 'the same defect in its string-literal spelling',
    file: FIGURES_FILE,
    find: "const procedureShort = { prior: 'baseline', candidate: 'candidate', duration: '+ duration' };",
    replace: "const procedureShort = { prior: 'baseline', candidate: 'candidate', duration: '+ duration' };\n"
      + "const alsoLost = flag => (flag ? 'same' : 'same');",
    run: SOURCES,
    expect: /a conditional whose branches are identical/,
    guards: 'the same scan, on the spelling it could already see — both must fire, or widening it traded '
      + 'one blind spot for another',
  },
  {
    name: 'the lesson body reaches for a member of the global Math, which is shadowed there',
    file: TOPIC_FILE,
    find: 'const partition = formulationData.partition;',
    replace: 'const partition = formulationData.partition;\nconst shadowed = Math.round(2.4);',
    run: SOURCES,
    expect: /Math\.round in a module where `Math` is the KaTeX component/,
    guards: 'the shadowed-Math scan',
  },
  {
    name: 'the SVG layout rule is unscoped, putting it back in front of KaTeX\'s radicals',
    file: CSS_FILE,
    find: '.formulation-lesson svg.form-svg {',
    replace: '.formulation-lesson svg {',
    run: SOURCES,
    expect: /a bare svg selector|scoped to this lesson's own SVG class/,
    guards: 'the bare-selector scan, which is the only thing standing between a stylesheet and a wrong formula',
  },
  {
    name: 'a figure writes its own svg element, outside the wrapper that applies the layout class',
    file: FIGURES_FILE,
    find: "const procedureShort = { prior: 'baseline', candidate: 'candidate', duration: '+ duration' };",
    replace: "const procedureShort = { prior: 'baseline', candidate: 'candidate', duration: '+ duration' };\n"
      + 'const stray = <svg viewBox="0 0 10 10" />;',
    run: SOURCES,
    expect: /an <svg> element written outside the shared Drawing wrapper/,
    guards: 'the scan that forbids a hand-written svg, so the layout class cannot be forgotten by a new figure',
  },
  {
    name: 'the lesson reaches into another lesson\'s asset directory',
    file: TOPIC_FILE,
    find: 'const partition = formulationData.partition;',
    replace: "const partition = formulationData.partition;\nconst borrowed = '/learn-assets/evaluation-metrics/banknote-subset.csv';",
    run: SOURCES,
    expect: /belongs to another lesson/,
    guards: 'the own-assets-only scan',
  },
  {
    name: 'the ranked scores are printed in the six-decimal graded form, blunting the leak pin',
    file: SHARED_FILE,
    find: 'export const probabilityText = value => sign(value.toFixed(4));',
    replace: 'export const probabilityText = value => sign(value.toFixed(6));',
    run: SOURCES,
    expect: /probabilityText really is four decimals|printed through probabilityText/,
    guards: 'the convention that only a graded quantity is printed at six decimals, which is what the '
      + 'browser leak pin depends on',
  },
  {
    /* B1, the blocking one. The allowance must never be tighter than the
       precision the page prints; at 1e-9 the page showed a learner's 0.666667
       and its own 0.666667 as the same string and declared them different. */
    name: 'the graded allowance drops below the precision the page prints',
    file: MODELS_FILE,
    find: '  return Math.max(declared, displayTolerance(digits));',
    replace: '  return declared;',
    run: MODELS,
    expect: /a learner copying the page's own number|would be told it is outside/,
    guards: 'the printed-back sweep over every capacity the control offers: the page\'s own printed value, '
      + 'read back, must grade as correct',
  },
  {
    name: 'the age-window truncation flag goes back to being computed and unread',
    file: LABS_FILE,
    find: 'geometry.ageWindow.startsBeforeTheRuler',
    replace: 'false /* flag ignored */',
    run: MODELS,
    expect: /reads the truncation flag rather than leaving it computed and unused/,
    guards: 'the assertion that the figure consumes the flag, not only that the model computes it',
  },
  {
    name: 'the arrive-earlier preset goes back to matching the queried entity',
    file: MODELS_FILE,
    find: 'export function withArrival(fixture, { entity = fixture.entity, event, version, available }) {',
    replace: 'export function withArrival(fixture, { event, version, available }) {\n'
      + '  const entity = fixture.entity;',
    run: MODELS,
    expect: /left sensor A's delayed record at|moving a record that does not exist/,
    guards: 'the preset asserted to move the record it names under both entity selections',
  },
  {
    /* S9. A one-character JSX error in a sibling broke `vite build` repo-wide
       and was invisible to every offline verifier here. A parse failure in a
       file this checker owns must now fail this checker. */
    name: 'an unescaped apostrophe inside a single-quoted string breaks a component',
    file: FIGURES_FILE,
    find: "const procedureShort = { prior: 'baseline', candidate: 'candidate', duration: '+ duration' };",
    replace: "const procedureShort = { prior: 'baseline', candidate: 'that study's', duration: '+ duration' };",
    run: SOURCES,
    expect: /does not parse/,
    guards: 'the parse step over every owned .jsx, .js, .mjs and .cjs source',
  },
  {
    /* The floors sit just below the current values on purpose, so they catch a
     * WHOLESALE loss rather than the removal of one call. Breaking a single
     * `record()` therefore does not trip them -- correct behaviour, and the
     * reason this case neuters the counter instead. The limitation is stated
     * in the report rather than hidden by a floor tightened to today's number. */
    name: 'the model suite stops counting what it checked',
    file: 'scripts/verify-formulation-models.mjs',
    find: 'const record = name => { counts[name] = (counts[name] ?? 0) + 1; };',
    replace: 'const record = () => {};',
    run: MODELS,
    expect: /the suite has lost coverage|groups ran/,
    guards: 'the floors under the grouped-check and group counts, which stop a suite that has stopped '
      + 'checking from still printing PASS',
  },
  {
    name: 'the source-hygiene suite stops counting what it checked',
    file: 'scripts/verify-formulation-sources.py',
    find: '    checks += 1\n    if not condition:',
    replace: '    checks += 0\n    if not condition:',
    run: SOURCES,
    expect: /failable assertions ran; the suite has lost coverage/,
    guards: 'the floor under the hygiene-check count',
  },
];

/* Browser cases need a production build and a running preview, so they are off
   by default and run with `--browser`. Each one costs a rebuild. */
const browserCases = [
  {
    name: 'investigation 1 renders its reveal before a prediction is committed',
    file: LABS_FILE,
    find: '  const lookup = shown?.answer.lookup ?? null;',
    replace: '  const lookup = shown?.answer.lookup ?? (problem ? null : latestKnown(draft));',
    rebuild: true,
    run: BROWSER,
    expect: /shows its reveal before any prediction|renders "Selected record" before|shows its graded value/,
    guards: 'the leak property: the graded quantity absent before commitment, pinned numerically and by the '
      + 'phrases only the reveal renders',
  },
  {
    name: 'investigation 2 shows every case\'s outcome while the learner is still predicting',
    file: LABS_FILE,
    find: "        result ? (targetById[row.id] === 1 ? 'subscribed' : 'did not subscribe') : 'hidden until you apply',",
    replace: "        targetById[row.id] === 1 ? 'subscribed' : 'did not subscribe',",
    rebuild: true,
    run: BROWSER,
    expect: /renders "subscribed" before a prediction is recorded|hidden until you apply/,
    guards: 'the stronger form of the leak property on this page: NO outcome is rendered for any of the 824 '
      + 'cases until a prediction is committed',
  },
  {
    /* The one breakage that protects a MATHEMATICAL claim rather than a layout
       one: a stylesheet collapsing part of a formula, leaving the DOM correct
       and the rendered mathematics wrong.

       It is NOT the unscoped-`svg`-rule breakage a sibling lesson uses. This
       lesson's formulas contain no radical, no stretchy delimiter and no
       KaTeX-drawn accent, so they emit no `.katex svg` at all; unscoping the
       rule here would collapse nothing and prove nothing. What this page does
       use are fraction spans, and collapsing those is the same defect reaching
       the same reader through the element this page actually has. The
       bare-selector guard itself is falsified offline instead, where it fires. */
    name: 'a stylesheet collapses every rendered formula on the page',
    file: CSS_FILE,
    /* Two earlier choices for this breakage were INERT and were measured, not
       assumed. `.katex .mfrac { height: 0 }` does nothing because KaTeX sets
       those spans inline and `height` on an inline box is ignored.
       `.katex-display { height: 0 !important }` reached the built stylesheet --
       verified with grep -- and the browser still reported a 16 px box, so the
       collapse never happened and the guard was right to stay silent. A
       breakage that breaks nothing proves nothing about the guard it points at.
       `font-size: 0` on `.katex` is a collapse the browser cannot ignore: every
       glyph box goes to zero in BOTH dimensions, measured at 0 by 0 before this
       case was written. It is also the truer analogue of the original defect --
       mathematics that is present in the DOM, correct in its markup, and
       unreadable on screen. */
    find: '.formulation-lesson blockquote {',
    replace: '.formulation-lesson .katex { font-size: 0 !important; }\n'
      + '.formulation-lesson blockquote {',
    rebuild: true,
    run: BROWSER,
    expect: /KaTeX element carrying text paints with no height/,
    guards: 'the measurement of every KaTeX box carrying text, in BOTH dimensions, on the rendered page',
  },
  {
    name: 'a stylesheet rule paints a partition band black, so it vanishes into the page ground',
    file: CSS_FILE,
    find: '.formulation-lesson svg.form-svg .form-partition.is-development rect { fill: #24312b; stroke: #6f8f7d; }',
    replace: '.formulation-lesson svg.form-svg .form-partition.is-development rect { fill: #000000; stroke: none; }',
    rebuild: true,
    run: BROWSER,
    expect: /black fill with no stroke|attribute overridden/,
    guards: 'the painted-against-declared check, which compares computed styles rather than attributes',
  },
  {
    /* NOT the inset, and not the top padding. Both of those are now refused by
       a floor in the model layer, so the page never renders and the browser
       verifier fails on an unrelated timeout -- defence in depth, but useless
       as a proof that the BROWSER inspector works.

       This breaks something the model does not model: the drawing renders the
       readable sixteen-character identity instead of the compact one, while the
       model still computes its width from the compact form, so no floor fires.
       It is the exact defect the curve sampler found by looking -- the arrival
       diamond running through the label of the record it names. */
    /* The defect this lesson shipped three times before a guard existed: a JSX
       expression at the end of a line loses the space after it, and the value
       and the next word render glued. Invisible in the source, invisible in the
       DOM structure, plainly visible to a reader. Two were found by opening
       screenshots; this is the third, and it is the one the guard was written
       against. */
    name: 'a JSX expression loses the space after it and glues a value to the next word',
    file: FIGURES_FILE,
    find: "they come out at {round(costThreshold(OVERAGE, UNDERAGE), 6)}{' '}",
    replace: 'they come out at {round(costThreshold(OVERAGE, UNDERAGE), 6)}',
    rebuild: true,
    run: BROWSER,
    expect: /two words are glued together in the rendered prose/,
    guards: 'the rendered-prose scan for a value touching the word after it, over every prose element on the page',
  },
  {
    /* B2, the second blocking one. The caption must name the state the grader
       uses. It named capacity 50 unconditionally while the grader compared
       against the previously committed state, so from round two a learner
       following the instruction was marked wrong in a verdict that confirmed
       their number. */
    name: 'the baseline caption goes back to naming the section-5 starting point every round',
    file: LABS_FILE,
    find: '      {isFirstRound',
    replace: '      {true',
    rebuild: true,
    run: BROWSER,
    expect: /the second round still cites section 5 as the baseline|the caption must name the state the grader uses/,
    guards: 'the two-round browser case that reads the caption off the page and requires it to agree with the '
      + 'grading',
  },
  {
    /* MOVED, not deleted. Deleting it was inert: `Table` also renders its
       optional footnote with the caption class, so a check for "some caption"
       found the footnote and passed. The check is keyed on the labelling
       element now, and this breakage does what its name says. */
    name: 'a table caption is moved inside the scroll box it must sit outside',
    file: SHARED_FILE,
    find: '    <p className="formulation-caption" id={id}>{caption}</p>\n'
      + '    <div className="formulation-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>',
    replace: '    <div className="formulation-table-scroll" role="region" aria-labelledby={id} tabIndex={0}>\n'
      + '      <p className="formulation-caption" id={id}>{caption}</p>',
    rebuild: true,
    run: BROWSER,
    expect: /a table renders no caption at all|a caption sits inside the scroll box/,
    guards: 'the positive caption check, which replaced an absence assertion over a tag this lesson never emits',
  },
  {
    name: 'the timeline draws the readable identity, which overruns its gutter',
    file: LABS_FILE,
    find: '<text className="form-small" x={0} y={lane.markY + 3}>{lane.drawnLabel}</text>',
    replace: '<text className="form-small" x={0} y={lane.markY + 3}>{lane.label}</text>',
    rebuild: true,
    run: BROWSER,
    expect: /A curve runs through a label|Figure layout collides/,
    guards: 'the curve sampler, which walks every path and polyline against every label box because the '
      + 'shared inspector reads straight lines only',
  },
];

/* This harness's own report, which must NOT be snapshotted.
 *
 * It was, and the two safety mechanisms cancelled each other: the provisional
 * `status: running` record written before the first mutation exists so that a
 * throw mid-run cannot leave the previous green report on disk, but the
 * `finally` wrote the snapshot back -- the previous green report -- and the
 * throw then skipped the real write. The exact failure the provisional record
 * was written to prevent restored `guardsThatFired: 35` and exited non-zero.
 * Excluding it from the snapshot is the whole fix. */
const OWN_REPORT = 'docs/teaching/evidence/formulation-falsification.json';

/* ONLY this lesson's evidence. A harness that globbed the whole directory once
   came close to deleting 709 of a sibling lesson's captures. */
const evidenceFiles = [
  ...(fs.existsSync('docs/teaching/evidence')
    ? fs.readdirSync('docs/teaching/evidence')
      .filter(name => name.startsWith('formulation-') && name.endsWith('.json'))
      .map(name => `docs/teaching/evidence/${name}`)
      .filter(file => file !== OWN_REPORT)
    : []),
  ...(fs.existsSync('docs/teaching/evidence/screenshots')
    ? fs.readdirSync('docs/teaching/evidence/screenshots')
      .filter(name => name.startsWith('formulation-'))
      .map(name => `docs/teaching/evidence/screenshots/${name}`)
    : []),
];
const evidenceSnapshot = Object.fromEntries(evidenceFiles.map(file => [file, fs.readFileSync(file)]));

const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const results = [];
let fired = 0;
/* Recorded, not assumed: whether the build that puts the real page back
   actually succeeded. */
let restoringBuild = { attempted: false, ok: null };

/* ===================================================== surviving a kill
 *
 * Keeping the pre-mutation bytes only in this process's heap leaves any
 * termination that does not unwind the stack -- Ctrl-C, an OOM kill, a spend
 * limit -- with the breakage applied and nothing on disk saying so. The window
 * is a whole verifier run, and for a browser case a whole production build.
 * One of the breakages here restores the CSS rule that collapses KaTeX's drawn
 * glyphs, so a kill during it would silently reinstate wrong mathematics.
 *
 * Three things stand between a kill and a corrupted tree: a `.orig` sidecar
 * written before each mutation and removed only after a verified restore,
 * signal and exception handlers that restore from it, and a lock file so two
 * instances cannot interleave. On Windows a forced termination cannot be
 * trapped at all -- `taskkill /F` delivers no catchable signal -- so the
 * handlers are a courtesy and the SIDECAR is the actual protection.
 */
const LOCK = 'scratch/formulation-falsification.lock';
const SIDECAR = file => `${file}.formulation-falsify-orig`;
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
    + `Check for stale ${SIDECAR('<file>')} sidecars, restore any you find, then remove the lock, or run `
    + 'this harness with --recover.');
}
/* A stale sidecar means a previous run died mid-case. Restoring is not this
 * run's job to guess at -- it is reported, loudly, and the run refuses. */
const staleSidecars = [...new Set([...cases, ...browserCases].map(item => item.file))]
  .filter(file => fs.existsSync(SIDECAR(file)));
if (staleSidecars.length) {
  throw new Error(`stale sidecars from an interrupted run: ${staleSidecars.join(', ')}. `
    + 'Each holds the original bytes of the file beside it. Run --recover, or restore them by hand, '
    + 'before running again.');
}
fs.writeFileSync(LOCK, `${process.pid} ${new Date().toISOString()}\n`);

/* A PROVISIONAL report, written before the first mutation. The report at the
 * end is written before the inert check throws, so an inert guard is recorded;
 * what it does not cover is a throw from inside the loop -- a restore that did
 * not match, a build that failed -- which would otherwise leave the previous
 * report on disk claiming every guard fired. */
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync(OWN_REPORT, JSON.stringify({
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-formulation.mjs',
  status: 'running',
  note: 'Provisional record written before the first breakage was applied. If this is what is on disk, the '
    + 'run did not reach its end: a case threw, or it was killed. Check for a .formulation-falsify-orig '
    + 'sidecar and run --recover.',
  guardsThatFired: 0,
  breakagesApplied: 0,
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

const allCases = withBrowser ? [...cases, ...browserCases] : cases;
/* The whole loop sits inside a try/finally. An evidence restore that followed
   the loop would be skipped by a case throwing the restore-mismatch error,
   leaving the broken page's evidence on disk -- exactly what the snapshot
   exists to prevent. */
try {
  for (const item of allCases) {
    const original = fs.readFileSync(item.file);
    const before = crypto.createHash('sha256').update(original).digest('hex');
    const text = original.toString('utf8');
    if (!text.includes(item.find)) {
      results.push({
        name: item.name,
        file: item.file,
        guards: item.guards,
        applied: false,
        fired: false,
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
    // Measured, not assumed. `String.replace` with a string pattern replaces
    // the first occurrence only, so reporting a constant 1 would be a claim
    // about a substitution nobody counted.
    const replaced = occurrences - (mutated.split(item.find).length - 1);
    fs.writeFileSync(item.file, mutated, 'utf8');
    let outcome;
    try {
      if (item.rebuild) {
        const built = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-form'],
          { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
        if (built.status !== 0) throw new Error(`the build failed while applying "${item.name}"`);
      }
      const [command, argv] = item.run;
      const run = spawnSync(command, argv, {
        encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
        env: { ...process.env, DIST_DIR: 'dist-form' },
      });
      const output = `${run.stdout ?? ''}\n${run.stderr ?? ''}`;
      const failed = run.status !== 0;
      const named = item.expect.test(output);
      outcome = {
        name: item.name,
        file: item.file,
        guards: item.guards,
        expectSource: String(item.expect),
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
    /* SAME maxBuffer as the per-case builds. Every per-case build passed 64 MB
       and this one inherited Node's 1 MB default, so on a full asset listing it
       could die with ENOBUFS and leave the LAST BREAKAGE'S build on disk. The
       sources would restore correctly and the record would say so honestly,
       while any browser run afterwards tested sabotaged bytes. The status is
       checked rather than discarded, and a failure is reported rather than
       swallowed by the finally it sits in. */
    const restored = spawnSync('npx', ['vite', 'build', '--outDir', 'dist-form'],
      { encoding: 'utf8', shell: true, maxBuffer: 64 * 1024 * 1024 });
    restoringBuild = {
      attempted: true,
      status: restored.status,
      ok: restored.status === 0,
      error: restored.error ? String(restored.error && restored.error.message) : null,
      stderrTail: (restored.stderr || '').split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 300),
    };
    if (!restoringBuild.ok) {
      process.stderr.write('\n[falsify] THE RESTORING BUILD FAILED. dist-form still holds the last '
        + "breakage's bytes; rebuild before running any browser check.\n");
    }
  }
  for (const [file, bytes] of Object.entries(evidenceSnapshot)) fs.writeFileSync(file, bytes);
  fs.rmSync(LOCK, { force: true });
}

const report = {
  checkedAt: new Date().toISOString(),
  harness: 'scripts/falsify-formulation.mjs',
  harnessSha256: digest('scripts/falsify-formulation.mjs'),
  mode: withBrowser ? 'offline and browser' : 'offline only',
  breakagesApplied: results.length,
  guardsThatFired: fired,
  allRestoredExactly: results.every(result => result.restoredExactly !== false),
  restoringBuild,
  evidenceFilesRestored: evidenceFiles,
  results,
  browserCasesIncluded: withBrowser ? browserCases.length : 0,
  browserCasesDeferred: withBrowser ? [] : browserCases.map(item => item.name),
  method: 'Each breakage is applied alone to a clean tree, the relevant verifier is run, the file is restored '
    + 'and the restoration checked by SHA-256. A case counts as fired only when the verifier exits non-zero '
    + 'AND its output names the guard, so a run that fails for an unrelated reason is not credited.',
  limitations: [
    'The coverage floors in the model suite are set just below its current totals, so they catch a wholesale '
      + 'loss of checking rather than the removal of one assertion. Deleting a single record() call does NOT '
      + 'trip them, and the case above neuters the counter instead.',
    'Browser guards need a running preview and a production build; without --browser they are listed as '
      + 'deferred rather than silently omitted.',
    'Crash safety is best-effort: SIGKILL and a power loss cannot be trapped. In those cases the `.orig` '
      + 'sidecar beside the mutated file holds its original bytes, and the next run refuses to start until '
      + 'the sidecar is dealt with, rather than mutating a file that is already wrong. `--recover` restores '
      + 'from the sidecar and verifies the bytes.',
    'This harness touches only evidence files named formulation-*. A sibling lesson\'s captures are outside '
      + 'its reach by construction, not by care.',
    'A guard that fires here is a guard that can fail. It is not evidence that its domain covers every '
      + 'defect of its kind -- that is what the whole-grid sweeps in the model suite are for.',
  ],
};
fs.mkdirSync('docs/teaching/evidence', { recursive: true });
fs.writeFileSync(OWN_REPORT, JSON.stringify(report, null, 2) + '\n');

const inert = results.filter(result => !result.fired);
if (inert.length) {
  /* Two different failures wear the same word. A case where the verifier
     exited zero is a guard that did not fire. A case where it exited non-zero
     but the pattern did not match is usually PATTERN DRIFT -- the guard fired
     and said something the case no longer recognises -- which is the same
     bare-form defect in another costume, and it cost this lesson two rounds.
     They are named separately so the next reader does not have to work it
     out. */
  inert.forEach(result => {
    const kind = result.applied === false ? 'NOT APPLIED'
      : result.failedAsIntended && !result.namedTheGuard ? 'PATTERN DRIFT'
        : 'DID NOT FIRE';
    console.log(`INERT (${kind}) ${result.name}: ${result.note ?? JSON.stringify({
      exit: result.exitCode, failed: result.failedAsIntended, named: result.namedTheGuard,
      excerpt: result.excerpt,
    })}`);
    if (kind === 'PATTERN DRIFT') {
      console.log(`  the verifier DID fail; its output did not match ${result.expectSource}. `
        + 'Realign the pattern with the message the guard now emits, or the guard with its pattern.');
    }
  });
  throw new Error(`${inert.length} of ${results.length} breakages did not make their guard fire and name itself`);
}
console.log(`PASS: ${fired} of ${results.length} breakages made the intended guard fire and name itself; every `
  + `file was restored byte for byte, and ${evidenceFiles.length} evidence files were restored. `
  + (withBrowser
    ? `${browserCases.length} of them are browser guards, each rebuilt and run against the live preview.`
    : `${browserCases.length} browser guards are deferred and listed in the report; pass --browser to run them.`));
