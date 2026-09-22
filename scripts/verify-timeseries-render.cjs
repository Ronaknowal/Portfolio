// Render the forecasting lesson's body to static markup in Node.
//
// This exists because of the worst near-miss of this effort: a sibling lesson
// DID NOT RENDER AT ALL. Its body threw, the error boundary replaced the whole
// page, and three offline verifiers were green throughout. Nobody noticed until
// a screenshot was opened. Every check in this repository that reads the model
// layer, the recorded data or the source text is blind to that failure, because
// each of those artifacts was perfectly correct.
//
// So this verifier executes the actual React tree. Every component runs, every
// geometry function is called with the props the page really passes, and a
// throw anywhere in the tree fails here -- without a browser, a build or a
// preview, which means it can run in the same breath as the model checks rather
// than at the end of a phase.
//
// It renders the OPENING state only, and says so. Interaction, layout, paint,
// legibility and the KaTeX radical measurement are the browser verifier's job;
// this one answers a narrower question that nothing else answers: does the page
// exist, and does its first paint keep the investigation contract?
//
// Run:
//   node scripts/verify-timeseries-render.cjs
//   node scripts/verify-timeseries-render.cjs --no-evidence
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { build } = require('esbuild');

const ROOT = path.resolve(__dirname, '..');
const ENTRY = path.join(ROOT, 'src/learn/data/topics/time-series-validation-forecasting-baselines.jsx');
const BUNDLE = path.join(ROOT, 'scratch/timeseries/render-bundle.mjs');
const MARKUP = path.join(ROOT, 'scratch/timeseries/render.html');
const EVIDENCE = path.join(ROOT, 'docs/teaching/evidence/timeseries-render.json');
const keepEvidence = !process.argv.includes('--no-evidence');
const startedAt = new Date().toISOString();

const ownedFiles = [
  'src/learn/data/topics/time-series-validation-forecasting-baselines.jsx',
  'src/learn/data/timeseries-models.js',
  'src/learn/data/timeseries-data.js',
  'src/learn/data/timeseries-examples.js',
  'src/learn/components/lesson-labs/TimeSeriesShared.jsx',
  'src/learn/components/lesson-labs/TimeSeriesFigures.jsx',
  'src/learn/components/lesson-labs/TimeSeriesLabs.jsx',
];
const hash = file => createHash('sha256').update(fs.readFileSync(path.join(ROOT, file))).digest('hex');

const writeEvidence = payload => {
  if (!keepEvidence) return;
  fs.mkdirSync(path.dirname(EVIDENCE), { recursive: true });
  fs.writeFileSync(EVIDENCE, JSON.stringify(payload, null, 2) + '\n');
};
/* A provisional failing record goes to disk FIRST. A run that dies part way
   through -- which for this verifier means the lesson threw -- must not leave
   yesterday's success on disk describing today's tree. */
writeEvidence({
  checkedAt: startedAt,
  verifier: 'scripts/verify-timeseries-render.cjs',
  status: 'in progress: this record is provisional and is rewritten only after the final assertion',
  passed: false,
});

let checks = 0;
const check = (condition, label) => { checks += 1; assert.ok(condition, label); };
const equal = (actual, expected, label) => { checks += 1; assert.equal(actual, expected, label); };

(async () => {
  let stage = 'bundling';
  let rendered = null;
  try {
    await build({
      entryPoints: [ENTRY],
      bundle: true,
      format: 'esm',
      platform: 'node',
      outfile: BUNDLE,
      jsx: 'automatic',
      // Styles and fonts are not what this verifier is about; the browser
      // verifier measures what they do. Stubbing them keeps a missing font file
      // from being reported as a lesson defect.
      loader: { '.js': 'jsx', '.jsx': 'jsx', '.ttf': 'empty', '.woff': 'empty', '.woff2': 'empty', '.css': 'empty' },
      external: ['react', 'react/*', 'react-dom', 'react-dom/*'],
      logLevel: 'warning',
    });

    stage = 'importing the lesson module';
    const lesson = (await import('file://' + BUNDLE.replace(/\\/g, '/'))).default;
    check(lesson && typeof lesson === 'object', 'the lesson module has no default export');
    check(typeof lesson.content === 'function', 'the lesson exports no content function');
    // The metadata extractor reads these as static strings; a non-string here
    // would publish a lesson with no title or reading estimate.
    check(typeof lesson.title === 'string' && lesson.title.length > 0, 'the lesson has no title');
    check(typeof lesson.readTime === 'string' && /min/.test(lesson.readTime),
      `the reading estimate carries no units: ${lesson.readTime}`);

    stage = 'rendering the lesson body';
    const { renderToStaticMarkup } = await import('react-dom/server');
    /* THE ASSERTION THIS FILE EXISTS FOR. If the body throws, this line throws,
       and the catch below records which stage failed instead of leaving a green
       record beside a page that does not exist. */
    rendered = renderToStaticMarkup(lesson.content());
    check(typeof rendered === 'string' && rendered.length > 50000,
      `the lesson rendered only ${rendered ? rendered.length : 0} bytes of markup, which is not a whole lesson`);

    stage = 'inspecting the first paint';
    fs.mkdirSync(path.dirname(MARKUP), { recursive: true });
    fs.writeFileSync(MARKUP, rendered);

    const countOf = pattern => (rendered.match(pattern) || []).length;
    const counts = {
      figures: countOf(/class="ts-figure"/g),
      investigations: countOf(/class="ts-investigation"/g),
      practices: countOf(/class="ts-practice"/g),
      programs: countOf(/class="python-example"/g),
      diagrams: countOf(/class="ts-diagram /g),
      tables: countOf(/<table/g),
    };
    equal(counts.figures, 5, `the page renders ${counts.figures} figures, not 5`);
    equal(counts.investigations, 3, `the page renders ${counts.investigations} investigations, not 3`);
    equal(counts.practices, 8, `the page renders ${counts.practices} practice tasks, not 8`);
    equal(counts.programs, 3, `the page renders ${counts.programs} displayed programs, not 3`);
    check(counts.diagrams >= 6, `only ${counts.diagrams} tagged diagrams rendered`);
    check(counts.tables >= 8, `only ${counts.tables} tables rendered`);

    // Every svg must carry the layout class, EXCEPT KaTeX's own, which must
    // not. That separation is the whole point of scoping the stylesheet rule,
    // so both halves are asserted here.
    const svgTags = rendered.match(/<svg[^>]*/g) || [];
    const katexOwn = svgTags.filter(tag => tag.includes('400000'));
    const mine = svgTags.filter(tag => tag.includes('ts-diagram'));
    const untagged = svgTags.filter(tag => !tag.includes('ts-diagram') && !tag.includes('400000'));
    equal(untagged.length, 0,
      `${untagged.length} svg elements carry neither this lesson's layout class nor KaTeX's own geometry`);
    check(katexOwn.length >= 2,
      `only ${katexOwn.length} KaTeX radicals rendered; the browser verifier needs them to measure`);
    check(mine.length === counts.diagrams, 'a tagged diagram was counted inconsistently');
    equal(countOf(/katex-error/g), 0, 'the page renders a KaTeX error');

    // THE INVESTIGATION CONTRACT, at first paint. Nothing computed from a
    // graded answer may be on screen before a prediction is recorded.
    equal(countOf(/class="ts-verdict/g), 0, 'a verdict is rendered on first paint');
    equal(countOf(/class="ts-numeric-verdict/g), 0, 'numeric feedback is rendered on first paint');
    equal(countOf(/class="ts-reveal"/g), 0, 'a reveal panel is rendered on first paint');
    equal(countOf(/checked=""/g), 0, 'a prediction control is preselected on first paint');

    // Values that reach the page as text when a binding is wrong. Each of these
    // has shipped in this repository as a visible string.
    for (const [pattern, label] of [
      [/>undefined</g, 'literal "undefined" in rendered text'],
      [/NaN/g, 'NaN in rendered text'],
      [/\[object Object\]/g, '"[object Object]" in rendered text'],
      [/>null</g, 'literal "null" in rendered text'],
      [/\$\{/g, 'an unevaluated template placeholder in rendered text'],
    ]) {
      equal(countOf(pattern), 0, `${countOf(pattern)} occurrences of ${label}`);
    }

    // Every measured headline number must actually reach the page. A lesson
    // that renders but prints none of its results is a different failure from
    // one that throws, and neither is visible to the model checks.
    const { timeSeriesData } = await import('file://'
      + path.join(ROOT, 'src/learn/data/timeseries-data.js').replace(/\\/g, '/'));
    for (const row of [...timeSeriesData.development, ...timeSeriesData.final]) {
      check(rendered.includes(row.mae.toFixed(2)),
        `the measured MAE for ${row.key} (${row.mae.toFixed(2)}) is not rendered anywhere on the page`);
    }
    check(rendered.includes(timeSeriesData.provenance.sha256),
      'the served dataset digest is not rendered in the provenance callout');

    check(checks >= 30, `only ${checks} render checks ran; the suite has lost coverage`);

    writeEvidence({
      checkedAt: startedAt,
      completedAt: new Date().toISOString(),
      verifier: 'scripts/verify-timeseries-render.cjs',
      verifierSha256: createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),
      stage: 'server-side render of the lesson body; browser, independent and integration review are separate',
      sources: Object.fromEntries(ownedFiles.map(file => [file, hash(file)])),
      question: 'Does the page exist at all, and does its first paint keep the investigation contract?',
      whyItExists: 'A sibling lesson in this effort did not render at all -- its body threw and the error '
        + 'boundary replaced the whole page -- while three offline verifiers stayed green, because the model '
        + 'layer, the recorded data and the source text were each perfectly correct. Nothing but executing '
        + 'the React tree can see that failure.',
      checks,
      renderedBytes: rendered.length,
      counts,
      svg: {
        total: svgTags.length,
        lessonDiagrams: mine.length,
        katexOwn: katexOwn.length,
        untagged: untagged.length,
        note: 'KaTeX’s own radical SVGs must NOT carry this lesson’s layout class; that separation '
          + 'is what keeps the stylesheet away from them. Both halves are asserted.',
      },
      firstPaintContract: {
        verdicts: 0, numericFeedback: 0, revealPanels: 0, preselectedControls: 0,
        statement: 'Nothing computed from a graded answer is on screen before a prediction is recorded.',
      },
      markupWrittenTo: 'scratch/timeseries/render.html',
      limitations: [
        'The OPENING state only. Committed states, reveals, control interaction, narrow layouts and paint are '
          + 'the browser verifier’s job.',
        'Static markup, not a layout. It cannot see a collapsed SVG, an illegible label, a colour or an '
          + 'overflow; it can only see that the elements exist and carry the right classes.',
        'Styles are stubbed during bundling, so a stylesheet defect is invisible here by construction.',
      ],
      passed: true,
    });
    console.log(`PASS: the lesson body rendered to ${rendered.length.toLocaleString('en-US')} bytes of markup `
      + `(${checks} checks): ${counts.figures} figures, ${counts.investigations} investigations, `
      + `${counts.practices} practice tasks, ${counts.programs} programs, ${counts.diagrams} tagged diagrams, `
      + `${katexOwn.length} KaTeX radicals left untagged, and no verdict, numeric feedback or reveal on first `
      + 'paint.');
  } catch (error) {
    writeEvidence({
      checkedAt: startedAt,
      completedAt: new Date().toISOString(),
      verifier: 'scripts/verify-timeseries-render.cjs',
      stage,
      failedAt: stage,
      checksBeforeFailure: checks,
      renderedBytes: rendered ? rendered.length : 0,
      message: error?.message ?? String(error),
      stack: error?.stack ?? null,
      passed: false,
    });
    console.error(`FAIL while ${stage}: the lesson body did not render.`);
    console.error(error?.stack ?? error);
    process.exit(1);
  } finally {
    fs.rmSync(BUNDLE, { force: true });
  }
})();
