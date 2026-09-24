"""Bind final author evidence; no publication/integration or independent-review claim."""
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'scratch/ensemble-methods'
EVIDENCE=ROOT/'docs/teaching/evidence'
packet=EVIDENCE/'ensemble-methods-author-review.json'
if packet.exists():
    raise SystemExit('Preserve the original freeze; record a separate explicit amendment instead.')

production=[
    'src/learn/data/topics/ensemble-methods-stacking.jsx',
    'src/learn/data/ensemble-methods-models.js',
    'src/learn/data/ensemble-methods-examples.js',
    'src/learn/data/ensemble-prediction-map.json',
    'src/learn/components/lesson-labs/EnsembleMethodsLabs.jsx',
    'src/learn/components/lesson-labs/ensemble-methods-labs.css',
    'src/learn/data/curriculum/blueprints/ensemble-methods-stacking.js',
]
images=[
    'reading-start-1440.png','bootstrap-gap-390.png','boosting-second-fit-1440.png',
    'boosting-second-fit-390.png','oof-completed-390.png','oof-completed-320.png',
    'native-prediction-map-390.png','native-prediction-map-320.png','changed-residuals-390.png',
    'bound-derivation-390.png','final-equation-1-320.png','final-equation-4-320.png',
    'api-reading-320.png','context-reading-320.png','calibration-law-390.png',
    'changed-report-output-390.png','resources-320.png',
]
def fingerprint(relative):
    content=(ROOT/relative).read_bytes()
    return {'path':relative,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
now=datetime.now(timezone.utc).isoformat()
native=json.loads((WORK/'native-results.json').read_text(encoding="utf-8"))
reading=json.loads((WORK/'browser/reading-results.json').read_text(encoding="utf-8"))
controls=json.loads((WORK/'browser/results.json').read_text(encoding="utf-8"))
assert native['status']=='passed'
assert [item['width'] for item in reading['records']]==[1440,390,320]
assert all(not item['errors'] and not item['wideMath'] and item['fonts'] for item in reading['records'])
assert controls['records'][0]['states']==92 and controls['records'][0]['width']==320
design=ROOT/'docs/teaching/ENSEMBLE-METHODS-LESSON-DESIGN.md'
text=design.read_text(encoding='utf-8')
text=text.replace('The complete implementation is installed; native/oracle checks passed and author browser review is in progress. This is not a production integration claim.', 'The complete implementation is author-reviewed; exact sources, passing numerical evidence, browser scope and opened images are bound in ENSEMBLE-METHODS-VERIFICATION.md and its author packet. Independent review and production integration remain separate.')
design.write_text(text,encoding='utf-8')
source_archive='docs/teaching/evidence/ensemble-methods-author-sources.json'
(ROOT/source_archive).write_text(json.dumps({'archivedAt':now,'sources':[{**fingerprint(relative),'base64':base64.b64encode((ROOT/relative).read_bytes()).decode()} for relative in production]},indent=2)+'\n')
support=[
    'docs/teaching/ENSEMBLE-METHODS-LESSON-DESIGN.md',
    'scripts/generate-ensemble-examples.py','scripts/verify-ensemble-methods.mjs',
    'scripts/verify-ensemble-methods.py','scripts/review-ensemble-methods.cjs',
    'scripts/freeze-ensemble-methods.py',
    'docs/teaching/topic-notes/ensemble-methods-stacking.md',
    'docs/teaching/topic-notes/cross-validation-hyperparameter-tuning.md',
    'docs/teaching/topic-notes/recommender-systems-collaborative-filtering-matrix-factorization.md',
]
record={
    'topicId':'ensemble-methods-stacking','authorFrozenAt':now,'status':'author-reviewed; independent review and production integration separate',
    'production':[fingerprint(relative) for relative in production],'sourceArchive':source_archive,
    'originalEvidence':'docs/teaching/evidence/ensemble-original-review.json',
    'support':[fingerprint(relative) for relative in support],
    'checks':{
        'native':native,'modelFormatRecheck':json.loads((WORK/'model-format-recheck.json').read_text(encoding="utf-8")),
        'recorded320Controls':controls,'finalThreeWidthReading':reading,
        'finalAttachments':json.loads((WORK/'browser/attachments-results.json').read_text(encoding="utf-8")),
        'earlierBrowserScope':'The author operated all control stages at 1440 and 390 before an unrelated Vite overlay interrupted the late 320 stage. That aborted run did not persist its completed-width JSON. Do not count it as a saved three-width full-suite result. The outstanding 320 pass and final three-width reading/output checks are separately recorded here.',
        'repairedFindings':['Shared RunnableExample expected/output property mapping repaired; all 11 actual outputs and questions subsequently visible and checked.', 'Bootstrap midpoint candidates now use represented input values; actual native and exact-multiset oracles agree.', 'Native signed labels are validated before integer truncation could change them.', 'Two narrow equations reflowed, preserving their calculations; final seven equations fit 1440/390/320.', 'Mobile API contract table replaced by a responsive definition list after actual screenshot inspection.'],
    },
    'openedImages':[fingerprint('scratch/ensemble-methods/browser/'+name) for name in images],
    'reviewScope':{
        'read':'Full original and rewritten lesson; all three original programs and all 11 final programs; models, labs, blueprint, current API and cited primary-source portions. All 12 changed practice solutions and two checkpoints reviewed.',
        'retained':'Exact original body/program/output archive; the useful full NumPy bagging/AdaBoost program is byte-preserved and its normalized stdout conserved in the actual lesson.',
        'resources':'Instructor index and matching Kamper algorithm notes substantively reviewed; full recordings not watched. Detailed source scopes and rejected candidate recorded in design.',
        'limits':'No production build or broader integration performed by this author. No user acceptance, learner study, population accuracy/calibration guarantee or arbitrary-range numeric guarantee. Detailed plot/code/table local scrolling remains where appropriate.',
    },
}
packet.write_text(json.dumps(record,indent=2)+'\n')
rows='\n'.join('| '+item['path']+' | `'+item['sha256']+'` |' for item in record['production'])
report=f'''# Ensemble Methods & Stacking — author verification

Author freeze: **{now}**. Stable identity and existing module order retained. This is author review, not independent review, production integration or user acceptance.

[Exact packet](evidence/ensemble-methods-author-review.json) binds seven production sources, the native fitted probability artifact, current support files and 17 actually opened final images. [Source archive](evidence/ensemble-methods-author-sources.json) preserves those bytes. Parent owns registration/build/integration and a separate independent review.

## Teaching and original conservation

The full original body, three programs, six exercises and visual claims were read before replacement. [Original archive](evidence/ensemble-original-review.json) preserves exact bytes and execution outcomes, including the original incomplete stacking program's missing dependencies. The full useful NumPy tree/stump program is retained byte-for-byte in an optional complete example; its stdout remains the same after newline normalization.

The new route uses six distinct investigations and four inline figures, 11 standalone executed programs, two early checkpoints and 12 changed practice tasks. It teaches signed-error cancellation; actual bootstrap multiplicities, fitted thresholds and OOB eligibility; weighted-stump fitting and AdaBoost's normalizer/weak-edge proof; OOF versus full-refit ownership; actual library shapes and legal group/time alternatives; an exact calibration counterexample; a native-fitted probability field; and a held-out report with a changed-input accepted result. Probability mixing versus likelihood multiplication, context interactions, annotation/sensor/recommendation applications and serving costs are explicit. The title and downstream Recommender Systems bridge are retained.

The [design and source ledger](ENSEMBLE-METHODS-LESSON-DESIGN.md) records retained/repaired coverage, local prerequisites and honest resource access. Incoming Naive Bayes findings are adapted; scoped destination notes preserve the general CV and recommendation-specific follow-ups without rewriting those topics here.

## Numerical evidence

`node scripts/verify-ensemble-methods.mjs` passed **{native['checkedAt']}**. The packet embeds its result; [raw result](../../scratch/ensemble-methods/native-results.json) and the paired JS/Python scripts retain reproducibility.

- All 11 actual stored programs execute and match displayed stdout; one original full program/output is conserved.
- 625 exact weighted-vote cases, 303 exact residual states, 461 exact bootstrap multisets and three actual sklearn preset fits.
- 254 changed native AdaBoost label laws against a Fraction-based normalized-weight recurrence, plus three visual traces with product-loss and stopping contracts.
- 48 OOF ownership/query states against independent least-squares and bounded minimization; all 3,125 probability-map entries agree with the actual fitted estimators.
- Two complete split/selection/action/metric audits and 15 invalid model inputs.

Conventional lab/model formatting preserved normalized AST. A separate [current-state equality check](../../scratch/ensemble-methods/model-format-recheck.json) at 14:14:20 UTC confirms the current model states and example records exactly equal the independently checked inputs; unchanged programs were not regenerated for prose/layout fixes.

## Actual browser and visual review

The author completed the desktop and 390px operated-control stages before a Vite overlay interrupted the late 320px stage. That aborted run did not persist its completed-width JSON; it is not represented as a saved all-width full-suite pass. A fresh probe showed a mounted lesson without an overlay, and the outstanding 320px pass completed **{controls['checkedAt']}**: 92 changed states, 33 disclosures, all questions/code/stdout, keyboard interaction and supported bounds. That record includes the two then-wide equations rather than silently hiding them.

The final focused 1440/390/320 reading pass completed **{reading['checkedAt']}** with actual Space Grotesk fonts, all 33 disclosures, 11 complete visible programs/outputs, 12 practice tasks, correct anchors and all seven equations fitting. No page/console errors, invalid controls or document overflow were reported. [Final reading result](../../scratch/ensemble-methods/browser/reading-results.json), [320 control result](../../scratch/ensemble-methods/browser/results.json) and the packet distinguish these scopes. Final calibration/report attachments add a separate 390px pass.

Actual opened images include the start/combination flow, signed residuals, omitted-row threshold fit, full desktop and narrow weighted-stump views, OOF matrices at 390/320, native probability maps at 390/320, the bound derivation, both repaired equations, mobile API definition list, context products, exact calibration law, changed report output and resource annotations. The packet identifies all 17 exact image paths/hashes. Dense stump geometry and code retain deliberate local scrolling; simple numerical plots and the API explanation fit phones.

Concrete author corrections are preserved in the packet: output-property mismatch, represented-value threshold midpoint, pre-cast label validation, narrow formula wrapping and the difficult mobile API table. These were actual implementation/reading findings, not claims of a learner study.

## Frozen production identity

| Source | SHA-256 |
| --- | --- |
{rows}

Temporary installed draft and generator-only Python copies may be removed after ownership/reference checks; reusable generators, necessary inputs, exact archives, final JSON and the 17 opened images remain. No shared ledger/handoff/manifest mutation belongs to this author freeze.
'''
(ROOT/'docs/teaching/ENSEMBLE-METHODS-VERIFICATION.md').write_text(report,encoding='utf-8')
print(json.dumps({'authorFrozenAt':now,'production':record['production'],'openedImages':len(images)},indent=2))
