"""Record the completed scoped author checks; does not certify shared integration."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
now = datetime.now(timezone.utc).isoformat()
design_path = ROOT / 'docs/teaching/CONDITIONING-STABILITY-LESSON-DESIGN.md'
design = design_path.read_text(encoding='utf-8')
design = design.replace('Status: design only, proposed for central review on 11 September 2026. The numerical fixtures below have been executed; the lesson, browser models and representations have not been implemented or reviewed. This document is not a publication or integration claim.',
    'Status: assessed design implemented and author-verified on 11 September 2026. The exact source freeze and actual native/browser checks are recorded in [CONDITIONING-STABILITY-VERIFICATION.md](CONDITIONING-STABILITY-VERIFICATION.md) and [the author packet](evidence/conditioning-stability-author-review.json). The original proposal and its design-only fixtures below are retained as design history; they are not substituted for the final production checks. Independent review and shared production integration remain separate parent-owned steps.')
design = design.replace('Root has agreed in principle; no shared prerequisite entry is changed by this design.', 'Root assessed and registered this prerequisite replacement. This author changed no shared prerequisite entry.')
design = design.replace('Both remain open proposals from a designed origin; neither claims implementation.', 'Both remain open for their destination authors; their origin status now links the completed authored lesson and final numerical/browser evidence. Neither destination is claimed implemented.')
design = design.replace('**Changed complete report:** use the explicit `ε=2^-16` scenario from section9, including both measurement bounds and the `1e-4` amount budget; provide the complete report with its actual repaired center and reproducible solver/reference path. The design fixture already checks both extreme allowed readings by exact solves. Authoring must execute the final displayed protocol and report, rather than substitute the earlier `ε=2^-20` worked example unchanged.',
    '**Changed complete report (final authoring refinement):** section 9 supplies the full executed `ε=2^-16` example. Independent closing practice changes it again to `ε=2^-18`, preserving the larger reading radius `2^-24` while changing the smaller one to `2^-34`. Actual unchanged report code with these two literal edits gives central fractions `2863311531/8589934592` and `5726623061/8589934592`, initial float32 values `(0.34375,0.65625)`, zero final central error for this fixture and amount bounds `1/64` (fail) and `1/65536` (pass). The separate changed-report verifier checks exact stored-input Cramer values and both interval conversions. The original assessed `ε=2^-16` design fixture remains historical evidence, not the independent task answer.')
design = design.replace('## Proposed file ownership and next action', '## Implemented ownership and independent-review handoff')
design = design.replace('Design and individual brief are ready for review. Proposed body', 'Root assessed and registered the brief and the complete body. Implemented body')
design = design.replace('This design does not add a manifest entry, update the blueprint index, rewrite other lessons or change reading order. After parent review/registration, implement the scoped source, continue checking ownership during writing, execute the final independent changed report and complete topic-native verification before requesting integrated review.',
    'The author changed only these scoped semantic files, individual evidence/scripts and outgoing notes. Root owns the manifest/index, generated catalogue, shared prerequisites and integration ledger. The final author packet is ready for a separate independent source review and production integration; no broader rollout completion or user acceptance is claimed.')
design_path.write_text(design, encoding='utf-8')

for name in ['floating-point-representation-numerical-error', 'numerical-pdes-grids-finite-elements-stability']:
    note_path = ROOT / f'docs/teaching/topic-notes/{name}.md'
    note = note_path.read_text(encoding='utf-8').replace('currently designed and fixture-checked, not implemented.', 'now authored with final native/browser evidence in [the verification record](../CONDITIONING-STABILITY-VERIFICATION.md); independent review/integration are separate.')
    note = note.replace('Origin proposes binary64', 'Origin now teaches binary64').replace('Those origin forms are not yet authored or integrated.', 'Those origin forms are authored and verified within the final scoped evidence; no destination implementation is claimed.')
    note = note.replace('Conditioning52 proposes the intervening', 'Conditioning52 now teaches the intervening').replace('Conditioning52 plans a complete', 'Conditioning52 now teaches a complete')
    note = note.replace('origin link above is design evidence only.', 'origin links now include actual production verification.').replace("Origin's design is linked, and its [fixtures](../../../scratch/conditioning-design/fixtures.json) verify only the scoped finite examples.", "Origin's [final verification](../CONDITIONING-STABILITY-VERIFICATION.md) distinguishes actual production checks from the preserved design-only fixtures.")
    note_path.write_text(note, encoding='utf-8')

production = [
    'src/learn/data/topics/conditioning-stability-numerical-analysis.jsx',
    'src/learn/data/conditioning-stability-models.js',
    'src/learn/data/conditioning-stability-examples.js',
    'src/learn/components/lesson-labs/ConditioningStabilityLabs.jsx',
    'src/learn/components/lesson-labs/conditioning-stability-labs.css',
    'src/learn/data/curriculum/blueprints/conditioning-stability-numerical-analysis.js',
]
support = [
    'docs/teaching/CONDITIONING-STABILITY-LESSON-DESIGN.md',
    'docs/teaching/CONDITIONING-STABILITY-VERIFICATION.md',
    'docs/teaching/evidence/conditioning-stability-original-plan.json',
    'docs/teaching/topic-notes/floating-point-representation-numerical-error.md',
    'docs/teaching/topic-notes/numerical-pdes-grids-finite-elements-stability.md',
    'scripts/generate-conditioning-stability-examples.py',
    'scripts/verify-conditioning-stability.mjs',
    'scripts/verify-conditioning-stability-native.py',
    'scripts/verify-conditioning-changed-report.py',
    'scripts/review-conditioning-stability-lesson.cjs',
    'scripts/review-conditioning-final-reading.cjs',
    'scripts/format-conditioning-source.mjs',
]
def fingerprint(name):
    data = (ROOT / name).read_bytes()
    return {'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

images = [
    'reading-2-320.png', 'reading-5-390.png', 'reading-7-320.png', 'reading-8-1440.png',
    'unstable-refinement-320.png', 'inline-2-390.png', 'reading-9-320.png', 'sources-390.png',
    'propagation-pulse-390.png', 'sum-tree-scrolled-320.png', 'cancellation-chart-320.png',
    'measurement-sensitive-320.png', 'sum-positive-320.png', 'backward-zero-entry-320.png',
    'rounding-tie-320.png', 'final-reference-branches-1440.png', 'final-reference-branches-320.png',
    'final-changed-report-solution-320.png', 'final-report-output-390.png',
]
native = load('scratch/conditioning-stability-native/results.json')
for name, sha in native['sourceHashes'].items():
    assert fingerprint(name)['sha256'] == sha
record = {
    'topicId': 'conditioning-stability-numerical-analysis', 'frozenAt': now,
    'status': 'author-verified; independent review and production integration pending',
    'productionSources': [fingerprint(name) for name in production],
    'supportingSources': [fingerprint(name) for name in support],
    'native': native,
    'changedReport': load('scratch/conditioning-stability-native/changed-report.json'),
    'browser': load('scratch/conditioning-stability-browser/results.json'),
    'finalReading': load('scratch/conditioning-stability-browser/final-reading-results.json'),
    'formatting': load('scratch/conditioning-stability-format/results.json'),
    'openedScreenshots': [fingerprint('scratch/conditioning-stability-browser/' + name) for name in images],
    'openedScreenshotAttribution': 'Author actually opened these 19 files using view_image. The final reference-branch images supersede the earlier inline flow; the unchanged-control full-browser images predate only that isolated figure amendment.',
    'reviewBounds': [
        'Finite bounded exact/Decimal/native-library oracles are not arbitrary-float certification.',
        'Full 82-state per-width browser run precedes the isolated reference-branch figure clarification; final three-width reading checks close that change.',
        'Actual reference/program/ordinary reading and keyboard review; no novice study, screen-reader audit, GPU performance benchmark or deployment.',
        'Official alternate-video page and relevant transcript inspected; no full-video playback claim.',
        'Parent owns independent review, shared production build/loading and rollout status.'
    ],
}
destination = ROOT / 'docs/teaching/evidence/conditioning-stability-author-review.json'
destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'frozenAt': now, 'productionSources': record['productionSources'], 'openedScreenshots': len(images)}, indent=2))
