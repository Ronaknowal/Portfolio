"""Append source-versioned endpoint evidence without relabelling the original suite."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


before_path = 'docs/teaching/evidence/numerical-pdes-endpoint-before.json'
before = read(before_path)
assert all(digest(file['archive']) == file['sha256'] for file in before['files'])
author_path = 'docs/teaching/evidence/numerical-pdes-author-review.json'
old_author_path = next(file['archive'] for file in before['files'] if file['source'] == author_path)
original = read(old_author_path)
assert digest(author_path) == digest(old_author_path), 'Do not overwrite an intervening amendment.'
model_path = 'scratch/numerical-pde-endpoint-amendment/model-results.json'
native_path = 'scratch/numerical-pde-endpoint-amendment/native-results.json'
browser_path = 'scratch/numerical-pde-endpoint-amendment/browser/results.json'
model, native, browser = map(read, (model_path, native_path, browser_path))
assert model['passed'] and native['passed'] and browser['passed'] and browser['errors'] == []
assert model['sourceHashes'] == native['sourceHashes'] == browser['sourceHashes']
assert all(digest(file) == sha for file, sha in model['sourceHashes'].items())
changed = [file for file, sha in model['sourceHashes'].items() if sha != original['sourceHashes'][file]]
assert changed == ['src/learn/data/numerical-pde-models.js', 'src/learn/data/numerical-pde-examples.js']
assert model['cases'] == 150 and model['unchangedExampleRecords'] == 15
assert native['counts']['complete displayed programs'] == 16
assert native['counts']['changed endpoint helper states'] == 105
assert [record['width'] for record in browser['records']] == [1440, 390, 320]
assert all(len(record['states']) == 4 for record in browser['records'])
opened = ['pinned-endpoint-program-320', 'unit-rod-boundary-390',
          'unchanged-program-output-1440', 'pinned-endpoint-program-1440',
          'unchanged-program-output-320']
captures = {name for record in browser['records'] for name in record['captures']}
assert all(name + '.png' in captures for name in opened)
amendment_path = 'docs/teaching/evidence/numerical-pdes-endpoint-amendment.json'
assert not (ROOT / amendment_path).exists()
now = datetime.now(timezone.utc).isoformat()
amendment = {
    'amendedAuthorFrozenAt': now,
    'topicId': original['topicId'],
    'originalAuthorFrozenAt': original['authorFrozenAt'],
    'originalPacketArchive': old_author_path,
    'beforeManifest': before_path,
    'sourceHashes': model['sourceHashes'],
    'changedProductionFiles': changed,
    'finding': 'A generated endpoint 3*0.7/3 is below 0.7; strict interpolation correctly rejected a curve evaluation beyond that generated domain.',
    'repair': 'Pin JS mesh and curve endpoints to the prescribed exact inputs. Pin the actual Python Poisson helper endpoints likewise; preserve strict outside-domain rejection and all interior arithmetic.',
    'model': model,
    'native': native,
    'browser': browser,
    'actuallyOpenedImages': [
        {'path': 'scratch/numerical-pde-endpoint-amendment/browser/' + name + '.png',
         'sha256': digest('scratch/numerical-pde-endpoint-amendment/browser/' + name + '.png'),
         'inspection': 'Opened with view_image after the targeted final run. Complete code/output retain intentional horizontal scrolling on phones.'}
        for name in opened],
    'evidenceHashes': {path: digest(path) for path in [before_path, old_author_path, model_path, native_path, browser_path, 'scratch/numerical-pde-native/example-runs.json']},
    'scriptHashes': {path: digest(path) for path in ['scripts/generate-numerical-pde-examples.py', 'scripts/verify-numerical-pde-endpoint-amendment.mjs', 'scripts/verify-numerical-pde-endpoint-native.py', 'scripts/review-numerical-pde-endpoint-amendment.cjs', 'scripts/finalize-numerical-pde-endpoint-amendment.py']},
    'harnessCorrection': 'The first amendment harness incorrectly expected the Poisson helper in the refinement record. Source inspection confirmed only the Poisson record changes; the harness now asserts the other fifteen entire example records remain unchanged. This was a test assumption, not a second production defect.',
    'scope': 'Narrow author amendment. Original full native/browser evidence remains bound to its original source hashes; no unrelated full-suite rerun is claimed. The independent reviewer and integrated production checks remain separately attributed.',
}
(ROOT / amendment_path).write_text(json.dumps(amendment, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
packet = dict(original)
packet['originalAuthorFrozenAt'] = original['authorFrozenAt']
packet['authorFrozenAt'] = now
packet['sourceHashes'] = model['sourceHashes']
packet['amendments'] = [{'record': amendment_path, 'sha256': digest(amendment_path), 'amendedAuthorFrozenAt': now, 'changedProductionFiles': changed}]
packet['evidenceApplicability'] = {
    'originalPacketArchive': old_author_path,
    'originalFullChecks': 'The native, browser, modelExport, formatConservation and original image/evidence fields retain their original frozen-source identities. They were not rerun or relabelled.',
    'currentAmendment': 'The amendments record supplies current-source model/native/browser checks, conservation, opened images and exact post-repair hashes.',
}
(ROOT / author_path).write_text(json.dumps(packet, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'amendedAuthorFrozenAt': now, 'changedSources': changed, 'openedImages': len(opened), 'model': model['checkedAt'], 'native': native['checkedAt'], 'browser': browser['checkedAt']}, indent=2))
