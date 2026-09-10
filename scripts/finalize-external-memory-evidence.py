"""Freeze source fingerprints only after actual native/browser/visual review."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))
def fingerprint(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

native = read('scratch/external-memory-verification/results.json')
browser = read('scratch/external-memory-browser/results.json')
reading = read('scratch/external-memory-browser/reading-results.json')
formatting = read('scratch/external-memory-verification/formatting-results.json')
assert browser['errors'] == []
assert [row['width'] for row in browser['results']] == [1440,390]
assert [row['width'] for row in reading['results']] == [1440,390,320]
for relative, digest in native['sha256'].items():
    assert fingerprint(relative) == digest, relative
runtime_sources = formatting['files']
owned = runtime_sources + [
    'docs/teaching/EXTERNAL-MEMORY-LESSON-DESIGN.md',
    'docs/teaching/EXTERNAL-MEMORY-VERIFICATION.md',
    'docs/teaching/topic-notes/external-memory-algorithms-b-trees-i-o-complexity.md',
    'scripts/prepare-external-memory-examples.py',
    'scripts/verify-external-memory.py',
    'scripts/review-external-memory.cjs',
    'scripts/review-external-memory-reading.cjs',
    'scripts/format-external-memory.cjs',
    'scripts/finalize-external-memory-evidence.py',
]
opened = [
    'reading-1-1440.png','buffer-initial-1440.png','tree-final-insert-1440.png',
    'reading-6-1440.png','bplus-contrast-390.png','tree-search-390.png',
    'merge-runs-390.png','crash-invalid-pages-390.png','reading-5-390.png','intro-390.png',
    'detail-large-tree-pages-320.png','detail-calculation-2-320.png',
    'detail-native-program-1440.png','detail-native-output-390.png','detail-sources-390.png',
    'detail-height-proof-390.png','detail-large-tree-controls-390.png','detail-merge-runs-final-390.png',
    'detail-bplus-contrast-320.png','detail-crash-invalid-320.png','detail-deletion-proof-390.png',
    'bplus-pages-390.png','buffer-initial-390.png','bplus-initial-390.png',
    'reading-8-390.png','reading-9-1440.png','detail-calculation-1-320.png',
]
for image in opened:
    assert (ROOT/'scratch/external-memory-browser'/image).exists()
record = {
    'topic_id':'external-memory-algorithms-b-trees-i-o-complexity',
    'status':'author-verified; root integration and user acceptance separate',
    'frozen_at':datetime.now(timezone.utc).isoformat(),
    'source_sha256':{relative:fingerprint(relative) for relative in runtime_sources},
    'owned_file_sha256':{relative:fingerprint(relative) for relative in owned},
    'native':native,'browser':browser,'ordinary_reading':reading,'formatting':formatting,
    'actually_opened_images':opened,
    'research':{
        'checked_on':'2026-09-10',
        'record':'docs/teaching/EXTERNAL-MEMORY-LESSON-DESIGN.md',
        'official_leetcode_ids':[146,23,148,378],
        'alternate_video':'MIT 6.851 Lecture7 official page and relevant scribe text inspected; no full playback',
        'written':'Open Data Structures B-Trees; Cornell external sorting; CMU shadow paging; SQLite atomic commit. Exact inspected sections recorded in design.',
        'limits':'No LeetCode submission/editorial review, physical I/O or timing benchmark, real fsync/crash experiment, strict Python process-memory cap, full B+ update/database engine or novice study claimed.'
    },
    'root_source_review':{
        'status':'no actionable source defect reported before final freeze',
        'scope':'full body, page/record units, B-tree repair/height proofs, B+ routing/payload bounds, materialized sorting and partial pages/RAM caveats, shadow-root proof and changed tasks',
        'limit':'bounded independent source/proof review; not additional numerical tests'
    },
    'refinements':[
        'Moved B-tree and B+ current-event controls above tall page hierarchies.',
        'Highlighted final-tree search path; cleared stale search results on changed keys/steps.',
        'Shortened root-order option labels and separated reset controls from following prose.',
        'Used native superscripts/subscripts for height and split sort notation into explicit scan/scale factors; checked final1440/390/320 reading.'
    ],
    'notes':'Incoming persistent-root note adapted with actual reachability verification; deeper I/O/storage-engine mechanisms deferred with explicit owner-reassessment reasoning in the same stable-ID note.'
}
target = ROOT/'docs/teaching/evidence/external-memory-author-review.json'
target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'record':str(target.relative_to(ROOT)),'frozen_at':record['frozen_at'],'source_sha256':record['source_sha256']},indent=2))
