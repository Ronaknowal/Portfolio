from pathlib import Path
import json,hashlib,shutil,datetime
root=Path('.')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=json.loads(Path('scratch/linked-traversal-extension-verification/results.json').read_text())
browser=json.loads(Path('scratch/linked-traversal-extension-browser/results.json').read_text())
assert native['status']==browser['status']=='passed'
assert native['productionSources']==browser['productionSources']
archive=Path('scratch/linked-traversal-extension-author-freeze')
assert not archive.exists(), 'Preserve the first immutable freeze; use an explicit amendment instead.'
archive.mkdir(parents=True)
for row in native['productionSources']:
    assert hashfile(row['path'])==row['sha256']
    destination=archive/row['path'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['path'],destination)
    row['archive']=destination.as_posix()
opened=['reading-9-1440','cycle-entry-1440','cycle-self-loop-320','middle-cut-1440','middle-cut-390','next-greater-multipop-1440','next-greater-result-320','histogram-rectangle-1440','histogram-tied-320','reading-11-390','reading-12-320','changed-stack-practice-390','histogram-scrolled-320','middle-cut-scrolled-390','reset-proof-320','zero-heights-320','references-390','signed-readings-390']
images=[]
for name in opened:
    path=f'scratch/linked-traversal-extension-browser/{name}.png'
    images.append({'path':path,'sha256':hashfile(path),'openedBy':'author using view_image; ordinary reading, operated geometry, keyboard-scroll or explanation reviewed'})
record={'frozenAt':now,'status':'author-verified scoped extension; independent review and production integration remain parent-owned','topicId':'linked-lists-stacks-queues','productionSources':native['productionSources'],'originalSnapshot':'docs/teaching/evidence/linked-traversal-extension-original.json','originalSnapshotSha256':hashfile('docs/teaching/evidence/linked-traversal-extension-original.json'),'native':native,'browser':browser,'openedImages':images,'reviewLimits':['No shared registry, generated catalogue, coverage map, ledger, historical record or unrelated lesson changed.','No broader build/integration or screen-reader session is claimed.','Exhaustive small successor graphs/arrays and changed native inputs complement local proofs; they do not exhaust arbitrary inputs.','Official new LeetCode public statements inspected; no editorial/submission execution or new video playback claimed. Existing annotated CS50 video and written alternatives are retained.'],'reviewNotes':['Original six programs/outputs, old model/labs and historical records are preserved; all 61 original meaningful teaching AST subtrees and ten placements are conserved. Only 141 transfer now mentions the proved constant-space follow-up.','Full final browser passed 86 operated states per width at 1440/390/320, all ten actual code/output blocks, fourteen placements, anchors and no document overflow or console/page errors.','Final image review prompted an explicit long-chain keyboard-scroll hint, an exact zero-height SVG representation, and a precise reset proof describing the meeting via slow\'s t-hop route. Those final sources passed the native and full browser rerun.','Two initial browser harness stops involved an old lab locator: Investigation uses data-investigation, and its implicit select labels are reliably found by accessible combobox role. Production old labs were not changed.'],'commands':['node scripts/verify-linked-traversal-extension.mjs','node scripts/review-linked-traversal-extension.cjs'],'supportingFiles':[]}
for path in ['scripts/snapshot-linked-traversal-extension.py','scripts/generate-linked-traversal-extension.py','scripts/verify-linked-traversal-extension.mjs','scripts/verify-linked-traversal-extension.py','scripts/review-linked-traversal-extension.cjs','scripts/format-linked-traversal-extension.cjs','docs/teaching/LINKED-TRAVERSAL-MONOTONIC-EXTENSION-DESIGN.md']:
    record['supportingFiles'].append({'path':path,'sha256':hashfile(path)})
Path('docs/teaching/evidence/linked-traversal-extension-author-review.json').write_text(json.dumps(record,indent=2)+'\n')
shutil.copy2('docs/teaching/evidence/linked-traversal-extension-author-review.json',archive/'author-review.json')
print(now)
print('Frozen',len(record['productionSources']),'sources;',len(images),'opened images; browser',browser['checkedAt'])
