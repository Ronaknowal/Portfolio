from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path(__file__).resolve().parents[1]
packet_path = root/'docs/teaching/evidence/recommender-systems-author-review.json'
packet = json.loads(packet_path.read_text(encoding='utf8'))
native = json.loads((root/'docs/teaching/evidence/recommender-fallback-native.json').read_text())
browser = json.loads((root/'docs/teaching/evidence/recommender-fallback-browser.json').read_text())
changed = 'src/learn/data/recommender-examples.js'
digest = hashlib.sha256((root/changed).read_bytes()).hexdigest()
assert digest == native['examplesSHA256'] == browser['sourceHash']
previous = dict(packet['sourceHashes'])
packet['sourceHashes'][changed] = digest
stamp = datetime.now(timezone.utc).isoformat()
image = 'scratch/recommender-browser/fallback-final-390.png'
amendment = {'frozenAt':stamp,'finding':'Displayed neighbor helper did not implement the described global/prior fallback for a cold user.',
             'previousPacket':'docs/teaching/archive/recommender-fallback-before/recommender-systems-author-review.json',
             'previousSourceHashes':previous,'changedSource':changed,'finalSHA256':digest,
             'native':'docs/teaching/evidence/recommender-fallback-native.json',
             'browser':'docs/teaching/evidence/recommender-fallback-browser.json',
             'openedImage':{'path':image,'sha256':hashlib.sha256((root/image).read_bytes()).hexdigest(),'actuallyOpened':True},
             'scope':'One displayed helper and its generator only. All14 stdout and13 other example records remain unchanged; no full author-suite rerun claimed.'}
packet.setdefault('amendments',[]).append(amendment)
packet['finalAmendedFreeze'] = stamp
packet_path.write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
(root/'docs/teaching/evidence/recommender-fallback-amendment.json').write_text(json.dumps(amendment,indent=2)+'\n',encoding='utf8')
record = root/'docs/teaching/RECOMMENDER-SYSTEMS-VERIFICATION.md'
with record.open('a',encoding='utf8') as stream:
    stream.write(f'''\n## Independent-review fallback amendment — {stamp}\n\nThe independent reviewer found that the displayed `neighbors` Python helper returned NaN for a wholly missing target-user row, although the body and JS model specify a training-global mean and then a declared prior. The original frozen packet, actual examples and generator are preserved under `archive/recommender-fallback-before/`. The corrected helper computes explicit observed counts, preserves an observed zero, and falls back to the training global mean (2.4 in the changed fixture) or declared prior3 for an entirely empty matrix.\n\nFocused native checks pass with warnings treated as failures. All14 printed outputs and the other13 example records are unchanged. Actual-font Edge checks at1440/390/320 confirm the exact amended code and unchanged output; the final390 capture was opened and inspected. Code remains intentionally horizontally scrollable. No body/model/lab/style change and no unrelated broad-suite rerun. Final examples SHA256 `{digest}`. Exact amendment and source-versioned evidence: [packet](evidence/recommender-fallback-amendment.json). The original full author tests retain their original source attribution.\n''')
print(stamp,digest)
