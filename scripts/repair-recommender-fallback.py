"""Narrow reviewed fallback correction; preserve the previous frozen identity."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import io
import json
import shutil
import warnings
import black
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'docs/teaching/archive/recommender-fallback-before'
archive.mkdir(parents=True,exist_ok=True)
sources = ['docs/teaching/evidence/recommender-systems-author-review.json',
           'src/learn/data/recommender-examples.js','scripts/generate-recommender-examples.py']
for name in sources:
    destination = archive / Path(name).name
    if destination.exists():
        raise RuntimeError('Preserved pre-amendment files already exist; do not overwrite.')
    shutil.copy2(ROOT/name,destination)
path = ROOT/'src/learn/data/recommender-examples.js'
prefix,body = path.read_text(encoding='utf8').split('export const recommenderExamples = ',1)
examples = json.loads(body.strip().removesuffix(';'))
original = json.loads(json.dumps(examples))
code = examples['neighbors']['code']
old = '    user_means = np.nanmean(ratings, axis=1)\n    fallback = float(user_means[user])'
new = '''    prior = 3.0  # Declared rating prior when the entire training matrix is empty.
    global_mean = float(ratings[observed].mean()) if observed.any() else prior
    counts = observed.sum(axis=1)
    user_means = np.divide(
        np.where(observed, ratings, 0.0).sum(axis=1),
        counts,
        out=np.full(len(ratings), global_mean),
        where=counts > 0,
    )
    fallback = float(user_means[user])'''
assert old in code
examples['neighbors']['code'] = black.format_str(code.replace(old,new),mode=black.Mode(line_length=88)).rstrip()
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout), warnings.catch_warnings():
    warnings.simplefilter('error')
    exec(examples['neighbors']['code'],namespace)
assert stdout.getvalue().rstrip() == original['neighbors']['expected']
cases = []
for ratings,user,item,expected in [
    ([[np.nan,np.nan,np.nan],[0.,2.,4.],[1.,np.nan,5.]],0,1,2.4),
    ([[np.nan,np.nan],[np.nan,np.nan]],0,1,3.),
    ([[0.,np.nan],[2.,4.]],0,1,0.),
]:
    namespace['ratings'] = np.array(ratings,dtype=float)
    namespace['observed'] = np.isfinite(namespace['ratings'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = namespace['predict'](user,item)[0]
    assert np.isclose(actual,expected), (actual,expected)
    cases.append({'expected':expected,'actual':actual})
for key in examples:
    if key != 'neighbors': assert examples[key] == original[key]
for key in ['title','question','expected','language']:
    assert examples['neighbors'][key] == original['neighbors'][key]
path.write_text(prefix+'export const recommenderExamples = '+json.dumps(examples,indent=2,ensure_ascii=False)+';\n',encoding='utf8')
generator = ROOT/'scripts/generate-recommender-examples.py'
text = generator.read_text(encoding='utf8')
assert old in text
generator.write_text(text.replace(old,new),encoding='utf8')
record = {'checkedAt':datetime.now(timezone.utc).isoformat(),'cases':cases,
          'unchangedExampleRecords':13,'all14StdoutConserved':True,
          'originalArchive':str(archive.relative_to(ROOT)),
          'examplesSHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'scope':'Only the actual neighbors code and matching generator change; no model/body/style changes.'}
(ROOT/'docs/teaching/evidence/recommender-fallback-native.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record,indent=2))
