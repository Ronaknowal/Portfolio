"""Fetch only the official AC Library v1.6 headers used by the range lesson."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
destination = root / 'public/learn-assets/segment-trees-fenwick-trees-range-queries'
include = destination / 'ac-library-v1.6'
base = 'https://raw.githubusercontent.com/atcoder/ac-library/v1.6/'
names = ['segtree', 'lazysegtree', 'fenwicktree', 'internal_bit', 'internal_type_traits']
files = ['LICENSE'] + [f'atcoder/{name}{suffix}' for name in names for suffix in ['', '.hpp']]
records = []
for name in files:
    url = base + name
    with urlopen(url, timeout=40) as response:
        content = response.read()
    path = include / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    records.append({'path': name, 'url': url, 'sha256': hashlib.sha256(content).hexdigest()})
with ZipFile(destination / 'ac-library-v1.6-range-headers.zip', 'w', ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(include / name, 'ac-library-v1.6/' + name)
(destination / 'ac-library-provenance.json').write_text(json.dumps({
    'release': 'v1.6', 'repository': 'https://github.com/atcoder/ac-library',
    'license': 'CC0-1.0', 'retrieved': '2026-09-22', 'files': records,
}, indent=2) + '\n', encoding='utf-8')
print('Fetched', len(records), 'official pinned files; archive includes the license.')
