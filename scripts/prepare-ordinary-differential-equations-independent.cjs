const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const directory = 'scratch/ordinary-differential-equations-independent';
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/ordinary-differential-equations-author-review.json', 'utf8'));
fs.mkdirSync(`${directory}/author-sources`, { recursive: true });
assert(!fs.existsSync(`${directory}/author-baseline.json`), 'Preserve the original author baseline.');
const sources = author.sources.map(source => {
  const bytes = fs.readFileSync(source.path);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), source.sha256, source.path);
  const archive = `${directory}/author-sources/${path.basename(source.path)}`;
  fs.writeFileSync(archive, bytes, { flag: 'wx' });
  return { ...source, archive };
});
fs.writeFileSync(`${directory}/author-baseline.json`, JSON.stringify({ at: new Date().toISOString(), sources }, null, 2) + '\n');
console.log('Archived seven final author sources before independent review amendments.');
