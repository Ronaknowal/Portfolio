const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file));
const hashText = text => createHash('sha256').update(text).digest('hex');
const entry = path => ({ path, sha256: hashText(fs.readFileSync(path)), bytes: fs.statSync(path).size });
const priorPath = 'docs/teaching/evidence/category-theory-author-review-before-entity-fix.json';
const prior = read(priorPath);
const body = prior.production[0].path;
const source = fs.readFileSync(body, 'utf8');
const reconstructedPrior = source.replace('For q(y)&gt; 0,', 'For q(y)&gt ; 0,').replace(/(    <Prose>With prior p\(x\).*<\/Prose>)\n/, '$1\r\n');
assert.equal(hashText(reconstructedPrior), prior.production[0].sha256);
for (const row of prior.production.slice(1)) assert.equal(entry(row.path).sha256, row.sha256);
const focused = read('scratch/category-theory-browser/conditional-text-results.json');
assert(focused.passed && focused.bodySha256 === entry(body).sha256);
const record = {
  ...prior,
  authorFreezeAt: new Date().toISOString(),
  production: prior.production.map(row => entry(row.path)),
  support: prior.support.map(row => entry(row.path)),
  amendment: {
    priorRecord: entry(priorPath),
    reason: 'Independent reviewer found a malformed greater-than entity in the positive-evidence condition.',
    exactChange: 'For q(y)&gt ; 0, -> For q(y)&gt; 0,; the edited line ending changed from CRLF to LF. Reversing both reconstructs the prior exact body SHA256.',
    priorBodySha256: prior.production[0].sha256,
    focused,
    actuallyOpenedImage: entry('scratch/category-theory-browser/conditional-text-320.png'),
    scope: 'Only the entity substring and its line ending changed; all other five production files are byte-identical. Earlier native/browser/model evidence retains its earlier identity, rather than being claimed as rerun.',
  },
};
fs.writeFileSync('docs/teaching/evidence/category-theory-author-review.json', JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ authorFreezeAt: record.authorFreezeAt, bodySha256: record.production[0].sha256 }));
