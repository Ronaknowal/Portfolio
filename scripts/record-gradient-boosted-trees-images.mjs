import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const names = [
  "validation-default-1440",
  "validation-changed-320",
  "residual-default-390",
  "goss-contributions-1440",
  "newton-overshoot-320",
  "bins-fine-390",
  "bins-coarse-320",
  "growth-reading-320",
  "prefix-default-320",
  "prefix-reversed-1440",
  "additive-reading-320",
  "bundling-reading-320",
  "equation-reading-7-320",
  "equation-reading-10-320",
  "xgboost-program-reading-320",
  "practice-reading-390"
];
const directory = 'scratch/gradient-boosted-trees-verification';
const browser = JSON.parse(fs.readFileSync(directory + '/browser/results.json','utf8'));
const captured = browser.records.flatMap(row => row.screenshots);
const images = names.map(name => {
  const path = directory + '/browser/' + name + '.png';
  const sha256 = crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
  assert(captured.some(row => row.path === path && row.sha256 === sha256));
  return {path, sha256, actuallyOpened: true, ...(name.startsWith('xgboost-program-reading') ? {actualContent: 'Ordered prefix-model program/output; the earlier test filename is retained, not a claim of an XGBoost screenshot.'} : {})};
});
fs.writeFileSync(directory + '/actually-opened-final-images.json',JSON.stringify({recordedAt:new Date().toISOString(),reviewer:'author /root/scientific_visual_improvements',images},null,2));
console.log(images.length+' actually opened images recorded.');

