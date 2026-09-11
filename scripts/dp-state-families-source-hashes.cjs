const fs = require('node:fs');
const crypto = require('node:crypto');
module.exports = () => [
  'src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx',
  'src/learn/data/curriculum/blueprints/dynamic-programming-states-transitions-optimization.js',
  'src/learn/data/practice/dynamic-programming-states-transitions-optimization.js',
  'src/learn/data/dp-state-families-models.js',
  'src/learn/data/dp-state-families-examples.js',
  'src/learn/components/lesson-labs/DpStateFamiliesLabs.jsx',
  'src/learn/components/lesson-labs/dp-state-families-labs.css',
].map(path => ({path,sha256:crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')}));
