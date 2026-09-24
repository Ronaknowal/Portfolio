import fs from 'node:fs';
import * as model from '../src/learn/data/geometry-trigonometry-models.js';
import { geometryTrigonometryExamples } from '../src/learn/data/geometry-trigonometry-examples.js';
const directory = 'scratch/geometry-trigonometry-independent';
fs.mkdirSync(directory, { recursive: true });
const frames = [];
for (const [px, py, ox, oy] of [[-4, -4, 2, 2], [4, -4, -2, 2], [0, 0, 0, 0], [1, -3, -2, -1]]) {
  for (const degrees of [-165, -105, -45, 0, 75, 135, 180]) for (const mode of ['active', 'passive']) {
    frames.push({ inputs: [px, py, ox, oy, degrees, mode], result: model.frameState(px, py, ox, oy, degrees, mode) });
  }
}
const circles = [-345, -255, -195, -105, -15, 15, 75, 165, 255, 345].map(degrees => model.circleComponents(degrees));
fs.writeFileSync(directory + '/fixtures.json', JSON.stringify({ frames, circles, ambiguous: model.ambiguousTriangleState(), examples: geometryTrigonometryExamples }));
console.log(`Exported ${frames.length} changed frames and ${circles.length} circle states for independent high-precision checks.`);
