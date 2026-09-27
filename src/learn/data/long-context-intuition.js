import { recurrenceTrace, stableSoftmax } from './long-context-models.js';

// Small constructed examples. They are separate from the measured Libras records.
export const entranceMessages = [
  { position: 0, text: 'Use the east entrance.', status: 'Outside the current window', available: false },
  { position: 1, text: 'The meeting starts at ten.', status: 'Outside the current window', available: false },
  { position: 2, text: 'Lunch is in the courtyard.', status: 'Outside the current window', available: false },
  { position: 3, text: 'Bring your badge.', status: 'In the current window', available: true },
  { position: 4, text: 'Which entrance should I use?', status: 'Current question', available: true },
];

const supports = [2, 1, 3];
const values = [2, 4, 8];
const weights = stableSoftmax(supports.map(Math.log));
export const weightedReadExample = supports.map((support, i) => ({
  donor: String.fromCharCode(65 + i), support, value: values[i],
  weight: weights[i], contribution: weights[i] * values[i],
}));

export const silentStepExamples = [
  { label: 'Ordinary silent step', input: 1, recurrence: 1 / 8 },
  { label: 'Close the input gate only', input: 0, recurrence: 1 / 8 },
  { label: 'Exact hold limit', input: 1, recurrence: 0 },
].map(event => ({ ...event, ...recurrenceTrace([{ x: 0, ...event }], .8, .6)[0] }));

export const sameMeanPaths = [
  { label: 'A · straight path', heights: [.5, .5, .5, .5, .5] },
  { label: 'B · changing direction', heights: [.5, .75, 0, .75, .5] },
].map(path => {
  const points = path.heights.map((y, i) => ({ x: i / 4, y }));
  return { ...path, points, mean: {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  } };
});
