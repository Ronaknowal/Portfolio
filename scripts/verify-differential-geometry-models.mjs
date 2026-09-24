import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as g from '../src/learn/data/differential-geometry-models.js';

const fixtures = { charts:[], metrics:[], patches:[], maps:[], steps:[], polar:[], transport:[], curvature:[], covariance:[] };
for (let angle=0; angle<=360; angle+=5) fixtures.charts.push(g.circleCharts(angle));
for (const shear of [-1.5,-.5,0,1,1.5]) for (const cost of [.5,1,2,3]) {
  for (const angle of [0,35,90,180,315]) fixtures.metrics.push(g.metricDifferential(shear,cost,angle));
}
for (const theta of [15,45,60,90,135,165]) for (const phi of [-180,-75,0,35,120,180]) {
  for (const radius of [.5,1,3]) fixtures.patches.push(g.spherePatch(theta,phi,radius));
}
for (const raw of [[1,0,0],[0,0,1],[1,2,3],[-2,1,-1]]) {
  const point = g.scale(raw,1/g.norm(raw));
  const axis = Math.abs(point[2]) < .8 ? [0,0,1] : [0,1,0];
  const rawTangent = g.cross(point,axis), direction = g.scale(rawTangent,1/g.norm(rawTangent));
  // Rotated floating-point unit vectors can round a nominal norm of 8 upward;
  // keep these fixtures inside the strict input contract and test its exact
  // boundary separately with an axis-aligned tangent below.
  for (const length of [0,1e-12,1e-8,1e-4,.5,2,Math.PI-1e-6,Math.PI,4,7.5]) {
    const tangent = g.scale(direction,length), exponential = g.sphereExp(point,tangent);
    let logarithm = null;
    if (length < Math.PI-1e-7) logarithm=g.sphereLog(point,exponential);
    fixtures.maps.push({point,tangent,length,exponential,logarithm,
      retracted:g.sphereRetract(point,tangent),angle:g.sphereAngle(point,exponential)});
  }
}
assert.doesNotThrow(() => g.sphereExp([1,0,0],[0,8,0]));
assert.throws(() => g.sphereExp([1,0,0],[0,8.000001,0]),RangeError);
for (const angle of [-180,-90,-30,0,45,90,180]) for (const rate of [0,.1,.5,1,1.5]) fixtures.steps.push(g.sphereStep(angle,rate));
for (const time of [-2,-1.25,-.1,0,.3,1,2]) for (const height of [.25,.5,1,2]) fixtures.polar.push(g.polarConnection(time,height));
for (const angle of [15,45,90,150]) for (const initial of [-135,0,60]) for (const reverse of [false,true]) {
  for (const progress of [0,.25,1,1.5,2.75,3]) fixtures.transport.push(g.transportTriangle(angle,initial,progress,reverse,2));
}
for (const radius of [.5,1,2,3]) for (const distance of [.05,.1,.5,1,1.5]) fixtures.curvature.push(g.curvatureComparison(radius,distance));
for (const ratio of [2,4,9]) for (const fraction of [0,.1,.5,.8,1]) fixtures.covariance.push(g.covariancePaths(ratio,fraction));

const failures = [
  () => g.circleCharts(-1), () => g.circleCharts(NaN), () => g.circleCharts(361),
  () => g.metricDifferential(2), () => g.metricDifferential(0,0),
  () => g.spherePatch(0), () => g.spherePatch(90,0,0),
  () => g.sphereExp([2,0,0],[0,1,0]), () => g.sphereExp([1,0,0],[1,0,0]),
  () => g.sphereExp([1,0,0],[0,9,0]),
  () => g.sphereLog([1,0,0],[-1,0,0]),
  () => g.sphereTransport([1,0,0],[-1,0,0],[0,1,0]),
  () => g.sphereProjection([1,0,0],[Infinity,0,0]),
  () => g.sphereArc(181), () => g.sphereStep(0,2),
  () => g.polarConnection(0,0), () => g.transportTriangle(0),
  () => g.transportTriangle(90,0,3,'false'),
  () => g.curvatureComparison(1,2), () => g.warpedMetric('unknown'),
  () => g.covariancePaths(1),
];
failures.forEach(check => assert.throws(check,RangeError));
assert(Object.isFrozen(fixtures.transport[0]) && Object.isFrozen(fixtures.transport[0].vectors[0]));
assert.throws(() => { fixtures.metrics[0].metric[0][0] = 0; }, TypeError);
fs.mkdirSync('scratch/differential-geometry-verification',{recursive:true});
fs.writeFileSync('scratch/differential-geometry-verification/model-fixtures.json',JSON.stringify(fixtures));
console.log(JSON.stringify({fixtureCounts:Object.fromEntries(Object.entries(fixtures).map(([key,value])=>[key,value.length])),rejected:failures.length}));
