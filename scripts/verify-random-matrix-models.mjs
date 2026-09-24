import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as model from '../src/learn/data/random-matrix-models.js';
import {randomMatrixExamples} from '../src/learn/data/random-matrix-examples.js';

const covariance = [];
for (const [rows,columns] of [[4,2],[5,8],[32,8],[64,16],[48,48],[24,48],[64,32]]) {
  for (const law of ['gaussian','sign']) for (const centered of [false,true]) for (const seed of [7,8,19]) {
    const state=model.randomMatrixSpectrum({rows,columns,law,centered,seed});
    const bins=model.randomMatrixBins(state.values,state.gamma);
    assert(Math.abs(bins.reduce((sum,bin)=>sum+bin.observed,0)+state.zeroCount/columns-1)<1e-12);
    assert(Math.abs(bins.reduce((sum,bin)=>sum+bin.theoretical,0)+model.marchenkoPastur(state.gamma).atom-1)<1e-11);
    covariance.push(state);
  }
}
const spikes=[];
for(const columns of [16,32]) for(const population of [1,1.1,1.5,1+Math.sqrt(columns/64),2,3,5]) for(const seed of [7,8]) {
  spikes.push({state:model.randomMatrixSpectrum({rows:64,columns,seed,spike:population}),limit:model.spikeLimits(columns/64,population)});
}
const intervals=[];
for(const gamma of [.02,.25,.8,1-1e-10,1,1+1e-10,1.1,4,20]) for(const variance of [.01,1,9,100]) {
  const law=model.marchenkoPastur(gamma,variance);
  for(const [left,right] of [[0,1],[.001,.2],[.2,.65],[.65,1]]) {
    const lower=law.lower+(law.upper-law.lower)*left;
    const upper=law.lower+(law.upper-law.lower)*right;
    intervals.push({gamma,variance,lower,upper,mass:model.mpIntervalMass(gamma,lower,upper,variance)});
  }
}
const wigner=[];
for(const size of [2,12,32,48]) for(const law of ['gaussian','sign']) for(const seed of [7,8,19]) wigner.push(model.wignerSpectrum(size,seed,law));
const levels=[];
for(const offset of [-10,0,3]) for(const difference of [-2,0,.5,2]) for(const coupling of [-1,0,.5,1.5]) levels.push({offset,difference,coupling,...model.twoLevelSpectrum(offset,difference,coupling)});
const bounds=[];
for(const [rows,columns] of [[25,20],[40,12],[1000,250],[100,100]]) for(const delta of [.05,1e-10,Number.MIN_VALUE]) bounds.push({rows,columns,delta,...model.gaussianSpectrumBound(rows,columns,delta)});
const calibration=[];
for(const nullModel of ['iid','duplicate']) for(const observedModel of ['iid','duplicate','spike']) {
  const result=model.matrixNullCalibration(nullModel,observedModel);
  assert.equal(result.rankScore,(1+result.maxima.filter(value=>value>=result.observed).length)/60);
  calibration.push(result);
}
const nullStates=[];
for(const duplicate of [false,true]) for(let index=0;index<59;index++) nullStates.push(model.randomMatrixSpectrum({rows:40,columns:12,seed:2000+index*7919,duplicate}));
const eigenFixtures=[[[2,0],[0,2]],[[0,0],[0,0]],[[1,.5],[.5,1]],[[-3,2,1],[2,5,-2],[1,-2,4]]].map(matrix=>({matrix,...model.symmetricRandomMatrixEigen(matrix)}));
const invalid=[
  ()=>model.randomMatrixSpectrum({rows:3}),()=>model.randomMatrixSpectrum({columns:49}),
  ()=>model.randomMatrixSpectrum({law:'uniform'}),()=>model.randomMatrixSpectrum({seed:0}),
  ()=>model.randomMatrixSpectrum({centered:'yes'}),()=>model.randomMatrixSpectrum({spike:Infinity}),
  ()=>model.symmetricRandomMatrixEigen([[1,2],[3,4]]),()=>model.symmetricRandomMatrixEigen([[NaN]]),
  ()=>model.marchenkoPastur(0),()=>model.marchenkoPastur(1,0),
  ()=>model.mpIntervalMass(1,2,1),()=>model.spikeLimits(1,2),
  ()=>model.gaussianSpectrumBound(10,11,.05),()=>model.gaussianSpectrumBound(10,2,0),
  ()=>model.wignerSpectrum(1),()=>model.twoLevelSpectrum(0,NaN,1),
];
invalid.forEach(fn=>assert.throws(fn,RangeError));
assert.notEqual(model.formatRandomMatrix(1e-12),'0');
assert.equal(model.formatRandomMatrix(0),'0');
const result={checkedAt:new Date().toISOString(),covariance,spikes,intervals,wigner,levels,bounds,calibration,nullStates,eigenFixtures,examples:randomMatrixExamples,rejections:invalid.length};
fs.mkdirSync('scratch/random-matrix-verification',{recursive:true});
fs.writeFileSync('scratch/random-matrix-verification/model-fixtures.json',JSON.stringify(result));
console.log(JSON.stringify({covariance:covariance.length,spikes:spikes.length,mpIntervals:intervals.length,wigner:wigner.length,levels:levels.length,bounds:bounds.length,calibrations:calibration.length,rejections:invalid.length}));
