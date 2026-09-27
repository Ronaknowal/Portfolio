import fs from 'node:fs';
const dir='docs/teaching/drafts/hyena-long-convolution-models/';
const mechanisms=JSON.parse(fs.readFileSync(dir+'mechanism-results.json'));
const author=JSON.parse(fs.readFileSync(dir+'author-results.json'));
const results=JSON.parse(fs.readFileSync(dir+'splice-results.json'));
const display={mechanisms,workedSequence:author.sequence_fixtures[0],freshSequence:author.sequence_fixtures[1],allN:author.all_ambiguity,fits:results.fits,data:{rawRows:results.data.raw_rows,retainedRows:results.data.retained_rows,conflicts:results.data.conflicting_source_ids,classCounts:results.data.class_counts,roleClassCounts:results.data.role_class_counts},attribution:'Towell,Noordewier,Shavlik; UCI dataset69; CC BY4.0'};
fs.writeFileSync('src/learn/data/hyena-example-inputs.js','// Lossless worked inputs and complete measured curves; selected model weights are loaded separately.\nexport const hyenaExamples='+JSON.stringify(display)+';\n');
console.log('Selected display data and all320 actual validation epochs extracted without downsampling.');
