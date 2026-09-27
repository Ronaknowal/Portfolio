import fs from 'node:fs';
import assert from 'node:assert/strict';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id='spectral-normalization-gradient-penalty', source='docs/teaching/drafts/'+id, destination='public/learn-code/'+id;
fs.mkdirSync(destination,{recursive:true});
const files=['critic-regularization-study.py','sensitivity-calculations.py','calculated-inputs.json','sensitivity-results.json','digits-400.csv','data-provenance.md'];
for(const file of files)fs.copyFileSync(source+'/'+file,destination+'/'+file);
const data=JSON.parse(fs.readFileSync(source+'/calculated-inputs.json'));
const csv=fs.readFileSync(source+'/digits-400.csv','utf8').trim().split(/\r?\n/).slice(1).map(line=>line.split(',').map(Number));
assert.equal(csv.length,400);
const rows=csv.map((row,i)=>{const[sourceId,...rest]=row,pixels=rest.slice(0,64),digit=rest[64];const profile=[0,1].map(half=>pixels.reduce((s,v,j)=>s+(Math.floor(j%8/4)===half?v:0),0)/512);assert.deepEqual(profile,data.measurements[i]);const roles=Object.entries(data.protocol.roles).filter(([,ids])=>ids.includes(sourceId));assert.equal(roles.length,1);return {sourceId,pixels,digit,profile,role:roles[0][0]};});
const dataset={rows,bootstrap:data.bootstrap,fits:data.fits.map(({method,seed,metrics})=>({method,seed,metrics})),csvSha256:data.protocol.csv_sha256};
fs.writeFileSync(destination+'/dataset.json',JSON.stringify(dataset));
for(const fit of data.fits)fs.writeFileSync(destination+'/model-'+fit.method+'-'+fit.seed+'.json',JSON.stringify(fit));
let manuscript=fs.readFileSync(source+'/lesson.md','utf8').replaceAll('\r\n','\n');
const fence=String.fromCharCode(96).repeat(3),begin=manuscript.indexOf(fence+'python\n'),end=manuscript.indexOf(fence,begin+fence.length);
assert.ok(begin>=0&&end>begin);assert.equal(manuscript.slice(begin+10,end).trim(),fs.readFileSync(source+'/critic-regularization-study.py','utf8').trim());
manuscript=manuscript.slice(0,begin)+'SPECTRAL_COMPLETE_PROGRAM\n'+manuscript.slice(end+fence.length);
const figures=['CriticDerivativeFigure','SpectralCompositionFigure','PenaltyLocationsFigure','BatchGradientFigure','SpectralGroupSortFigure','SpectralOdeFigure'];
const labs=['SpectralSlopeLab','SpectralCircleFigure','SpectralMatrixLab','SpectralDerivativeLab','SpectralConvolutionLab','SpectralPenaltyLab','SpectralLinearPenaltyLab','SpectralLibraryLab','SpectralMarginLab'];
const rendered=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:'/learn-code/'+id+'/',replacements:[
 ['**Visual: two update paths.','<CriticDerivativeFigure />'],
 ['**Visual: slope envelope.','<SpectralSlopeLab />'],
 ['**Visual: circle, ellipse and spectrum.','<SpectralCircleFigure />'],
 ['**Investigation: hide the strongest direction.','<SpectralMatrixLab />'],
 ['**Visual: overlapping stencils become a matrix.','<SpectralConvolutionLab />'],
 ['**Investigation: move the unobserved kink.','<SpectralPenaltyLab />'],
 ['**Visual investigation: points, surface and derivatives.','<SpectralMeasuredLab />'],
 ['SPECTRAL_COMPLETE_PROGRAM','<SpectralProgram />'],
],additions:[
 ['Bounds can be loose.','<SpectralCompositionFigure />'],
 ['The companion calculation checks the derivative','<SpectralDerivativeLab /><SpectralProgram filename="sensitivity-calculations.py" />'],
 ['A penalty-only descent step','<SpectralLinearPenaltyLab />'],
 ['R1 uses','<PenaltyLocationsFigure />'],
 ['For two scalar inputs, define','<BatchGradientFigure />'],
 ['At export, evaluation mode','<SpectralLibraryLab />'],
 ['This is a useful application of the same geometry','<SpectralMarginLab />'],
 ['For two values','<SpectralGroupSortFigure />'],
 ['The later [Neural ODE','<SpectralOdeFigure />'],
]});
const output='// Generated from the complete prepared manuscript by scripts/generate-spectral-lesson.mjs.\n'+
 "import { Prose,H2,H3,CodeBlock } from '../../components/content';\n"+
 "import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';\n"+
 "import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';\n"+
 "import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';\n"+
 'import { '+figures.join(', ')+' } from \'../../components/lesson-labs/SpectralMechanismFigures.jsx\';\n'+
 'import { '+labs.join(', ')+' } from \'../../components/lesson-labs/SpectralMechanismLabs.jsx\';\n'+
 "import { SpectralMeasuredLab,SpectralProgram } from '../../components/lesson-labs/SpectralMeasuredLab.jsx';\n"+
 'export default {title:"Spectral Normalization & Gradient Penalty",readTime:"~80 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral spectral-lesson"><LessonIntro prerequisites="Matrix multiplication, vector lengths and derivatives; the loss and derivative routes are developed here." sections={'+JSON.stringify(rendered.sections)+'}>Shape a learning signal by controlling what can stretch and by measuring where it changes.</LessonIntro>\n'+rendered.jsx+'\n</div>};\n';
fs.writeFileSync('src/learn/data/topics/'+id+'.jsx',output);
console.log('Generated complete spectral lesson; '+rows.length+' actual images and '+data.fits.length+' lossless selected-fit assets.');
