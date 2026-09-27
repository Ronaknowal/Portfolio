import fs from 'node:fs';
import {parse} from '@babel/parser';
import {renderPreparedLesson} from './lib/prepared-lesson-renderer.mjs';
const id='hyena-long-convolution-models',draft='docs/teaching/drafts/'+id,publicFolder='public/learn-code/'+id;
fs.mkdirSync(publicFolder,{recursive:true});
for(const name of ['convolution_mechanisms.py','splice_models.py','author_calculations.py','blocked_convolution.py','splice-fits.npz','splice-results.json','splice.data','splice.names','data-provenance.md'])fs.copyFileSync(draft+'/'+name,publicFolder+'/'+name);
const figures=['DelayAddressFigure','EchoFigure','ToeplitzFigure','PaddingFigure','GateRailsFigure','CoefficientFigure','HierarchyFigure','LearnedFilterFigure','CoordinateFigure','GradientFigure','DnaWindowFigure','DataRolesFigure','LearningCurvesFigure','CounterfactualFigure','OverlapFigure','ModesFigure','ApproximationFigure','MechanismMapFigure','StripedFigure','BenchmarkFigure'];
// The packet's inline dollar notation becomes the renderer's inline-math syntax.
// Code fences are preserved byte-for-byte (apart from normalized line endings).
let fenced=false;
const manuscript=fs.readFileSync(draft+'/lesson.md','utf8').replaceAll('\r\n','\n').split('\n').map(line=>{if(line.startsWith('```')){fenced=!fenced;return line;}return fenced?line:line.replace(/\$([^$\n]+)\$/g,(_,math)=>'\\('+math+'\\)');}).join('\n');
const rendered=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:'/learn-code/'+id+'/',replacements:[...figures.map((name,i)=>['**Figure H'+String(i+1).padStart(2,'0')+' —',`<${name}/>`]),['**Investigation HA —','<ConvolutionLab/>'],['**Investigation HB —','<GateLab/>'],['**Investigation HC —','<DnaStudy/>'],['**Investigation HD —','<StreamingLab/>']],additions:[['The arrays both display','<HyenaProgram filename="convolution_mechanisms.py"/>'],['The training program contains','<SequenceBlockFigure/><HyenaProgram filename="splice_models.py"/><HyenaProgram filename="author_calculations.py"/>'],['For input length N, filter length M, block size B','<BlockedFftLab/><HyenaProgram filename="blocked_convolution.py"/>']]});
const output='// Generated from the complete prepared manuscript by scripts/generate-hyena-lesson.mjs.\n'+
"import {Prose,H2,H3,CodeBlock} from '../../components/content';\n"+
"import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';\n"+
"import {LessonIntro} from '../../components/lesson-labs/LessonElements.jsx';\n"+
"import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';\n"+
`import {${figures.filter(n=>n!=='LearnedFilterFigure').join(',')},SequenceBlockFigure} from '../../components/lesson-labs/HyenaFigures.jsx';\n`+
"import {ConvolutionLab,GateLab,BlockedFftLab,StreamingLab} from '../../components/lesson-labs/HyenaMechanismLabs.jsx';\n"+
"import {DnaStudy,LearnedFilterFigure,HyenaProgram} from '../../components/lesson-labs/HyenaStudy.jsx';\n"+
'export default {title:"Hyena & Long Convolution Models",readTime:"~75 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral hyena-lesson"><LessonIntro prerequisites="Weighted sums, matrix shapes and learning by reducing a loss. Fourier and signal-processing ideas are introduced here." sections={'+JSON.stringify(rendered.sections)+'}>Follow delayed contributions, input-dependent gates and the state that carries a filter across chunks.</LessonIntro>\n'+rendered.jsx+'\n</div>};\n';
parse(output,{sourceType:'module',plugins:['jsx']});
const path='src/learn/data/topics/'+id+'.jsx';fs.writeFileSync(path+'.tmp',output);fs.renameSync(path+'.tmp',path);
console.log('Complete Hyena manuscript,20 specified figures,block/application/Hankel insets,4 main investigations,blockedFFT workspace and4 complete program readers generated.');
