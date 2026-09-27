import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id='modern-hopfield-networks',source='docs/teaching/drafts/'+id,destination='public/learn-code/'+id;
fs.mkdirSync(destination,{recursive:true});
// Public learner downloads, never author checks, private fixtures or draft specs.
for(const file of ['associative_memory.py','digit_memory.py','digit-memory-fits.npz','digit-results.json','mechanism-results.json','optdigits.tra','optdigits.tes','optdigits.names','data-provenance.md'])fs.copyFileSync(source+'/'+file,destination+'/'+file);
const figures=['LookupFigure','NormTrapFigure','SignedMemoryFigure','BinaryWorkedFigure','BinaryEnergyFigure','BinaryCapacityFigure','ContinuousWorkedFigure','CobwebFigure','EnergyLandscapeFigure','KeyValueFigure','ModuleOwnershipFigure','QueryTrainingFigure','DigitRolesFigure','DigitArchitectureFigure','DigitMetricsFigure','DigitStoriesFigure','MarginFigure','CapacityAxesFigure','ParityFigure','BagPoolingFigure','HopularFigure'];
const manuscript=fs.readFileSync(source+'/lesson.md','utf8').replaceAll('\r\n','\n').replace(/^~~~/gm,'```');
const rendered=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:'/learn-code/'+id+'/',replacements:[...figures.map((name,i)=>['[Figure '+(i+1)+':',`<${name} />`]),['[Investigation A:','<BinaryMemoryLab />'],['[Investigation B:','<ContinuousMemoryLab />'],['[Investigation C:','<AssociationLab />'],['[Investigation D:','<DigitMemoryLab />']],additions:[
 ['In our two-memory example, F′(0)','<SensitivityFigure />'],
 ['For the smaller exact mechanisms,','<HopfieldProgram filename="associative_memory.py" /><HopfieldProgram filename="digit_memory.py" />'],
 ['There is also a genuine architecture consequence','<ChangingBankFigure />'],
]});
const output='// Generated from the full prepared manuscript by scripts/generate-hopfield-lesson.mjs.\n'+
"import {Prose,H2,H3,CodeBlock} from '../../components/content';\n"+
"import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';\n"+
"import {LessonIntro} from '../../components/lesson-labs/LessonElements.jsx';\n"+
"import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';\n"+
'import {'+figures.filter(n=>n!=='DigitStoriesFigure').join(',')+",SensitivityFigure,ChangingBankFigure} from '../../components/lesson-labs/HopfieldFigures.jsx';\n"+
"import {BinaryMemoryLab,ContinuousMemoryLab,AssociationLab} from '../../components/lesson-labs/HopfieldLabs.jsx';\n"+
"import {DigitMemoryLab,DigitStoriesFigure,HopfieldProgram} from '../../components/lesson-labs/HopfieldDigitLab.jsx';\n"+
'export default {title:"Modern Hopfield Networks",readTime:"~70 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral hopfield-lesson"><LessonIntro prerequisites="Vectors, dot products and weighted averages; derivatives and the energy argument are developed here." sections={'+JSON.stringify(rendered.sections)+'}>Turn a partial cue into a memory read, then test what that read actually improves.</LessonIntro>\n'+rendered.jsx+'\n</div>};\n';
const path='src/learn/data/topics/'+id+'.jsx';fs.writeFileSync(path+'.tmp',output);fs.renameSync(path+'.tmp',path);console.log('Generated all21 prepared figures,4 distinct investigations and the complete manuscript.');
