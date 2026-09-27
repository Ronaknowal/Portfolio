import fs from 'node:fs';
import {parse} from '@babel/parser';
import {renderPreparedLesson} from './lib/prepared-lesson-renderer.mjs';
const id='xlstm-extended-lstm',source='docs/teaching/drafts/'+id,destination='public/learn-code/'+id;
fs.mkdirSync(destination,{recursive:true});
// Explicit reader-facing programs, data, saved fits and attribution only.
for(const file of ['memory_mechanisms.py','row_sequence_models.py','author_calculations.py','row-sequence-fits.npz','row-sequence-results.json','optdigits.tra','optdigits.tes','optdigits.names','data-provenance.md'])fs.copyFileSync(source+'/'+file,destination+'/'+file);
const figures=['XOpeningFigure','OrdinaryCellFigure','ScalarWorkedFigure','ScalarMixingFigure','ScalarScaleFigure','GateLearningFigure','OuterProductFigure','MatrixWorkedFigure','SignedReadFigure','MatrixFloorFigure','CausalWorkedFigure','ChunkWorkedFigure','ReaderBlockFigure','DigitScanFigure','RowDataRolesFigure','RowMetricsFigure','WorkedDigitTraceFigure','VersionBlocksFigure','StateAccountingFigure','ApplicationFlowsFigure'];
const manuscript=fs.readFileSync(source+'/lesson.md','utf8').replaceAll('\r\n','\n').replace(/^~~~/gm,'```');
const rendered=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:'/learn-code/'+id+'/',replacements:[...figures.map((name,i)=>['[Figure X'+String(i+1).padStart(2,'0')+':',`<${name} />`+(i===15?'<RowLearningCurve />':'')]),['[Investigation XA:','<ScalarLedgerLab />'],['[Investigation XB:','<MatrixAddressLab />'],['[Investigation XC:','<CausalChunkLab />'],['[Investigation XD:','<RowReaderLab />']],additions:[['The complete source is included with this lesson.','<XlstmProgram filename="memory_mechanisms.py" /><XlstmProgram filename="row_sequence_models.py" /><XlstmProgram filename="author_calculations.py" /><p>To inspect the retained experiment without refitting, download the <a href="/learn-code/xlstm-extended-lstm/row-sequence-fits.npz">six selected fits and inspection arrays</a> and <a href="/learn-code/xlstm-extended-lstm/row-sequence-results.json">complete measured curves and confusion matrices</a>. The calculation program reads the fit archive; the training program generates it.</p>'],['There is also a later sigmoid-input matrix variant.','<SigmoidVariantFigure />']]});
const output='// Generated from the complete prepared manuscript by scripts/generate-xlstm-lesson.mjs.\n'+
"import {Prose,H2,H3,CodeBlock} from '../../components/content';\n"+
"import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';\n"+
"import {LessonIntro} from '../../components/lesson-labs/LessonElements.jsx';\n"+
"import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';\n"+
`import {${figures.join(',')},SigmoidVariantFigure} from '../../components/lesson-labs/XlstmFigures.jsx';\n`+
"import {ScalarLedgerLab,MatrixAddressLab,CausalChunkLab} from '../../components/lesson-labs/XlstmMechanismLabs.jsx';\n"+
"import {RowReaderLab,RowLearningCurve,XlstmProgram} from '../../components/lesson-labs/XlstmStudy.jsx';\n"+
'export default {title:"xLSTM (Extended LSTM)",readTime:"~75 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral xlstm-lesson"><LessonIntro prerequisites="Weighted sums, vectors and basic neural networks; the cell mechanics and state invariants are developed here." sections={'+JSON.stringify(rendered.sections)+'}>Decide what a recurrent memory stores, how a query reads it, and which state a chunk must preserve.</LessonIntro>\n'+rendered.jsx+'\n</div>};\n';
parse(output,{sourceType:'module',plugins:['jsx']});
const path='src/learn/data/topics/'+id+'.jsx';fs.writeFileSync(path+'.tmp',output);fs.renameSync(path+'.tmp',path);
console.log('Complete xLSTM manuscript,20 specified figures,version/application insets,4 investigations and3 program readers generated.');
