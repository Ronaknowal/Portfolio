import fs from 'node:fs';
import {parse} from '@babel/parser';
import {renderPreparedLesson} from './lib/prepared-lesson-renderer.mjs';
const id='mini-batches-training-loops-gradient-accumulation',source=`docs/teaching/drafts/${id}`;
let fenced=false;
const manuscript=fs.readFileSync(`${source}/lesson.md`,'utf8').replaceAll('\r\n','\n').split('\n').map(line=>{if(line.startsWith('```')){fenced=!fenced;return line;}return fenced?line:line==='$$'?'\\[':line.replace(/\$([^$]+)\$/g,(_,math)=>`\\(${math}\\)`);}).join('\n');
// Pair any display-dollar delimiters without altering complete Python programs.
let display=false;
const normalized=manuscript.split('\n').map(line=>{if(line==='\\['){display=!display;return display?'\\[':'\\]';}return line;}).join('\n');
const {jsx,sections}=renderPreparedLesson(normalized,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:`/learn-assets/${id}/`,replacements:[
 ['**Figure A —','<MinibatchClocksFigure/>'],['**Figure B —','<MinibatchStoresFigure/>'],['**Figure C —','<MinibatchCoefficientsFigure/>'],['**Figure D —','<MinibatchTargetSlotsFigure/>'],['**Figure E —','<MinibatchIrisHistory/><MinibatchIrisLab/>'],['**Figure F and investigation 3','<MinibatchNormalizationLab/>'],['**Figure G —','<MinibatchBoundaryFigure/>'],['**Investigation 1 —','<MinibatchStateLab/>'],['**Investigation 2 —','<MinibatchMassLab/>']
],additions:[
 ['Here \\(C\\) is','<MinibatchShapeFigure/>'],['\\(\\theta\\) denotes','<MinibatchGraphLifetimeFigure/>'],['The full runnable program is','<MinibatchProgram file="train_iris.py"/>'],['**Changed-constraint exercise.**','<MinibatchSevenFigure/>'],['using \\(\\epsilon=10^{-5}\\)','<MinibatchNormalizationWorked/>'],['For a constructed check, take','<MinibatchDropoutFigure/>'],['You can see the finite case exactly','<MinibatchVarianceFigure/>'],['Gradient clipping limits','<MinibatchClippingFigure/>'],['For two ranks holding','<MinibatchDdpFigure/>']
]});
const body=`// Generated from the complete ten-section manuscript, engine bridge and changed practice.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {MinibatchStateLab,MinibatchMassLab,MinibatchNormalizationLab} from '../../components/lesson-labs/MinibatchLoopLabs.jsx';
import {MinibatchClocksFigure,MinibatchStoresFigure,MinibatchCoefficientsFigure,MinibatchTargetSlotsFigure,MinibatchShapeFigure,MinibatchGraphLifetimeFigure,MinibatchNormalizationWorked,MinibatchDropoutFigure,MinibatchVarianceFigure,MinibatchBoundaryFigure,MinibatchClippingFigure,MinibatchDdpFigure,MinibatchSevenFigure} from '../../components/lesson-labs/MinibatchLoopDiagrams.jsx';
import {MinibatchIrisHistory,MinibatchIrisLab,MinibatchProgram,MinibatchDownloads} from '../../components/lesson-labs/MinibatchLoopStudy.jsx';
export default {title:'Mini-batches, Training Loops and Gradient Accumulation',readTime:'~75 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral minibatch-lesson">
${jsx}
<MinibatchProgram file="trace_update.py"/><MinibatchDownloads/>
</div>};
`;
for(const component of ['MinibatchLoopLabs','MinibatchLoopDiagrams','MinibatchLoopStudy'])parse(fs.readFileSync(`src/learn/components/lesson-labs/${component}.jsx`,'utf8'),{sourceType:'module',plugins:['jsx']});
parse(body,{sourceType:'module',plugins:['jsx']});fs.writeFileSync(`src/learn/data/topics/${id}.jsx`,body);console.log(JSON.stringify({sections:sections.length,investigations:4,completePrograms:2}));
