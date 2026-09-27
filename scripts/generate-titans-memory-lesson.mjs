import fs from 'node:fs';
import {parse} from '@babel/parser';
import {renderPreparedLesson} from './lib/prepared-lesson-renderer.mjs';
const id='titans-multi-memory-architecture',source=`docs/teaching/drafts/${id}`;
const manuscript=fs.readFileSync(`${source}/lesson.md`,'utf8');
const {jsx,sections}=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:`/learn-assets/${id}/`,replacements:[
 ['**Follow one sequence.**','<TitansLifetimesFigure/>'],['**The write direction.**','<TitansWriteFigure/>'],['**Add the actual vectors.**','<TitansUpdateFigure/>'],['**Three different junctions.**','<TitansJunctionsFigure/>'],['**A gradient through a gradient.**','<TitansOuterFigure/>'],['**Predict, observe, write.**','<TitansTimelineFigure/>'],['**Separate storage bands.**','<TitansPayloadFigure/>'],['**Measured figure — paired outcomes.**','<TitansPairedFigure/><TitansRentalLab/>'],
 ['**Investigation — which answer','<TitansWriteLab/>'],['**Investigation — what belongs','<TitansGatedLab/>'],['**Investigation — move the gradient','<TitansChunkLab/>']
],additions:[
 ['Thus repeated updates reduce','<TitansStabilityFigure/>'],['Here \\(W_1\\) has shape','<TitansNonlinearFigure/>'],['The supplied [neural_memory.py]','<TitansProgram file="neural_memory.py"/>'],['For autoregressive prediction','<TitansProjectionFigure/>'],['In PyTorch, `autograd.grad','<TitansOuterLab/>'],['Here `observed_value`','<TitansProgram file="rental_memory_study.py"/>'],['[memory_mechanisms.py](memory_mechanisms.py) owns','<TitansProgram file="memory_mechanisms.py"/>'],['With fixed gradient inputs','<TitansScanFigure/>'],['The TTT research line also explores','<TitansResearchFigure/>']
]});
const body=`// Generated from the complete twelve-section prepared manuscript and all changed practice.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {TitansWriteLab,TitansGatedLab,TitansChunkLab,TitansOuterLab} from '../../components/lesson-labs/TitansMemoryLabs.jsx';
import {TitansLifetimesFigure,TitansWriteFigure,TitansUpdateFigure,TitansStabilityFigure,TitansNonlinearFigure,TitansProjectionFigure,TitansJunctionsFigure,TitansOuterFigure,TitansTimelineFigure,TitansPayloadFigure,TitansScanFigure,TitansResearchFigure} from '../../components/lesson-labs/TitansMemoryDiagrams.jsx';
import {TitansPairedFigure,TitansRentalLab,TitansProgram,TitansDownloads} from '../../components/lesson-labs/TitansMemoryStudy.jsx';
export default {title:'Titans: Writing into a Multi-Memory Architecture',readTime:'~90 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral titans-lesson">
${jsx}
<TitansProgram file="check_author_packet.py"/><TitansDownloads/>
</div>};
`;
for(const component of ['TitansMemoryLabs','TitansMemoryDiagrams','TitansMemoryStudy'])parse(fs.readFileSync(`src/learn/components/lesson-labs/${component}.jsx`,'utf8'),{sourceType:'module',plugins:['jsx']});
parse(body,{sourceType:'module',plugins:['jsx']});fs.writeFileSync(`src/learn/data/topics/${id}.jsx`,body);console.log(JSON.stringify({sections:sections.length,investigations:5,completePrograms:4}));
