import fs from 'node:fs';
import {parse} from '@babel/parser';
import {renderPreparedLesson} from './lib/prepared-lesson-renderer.mjs';
const id='ring-attention-sequence-parallelism',base=`docs/teaching/drafts/${id}`;
let manuscript=fs.readFileSync(`${base}/lesson.md`,'utf8').replaceAll('\r\n','\n').replace(/<!--[^]*?-->/g,'');
const programs=['ring-attention-reference.py','distributed_ring.py'];let index=0;
manuscript=manuscript.replace(/```python\n([^]*?)```/g,(_,code)=>{const file=programs[index];if(!file||code.trim()!==fs.readFileSync(`${base}/${file}`,'utf8').replaceAll('\r\n','\n').trim())throw Error('Full embedded source changed: '+file);return `RING_PROGRAM_${index++}`;});
if(index!==2)throw Error('Expected both full explained programs.');
const {jsx,sections}=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:`/learn-assets/${id}/`,replacements:[
 ['**Visual — memory inventory and token ownership.**','<RingMemoryFigure/>'],
 ['**Investigation — summary merger.**','<RingSummaryLab/>'],
 ['**Visual — identity travels with the packet.**','<RingIdentityInsets/><RingRotaryFigure/><RingIdentityLab/>'],
 ['**Investigation — causal workbench.**','<RingWorkLab/>'],
 ['**Investigation — schedule and capacity calculator.**','<RingCostLab/>'],
 ['**Visual — tensor reassembly puzzle.**','<RingUlyssesLab/>'],
 ['**Visual — gradient return map.**','<RingBackwardFigure/><RingGradientLab/>'],
 ['**Investigation — follow a real query around the ring.**','<RingMovementLab/>'],
 ['RING_PROGRAM_0','<RingProgram file="ring-attention-reference.py" title="Read the complete scratch reference: stable forward and recomputed backward"/><RingProgram file="attention-partition-study.py" title="Read the complete real-input and derivative study"/><RingProgram file="systems-calculations.py" title="Read the complete ownership, work and resource calculations"/>'],
 ['RING_PROGRAM_1','<RingProgram file="distributed_ring.py" title="Read the complete ordinary PyTorch CPU/Gloo multi-process program"/>']
],additions:[
 ['Why circulate K and V together?','<RingOwnershipFigure/>'],
 ['The rescaling line changes','<RingWorkedMergeFigure/>'],
 ['The subtraction expresses competition','<RingScalarGradientFigure/>'],
 ['The model maps each coordinate pair','<RingModelFigure/>'],
 ['Forward processing needs P block visits','<RingCodeBuffers/>'],
 ['One alternative is to keep KV shards stationary','<RingDecodeFigure/>']
]});
const figureNames=['RingMemoryFigure','RingOwnershipFigure','RingWorkedMergeFigure','RingIdentityInsets','RingRotaryFigure','RingBackwardFigure','RingScalarGradientFigure','RingModelFigure','RingCodeBuffers','RingDecodeFigure'];
const body=`// Generated from the complete fifteen-section manuscript and all eight changed-input exercises.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {${figureNames.join(',')}} from '../../components/lesson-labs/RingAttentionFigures.jsx';
import {RingSummaryLab,RingIdentityLab,RingGradientLab} from '../../components/lesson-labs/RingAttentionLabs.jsx';
import {RingWorkLab,RingCostLab,RingUlyssesLab} from '../../components/lesson-labs/RingAttentionSystems.jsx';
import {RingMovementLab,RingProgram} from '../../components/lesson-labs/RingAttentionStudy.jsx';
import '../../components/lesson-labs/neural-lesson-neutral.css';
import '../../components/lesson-labs/ring-attention.css';
export default {title:'Ring Attention & Sequence Parallelism',readTime:'~90 min read + investigations, programs and practice',content:()=> <div className="neural-lesson neural-lesson-neutral ring-attention">
${jsx}
</div>};
`;
for(const name of ['Figures','Labs','Systems','Study'])parse(fs.readFileSync(`src/learn/components/lesson-labs/RingAttention${name}.jsx`,'utf8'),{sourceType:'module',plugins:['jsx']});parse(body,{sourceType:'module',plugins:['jsx']});if(sections.length!==15)throw Error('Lost section');fs.writeFileSync(`src/learn/data/topics/${id}.jsx`,body);console.log(JSON.stringify({sections:sections.length,completePrograms:index+2,coreLabs:7}));
