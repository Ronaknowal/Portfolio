import fs from 'node:fs';
import { parse } from '@babel/parser';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id='hybrid-ssm-transformer-architectures-jamba',source=`docs/teaching/drafts/${id}`;
let fenced=false,displayMath=false;
const manuscript=fs.readFileSync(`${source}/lesson.md`,'utf8').split(/\r?\n/).map(line=>{
 if(/^(?:```|~~~)/.test(line)){fenced=!fenced;return line.replace(/^~~~/,'```');}
 if(fenced)return line;
 if(line==='$$'){displayMath=!displayMath;return displayMath?'\\[':'\\]';}
 if(displayMath)return line;
 return line.replace(/\$([^$\n]+)\$/g,(_,math)=>`\\(${math}\\)`);
}).join('\n');
const {jsx,sections}=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:`/learn-assets/${id}/`,replacements:[
 ['[Figure J01','<JambaOpeningFigure/>'],['[Figure J02','<JambaWorkedReadFigure/>'],['[Figure J03','<JambaCollisionFigure/>'],['[Figure J04','<JambaResidualFigure/>'],['[Figure J05','<JambaDepthFigure/>'],['[Figure J06','<JambaStateFigure/>'],['[Figure J07','<JambaHeadsFigure/>'],['[Figure J08','<JambaMemoryFigure/>'],['[Figure J09','<JambaArithmeticFigure/>'],['[Figure J10','<JambaRouterWorked/>'],['[Figure J11','<JambaParameterFigure/>'],['[Figure J12','<JambaReleaseFigure/>'],['[Figure J13','<JambaWorkedStrokeFigure/>'],['[Figure J14','<JambaCandidatesFigure/>'],['[Figure J15','<JambaTrainingFigure/>'],['[Figure J16','<JambaContinuationFigure/>'],['[Figure J17','<JambaRequestsFigure/>'],['[Figure J18','<JambaAlternativesFigure/>'],
 ['**Investigation JA','<HybridMemoryLab/>'],['**Investigation JC','<HybridBudgetLab/>'],['**Investigation JD','<HybridRouterLab/>'],['**Investigation JB','<HybridStrokeLab/>']
],additions:[
 ['With fixed channel and state dimensions','<JambaSelectivePipeline/><JambaScanFigure/>'],
 ['The recurrent mixer includes input-dependent','<JambaPositionFigure/>'],
 ['At 52B parameters','<JambaPrecisionFigure/>'],
 ['Rewinding is different','<JambaRewindFigure/>'],
 ["Hymba sends the same layer input",'<JambaSharedVariantsFigure/>'],
 ['The channel method normalizes','<JambaProgram file="stroke_models.py"/><JambaProgram file="inspect_stroke.py"/>'],
 ['The optional [deployment program]','<JambaProgram file="deployment_example.py"/>']
]});
const body=`// Generated from the complete prepared manuscript, preserving all twelve sections and changed practice.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {HybridMemoryLab,HybridBudgetLab,HybridRouterLab} from '../../components/lesson-labs/HybridJambaLabs.jsx';
import {JambaOpeningFigure,JambaWorkedReadFigure,JambaCollisionFigure,JambaResidualFigure,JambaDepthFigure,JambaStateFigure,JambaHeadsFigure,JambaMemoryFigure,JambaArithmeticFigure,JambaRouterWorked,JambaParameterFigure,JambaReleaseFigure,JambaCandidatesFigure,JambaRequestsFigure,JambaAlternativesFigure} from '../../components/lesson-labs/HybridJambaDiagrams.jsx';
import {JambaSelectivePipeline,JambaScanFigure,JambaPositionFigure,JambaPrecisionFigure,JambaRewindFigure,JambaSharedVariantsFigure} from '../../components/lesson-labs/HybridJambaMechanisms.jsx';
import {JambaWorkedStrokeFigure,JambaTrainingFigure,JambaContinuationFigure,HybridStrokeLab,JambaProgram,JambaDownloads} from '../../components/lesson-labs/HybridJambaStudy.jsx';
export default {title:'Hybrid SSM–Transformer Architectures: Jamba and Complementary Memory',readTime:'~90 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral jamba-lesson">
${jsx}
<JambaProgram file="hybrid_mechanisms.py"/><JambaDownloads/>
</div>};
`;
for(const file of ['HybridJambaLabs','HybridJambaDiagrams','HybridJambaMechanisms','HybridJambaStudy'])parse(fs.readFileSync(`src/learn/components/lesson-labs/${file}.jsx`,'utf8'),{sourceType:'module',plugins:['jsx']});
parse(body,{sourceType:'module',plugins:['jsx']});fs.writeFileSync(`src/learn/data/topics/${id}.jsx`,body);
console.log(JSON.stringify({sections:sections.length,originalFigures:18,additionalMechanismFigures:6,investigations:4,selectedModelAssets:5}));
