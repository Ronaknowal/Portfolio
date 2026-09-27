import fs from 'node:fs';import {parse} from '@babel/parser';import {renderPreparedLesson} from './lib/prepared-lesson-renderer.mjs';
const id='neural-training-diagnostics-reproducible-experiments',source=`docs/teaching/drafts/${id}`,manuscript=fs.readFileSync(`${source}/lesson.md`,'utf8').replaceAll('\r\n','\n').replace(/^> /gm,'');
const {jsx,sections}=renderPreparedLesson(manuscript,{ preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,assetBase:`/learn-assets/${id}/`,replacements:[
 ['**Inline figure — the evidence chain.**','<DiagnosticEvidenceFigure/>'],['**Inline figure — information permissions.**','<DiagnosticsSplitFigure/>'],['**Inline figure — values and derivatives together.**','<DiagnosticsActivationFigure/>'],['**Inline figure — measured loss curves.**','<DiagnosticsWineStudy/>'],['**Inline figure — first divergence after restart.**','<DiagnosticsRestartWorked/>'],['**Investigation — two switches, one forward pass.**','<DiagnosticsModeLab/>'],['**Investigation — choose what your experiment can answer.**','<DiagnosticsPairsLab/>'],['**Investigation — rebuild the continuation.**','<DiagnosticsRestartLab/><DiagnosticsReplayStudy/>']
],additions:[
 ['Losses are evaluation-mode means','<DiagnosticsTinyFigure/>'],['Use the scalar evidence chain','<DiagnosticsScalarLab/>'],['The [calculation record]','<DiagnosticsFiniteFigure/><DiagnosticsProgram file="calculations.py"/>'],['For diagnosis, it helps','<DiagnosticsProtocolFigure/>'],['The supplied program has all imports','<DiagnosticsProgram file="wine_diagnostics.py"/>'],['These repeats vary initialization','<DiagnosticsVarianceSourceFigure/>'],['The reference continuation uses','<DiagnosticsProgram file="checkpoint_replay.py"/>'],['**Extend the checkpoint boundary.**','<DiagnosticsCheckpointBoundary/>'],['Use separate named generators','<DiagnosticsRngFigure/>']
]});
const body=`// Complete eleven-section prepared manuscript, all case files, synthesis and code bridges.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {DiagnosticEvidenceFigure,DiagnosticsScalarLab,DiagnosticsModeLab,DiagnosticsRestartLab} from '../../components/lesson-labs/TrainingDiagnosticsLabs.jsx';
import {DiagnosticsSplitFigure,DiagnosticsTinyFigure,DiagnosticsActivationFigure,DiagnosticsFiniteFigure,DiagnosticsProtocolFigure,DiagnosticsVarianceSourceFigure,DiagnosticsRestartWorked,DiagnosticsCheckpointBoundary,DiagnosticsRngFigure} from '../../components/lesson-labs/TrainingDiagnosticsDiagrams.jsx';
import {DiagnosticsWineStudy,DiagnosticsPairsLab,DiagnosticsReplayStudy,DiagnosticsProgram,DiagnosticsDownloads} from '../../components/lesson-labs/TrainingDiagnosticsStudy.jsx';
export default {title:'Neural Training Diagnostics & Reproducible Experiments',readTime:'~75 min read + investigations, programs and practice',content:()=> <div className="neural-lesson neural-lesson-neutral diagnostics-lesson">
${jsx}
<DiagnosticsDownloads/>
</div>};
`;
for(const file of ['Labs','Diagrams','Study'])parse(fs.readFileSync(`src/learn/components/lesson-labs/TrainingDiagnostics${file}.jsx`,'utf8'),{sourceType:'module',plugins:['jsx']});parse(body,{sourceType:'module',plugins:['jsx']});fs.writeFileSync(`src/learn/data/topics/${id}.jsx`,body);console.log(JSON.stringify({sections:sections.length,investigations:4,evidenceExplorers:2,completePrograms:3}));
