import { formatLabValue, LabChoices, LabFeedback } from "./LabControls.jsx";
import { useState } from "react";
import { Investigation } from "./LessonInvestigation.jsx";
import { initialNotebook, notebookAction, randomConsumers, provenanceModel } from "../../data/notebook-models.js";
import './reliability.css';
export function NotebookKernelLab() {
  const [notebookState, setNotebookState] = useState(initialNotebook);
  const executeNotebookAction = (a, v) => setNotebookState(x => notebookAction(x, a, v));
  return <Investigation id="notebook-kernel" kicker="Document ≠ live memory ≠ saved output" title="Change the source without changing the answer">
 <p className="lesson-note">Run all, change offset to 5 and run only Display. Inspect stored values, then rerun the dependency that recomputes the result.</p><div className="nt-controls"><LabChoices label="Offset written in input cell" value={notebookState.sourceOffset} onChange={v => executeNotebookAction('edit', Number(v))} items={[[2, '2 ms'], [5, '5 ms']]} /></div>
 <div className="rel-notebook"><div><h4>Document cells</h4>{[['inputs', `values = [10,20,30]; offset = ${notebookState.sourceOffset}`], ['calculate', 'mean = sum(values) / len(values) - offset'], ['display', 'print(mean)']].map(([a, code], i) => <div className="rel-cell" key={a}><span>In [{notebookState.counts[i] ?? ' '}]</span><code>{code}</code><button type="button" onClick={() => executeNotebookAction(a)}>Run {['Inputs', 'Calculate', 'Display'][i]}</button></div>)}</div><div className="rel-memory"><h4>Kernel stores</h4><dl><dt>offset</dt><dd>{formatLabValue(notebookState.rate)}</dd><dt>mean</dt><dd>{formatLabValue(notebookState.mean)}</dd></dl><h4>Saved display output</h4><output>{notebookState.output === null ? 'no output' : notebookState.output}</output></div></div>
 <LabFeedback>{notebookState.note}</LabFeedback><div className="nt-stepper"><button type="button" onClick={() => executeNotebookAction('all')}>Run all</button><button type="button" onClick={() => executeNotebookAction('restart')}>Restart kernel</button><button type="button" onClick={() => setNotebookState(initialNotebook())}>Reset investigation</button></div><p className="lesson-note">Three fixed cells, no arbitrary Python. Restart intentionally retains saved output so you can see why a screenshot is historical evidence. Transfer: restart, then Calculate before Inputs; explain the failure.</p></Investigation>;
}
export function RandomStreamLab() {
  const [separate, setSeparate] = useState(false),
    [extra, setExtra] = useState(false),
    streamModel = randomConsumers(separate, extra);
  return <Investigation id="notebook-streams" kicker="Random generators remember consumption" title="Can an extra draw change a different experiment?">
 <p className="lesson-note">Give preprocessing one extra random draw and compare the model input. Switch to independent streams to separate unrelated consumers.</p><div className="nt-controls"><LabChoices label="Generator ownership" value={String(separate)} onChange={v => setSeparate(v === 'true')} items={[["false", "One shared stream"], ["true", "Separate child streams"]]} /><LabChoices label="Preprocessing draws" value={String(extra)} onChange={v => setExtra(v === 'true')} items={[["false", "One draw"], ["true", "Two draws"]]} /></div>
 <div className="rel-stream"><strong>{separate ? 'Preprocessing stream' : 'Shared stream'}</strong><div className="rel-tokens">{streamModel.splitTape.map((v, i) => <span key={i} data-active={i < streamModel.split.length}>{v}<small>{i < streamModel.split.length ? 'preprocess' : !separate && i === streamModel.modelIndex ? 'model' : 'unread'}</small></span>)}</div>{separate && <><strong>Model stream</strong><div className="rel-tokens">{streamModel.modelTape.map((v, i) => <span key={i} data-active={i === 0}>{v}<small>{i === 0 ? 'model' : 'unread'}</small></span>)}</div></>}<p>Preprocessing receives {streamModel.split.join(', ')} → model receives <strong>{streamModel.model}</strong>.</p></div>
 <LabFeedback>{separate ? 'Preprocessing advances only its own cursor. Its extra draw leaves the model at 7.' : 'Both consumers advance one cursor. An extra preprocessing draw shifts which tape value reaches the model.'}</LabFeedback><button type="button" onClick={() => {
      setSeparate(false);
      setExtra(false);
    }}>Reset</button><p className="lesson-note">The numbers form an invented deterministic tape; they are not NumPy samples, statistical independence evidence or a random-number algorithm. The native NumPy example below uses SeedSequence.spawn and checks real generator state.</p></Investigation>;
}
export function ProvenanceLab() {
  const [change, setChange] = useState('none'),
    [complete, setComplete] = useState(false),
    provenance = provenanceModel(change, complete);
  return <Investigation id="notebook-provenance" kicker="Trace the identity of a result" title="Does the same filename mean the same experiment?">
 <p className="lesson-note">Compare data [10,20,30] with [9,20,31]. Inspect both the unchanged mean and the changed data identity.</p><div className="nt-controls"><LabChoices label="New run" value={change} onChange={setChange} items={[["none", "Identical inputs"], ["sameMean", "Different data, same mean"], ["data", "Different data and mean"], ["offset", "Offset becomes 5"], ["code", "Code revision changes"]]} /><LabChoices label="Cache key ingredients" value={String(complete)} onChange={v => setComplete(v === 'true')} items={[["false", "Filename only"], ["true", "Bytes + config + code + fixed environment"]]} /></div>
 <div className="rel-pipeline"><div><strong>Current ingredients</strong><code>{provenance.current.data}</code><p>offset {provenance.current.offset} · {provenance.current.code}</p></div><b aria-hidden="true">→</b><div><strong>Lookup decision</strong><output>{provenance.hit ? 'REUSE 18' : 'RECOMPUTE'}</output></div><b aria-hidden="true">→</b><div><strong>Fresh result</strong><output>{provenance.result}</output></div></div>
 <LabFeedback>{provenance.hit ? provenance.result !== 18 ? 'Wrong reuse: the weak key selects 18 even though the current run produces ' + provenance.result + '.' : provenance.sameBytes && change === 'none' ? 'The recorded ingredients match in this fixed example.' : 'The reused number happens to agree. That does not establish that the recorded inputs or code match.' : 'The declared identity changed. Recompute and save a new manifest, even when the numeric answer happens to agree.'}</LabFeedback><button type="button" onClick={() => {
      setChange('none');
      setComplete(false);
    }}>Reset</button><p className="lesson-note">This model compares complete ingredient strings, not browser-computed hashes. The Python manifest hashes actual bytes. The environment is held fixed; external services, nondeterminism and undeclared inputs need their own policies.</p></Investigation>;
}
