import { useId, useState } from "react";
import { Investigation, Stepper } from "./LessonInvestigation.jsx";
import { stagingTrace, branchTrace, remoteTrace, conflictVersions, conflictTrace } from "../../data/git-foundations-model";
import { GitRemotePicture } from "./GitFigures.jsx";
export function GitStagingLab() {
  const [step, setStep] = useState(0),
    [restage, setRestage] = useState(false);
  const trace = stagingTrace(restage),
    state = trace[step];
  return <Investigation id="git-staging" kicker="EDIT → CAPTURE → COMMIT" title="Does committing save the file you are looking at?">
    <p>One tracked file begins at version 1. Stage version 2, then edit it again. Keep the three stored versions visible as you follow the arrows.</p><p className="lesson-note">Change the working file and stage it separately. Compare the working tree, index and next commit to see exactly what each command records.</p>
    <div className="nt-controls"><label>Before committing<select value={String(restage)} onChange={e => {
          setRestage(e.target.value === 'true');
          setStep(0);
        }}><option value="false">Inspect without staging again</option><option value="true">Stage version 3 again</option></select></label></div>
    <pre className="nt-code">{state.command}</pre><div className="nt-snapshots">{['HEAD snapshot', 'Index: proposed snapshot', 'Working file'].map((label, i) => <div className={'nt-snapshot ' + (i === 2 && state.transfer.startsWith('Edit') || i === 1 && state.transfer.includes('→ index') || i === 0 && state.transfer.includes('→ new') ? 'is-active' : '')} key={label}><h4>{label}</h4><pre>version {state.versions[i]}</pre><small>{i === 0 ? 'Last committed content' : i === 1 ? 'What a normal commit will record' : 'What your editor currently shows'}</small></div>)}</div>
    <p className="nt-transfer">{state.transfer}</p><p className="nt-feedback" aria-live="polite">{state.note}</p><p>Short status: <code>{state.status === '  ' ? '(clean)' : JSON.stringify(state.status) + ' report.txt'}</code>. First column: index vs HEAD. Second: working file vs index.</p>
    <Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">This fixed model contains one already-tracked file and ordinary commits. It does not model untracked files, partial hunks or operate on your repository.</p>
  </Investigation>;
}
export function GitBranchLab() {
  const uid = useId(),
    [step, setStep] = useState(0),
    [diverged, setDiverged] = useState(false);
  const trace = branchTrace(diverged),
    state = trace[step],
    positions = {
      A: [52, 143],
      B: [167, 65],
      C: [167, 225],
      M: [293, 143]
    };
  const checked = state[state.head],
    files = state.nodes.find(n => n.id === checked).files;
  return <Investigation id="git-branches" kicker="SNAPSHOTS STAY; REFERENCES MOVE" title="Why can one merge just move a branch name?">
    <p>Feature adds a note. Main may also add units. Watch commits, parent arrows and the branch that HEAD names. Files shown below belong to the checked-out snapshot.</p><p className="lesson-note">Create a branch and advance each pointer. Follow the commit graph to distinguish a new branch name from a new commit and inspect fast-forward eligibility.</p>
    <div className="nt-controls"><label>Main branch activity<select value={String(diverged)} onChange={e => {
          setDiverged(e.target.value === 'true');
          setStep(0);
        }}><option value="false">Main stays at A</option><option value="true">Main makes its own commit C</option></select></label></div><pre className="nt-code">{state.command}</pre>
    <svg className="nt-diagram" viewBox="0 0 360 295" role="img" aria-label={`${state.nodes.length} commits. main at ${state.main}; feature at ${state.feature ?? 'not created'}; HEAD names ${state.head}. Arrows point to parents.`}>
      <defs><marker id={uid} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#a99260" /></marker></defs>
      {state.nodes.flatMap(n => n.parents.map(parent => {
        const [x, y] = positions[n.id],
          [px, py] = positions[parent],
          length = Math.hypot(px - x, py - y),
          ux = (px - x) / length,
          uy = (py - y) / length;
        return <line key={n.id + parent} x1={x + ux * 23} y1={y + uy * 23} x2={px - ux * 25} y2={py - uy * 25} stroke="#a99260" strokeWidth="2" markerEnd={`url(#${uid})`} />;
      }))}
      {state.nodes.map(n => {
        const [x, y] = positions[n.id];
        return <g key={n.id}><circle cx={x} cy={y} r="22" fill={checked === n.id ? '#3c3019' : '#1a211b'} stroke={checked === n.id ? '#e2bb68' : '#8dab91'} strokeWidth="2" /><text x={x} y={y + 6} textAnchor="middle" className="nt-emphasis">{n.id}</text><text x={x} y={y + 43} textAnchor="middle" className="nt-small">{[state.main === n.id ? 'main' : '', state.feature === n.id ? 'feature' : ''].filter(Boolean).join(', ')}</text></g>;
      })}
      <text x="12" y="18" className="nt-small">Arrows: commit → parent</text>
    </svg>
    <p><strong>HEAD → {state.head} → {checked}</strong></p><p>Checked-out tracked files: {files.map(f => <span key={f} className="nt-chip">{f}</span>)}</p><p className="nt-feedback" aria-live="polite">{state.note}</p><Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">Letters stand for real commit IDs. The two-branch examples use a clean working tree and nonconflicting files; a separate investigation handles conflicting edits. File contents, not visual branch position, determine a merge result.</p>
  </Investigation>;
}
export function GitRemoteLab() {
  const [step, setStep] = useState(0),
    [localChange, setLocalChange] = useState(false);
  const trace = remoteTrace(localChange),
    state = trace[step];
  return <Investigation id="git-remotes" kicker="SHARED STATE ≠ YOUR LAST FETCH" title="Did fetch update your files or your knowledge?">
    <p>A colleague publishes B after your clone of A. Origin/main is a reference stored on your machine; it records what you last learned about the shared branch.</p><p className="lesson-note">Compare a colleague push, your fetch and your local commit. Follow the remote branch, remote-tracking reference and local branch separately.</p>
    <div className="nt-controls"><label>Your local work<select value={String(localChange)} onChange={e => {
          setLocalChange(e.target.value === 'true');
          setStep(0);
        }}><option value="false">No new local commit</option><option value="true">Commit a local note as C</option></select></label></div><pre className="nt-code">{state.command}</pre>
    <GitRemotePicture state={state} step={step} localChange={localChange} />
    <p className="nt-feedback" aria-live="polite">{state.note}</p><Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">This model assumes the usual origin fetch mapping; custom refspecs, shallow/partial clones and hosting permissions need separate treatment.</p>
  </Investigation>;
}
export function GitConflictLab() {
  const [step, setStep] = useState(0),
    [resolution, setResolution] = useState('combined');
  const trace = conflictTrace(resolution),
    state = trace[step];
  return <Investigation id="git-conflict" kicker="BASE + TWO CHANGES → A DECISION" title="Can a successful resolution still lose the intended meaning?">
    <p>One branch adds “by model”; the other adds “by region” to the same title. The intended combined report groups by both. Compare the three source versions before editing.</p><p className="lesson-note">Edit the conflicting content, stage the resolution and follow the remaining merge steps. Removing markers and recording a resolution are separate operations.</p>
    <div className="nt-controls"><label>Chosen resolution<select value={resolution} onChange={e => {
          setResolution(e.target.value);
          setStep(0);
        }}><option value="combined">Describe model and region</option><option value="ours">Keep only the model wording</option></select></label></div>
    <div className="nt-snapshots">{Object.entries(conflictVersions).map(([name, text]) => <div className="nt-snapshot" key={name}><h4>{name === 'base' ? 'Common base' : name === 'ours' ? 'Ours: main' : 'Theirs: feature'}</h4><pre>{text}</pre></div>)}</div>
    <p className="nt-transfer">{state.phase}</p><h4>Working title.txt</h4><pre className="nt-code">{state.working}</pre><p>Index: <strong>{state.unmerged ? 'unmerged stages (base, ours, theirs)' : step === 0 ? 'ordinary base snapshot' : 'ordinary resolved entry'}</strong> · merge committed: <strong>{state.committed ? 'yes' : 'no'}</strong></p><p className="nt-feedback" aria-live="polite">{state.note}</p><Stepper step={step} count={trace.length} setStep={setStep} /><p className="lesson-note">These “ours/theirs” labels describe a normal merge while on main. Rebase can change their intuitive meaning. This example models one textual conflict, not every merge driver or rename case.</p>
  </Investigation>;
}
