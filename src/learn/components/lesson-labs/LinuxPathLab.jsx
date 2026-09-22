import { useId, useMemo, useState } from "react";
import { linuxPathEntries, linuxPathPresets, resolveLinuxPath } from "../../data/linux-path-model";
import "./linux-path-lab.css";

function DirectoryBranch({ path, current, shell, visited }) {
  const entry = linuxPathEntries.find(item => item.path === path);
  const children = linuxPathEntries.filter(item => item.parent === path);
  return <li className="linux-path-branch">
    <div className={`linux-path-node${current === path ? " is-current" : ""}${visited.has(path) ? " is-visited" : ""}`}>
      <span className={`linux-path-icon is-${entry.kind}`} aria-hidden="true" />
      <span className="linux-path-node-name">{entry.name}{entry.kind === "directory" && path !== "/" ? "/" : ""}</span>
      <span className="linux-path-node-tags">
        {shell === path && <span className="linux-path-shell-tag">shell here</span>}
        {current === path && <span className="linux-path-lookup-tag">← lookup</span>}
      </span>
      <span className="linux-path-sr-only">{entry.kind}</span>
    </div>
    {children.length > 0 && <ul>{children.map(child => <DirectoryBranch key={child.path} path={child.path} current={current} shell={shell} visited={visited} />)}</ul>}
  </li>;
}

export default function LinuxPathLab() {
  const uid = useId();
  const [choice, setChoice] = useState(linuxPathPresets[0]);
  const [cursor, setCursor] = useState(null);
  const resolution = useMemo(() => resolveLinuxPath(choice), [choice]);
  const activeStep = cursor === null ? resolution.steps.length - 1 : Math.min(cursor, resolution.steps.length - 1);
  const step = resolution.steps[activeStep];
  const finished = activeStep === resolution.steps.length - 1;
  const visited = new Set(resolution.steps.slice(0, activeStep + 1).map(item => item.location));
  const update = patch => { setChoice(previous => ({ ...previous, ...patch, id: "custom" })); setCursor(null); };
  const applyPreset = id => { setChoice(linuxPathPresets.find(item => item.id === id)); setCursor(null); };
  const quotedPath = `'${choice.path.replaceAll("'", "'\\''")}'`;

  return <section className="lesson-lab linux-path-lab" data-live-exploration aria-labelledby={`${uid}-title`}>
    <div className="lesson-eyebrow">PATH LAB · FOLLOW EACH BRANCH</div>
    <h3 id={`${uid}-title`}>Where does this path actually go?</h3>
    <p>The first investigation starts in <code>/project/data/raw</code>, with your report folder on another branch. Change the path, starting directory or operation and see the destination update immediately. Use the step controls to follow each path component through the tree.</p>

    <div className="linux-path-setup">
      <label>Investigation<select aria-label="Path investigation" value={choice.id} onChange={event => applyPreset(event.target.value)}>
        {choice.id === "custom" && <option value="custom">Your variation</option>}
        {linuxPathPresets.map(preset => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
      </select></label>
      <label>Starting directory<select aria-label="Starting directory" value={choice.cwd} onChange={event => update({ cwd: event.target.value })}>
        {linuxPathEntries.filter(entry => entry.kind === "directory").map(entry => <option key={entry.path}>{entry.path}</option>)}
      </select></label>
      <label className="linux-path-input-label">Path value<input aria-label="Path value" value={choice.path} maxLength={160} onChange={event => update({ path: event.target.value })} spellCheck="false" autoComplete="off" aria-describedby={`${uid}-input-note`} /></label>
      <label>Operation<select aria-label="Path operation" value={choice.operation} onChange={event => update({ operation: event.target.value })}>
        <option value="cd">cd: enter a directory</option><option value="locate">Locate a file or directory</option>
      </select></label>
    </div>
    <p className="lesson-note" id={`${uid}-input-note`}>Enter a path only, without shell quotes. Spaces stay inside this one argument; commands and shell expansions are not executed.</p>
    <div className="linux-path-command"><span>{choice.operation === "cd" ? "Command being modelled" : "One pathname to look up"}</span><code>{choice.operation === "cd" ? `cd -- ${quotedPath}` : quotedPath}</code></div>

    <div className="linux-path-workspace">
      <figure className="linux-path-tree">
        <figcaption>The fixed filesystem tree</figcaption>
        <ul className="linux-path-root"><DirectoryBranch path="/" current={step?.location} shell={step?.cwd || choice.cwd} visited={visited} /></ul>
        <p className="lesson-note">Indented entries live inside the directory above. Amber follows lookup; the green “shell here” label marks the working directory.</p>
      </figure>

      <div className="linux-path-walk">
        <p className="linux-path-kind">{resolution.absolute ? "Absolute path · start at /" : "Relative path · start where the shell is"}</p>
        <ol className="linux-path-segments" aria-label="Path components">
          {resolution.absolute && <li className={activeStep === 0 ? "is-active" : ""}><code>/</code><span className="linux-path-sr-only">root start</span></li>}
          {resolution.segments.map((segment, index) => <li key={`${index}-${segment}`} className={step?.segmentIndex === index ? "is-active" : step && step.segmentIndex > index ? "is-done" : ""} aria-current={step?.segmentIndex === index ? "step" : undefined}><code>{segment}</code></li>)}
          {!choice.path && <li>empty path</li>}
        </ol>
        <div className={`linux-path-step${step?.kind === "error" ? " is-error" : ""}`} role="status" aria-live="polite" aria-atomic="true">
          <span className="lesson-eyebrow">{step ? `STEP ${activeStep + 1} OF ${resolution.steps.length}` : "PATH RESULT"}</span>
          <h4>{step?.title || "Follow the route"}</h4>
          <p>{step?.explanation || "Which entry will be reached? Does the shell move while the path is being checked, or only after cd succeeds?"}</p>
          {step && <p className="linux-path-location">Lookup is at <code>{step.location}</code></p>}
        </div>
        <div className="linux-path-actions">
          <button type="button" onClick={() => setCursor(0)}>Walk from the start</button><button type="button" disabled={activeStep === 0} onClick={() => setCursor(activeStep - 1)}>Back</button>
          <button type="button" className="linux-path-next" disabled={finished} onClick={() => setCursor(activeStep + 1)}>{finished ? "Walk complete" : "Next step"}</button>
          <button type="button" onClick={() => applyPreset(linuxPathPresets[0].id)}>Reset path lab</button>
        </div>
      </div>
    </div>

    <details className="linux-path-transfer"><summary>Try a variation: does an absolute path care where you start?</summary>
      <p>Choose “Start from the root”, then change the starting directory to <code>/project/reports</code>. Watch the destination, then remove the first slash and compare.</p>
      <details><summary>Hint</summary><p>Choose the starting point before following the first name. Without the slash, which directory would need to contain an entry called <code>project</code>?</p></details>
      <details><summary>Explain the result</summary><p><code>/project/reports</code> starts at <code>/</code> and reaches the same directory from either starting point. The relative value <code>project/reports</code> searches inside your starting directory first; the pictured tree has no nested <code>project</code> entry there, so it fails.</p></details>
    </details>
    <p className="lesson-note">This is an in-browser pathname model. The tree is fixed, all directory search permissions are granted, and there are no symlinks, mounts, CDPATH, or logical/physical path differences. Repeated slashes are treated as Linux separators; <code>..</code> at <code>/</code> stays at <code>/</code>. The next permission activity adds access checks.</p>
  </section>;
}
