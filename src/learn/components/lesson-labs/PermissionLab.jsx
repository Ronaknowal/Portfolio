import { useState } from "react";
import { LessonTable } from "./LessonElements";
import { permissionModel } from "../../data/system-lesson-models";
import { initialPathPermissions, pathPermissionModel } from "../../data/linux-permission-model";
import "./linux-permission-lab.css";

const permissionLabels = ["Read", "Write", "Execute / search"];
const bitValues = [4, 2, 1];
const bitLetters = ["r", "w", "x"];
const gateLabels = { passed: "Passed", blocked: "Blocked here", "not-reached": "Not reached" };

function PermissionBits({ digit, label }) {
  return <span className="permission-lab__bits" role="img" aria-label={`${label}: ${permissionLabels.map((name, index) => `${name} ${digit & bitValues[index] ? "allowed" : "not allowed"}`).join(", ")}`}>
    {bitLetters.map((letter, index) => <span key={letter} className={digit & bitValues[index] ? "is-allowed" : "is-absent"} aria-hidden="true">{digit & bitValues[index] ? letter : "−"}</span>)}
  </span>;
}

function PermissionGate({ name, path, state, children }) {
  return <li className={`permission-lab__gate permission-lab__gate--${state}`} data-permission-gate={name} data-state={state}>
    <div className="permission-lab__gate-head">
      <span className="permission-lab__gate-mark" aria-hidden="true">{state === "passed" ? "✓" : state === "blocked" ? "×" : "·"}</span>
      <span>{gateLabels[state]}</span>
    </div>
    <code className="permission-lab__path">{path}</code>
    {children}
  </li>;
}

export default function PermissionLab() {
  const [permissions, setPermissions] = useState(initialPathPermissions);
  const [mode, setMode] = useState("640");
  const [subject, setSubject] = useState("owner");
  const [kind, setKind] = useState("file");
  const access = pathPermissionModel(permissions);
  const rows = permissionModel(mode, subject, kind === "directory");
  const changePermission = (name, checked) => setPermissions(current => ({ ...current, [name]: checked }));

  return <section className="lesson-lab permission-lab" aria-label="Linux permission explorer">
    <h3>Why can a readable file still refuse to open?</h3>
    <p>To read a file, Linux must first follow its path through directories. The directory decides whether you can look up the next name; the file decides whether you can read its contents.</p>

    <dl className="permission-lab__key" aria-label="What the three permission letters mean">
      <div><dt><code>r</code> Read</dt><dd><strong>File:</strong> read contents.<br /><strong>Directory:</strong> list entry names.</dd></div>
      <div><dt><code>w</code> Write</dt><dd><strong>File:</strong> change contents.<br /><strong>Directory:</strong> change entries, with search permission too.</dd></div>
      <div><dt><code>x</code> Execute / search</dt><dd><strong>File:</strong> allow an execution attempt.<br /><strong>Directory:</strong> look up a known name inside it.</dd></div>
    </dl>
    <p className="lesson-note">Directory <code>x</code> means search, not “run this folder.” A file's <code>x</code> bit still needs a runnable program or interpreter. A dash means the permission is absent.</p>
    <p className="permission-lab__identity"><strong>You are a member of the owning group, and are not the owner.</strong> For both <code>data</code> and <code>run.csv</code>, Linux therefore checks the group bits. It does not add owner, group and other permissions together.</p>
    <p><strong>Predict first:</strong> the file allows reading, but its directory does not allow search. Which gate will stop <code>cat /project/data/run.csv</code>?</p>

    <fieldset className="permission-lab__controls">
      <legend>Change the group's permissions</legend>
      <label><input type="checkbox" checked={permissions.directoryRead} onChange={event => changePermission("directoryRead", event.target.checked)} /><span><strong>Directory read · r</strong><small>List names in data</small></span></label>
      <label><input type="checkbox" checked={permissions.directorySearch} onChange={event => changePermission("directorySearch", event.target.checked)} /><span><strong>Directory search · x</strong><small>Look up run.csv inside data</small></span></label>
      <label><input type="checkbox" checked={permissions.fileRead} onChange={event => changePermission("fileRead", event.target.checked)} /><span><strong>File read · r</strong><small>Read run.csv contents</small></span></label>
    </fieldset>

    <figure className="permission-lab__journey">
      <figcaption><code>cat /project/data/run.csv</code><span>Follow the checks in order →</span></figcaption>
      <ol className="permission-lab__gates" aria-label="Checks for reading the known file path">
        <PermissionGate name="ancestors" path="/ → /project" state={access.gates.ancestors}>
          <p>Search both ancestors.<br /><strong>Allowed and fixed in this model.</strong></p>
        </PermissionGate>
        <PermissionGate name="directory" path="/project/data" state={access.gates.directory}>
          <div className="permission-lab__gate-bits"><span>Group bits</span><PermissionBits digit={access.directoryGroupDigit} label="Directory group permissions" /></div>
          <p>Needs <strong>x</strong> to find the entry <code>run.csv</code>. Its <code>r</code> bit is not needed for this known name.</p>
        </PermissionGate>
        <PermissionGate name="file" path="run.csv" state={access.gates.file}>
          <div className="permission-lab__gate-bits"><span>Group bits</span><PermissionBits digit={access.fileGroupDigit} label="File group permissions" /></div>
          <p>{access.fileReadChecked ? <>Now check the file's <strong>r</strong> bit to read its contents.</> : <>The directory search failed, so this file's read permission has not been checked.</>}</p>
        </PermissionGate>
      </ol>
    </figure>

    <div className="permission-lab__outcome" role="status" aria-live="polite" aria-atomic="true">
      <p><strong>{access.canReadFile ? "The file can be read." : access.blockedAt === "directory-search" ? "Stopped at /project/data: search is denied." : "Stopped at run.csv: reading is denied."}</strong> {access.blockedAt === "directory-search" ? "Changing only the file's read bit cannot repair this earlier gate." : access.blockedAt === "file-read" ? "The path is reachable; the file now needs its own read permission." : "Both the directory search check and the file read check passed."}</p>
      <div className="permission-lab__observations">
        <div><span className="permission-lab__observation-label">List only the names in data</span><output>{access.canListNames ? "run.csv" : "Names cannot be listed"}</output><p>{access.canListNames ? "Directory r allows the names to be listed. This alone does not allow opening them." : "Directory r is absent. Knowing a filename can still let you use it when directory x allows lookup."}</p></div>
        <div><span className="permission-lab__observation-label">Read the known file with cat</span><output className={access.canReadFile ? "is-readable" : ""}>{access.canReadFile ? "latency_ms\n10\n20" : "Permission denied"}</output><p>{access.canReadFile ? "These are invented file contents. Directory r is not part of this read check." : "The first blocked gate above explains why no contents are shown."}</p></div>
      </div>
    </div>

    <div className="permission-lab__try">
      <p><strong>Try the surprising case:</strong> enable directory search and file read, then disable directory read. Compare the two results. Next, deny file read while keeping search: where does the failure move?</p>
      <div className="lesson-controls"><button type="button" onClick={() => setPermissions(initialPathPermissions)}>Reset path permissions</button></div>
    </div>
    <p className="lesson-note">This is a browser model of an existing regular file and ordinary group permissions. <code>/</code> and <code>/project</code> remain searchable. Name listing means entries alone, not an <code>ls -l</code> metadata listing. No root/capabilities, ACLs, symlinks, special mode bits, security modules or mount restrictions are modeled. The controls do not change your computer's files.</p>

    <details className="permission-lab__advanced">
      <summary>Deeper: read an ordinary permission mode</summary>
      <p>Each object stores three groups of bits: owner, group, other. Select exactly one class for the current user on that object: owner first; otherwise matching group; otherwise other. A denied owner cannot fall back to group or other.</p>
      <p>Within each class, <code>r = 4</code>, <code>w = 2</code>, <code>x = 1</code>. Add the allowed values to get one octal digit: <code>rw− = 4 + 2 = 6</code>. These digits describe permission bits; the meaning of each bit still depends on whether the object is a file or directory.</p>
      <div className="lesson-controls">
        <label>Mode (octal)<select aria-label="Permission mode" value={mode} onChange={event => setMode(event.target.value)}>{["600", "640", "700", "750", "755", "047"].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
        <label>Applicable class<select aria-label="Permission class" value={subject} onChange={event => setSubject(event.target.value)}><option value="owner">Owner</option><option value="group">Matching group (not owner)</option><option value="other">Other user</option></select></label>
        <label>Object<select aria-label="Filesystem object" value={kind} onChange={event => setKind(event.target.value)}><option value="file">Regular file</option><option value="directory">Directory</option></select></label>
        <button type="button" onClick={() => { setMode("640"); setSubject("owner"); setKind("file"); }}>Reset permissions</button>
      </div>
      <div className="permission-lab__classes" aria-label="Permission bit groups and selected class">
        {["owner", "group", "other"].map((name, index) => <div key={name} className={subject === name ? "is-selected" : ""}><span>{name === "other" ? "Other" : name === "group" ? "Group" : "Owner"}</span><PermissionBits digit={Number(mode[index])} label={`${name} permissions`} /><span>Digit {mode[index]} · {subject === name ? "Selected" : "Not used"}</span></div>)}
      </div>
      <div className="lesson-results" aria-live="polite">
        <p>Mode {mode}, {subject} class, {kind}. “Yes” means these bits allow the operation under this model's assumptions. All parent paths are assumed searchable in this separate explorer.</p>
        <LessonTable caption="Operations allowed by these bits" headers={["Operation", "Allowed?", "Interpretation"]} rows={rows.map(([name, allowed, note]) => [name, allowed ? "Yes" : "No", note])} />
      </div>
      <p className="lesson-note">Test the class rule: choose <code>047</code> and compare owner, group and other. The owner has no access bits; the group can read but does not inherit the other class's write or execute bits. Creating/removing directory entries normally needs both write and search; sticky-bit rules can restrict removal further.</p>
    </details>
  </section>;
}
