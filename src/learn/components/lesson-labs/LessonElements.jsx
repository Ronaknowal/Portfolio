import "./lessons.css";

export function LessonIntro({ children, prerequisites, sections, exampleKind = "Python" }) {
  return <aside className="lesson-intro" aria-label="Lesson route">
    <p className="lesson-eyebrow">UNDERSTAND · EXPLORE · PRACTISE</p>
    <p>{children}</p>
    <p><strong>Before you start:</strong> {prerequisites}</p>
    <nav aria-label="In this lesson"><ol>{sections.map(([id, title]) => <li key={id}><a href={`#${id}`}>{title}</a></li>)}</ol></nav>
    <p className="lesson-note">First time? Follow the route in order. Revising? Jump to the worked example or practise section. The labs run here; {exampleKind} examples run in your own environment.</p>
  </aside>;
}

export function Checkpoint({ prompt, children }) {
  return <div className="lesson-check"><p><strong>Try it first.</strong> {prompt}</p>
    <details><summary>Show explanation</summary><div>{children}</div></details>
  </div>;
}

export function LessonTable({ caption, headers, rows }) {
  return <div className="lesson-table-wrap" tabIndex={0} role="region" aria-label={caption}>
    <table><caption>{caption}</caption><thead><tr>{headers.map(h => <th scope="col" key={h}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{row.map((v, j) => j === 0 ? <th scope="row" key={j}>{v}</th> : <td key={j}>{v}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export function Sources({ children, alternatives }) {
  return <aside className="lesson-sources"><h3>References & another way to learn it</h3>{alternatives}{alternatives && <h4>Technical references</h4>}<ul>{children}</ul>
    <p className="lesson-note">These are optional deeper references. The explanation, labs, code and exercises above are self-contained.</p></aside>;
}
