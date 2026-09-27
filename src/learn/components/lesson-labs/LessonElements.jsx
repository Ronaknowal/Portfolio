import "./lessons.css";
import { LessonOrientation } from "../LessonOpening.jsx";
import { H2 } from "../content/Headings.jsx";

export function LessonIntro({ children, prerequisites, exampleKind = "Python" }) {
  return <LessonOrientation prerequisites={prerequisites} exampleKind={exampleKind}>{children}</LessonOrientation>;
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
  return <>
    {alternatives && <section className="lesson-ending lesson-ending--resources" data-lesson-ending="further-learning">
      <H2>Further learning</H2>
      <div className="lesson-resource-list">{alternatives}</div>
    </section>}
    <section className="lesson-ending lesson-ending--resources" data-lesson-ending="references">
      <H2>Technical references</H2>
      <div className="lesson-resource-list"><ul>{children}</ul></div>
      <p className="lesson-note">These are optional deeper references. The explanation, labs, code and exercises above are self-contained.</p>
    </section>
  </>;
}
