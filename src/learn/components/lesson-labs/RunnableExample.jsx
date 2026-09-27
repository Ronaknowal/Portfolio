import {CodeBlock,H3} from '../content';
export function RunnableExample({example,children}) {
  return <div className="python-example"><H3>{example.title}</H3>{example.file&&<p className="lesson-note">Save as <code>{example.file}</code></p>}<CodeBlock language={example.language||'python'} filename={example.file} title={example.title}>{example.code}</CodeBlock><p className="lesson-note">Expected result</p><CodeBlock language="output" filename={example.file ? `${example.file.replace(/\.[^.]+$/, '')}-output.txt` : undefined}>{example.expected}</CodeBlock>{children}</div>;
}
