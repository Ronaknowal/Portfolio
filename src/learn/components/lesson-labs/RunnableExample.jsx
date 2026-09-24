import {CodeBlock,H3} from '../content';
export function RunnableExample({example,children}) {
  return <div className="python-example"><H3>{example.title}</H3>{example.file&&<p className="lesson-note">Save as <code>{example.file}</code></p>}<CodeBlock language={example.language||'python'}>{example.code}</CodeBlock><p className="lesson-note">Expected result</p><CodeBlock language="output">{example.expected}</CodeBlock>{children}</div>;
}
