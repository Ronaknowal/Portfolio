import { CodeBlock } from "../content";

export default function PythonExample({ example, children }) {
  return <div className="python-example">
    {Object.entries(example.files || {}).map(([filename, code]) => <div key={filename}>
      <p className="lesson-note">Save as <strong>{filename}</strong></p>
      <CodeBlock language="python">{code.trim()}</CodeBlock>
    </div>)}
    {example.filename && <p className="lesson-note">Save as <strong>{example.filename}</strong> in the same folder. Run <strong>python {example.filename}</strong>.</p>}
    <CodeBlock language="python">{example.code}</CodeBlock>
    <p className="lesson-note">Expected output</p>
    <div className="python-example__output"><CodeBlock language="output">{example.output}</CodeBlock></div>
    {children && <div className="python-example__explanation">{children}</div>}
  </div>;
}
