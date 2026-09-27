import { CodeBlock } from "../content";

export default function TerminalExample({ example, children }) {
  return <div className="terminal-example">
    <CodeBlock language={example.language || "bash"} filename={example.filename || example.file}>{example.code}</CodeBlock>
    <p className="lesson-note">Expected standard output. Successful commands may print nothing; incidental diagnostic messages may vary.</p>
    <div className="terminal-example__output"><CodeBlock language="output" filename={example.filename || example.file ? `${(example.filename || example.file).replace(/\.[^/.]+$/, '')}-output.txt` : undefined}>{example.output}</CodeBlock></div>
    {children}
  </div>;
}
