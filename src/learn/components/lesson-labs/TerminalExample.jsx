import { CodeBlock } from "../content";

export default function TerminalExample({ example, children }) {
  return <div className="terminal-example">
    <CodeBlock language={example.language || "bash"}>{example.code}</CodeBlock>
    <p className="lesson-note">Expected standard output. Successful commands may print nothing; incidental diagnostic messages may vary.</p>
    <div className="terminal-example__output"><CodeBlock language="output">{example.output}</CodeBlock></div>
    {children}
  </div>;
}
