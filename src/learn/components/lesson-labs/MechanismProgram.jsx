import { CodeBlock } from '../content/Code.jsx';
import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import './mechanism-program.css';

export default function MechanismProgram({ source, title, output, language = 'Python' }) {
  return <section className="mechanism-program">
    <RemoteCodeBlock source={source} title={title} language={language} />
    <p>Recorded output from the tested CPU environment</p>
    <CodeBlock language="output" filename={`${source.split('/').at(-1).replace(/\.[^.]+$/, '')}-output.txt`}>{output}</CodeBlock>
  </section>;
}
