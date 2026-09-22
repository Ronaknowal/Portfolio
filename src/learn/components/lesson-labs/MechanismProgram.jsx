import { useEffect, useState } from 'react';
import { fonts } from '../../styles';
import './mechanism-program.css';

function ProgramView({ source, title, output, language = 'Python' }) {
  const [open, setOpen] = useState(false);
  const [code, setCode] = useState(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || code !== null) return;
    const controller = new AbortController();
    setFailed(false);
    fetch(source, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Program unavailable');
        return response.text();
      })
      .then(text => { if (!controller.signal.aborted) setCode(text); })
      .catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, [source, open, code, attempt]);
  return <div className="mechanism-program">
    <p><a href={source} download>Download {source.split('/').at(-1)}</a></p>
    <details onToggle={event => setOpen(event.currentTarget.open)}>
      <summary>{title}</summary>
      {open && <>
        {code !== null ? <pre tabIndex={0} role="region" aria-label={`Complete ${language} program; scroll horizontally when needed`} style={{ fontFamily: fonts.mono }}><code>{code}</code></pre> : failed ? <p role="alert">The code view could not load. <button onClick={() => setAttempt(value => value + 1)}>Retry code view</button></p> : <p role="status">Loading complete program…</p>}
        <p>Recorded output from the tested CPU environment</p>
        <pre tabIndex={0} role="region" aria-label="Recorded program output; scroll horizontally when needed" style={{ fontFamily: fonts.mono }}><code>{output}</code></pre>
      </>}
    </details>
  </div>;
}

export default function MechanismProgram(props) {
  return <ProgramView key={props.source} {...props} />;
}
