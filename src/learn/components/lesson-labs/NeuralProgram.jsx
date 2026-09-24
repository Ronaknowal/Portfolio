import { useEffect, useState } from 'react';
import { fonts } from '../../styles';

const loaders = {
  loss: () => import('../../data/loss-functions-program.js'),
  normalization: () => import('../../data/normalization-program.js'),
  'loss-mechanisms': () => import('../../data/loss-mechanisms-program.js'),
  'normalization-backward': () => import('../../data/normalization-backward-program.js'),
};

export default function NeuralProgram({ topic, title = 'Read the complete CPU program' }) {
  const [open, setOpen] = useState(false);
  const [code, setCode] = useState(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || code) return;
    let current = true;
    setFailed(false);
    loaders[topic]().then(module => { if (current) setCode(module.default); })
      .catch(() => { if (current) setFailed(true); });
    return () => { current = false; };
  }, [topic, open, code, attempt]);
  return <details className="neural-program" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>{title}</summary>
    {open && (code ? <pre className="neural-program-source" tabIndex={0} role="region" aria-label="Complete Python program; scroll horizontally when needed" style={{ fontFamily: fonts.mono }}><code>{code}</code></pre> : failed ? <p role="alert">The code view could not load. The program download remains available. <button onClick={() => setAttempt(value => value + 1)}>Retry code view</button></p> : <p role="status">Loading complete program…</p>)}
  </details>;
}
