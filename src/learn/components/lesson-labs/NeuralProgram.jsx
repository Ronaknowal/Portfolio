import { CodeBlock } from '../content/Code.jsx';
import useLessonViewport from './useLessonViewport.js';
import { useEffect, useState } from 'react';


const loaders = {
  loss: () => import('../../data/loss-functions-program.js'),
  normalization: () => import('../../data/normalization-program.js'),
  'loss-mechanisms': () => import('../../data/loss-mechanisms-program.js'),
  'normalization-backward': () => import('../../data/normalization-backward-program.js'),
};

export default function NeuralProgram({ topic, title = 'Read the complete CPU program' }) {
  const [container, ready] = useLessonViewport();
  const [code, setCode] = useState(null), [failed, setFailed] = useState(false), [attempt, setAttempt] = useState(0);
  const filenames = { loss: 'loss-experiments.py', normalization: 'normalization-experiments.py', 'loss-mechanisms': 'loss-mechanisms.py', 'normalization-backward': 'normalization-backward.py' };
  useEffect(() => {
    if (!ready) return undefined;
    let current = true;
    setCode(null); setFailed(false);
    loaders[topic]().then(module => { if (current) setCode(module.default); })
      .catch(() => { if (current) setFailed(true); });
    return () => { current = false; };
  }, [topic, ready, attempt]);
  return <section ref={container} className="lesson-teaching-section" data-lesson-teaching="code">
    <h4 className="lesson-teaching-section__title">{title}</h4>
    {code ? <CodeBlock language="python" filename={filenames[topic]}>{code}</CodeBlock>
      : failed ? <p role="alert">The code view could not load. The program download remains available. <button onClick={() => setAttempt(value => value + 1)}>Retry code view</button></p>
        : <p role="status">Loading complete program…</p>}
  </section>;
}
