import { useCallback, useEffect, useRef, useState } from 'react';
import { CodeBlock } from './Code.jsx';
import { useCodeDownload } from './LessonCodeDownloads.jsx';

function RemoteCode({ source, filename, language = 'python', title }) {
  const container = useRef(null), [near, setNear] = useState(false);
  const [code, setCode] = useState(null), [error, setError] = useState(false), [attempt, setAttempt] = useState(0);
  const file = filename || source.split('/').at(-1);
  const getText = useCallback(() => code || '', [code]);
  const inPractice = useCallback(() => container.current?.closest('details, [data-lesson-exercise], [data-lesson-ending="practice"], .lesson-check, .neural-practice'), []);
  useCodeDownload({ filename: file, language, downloadUrl: source, title, getText, inPractice });
  useEffect(() => {
    if (!('IntersectionObserver' in window)) { setNear(true); return; }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setNear(true); observer.disconnect(); }
    }, { rootMargin: '600px' });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!near) return;
    const controller = new AbortController();
    setError(false);
    fetch(source, { signal: controller.signal }).then(response => {
      if (!response.ok || /text\/html/i.test(response.headers.get('content-type') || '')) throw new Error('Code unavailable');
      return response.text();
    }).then(text => { if (!controller.signal.aborted) setCode(text); })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, [source, near, attempt]);
  return <div ref={container} className="lesson-code-source">
    {code !== null ? <CodeBlock filename={file} language={language} title={title} downloadUrl={source}>{code}</CodeBlock>
      : <div className="lesson-code-source__pending"><p>{title || file}</p>
        {error ? <p role="alert">The program could not load. <button type="button" onClick={() => setAttempt(value => value + 1)}>Retry</button></p>
          : <p role="status">{near ? 'Loading program…' : 'The program loads here as you reach this part of the lesson.'}</p>}
        <a href={source} download={file}>Download {file}</a>
      </div>}
  </div>;
}
export default function RemoteCodeBlock(props) { return <RemoteCode key={props.source} {...props} />; }
