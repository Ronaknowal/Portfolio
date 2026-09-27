import { colors, fonts } from "../../styles";
import { useCallback, useEffect, useRef, useState } from 'react';
import { useCodeDownload } from './LessonCodeDownloads.jsx';
import { downloadText } from './code-downloads.js';
import './lesson-code.css';

export function Code({ children }) {
  return (
    <code style={{
      fontFamily: fonts.mono,
      fontSize: 12,
      color: colors.gold,
      background: "rgba(226,181,90,0.08)",
      padding: "1px 5px",
      borderRadius: 3,
    }}>
      {children}
    </code>
  );
}

export function CodeBlock({ children, language, filename, downloadUrl, title, kind }) {
  const codeRef = useRef(null), textRef = useRef(children);
  textRef.current = children;
  const getText = useCallback(() => typeof textRef.current === 'string' ? textRef.current : codeRef.current?.textContent || '', []);
  const inPractice = useCallback(() => codeRef.current?.closest('details, [data-lesson-exercise], [data-lesson-ending="practice"], .lesson-check, .neural-practice'), []);
  const output = kind === 'output' || /^(?:output|text output|console output)$/i.test(language || '');
  const file = useCodeDownload({ filename, language, downloadUrl, title, kind: output ? 'output' : 'code', getText, inPractice });
  const [message, setMessage] = useState('');
  const copyTimer = useRef(null), copyRequest = useRef(0);
  useEffect(() => () => {
    copyRequest.current += 1;
    window.clearTimeout(copyTimer.current);
  }, []);
  async function copy() {
    const request = ++copyRequest.current;
    window.clearTimeout(copyTimer.current);
    setMessage('');
    try {
      await navigator.clipboard.writeText(getText());
      if (request !== copyRequest.current) return;
      setMessage('Copied');
      copyTimer.current = window.setTimeout(() => setMessage(''), 2200);
    } catch {
      if (request === copyRequest.current) setMessage('Copy unavailable. Select the code and copy it manually.');
    }
  }
  return <div className="lesson-code" data-code-kind={output ? 'output' : 'code'}>
    <div className="lesson-code__toolbar">
      <div className="lesson-code__identity"><span>{title || (output ? 'Output' : language || 'Code')}</span><span className="lesson-code__filename">{file}</span></div>
      <div className="lesson-code__actions"><button type="button" className="lesson-code__copy" data-copied={message === 'Copied' ? '' : undefined} onClick={copy} aria-label={`Copy ${file}`}>{message === 'Copied' ? <><span aria-hidden="true">✓</span> Copied</> : 'Copy'}</button>
        {downloadUrl ? <a href={downloadUrl} download={file} aria-label={`Download ${file}`}>Download</a>
          : <button type="button" onClick={() => downloadText(getText(), file)} aria-label={`Download ${file}`}>Download</button>}</div>
      <span className={`lesson-code__status${message === 'Copied' ? ' lesson-code__status--success' : ''}`} role="status" aria-live="polite">{message}</span>
    </div>
    <pre tabIndex={0} aria-label={`${title || language || (output ? 'Output' : 'Code')}; scroll horizontally when needed`}><code ref={codeRef}>{children}</code></pre>
  </div>;
}
