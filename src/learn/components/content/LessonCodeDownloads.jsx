import { createContext, useCallback, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import { codeFilename, downloadText, localCodeAsset } from './code-downloads.js';

const RegistrationContext = createContext(null);
const EntriesContext = createContext([]);

export function LessonCodeProvider({ topicId, children }) {
  const [entries, setEntries] = useState(new Map());
  const ordinals = useRef(new Map());
  const register = useCallback((id, entry) => {
    if (!ordinals.current.has(id)) ordinals.current.set(id, ordinals.current.size + 1);
    const filename = codeFilename({ ...entry, topicId, ordinal: ordinals.current.get(id) });
    setEntries(previous => new Map(previous).set(id, { ...entry, id, filename }));
    return filename;
  }, [topicId]);
  const unregister = useCallback(id => setEntries(previous => {
    if (!previous.has(id)) return previous;
    const next = new Map(previous); next.delete(id); return next;
  }), []);
  const registration = useMemo(() => ({ topicId, register, unregister }), [topicId, register, unregister]);
  const list = useMemo(() => [...entries.values()], [entries]);
  return <RegistrationContext.Provider value={registration}><EntriesContext.Provider value={list}>{children}</EntriesContext.Provider></RegistrationContext.Provider>;
}

export function useCodeDownload({ filename, language, kind = 'code', downloadUrl, title, getText, inPractice }) {
  const context = useContext(RegistrationContext), id = useId();
  const [registeredFilename, setRegisteredFilename] = useState(null);
  useEffect(() => {
    if (!context) return;
    setRegisteredFilename(context.register(id, { filename, language, kind, downloadUrl, title, getText, practice: Boolean(inPractice?.()) }));
    return () => context.unregister(id);
  }, [context, id, filename, language, kind, downloadUrl, title, getText, inPractice]);
  return filename ? codeFilename({ filename }) : registeredFilename || codeFilename({ language, kind, topicId: context?.topicId });
}

/** One active lesson only. Existing canonical asset links are indexed without
 * fetching programs or loading other lessons. Inline examples register directly. */
export function LessonCodeDownloads({ articleRef, content }) {
  const entries = useContext(EntriesContext);
  const [assets, setAssets] = useState([]);
  useEffect(() => {
    const article = articleRef.current;
    if (!article || !content) return;
    const found = new Map();
    for (const anchor of article.querySelectorAll('a[href]')) {
      if (anchor.closest('[data-code-downloads], .lesson-code')) continue;
      const downloadUrl = localCodeAsset(anchor.getAttribute('href'), window.location.href);
      if (!downloadUrl || found.has(downloadUrl)) continue;
      const filename = decodeURIComponent(downloadUrl.split('?')[0].split('/').at(-1));
      found.set(downloadUrl, { downloadUrl, filename, title: anchor.textContent.trim(), kind: 'asset', practice: Boolean(anchor.closest('details, [data-lesson-exercise], [data-lesson-ending="practice"], .lesson-check, .neural-practice')) });
    }
    setAssets([...found.values()]);
  }, [articleRef, content]);
  const files = useMemo(() => {
    const seen = new Set();
    return [...assets, ...entries.filter(entry => entry.kind !== 'output')].filter(entry => {
      const key = entry.downloadUrl || entry.id;
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
  }, [assets, entries]);
  if (!files.length) return null;
  const renderFiles = items => <ul>{items.map(entry => <li key={entry.downloadUrl || entry.id}>
    <div><span className="lesson-code-downloads__filename">{entry.filename}</span>
      <span className="lesson-code-downloads__description">{entry.title && entry.title !== entry.filename ? entry.title : entry.downloadUrl ? 'Lesson file' : `${entry.language || 'Text'} example`}</span></div>
    {entry.downloadUrl ? <a href={entry.downloadUrl} download={entry.filename} aria-label={`Download ${entry.filename}`}>Download</a>
      : <button type="button" onClick={() => downloadText(entry.getText(), entry.filename)} aria-label={`Download ${entry.filename}`}>Download</button>}
  </li>)}</ul>;
  const practice = files.filter(entry => entry.practice);
  return <aside className="lesson-code-downloads" data-code-downloads="" aria-labelledby="lesson-code-downloads-title">
    <h2 id="lesson-code-downloads-title">Code &amp; supporting files</h2>
    <p>Download a complete program or save an example to work with locally. Example snippets keep the setup and dependencies explained beside them; saving a snippet does not make it a standalone program.</p>
    {renderFiles(files.filter(entry => !entry.practice))}
    {practice.length > 0 && <details><summary>Practice and optional files · {practice.length}</summary>{renderFiles(practice)}</details>}
  </aside>;
}
