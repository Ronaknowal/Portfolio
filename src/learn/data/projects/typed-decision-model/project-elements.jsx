import { useState } from 'react';
import { Link } from 'react-router-dom';
import { topicMap } from '../../catalogue.js';

export const projectAssets = '/learn-projects/typed-decision-model';
const sourceRequests = new Map();

export function Commands({ children, label = 'Terminal commands' }) {
  return <pre className="tdp-code" tabIndex={0} aria-label={label}><code>{children}</code></pre>;
}

export function Source({ file = 'typed_decision.py', start, end, title }) {
  const [code, setCode] = useState('');
  const [state, setState] = useState('idle');
  const url = `${projectAssets}/${file}`;
  async function loadSource() {
    setState('loading');
    try {
      if (!sourceRequests.has(url)) {
        sourceRequests.set(url, fetch(url).then(response => {
          if (!response.ok) throw new Error('Source unavailable');
          return response.text();
        }).catch(error => { sourceRequests.delete(url); throw error; }));
      }
      const source = await sourceRequests.get(url);
      const first = start ? source.indexOf(start) : 0;
      const last = end ? source.indexOf(end, first + (start?.length || 0)) : source.length;
      if (first < 0 || last < first) throw new Error('Source section unavailable');
      setCode(source.slice(first, last).trim());
      setState('ready');
    } catch {
      setState('error');
    }
  }
  return <details className="tdp-source" onToggle={event => {
    if (event.currentTarget.open && state === 'idle') loadSource();
  }}>
    <summary>{title}</summary>
    {state === 'loading' && <p role="status">Loading the downloadable source…</p>}
    {state === 'error' && <p role="alert">The source could not be loaded. <button type="button" onClick={loadSource}>Try again</button></p>}
    {state === 'ready' && <Commands label={title}>{code}</Commands>}
    <a href={url} download>Download {file}</a>
  </details>;
}

export function ConceptLinks({ items }) {
  return <aside className="tdp-concept-links" aria-label="Understand the mechanisms in this stage">
    <span className="tdp-eyebrow">Go deeper at this step</span>
    <ul>{items.map(({ id, reason }) => {
      const topic = topicMap[id];
      if (!topic) throw new Error(`Unknown project concept: ${id}`);
      return <li key={id}><Link to={`/learn/topic/${id}`}>{topic.title}</Link>
        {topic.status !== 'published' && <span className="tdp-small"> · planned outline</span>}
        <p>{reason}</p>
      </li>;
    })}</ul>
  </aside>;
}

export function Practice({ title, children }) {
  return <details className="tdp-practice"><summary>{title}</summary>{children}</details>;
}
