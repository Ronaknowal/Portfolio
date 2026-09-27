const extensions = { python: 'py', py: 'py', javascript: 'js', js: 'js', typescript: 'ts', ts: 'ts', jsx: 'jsx', tsx: 'tsx', bash: 'sh', shell: 'sh', sh: 'sh', sql: 'sql', json: 'json', yaml: 'yaml', yml: 'yml', c: 'c', cpp: 'cpp', 'c++': 'cpp', cuda: 'cu', rust: 'rs', go: 'go', java: 'java', r: 'R', julia: 'jl', matlab: 'm', html: 'html', css: 'css', markdown: 'md' };

export function codeFilename({ filename, topicId = 'lesson', language, kind = 'code', ordinal = 1 }) {
  if (filename) return String(filename).split(/[\\/]/).at(-1).replace(/[<>:"|?*\u0000-\u001f]/g, '-');
  const extension = kind === 'output' ? 'txt' : extensions[String(language || '').toLowerCase()] || 'txt';
  return `${topicId}-${kind === 'output' ? 'output' : 'example'}-${String(ordinal).padStart(2, '0')}.${extension}`;
}

export function downloadText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Keep the URL alive long enough for the browser to start the download.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function localCodeAsset(href, base) {
  try {
    const url = new URL(href, base);
    if (url.origin !== new URL(base).origin || !/^\/(?:learn-code|learn-assets|learn\/examples)\//.test(url.pathname)) return null;
    if (!/\.(?:py|ipynb|js|mjs|cjs|ts|jsx|tsx|sh|bash|sql|r|jl|m|c|h|cpp|hpp|cu|rs|go|java|json|csv|tsv|txt|md|yaml|yml|zip|npz|npy)$/i.test(url.pathname)) return null;
    return url.pathname + url.search;
  } catch { return null; }
}
