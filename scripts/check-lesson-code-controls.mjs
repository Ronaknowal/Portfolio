import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { build } from 'esbuild';
import { codeFilename, localCodeAsset } from '../src/learn/components/content/code-downloads.js';

// Execute the actual component event handlers in a small host harness. Browser
// layout and native clipboard/download integration have separate observations.
const base = 'https://example.test/learn/path/full-curriculum/example';
assert.equal(codeFilename({ filename: 'folder/matrix.py' }), 'matrix.py');
assert.equal(codeFilename({ filename: 'C:\\work\\matrix.py' }), 'matrix.py');
assert.equal(codeFilename({ topicId: 'arrays', language: 'python', ordinal: 3 }), 'arrays-example-03.py');
assert.equal(codeFilename({ topicId: 'arrays', language: 'python', kind: 'output' }), 'arrays-output-01.txt');
assert.equal(codeFilename({ topicId: 'arrays', language: 'unrecognized' }), 'arrays-example-01.txt');
for (const prefix of ['learn-code', 'learn-assets', 'learn/examples']) {
  assert.equal(localCodeAsset(`/${prefix}/test/code.py`, base), `/${prefix}/test/code.py`);
}
for (const url of ['https://elsewhere.test/learn-code/test.py', '/private/test.py', '/learn-code/test.png', 'javascript:alert(1)']) assert.equal(localCodeAsset(url, base), null);

const bundled = await build({ entryPoints: ['src/learn/components/content/Code.jsx'], bundle: true, write: false, format: 'cjs', jsx: 'automatic', plugins: [{ name: 'component-host', setup(plugin) {
  plugin.onResolve({ filter: /^(react|react\/jsx-runtime)$|LessonCodeDownloads\.jsx$|styles$/ }, args => ({ path: args.path, namespace: 'host' }));
  plugin.onLoad({ filter: /.*/, namespace: 'host' }, args => ({ contents: args.path === 'react'
    ? 'export const useCallback=f=>f; export const useRef=v=>({current:v}); export const useState=v=>[v,x=>globalThis.messages.push(x)]; export const useEffect=f=>globalThis.cleanups.push(f());'
    : args.path === 'react/jsx-runtime' ? 'export const jsx=(type,props)=>({type,props}); export const jsxs=jsx; export const Fragment="fragment";'
      : args.path.endsWith('styles') ? 'export const colors={gold:"amber"}; export const fonts={mono:"monospace"};'
        : 'export const useCodeDownload=entry=>{globalThis.registration=entry;return entry.filename||"example.py";};' }));
  plugin.onLoad({ filter: /\.css$/ }, () => ({ contents: '', loader: 'js' }));
} }] });
let written, savedBlob, clicked = 0, revoked = 0, removed = 0, rejectCopy = false;
const anchors = [];
const timers = new Map(); let nextTimer = 0;
const host = {
  exports: {}, messages: [], cleanups: [], Blob,
  navigator: { clipboard: { writeText: async value => { if (rejectCopy) throw new Error('Permission denied'); written = value; } } },
  URL: { createObjectURL: value => { savedBlob = value; return 'blob:test'; }, revokeObjectURL: () => revoked++ },
  document: { body: { appendChild: () => {} }, createElement: () => { const a = { click: () => clicked++, remove: () => removed++ }; anchors.push(a); return a; } },
  window: { setTimeout: (callback, delay) => { const id = ++nextTimer; timers.set(id, {callback, delay}); return id; }, clearTimeout: id => timers.delete(id) },
};
host.module = { exports: host.exports };
vm.runInNewContext(bundled.outputFiles[0].text, host);
host.exports = host.module.exports;
function descendants(node) { return node && typeof node === 'object' ? [node, ...[node.props?.children].flat(Infinity).flatMap(descendants)] : []; }
const exact = '\n    x = "λ"\r\n\tprint(x)\n\n';
let nodes = descendants(host.exports.CodeBlock({ children: exact, filename: 'example.py', language: 'python' }));
await nodes.find(n => n.props?.['aria-label'] === 'Copy example.py').props.onClick();
assert.equal(written, exact, 'Copy must preserve whitespace, Unicode and line endings');
assert.equal(host.messages.at(-1), 'Copied');
await nodes.find(n => n.props?.['aria-label'] === 'Copy example.py').props.onClick();
assert.equal(timers.size, 1, 'Repeated copy resets the feedback timeout');
const feedbackTimer = [...timers.entries()].find(([, timer]) => timer.delay === 2200);
feedbackTimer[1].callback(); timers.delete(feedbackTimer[0]);
assert.equal(host.messages.at(-1), '', 'Success feedback returns to Copy');
nodes.find(n => n.props?.['aria-label'] === 'Download example.py').props.onClick();
assert.equal(await savedBlob.text(), exact, 'Downloaded snippet must equal displayed source');
assert.equal(anchors.at(-1).download, 'example.py');
for (const [id, timer] of timers) { timer.callback(); timers.delete(id); }
assert.equal(clicked, 1); assert.equal(removed, 1); assert.equal(revoked, 1);
rejectCopy = true;
await nodes.find(n => n.props?.['aria-label'] === 'Copy example.py').props.onClick();
assert.match(host.messages.at(-1), /Copy unavailable/);
rejectCopy = false;
await nodes.find(n => n.props?.['aria-label'] === 'Copy example.py').props.onClick();
host.cleanups[0]();
assert.equal(timers.size, 0, 'Unmount clears the feedback timer');
nodes = descendants(host.exports.CodeBlock({ children: exact, filename: 'original.py', downloadUrl: '/learn-code/topic/original.py' }));
const canonical = nodes.find(n => n.props?.['aria-label'] === 'Download original.py');
assert.equal(canonical.type, 'a'); assert.equal(canonical.props.href, '/learn-code/topic/original.py'); assert.equal(canonical.props.download, 'original.py');
host.exports.CodeBlock({ children: '1\n', language: 'output' });
assert.equal(host.registration.kind, 'output');
const result = { passed: true, scope: 'Filename/asset routing and actual CodeBlock copy/download handlers; host stub does not certify native browser clipboard or file-save behavior', exactWhitespaceUnicodeLineEndings: true, canonicalFilenameAndUrl: true, clipboardFailure: true, outputMetadata: true, feedbackResets: true, repeatedClickRestartsTimer: true, unmountClearsTimer: true };
if (process.argv.includes('--record')) fs.writeFileSync('docs/teaching/lesson-code-access/code-control-checks.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
