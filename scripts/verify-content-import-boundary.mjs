import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parse } from '@babel/parser';

const root = path.resolve('src');
const barrel = path.resolve('src/learn/components/content/index.js');
const isMathModule = specifier => /(?:^|\/)Math(?:\.[jt]sx?)?$/.test(specifier) || /(?:^|\/)katex(?:\/|$)/i.test(specifier);
const isContentBarrel = specifier => /(?:^|\/)content(?:\/index(?:\.[jt]s)?)?$/.test(specifier);
const barrelAst = parse(fs.readFileSync(barrel, 'utf8'), { sourceType: 'module' });
for (const statement of barrelAst.program.body) {
  assert.ok(!statement.source || !isMathModule(statement.source.value), 'Generic content barrel must not import or re-export the math renderer or KaTeX');
  if (statement.type === 'ExportNamedDeclaration') for (const specifier of statement.specifiers) {
    assert.ok(!['Math', 'MathBlock'].includes(specifier.exported?.name), 'Math exports must stay outside the generic content barrel');
  }
}

// Follow static local dependencies too: moving a Math import into Prose (or
// another barrel member) would otherwise restore the same eager payload.
const lightweightClosure = new Set();
function checkLightweightClosure(filename) {
  if (lightweightClosure.has(filename)) return;
  lightweightClosure.add(filename);
  const ast = parse(fs.readFileSync(filename, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  for (const statement of ast.program.body) {
    if (!statement.source) continue;
    const specifier = statement.source.value;
    assert.ok(!isMathModule(specifier), `${filename}: a generic content dependency must not statically pull in the math renderer or KaTeX`);
    if (!specifier.startsWith('.')) continue;
    const base = path.resolve(path.dirname(filename), specifier);
    const target = [base, base + '.js', base + '.jsx', path.join(base, 'index.js')].find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
    assert.ok(target, 'Resolve static content dependency: ' + specifier);
    if (/\.[jt]sx?$/.test(target)) checkLightweightClosure(target);
  }
}
checkLightweightClosure(barrel);

function files(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? files(path.join(directory, entry.name)) : /\.(jsx?|tsx?)$/.test(entry.name) ? [path.join(directory, entry.name)] : []);
}
// Match and parse import/export declarations alone so pre-existing invalid JSX
// in unpublished drafts does not prevent checking their dependency boundary.
const namedDeclaration = /^[ \t]*(?:import|export)\s+(?:[\w$]+\s*,\s*)?\{[^}]*\}\s+from\s+(['"])([^'"\r\n]+)\1[ \t]*;?/gm;
const namespaceImport = /^[ \t]*import\s+\*\s+as\s+\w+\s+from\s+(['"])([^'"\r\n]+)\1/gm;
let scanned = 0, directMathImports = 0;
for (const filename of files(root)) {
  const source = fs.readFileSync(filename, 'utf8');
  scanned++;
  for (const match of source.matchAll(namedDeclaration)) {
    const specifier = match[2];
    if (isMathModule(specifier)) directMathImports++;
    if (!isContentBarrel(specifier)) continue;
    const statement = parse(match[0], { sourceType: 'module' }).program.body[0];
    for (const entry of statement.specifiers) {
      const imported = entry.imported?.name ?? entry.local?.name;
      assert.ok(!['Math', 'MathBlock'].includes(imported), `${filename}: import math directly from components/content/Math.jsx`);
    }
  }
  for (const match of source.matchAll(namespaceImport)) {
    assert.ok(!isContentBarrel(match[2]), `${filename}: use explicit lightweight content imports so dependency ownership is visible`);
  }
}
console.log(JSON.stringify({ passed: true, sourceFilesScanned: scanned, directMathImports, lightweightDependencyModules: lightweightClosure.size, genericContentExports: barrelAst.program.body.filter(node => node.type === 'ExportNamedDeclaration').flatMap(node => node.specifiers.map(specifier => specifier.exported.name)), checks: 'No direct/transitive static math or KaTeX barrel dependency; all named consumers use the direct math module, including unpublished drafts.' }));
