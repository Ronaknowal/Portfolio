// Read-only triage: source signals are not a factual or pedagogical certification.
import fs from 'node:fs';
import { trackDefinitions } from '../src/learn/data/track-definitions.js';

const mappings = JSON.parse(fs.readFileSync("src/learn/data/lesson-manifest.json", "utf8"));
const slugify = s => s.toLowerCase().replace(/[()]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
const seen = new Set();
const rows = [];
for (const track of trackDefinitions) for (const section of track.sections) for (const t of section.topics) {
  const title = typeof t === 'string' ? t : t.title;
  const slug = slugify(title);
  if (seen.has(slug)) continue;
  seen.add(slug);
  if (!mappings[slug]) continue;
  const file = `src/learn/data/${mappings[slug]}`;
  const source = fs.readFileSync(file, 'utf8');
  rows.push({ track: track.title, title, file,
    headings: [...source.matchAll(/<H[23][^>]*>([^<]+)<\/H[23]>/g)].map(m => m[1]),
    output: /language=["']output["']/.test(source),
    visual: /<svg|Visualizer|Visualiser|Diagram|<\w+Lab\b|<\w+Visual\b/.test(source),
    solution: /Checkpoint|type="answer"|<summary>|worked solution|answer key|solution:/i.test(source),
    shared: /ProgrammingReference|ReferenceArticle|ProgrammingTopic/.test(source),
  });
}
console.log(JSON.stringify({ total: rows.length, tracks: [...new Set(rows.map(r => r.track))].map(track => {
  const items = rows.filter(r => r.track === track);
  return { track, lessons: items.length, withoutExplicitOutput: items.filter(r => !r.output).length,
    withoutVisualSignal: items.filter(r => !r.visual).length,
    withoutSolutionSignal: items.filter(r => !r.solution).length };
}), rows: rows.map(({ headings, ...row }) => process.argv.includes('--headings') ? { ...row, headings } : row) }, null, 2));
