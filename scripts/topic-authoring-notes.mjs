import fs from 'node:fs';
import path from 'node:path';

// Markdown remains the editable source. Retrieval does not interpret a proposal
// as a verified fact, change its status or write notes into learner content.
export function readTopicAuthoringNotes(root, topicId) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(topicId)) throw new Error('Expected a canonical topic ID for authoring notes');
  const read = relativePath => {
    try {
      return { path: relativePath, exists: true, markdown: fs.readFileSync(path.join(root, relativePath), 'utf8') };
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      return { path: relativePath, exists: false, markdown: null };
    }
  };
  return {
    instruction: 'Read destination notes and relevant unresolved routing notes. Reassess evidence and learner benefit; record inclusion, adaptation, rerouting, reasoned deferral or rejection. A saved proposal is not an automatic content requirement.',
    workflow: 'docs/teaching/topic-notes/README.md',
    destination: read('docs/teaching/topic-notes/' + topicId + '.md'),
    unassigned: read('docs/teaching/topic-notes/UNASSIGNED.md'),
  };
}
