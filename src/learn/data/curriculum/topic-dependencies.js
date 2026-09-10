// Stable depth-first traversal inserts each recorded prerequisite before its
// dependent, including prerequisites outside a path's selected modules.
// Unknown older dependencies remain explicitly unreviewed in the inventory.
export function orderWithPrerequisites(ids, catalogue) {
  const ordered = [];
  const visited = new Set();
  const visiting = new Set();
  function visit(id, chain = []) {
    if (visited.has(id)) return;
    if (visiting.has(id)) throw new Error(`Curriculum prerequisite cycle: ${[...chain, id].join(" → ")}`);
    const topic = catalogue[id];
    if (!topic) throw new Error(`Unknown curriculum prerequisite: ${id} (${chain.join(" → ")})`);
    visiting.add(id);
    for (const prerequisite of topic.prerequisiteIds) visit(prerequisite, [...chain, id]);
    visiting.delete(id);
    visited.add(id);
    ordered.push(id);
  }
  ids.forEach((id) => visit(id));
  return ordered;
}
