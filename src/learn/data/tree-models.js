// Bounded ordered-set teaching model. Node IDs are identities; keys are data.
// These pure operations do not execute learner code or balance the tree.
export const TREE_SAMPLE = [8, 3, 10, 1, 6, 14, 4, 7, 13];
export const TREE_LIMIT = 12;
export const TREE_KEY_MIN = -99;
export const TREE_KEY_MAX = 99;
const copy = value => structuredClone(value);
const nodeAt = (tree, id) => tree.nodes.find(node => node.id === id);
const emptyTree = () => ({ rootId: null, nodes: [], nextId: 1 });

function checkKey(key) {
  if (!Number.isSafeInteger(key)) throw new TypeError('A key must be a safe integer.');
}

export function parseTreeKeys(text) {
  if (!text.trim()) return { valid: true, values: [], error: null };
  const tokens = text.split(',').map(token => token.trim());
  if (tokens.length > TREE_LIMIT) return { valid: false, values: [], error: `Use at most ${TREE_LIMIT} entries in this browser investigation.` };
  if (tokens.some(token => !/^-?\d+$/.test(token))) return { valid: false, values: [], error: 'Separate whole-number keys with commas; do not leave an empty entry.' };
  const values = tokens.map(Number);
  if (values.some(key => !Number.isSafeInteger(key) || key < TREE_KEY_MIN || key > TREE_KEY_MAX)) return { valid: false, values: [], error: `Use integer keys from ${TREE_KEY_MIN} to ${TREE_KEY_MAX}.` };
  return { valid: true, values, error: null };
}

export function parseTreeTarget(text) {
  const parsed = parseTreeKeys(text);
  return parsed.valid && parsed.values.length === 1 ? { valid: true, key: parsed.values[0], error: null } : { valid: false, key: null, error: `Enter one whole-number key from ${TREE_KEY_MIN} to ${TREE_KEY_MAX}.` };
}

export function buildTree(values = TREE_SAMPLE) {
  if (!Array.isArray(values)) throw new TypeError('Build a tree from an array of integer keys.');
  const tree = emptyTree();
  for (const key of values) {
    checkKey(key);
    let parent = null, currentId = tree.rootId, side = null;
    while (currentId !== null) {
      const current = nodeAt(tree, currentId);
      if (key === current.key) break;
      parent = current;
      side = key < current.key ? 'leftId' : 'rightId';
      currentId = current[side];
    }
    if (currentId !== null) continue;
    const node = { id: `n${tree.nextId++}`, key, leftId: null, rightId: null };
    tree.nodes.push(node);
    if (parent === null) tree.rootId = node.id;
    else parent[side] = node.id;
  }
  return tree;
}

export function inorderKeys(tree) {
  const output = [];
  function visit(id) { if (id === null) return; const node = nodeAt(tree, id); visit(node.leftId); output.push(node.key); visit(node.rightId); }
  visit(tree.rootId);
  return output;
}

export function validateBST(tree) {
  const errors = [], seen = new Set(), ids = new Set();
  for (const node of tree.nodes) {
    if (ids.has(node.id)) errors.push(`Repeated node identity ${node.id}.`);
    ids.add(node.id);
  }
  function visit(id, lower, upper) {
    if (id === null) return;
    const node = nodeAt(tree, id);
    if (!node) { errors.push(`Dangling link to ${id}.`); return; }
    if (seen.has(id)) { errors.push(`Node ${id} is reached twice: cycle or shared child.`); return; }
    seen.add(id);
    if (!Number.isSafeInteger(node.key) || node.key <= lower || node.key >= upper) errors.push(`${id} key ${node.key} violates the strict ancestor interval (${lower}, ${upper}).`);
    visit(node.leftId, lower, node.key);
    visit(node.rightId, node.key, upper);
  }
  visit(tree.rootId, -Infinity, Infinity);
  for (const id of ids) if (!seen.has(id)) errors.push(`Node ${id} is unreachable from the root.`);
  return { valid: errors.length === 0, errors };
}

export function treeMetrics(tree) {
  let diameter = 0;
  function height(id) { if (id === null) return -1; const node = nodeAt(tree, id), left = height(node.leftId), right = height(node.rightId); diameter = Math.max(diameter, left + right + 2); return 1 + Math.max(left, right); }
  const rootHeight = height(tree.rootId);
  return { size: tree.nodes.length, height: rootHeight, diameter, leafCount: tree.nodes.filter(node => node.leftId === null && node.rightId === null).length };
}

export function invalidBoundsExample() {
  return { rootId: 'n1', nextId: 4, nodes: [{ id: 'n1', key: 10, leftId: 'n2', rightId: null }, { id: 'n2', key: 5, leftId: null, rightId: 'n3' }, { id: 'n3', key: 12, leftId: null, rightId: null }] };
}

export function rotationExample() {
  const before = buildTree([30, 20, 40, 10, 25]), after = copy(before);
  const oldRoot = nodeAt(after, after.rootId), promoted = nodeAt(after, oldRoot.leftId);
  oldRoot.leftId = promoted.rightId;
  promoted.rightId = oldRoot.id;
  after.rootId = promoted.id;
  return { before, after, transferredId: 'n5' };
}

export function treeLayout(tree) {
  const positioned = [], width = Math.max(360, tree.nodes.length * 58 + 44);
  let rank = 0, deepest = 0;
  function visit(id, depth) {
    if (id === null) return;
    const node = nodeAt(tree, id);
    visit(node.leftId, depth + 1);
    const x = 36 + (tree.nodes.length < 2 ? (width - 72) / 2 : rank * (width - 72) / (tree.nodes.length - 1));
    positioned.push({ ...node, x, y: 55 + depth * 88, depth });
    rank++; deepest = Math.max(deepest, depth);
    visit(node.rightId, depth + 1);
  }
  visit(tree.rootId, 0);
  const edges = positioned.flatMap(node => ['leftId', 'rightId'].filter(side => node[side] !== null).map(side => ({ fromId: node.id, toId: node[side], side: side === 'leftId' ? 'left' : 'right' })));
  return { nodes: positioned, edges, width, height: deepest * 88 + 115 };
}

function snapshot(tree, fields) {
  return { tree: copy(tree), phase: '', note: '', activeId: null, visitedIds: [], lower: null, upper: null,
    comparisonCount: 0, result: null, edge: null, frontier: [], output: [], outputIds: [], transient: false, ...copy(fields) };
}

export function searchTrace(tree, key) {
  checkKey(key);
  const trace = [], visitedIds = [];
  let currentId = tree.rootId, lower = null, upper = null, parentId = null, side = null, comparisonCount = 0;
  const save = fields => trace.push(snapshot(tree, { activeId: currentId, visitedIds, lower, upper, comparisonCount, parentId, side, key, ...fields }));
  save({ phase: 'Start at the root', note: 'Only the root reference is needed to enter the tree. Each comparison chooses one child; the other subtree cannot contain the target.' });
  while (currentId !== null) {
    const node = nodeAt(tree, currentId);
    comparisonCount++; visitedIds.push(node.id);
    if (key === node.key) {
      save({ phase: 'Key found', result: 'found', note: `${key} equals the key at ${node.id}. Stop after ${comparisonCount} node comparison${comparisonCount === 1 ? '' : 's'}.` });
      return trace;
    }
    side = key < node.key ? 'leftId' : 'rightId';
    save({ phase: `Compare ${key} with ${node.key}`, note: `${key} ${side === 'leftId' ? '<' : '>'} ${node.key}: follow the ${side === 'leftId' ? 'left' : 'right'} child. Keep all earlier ancestor bounds too.`, edge: { fromId: node.id, toId: node[side], side: side === 'leftId' ? 'left' : 'right' } });
    if (side === 'leftId') upper = node.key; else lower = node.key;
    parentId = node.id; currentId = node[side];
  }
  save({ phase: 'Reach an empty child link', result: 'absent', note: parentId === null ? 'The root is empty, so the key is absent without any node comparison.' : `${parentId}.${side === 'leftId' ? 'left' : 'right'} is empty. The search has exhausted the only subtree where ${key} could belong.` });
  return trace;
}

export function insertionTrace(tree, key) {
  const trace = searchTrace(tree, key), terminal = trace.at(-1);
  if (terminal.result === 'found') {
    trace.push(snapshot(tree, { ...terminal, phase: 'Ignore the duplicate key', result: 'duplicate', note: `This is an ordered set: ${key} already exists. Keep the same node and all its links; do not allocate another node.` }));
    return trace;
  }
  if (tree.nodes.length >= TREE_LIMIT) {
    trace.push(snapshot(tree, { ...terminal, phase: 'Browser size limit', result: 'limit', note: `The investigation stops at ${TREE_LIMIT} nodes. Remove a key or rebuild a smaller tree; this is a teaching limit, not a BST rule.` }));
    return trace;
  }
  const updated = copy(tree), node = { id: `n${updated.nextId++}`, key, leftId: null, rightId: null };
  updated.nodes.push(node);
  if (terminal.parentId === null) updated.rootId = node.id;
  else nodeAt(updated, terminal.parentId)[terminal.side] = node.id;
  trace.push(snapshot(updated, { ...terminal, tree: updated, activeId: node.id, visitedIds: [...terminal.visitedIds, node.id], phase: 'Attach one new leaf', result: 'inserted', note: `Allocate ${node.id} with key ${key}; replace the empty ${terminal.parentId === null ? 'root reference' : terminal.parentId + '.' + (terminal.side === 'leftId' ? 'left' : 'right') + ' link'}. No existing node moves in memory, and both new child links are empty.` }));
  return trace;
}

export function traversalTrace(tree, order = 'inorder') {
  if (!['preorder', 'inorder', 'postorder', 'level-order'].includes(order)) throw new TypeError('Choose preorder, inorder, postorder or level-order.');
  const trace = [], frontier = [], output = [], outputIds = [];
  const save = (phase, note, activeId = null) => trace.push(snapshot(tree, { phase, note, activeId, frontier, output, outputIds, order }));
  if (order === 'level-order') {
    if (tree.rootId !== null) frontier.push({ nodeId: tree.rootId, phase: 'waiting' });
    save('Start the queue', 'The front is the next node to process. Enqueue children left before right so each level is read left to right.');
    while (frontier.length) {
      const { nodeId } = frontier.shift(), node = nodeAt(tree, nodeId);
      save('Dequeue the front', `${nodeId} leaves the front of the queue; later arrivals remain behind it.`, nodeId);
      output.push(node.key); outputIds.push(node.id);
      save('Emit the key', `Append ${node.key} to output. Output is the answer built so far, not the pending queue.`, nodeId);
      for (const childId of [node.leftId, node.rightId]) if (childId !== null) frontier.push({ nodeId: childId, phase: 'waiting' });
      save('Enqueue the children', `Append the nonempty left and right children of ${nodeId}, in that order.`, nodeId);
    }
  } else {
    save('Start the traversal', 'The call stack remembers unfinished ancestor calls. A key enters the output only at the chosen visit moment.');
    function visit(nodeId) {
      if (nodeId === null) return;
      const node = nodeAt(tree, nodeId), frame = { nodeId, phase: 'entered' };
      frontier.push(frame); save('Enter a call', `Begin the call for ${nodeId}, key ${node.key}. Its children are still to be considered.`, nodeId);
      const emit = () => { frame.phase = 'emit key'; output.push(node.key); outputIds.push(node.id); save('Emit the key', `Append ${node.key} ${order === 'preorder' ? 'before both subtrees' : order === 'inorder' ? 'after the left subtree and before the right' : 'after both subtrees'}.`, nodeId); };
      if (order === 'preorder') emit();
      frame.phase = 'resume after left'; visit(node.leftId);
      if (order === 'inorder') emit();
      frame.phase = 'resume after right'; visit(node.rightId);
      if (order === 'postorder') emit();
      frontier.pop(); save('Return to the caller', `${nodeId}'s subtree is finished. Remove its frame; the caller resumes from its saved phase.`, nodeId);
    }
    visit(tree.rootId);
  }
  save('Traversal complete', tree.nodes.length ? `Every node emitted exactly once: ${output.join(', ')}.` : 'An empty tree emits no keys. There are no pending nodes or calls.');
  trace.at(-1).result = 'complete';
  return trace;
}

export function deletionTrace(tree, key) {
  const trace = searchTrace(tree, key), found = trace.at(-1);
  if (found.result !== 'found') {
    trace.push(snapshot(tree, { ...found, phase: 'Nothing to delete', result: 'absent', note: `${key} is absent. Keep the root, every node identity and every link unchanged.` }));
    return trace;
  }
  const updated = copy(tree), target = nodeAt(updated, found.activeId);
  const targetId = target.id, visitedIds = [...found.visitedIds];
  const save = fields => trace.push(snapshot(updated, { targetId, activeId: targetId, visitedIds, key, comparisonCount: found.comparisonCount, ...fields }));
  let victim = target, victimParent = found.parentId === null ? null : nodeAt(updated, found.parentId), victimSide = found.side;
  if (target.leftId !== null && target.rightId !== null) {
    save({ phase: 'Two children need a successor', note: `Keep node identity ${targetId}. Find the smallest key in its right subtree: one step right, then as far left as possible.` });
    victimParent = target; victimSide = 'rightId'; victim = nodeAt(updated, target.rightId);
    save({ phase: 'Enter the right subtree', activeId: victim.id, successorId: victim.id, note: `${victim.id}, key ${victim.key}, starts the successor search. A successor may be below several left links.` });
    while (victim.leftId !== null) {
      victimParent = victim; victimSide = 'leftId'; victim = nodeAt(updated, victim.leftId); visitedIds.push(victim.id);
      save({ phase: 'Follow another left link', activeId: victim.id, successorId: victim.id, note: `Move left to ${victim.id}, key ${victim.key}. Everything farther right is larger.` });
    }
    save({ phase: 'Successor has no left child', activeId: victim.id, successorId: victim.id, note: `${victim.id}, key ${victim.key}, is the smallest key in the right subtree. It may still have a right child, which must be reattached.` });
    const oldKey = target.key;
    target.key = victim.key;
    save({ phase: 'Copy the successor key', successorId: victim.id, transient: true, note: `${targetId} keeps its identity and child links, but key ${oldKey} becomes ${victim.key}. This intermediate write leaves a temporary duplicate; unlink the successor before considering deletion complete.` });
  } else {
    save({ phase: target.leftId === null && target.rightId === null ? 'Remove a leaf' : 'Bypass a node with one child', note: target.leftId === null && target.rightId === null ? `${targetId} has no children. Its incoming link will become empty.` : `${targetId} has exactly one child. Its incoming link will point directly to that child; the child keeps its whole subtree.` });
  }
  const replacementId = victim.leftId ?? victim.rightId, successorId = victim.id === targetId ? null : victim.id;
  if (victimParent === null) updated.rootId = replacementId;
  else victimParent[victimSide] = replacementId;
  updated.nodes = updated.nodes.filter(node => node.id !== victim.id);
  save({ phase: 'Reconnect and finish', activeId: successorId ? targetId : replacementId, successorId, removedId: victim.id, replacementId, result: 'deleted', note: `${victimParent === null ? 'root' : victimParent.id + '.' + (victimSide === 'leftId' ? 'left' : 'right')} now points to ${replacementId ?? 'empty'}. Remove identity ${victim.id}. The remaining keys are exactly the original set without ${key}; strict ancestor ordering is restored.` });
  return trace;
}
