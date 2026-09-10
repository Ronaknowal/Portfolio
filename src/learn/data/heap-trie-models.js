// Exact, bounded teaching traces. These models do not run learner code.
export const HEAP_SAMPLE = [1, 2, 9, 7, 5];
export const HEAP_RAW_SAMPLE = [7, 2, 9, 1, 5];
export const TOP_K_SAMPLE = [5, 1, 9, 3, 9, 2];
export const TRIE_SAMPLE = ['car', 'cart', 'cat', 'dog'];
export const HEAP_LIMIT = 12;
export const STREAM_LIMIT = 16;
export const TRIE_NODE_LIMIT = 32;
const clone = value => structuredClone(value);
const numberKey = key => {
  if (!Number.isSafeInteger(key)) throw new TypeError('Use safe integer priorities.');
};
export function parseHeapValues(text, limit = HEAP_LIMIT) {
  if (!text.trim()) return {
    valid: true,
    values: [],
    error: null
  };
  const parts = text.split(',').map(part => part.trim());
  if (parts.length > limit || parts.some(part => !/^-?\d+$/.test(part) || Number(part) < -99 || Number(part) > 99)) return {
    valid: false,
    values: [],
    error: `Use at most ${limit} comma-separated integers from −99 to 99, with no empty entries.`
  };
  return {
    valid: true,
    values: parts.map(Number),
    error: null
  };
}
export function parseHeapKey(text) {
  const parsed = parseHeapValues(text, 1);
  return parsed.valid && parsed.values.length === 1 ? {
    valid: true,
    key: parsed.values[0],
    error: null
  } : {
    valid: false,
    key: null,
    error: 'Enter one integer from −99 to 99.'
  };
}
export function heapViolations(values) {
  const violations = [];
  for (let child = 1; child < values.length; child++) {
    const parent = Math.floor((child - 1) / 2);
    if (values[parent] > values[child]) violations.push({
      parent,
      child
    });
  }
  return violations;
}
export function heapOperationTrace(values = HEAP_SAMPLE, operation = 'push', key = 0) {
  if (!['push', 'pop', 'build'].includes(operation)) throw new TypeError('Choose push, pop or build.');
  values.forEach(numberKey);
  if (operation === 'push') numberKey(key);
  const heap = [...values],
    trace = [],
    settledRoots = [];
  let comparisonCount = 0,
    swapCount = 0,
    popped = null;
  const save = (phase, note, fields = {}) => trace.push(clone({
    heap,
    phase,
    note,
    operation,
    key,
    comparisonCount,
    swapCount,
    popped,
    settledRoots,
    activeIndices: [],
    comparedIndices: [],
    swappedIndices: [],
    result: null,
    ...fields
  }));
  save('Read the array as a complete tree', 'Array positions are filled level by level, left to right. Only parent–child order is required; the whole array need not be sorted.');
  if (heap.length > HEAP_LIMIT) {
    save('Browser size limit', `Use at most ${HEAP_LIMIT} values in this investigation.`, {
      result: 'limit'
    });
    return trace;
  }
  if (operation !== 'build' && heapViolations(heap).length) {
    save('Repair the starting heap first', 'Push and pop require a valid min-heap. Choose Build a heap to repair this unordered array.', {
      result: 'invalid'
    });
    return trace;
  }
  const swap = (first, second) => {
    [heap[first], heap[second]] = [heap[second], heap[first]];
    swapCount++;
    save('Swap the two values', `Indices ${first} and ${second} exchange their values. Array cells and tree positions keep the same indices.`, {
      activeIndices: [first, second],
      swappedIndices: [first, second]
    });
  };
  function siftDown(start) {
    let parent = start;
    while (2 * parent + 1 < heap.length) {
      const left = 2 * parent + 1,
        right = left + 1;
      let child = left;
      if (right < heap.length) {
        comparisonCount++;
        if (heap[right] < heap[left]) child = right;
        save('Choose the smaller child', `${heap[left]} at ${left} versus ${heap[right]} at ${right}: choose index ${child}. On equal priorities this implementation chooses the left child.`, {
          activeIndices: [parent, child],
          comparedIndices: [left, right]
        });
      } else save('Only a left child exists', `Index ${left} is the only child of ${parent}; there is no right child to compare.`, {
        activeIndices: [parent, child]
      });
      comparisonCount++;
      save('Compare parent with chosen child', `${heap[parent]} at ${parent} ${heap[parent] <= heap[child] ? '≤' : '>'} ${heap[child]} at ${child}. ${heap[parent] <= heap[child] ? 'Order holds, so stop this repair.' : 'Swap, then continue at the child position.'}`, {
        activeIndices: [parent, child],
        comparedIndices: [parent, child]
      });
      if (heap[parent] <= heap[child]) break;
      swap(parent, child);
      parent = child;
    }
  }
  if (operation === 'push') {
    if (heap.length === HEAP_LIMIT) {
      save('Browser size limit', `The investigation is limited to ${HEAP_LIMIT} nodes; rebuild a smaller heap.`, {
        result: 'limit'
      });
      return trace;
    }
    heap.push(key);
    let child = heap.length - 1;
    save('Append at the next complete-tree position', `Place ${key} at array index ${child}. Only its ancestor path can need repair.`, {
      activeIndices: [child]
    });
    while (child > 0) {
      const parent = Math.floor((child - 1) / 2);
      comparisonCount++;
      save('Compare with the parent', `${heap[child]} at ${child} ${heap[child] < heap[parent] ? '<' : '≥'} ${heap[parent]} at ${parent}. ${heap[child] < heap[parent] ? 'Swap upward.' : 'Stop: equal priorities are allowed.'}`, {
        activeIndices: [child, parent],
        comparedIndices: [child, parent]
      });
      if (heap[child] >= heap[parent]) break;
      swap(child, parent);
      child = parent;
    }
  } else if (operation === 'pop') {
    if (!heap.length) {
      save('No minimum to remove', 'The heap is empty. This browser reports the empty case; Python heapq.heappop would raise IndexError.', {
        result: 'empty'
      });
      return trace;
    }
    popped = heap[0];
    const tail = heap.pop();
    if (heap.length) {
      heap[0] = tail;
      save('Move the last value into the root slot', `Return minimum ${popped}; move last value ${tail} to index 0. Removing the last cell preserves complete shape. Repair may be needed below the root.`, {
        activeIndices: [0]
      });
      siftDown(0);
    } else save('Remove the only value', `Return ${popped}. The array is now empty.`);
  } else {
    for (let start = Math.floor(heap.length / 2) - 1; start >= 0; start--) {
      save('Repair the next internal subtree', `Start at index ${start}. Its child subtrees have already been heapified, or are leaves.`, {
        activeIndices: [start]
      });
      siftDown(start);
      settledRoots.push(start);
      save('This subtree now satisfies heap order', `The subtree rooted at index ${start} is a min-heap. Continue toward index 0.`, {
        activeIndices: [start]
      });
    }
  }
  save('Heap operation complete', `Every existing parent is ≤ each child. ${operation === 'pop' ? `Returned ${popped}. ` : ''}The resulting array is a heap, not necessarily a sorted sequence.`, {
    result: 'complete'
  });
  return trace;
}
export function heapLayout(values) {
  const deepest = values.length ? Math.floor(Math.log2(values.length)) : 0,
    width = Math.max(360, 2 ** deepest * 74);
  const nodes = values.map((value, index) => {
    const depth = Math.floor(Math.log2(index + 1)),
      rank = index - (2 ** depth - 1);
    return {
      index,
      value,
      depth,
      x: (rank + .5) * width / 2 ** depth,
      y: 45 + depth * 84
    };
  });
  return {
    nodes,
    width,
    height: 110 + 84 * deepest,
    edges: nodes.slice(1).map(node => ({
      parent: Math.floor((node.index - 1) / 2),
      child: node.index
    }))
  };
}
export function heapConstructionProfile(size = 15) {
  if (!Number.isInteger(size) || size < 0 || size > 1023) throw new TypeError('Use a size from 0 through 1023.');
  const heights = new Array(size).fill(0);
  for (let index = size - 1; index >= 0; index--) {
    const left = 2 * index + 1,
      right = left + 1;
    heights[index] = left < size ? 1 + Math.max(heights[left], right < size ? heights[right] : -1) : 0;
  }
  const rows = [...new Set(heights)].sort((a, b) => a - b).map(height => {
    const indices = heights.flatMap((value, index) => value === height ? [index] : []);
    return {
      height,
      indices,
      count: indices.length,
      downwardBudget: height * indices.length
    };
  });
  return {
    size,
    heights,
    rows,
    totalDownwardBudget: heights.reduce((sum, height) => sum + height, 0)
  };
}
export function topKStreamTrace(stream = TOP_K_SAMPLE, k = 3) {
  stream.forEach(numberKey);
  if (!Number.isInteger(k) || k < 1 || k > 8 || stream.length > STREAM_LIMIT) throw new TypeError('Use k from 1 through 8 and at most 16 stream values.');
  const heap = [],
    discarded = [],
    trace = [];
  let processed = 0;
  const save = (phase, note, current = null, fields = {}) => trace.push(clone({
    phase,
    note,
    stream,
    k,
    processed,
    heap,
    discarded,
    current,
    kth: processed >= k ? heap[0]?.value ?? null : null,
    result: null,
    ...fields
  }));
  const push = entry => {
    heap.push(entry);
    let child = heap.length - 1;
    while (child > 0) {
      const parent = Math.floor((child - 1) / 2);
      if (heap[parent].value <= heap[child].value) break;
      [heap[parent], heap[child]] = [heap[child], heap[parent]];
      child = parent;
    }
  };
  const sink = () => {
    let parent = 0;
    while (2 * parent + 1 < heap.length) {
      let child = 2 * parent + 1;
      if (child + 1 < heap.length && heap[child + 1].value < heap[child].value) child++;
      if (heap[parent].value <= heap[child].value) break;
      [heap[parent], heap[child]] = [heap[child], heap[parent]];
      parent = child;
    }
  };
  save('An empty retained set', 'Read one occurrence at a time. A min-heap will retain the largest k occurrences; repeated values count separately.');
  for (let index = 0; index < stream.length; index++) {
    const entry = {
      value: stream[index],
      sourceIndex: index
    };
    save('Inspect the next occurrence', `Occurrence #${index + 1} has value ${entry.value}. Predict whether the retained set has room or this value exceeds its minimum.`, entry);
    let note;
    if (heap.length < k) {
      push(entry);
      note = `There is room: retain occurrence #${index + 1}.`;
    } else if (entry.value > heap[0].value) {
      const removed = heap[0];
      heap[0] = entry;
      sink();
      discarded.push(removed);
      note = `${entry.value} exceeds boundary ${removed.value}; retain it and discard occurrence #${removed.sourceIndex + 1}. Repair the min-heap.`;
    } else {
      discarded.push(entry);
      note = `${entry.value} does not exceed boundary ${heap[0].value}. Discard this occurrence; a tie can keep either equal-valued occurrence.`;
    }
    processed++;
    save('Commit the prefix result', `${note} ${processed < k ? `Only ${processed} values have arrived: there is no kth-largest value yet.` : `The kth-largest value of this prefix is ${heap[0].value}.`}`, entry, {
      prefixComplete: true
    });
  }
  save('Stream complete', 'The retained occurrences are the largest k seen, or all values if fewer than k arrived. Heap order is not display ranking.', null, {
    result: 'complete'
  });
  return trace;
}
export function parseTrieWords(text) {
  if (!text.trim()) return {
    valid: true,
    words: [],
    error: null
  };
  const parts = text.split(',').map(part => part.trim()),
    words = parts.map(part => part === 'ε' ? '' : part);
  if (parts.length > 8 || parts.some(part => part !== 'ε' && !/^[a-z]{1,6}$/.test(part))) return {
    valid: false,
    words: [],
    error: 'Use at most 8 comma-separated lowercase a–z words, each 1–6 letters. Use ε for an empty stored word; do not leave an empty entry.'
  };
  if (buildTrie(words).nodes.length > TRIE_NODE_LIMIT) return {
    valid: false,
    words: [],
    error: `Shared paths may contain at most ${TRIE_NODE_LIMIT} visible nodes. Use fewer or shorter words.`
  };
  return {
    valid: true,
    words: [...new Set(words)],
    error: null
  };
}
export function parseTrieQuery(text) {
  const query = text === 'ε' ? '' : text;
  return /^[a-z]{0,6}$/.test(query) ? {
    valid: true,
    query,
    error: null
  } : {
    valid: false,
    query: null,
    error: 'Use 0–6 lowercase a–z letters. Blank or ε means the empty string.'
  };
}
const trieNode = (trie, id) => trie.nodes.find(node => node.id === id);
export function buildTrie(words = TRIE_SAMPLE) {
  const trie = {
    rootId: 'n0',
    nextId: 1,
    nodes: [{
      id: 'n0',
      prefix: '',
      terminal: false,
      children: {}
    }]
  };
  for (const word of words) {
    if (typeof word !== 'string') throw new TypeError('Trie keys must be strings.');
    let node = trieNode(trie, trie.rootId);
    for (const character of word) {
      if (!node.children[character]) {
        const child = {
          id: `n${trie.nextId++}`,
          prefix: node.prefix + character,
          terminal: false,
          children: {}
        };
        node.children[character] = child.id;
        trie.nodes.push(child);
      }
      node = trieNode(trie, node.children[character]);
    }
    node.terminal = true;
  }
  return trie;
}
export function trieWords(trie, startId = trie.rootId) {
  const output = [];
  function visit(id) {
    const node = trieNode(trie, id);
    if (node.terminal) output.push(node.prefix);
    for (const character of Object.keys(node.children).sort()) visit(node.children[character]);
  }
  visit(startId);
  return output;
}
export function validateTrie(trie) {
  const errors = [],
    seen = new Set(),
    ids = new Set();
  for (const node of trie.nodes) {
    if (ids.has(node.id)) errors.push(`Repeated identity ${node.id}.`);
    ids.add(node.id);
  }
  function visit(id, prefix) {
    const node = trieNode(trie, id);
    if (!node) {
      errors.push(`Missing child ${id}.`);
      return;
    }
    if (seen.has(id)) {
      errors.push(`Shared child or cycle at ${id}.`);
      return;
    }
    seen.add(id);
    if (node.prefix !== prefix) errors.push(`Wrong prefix at ${id}.`);
    if (typeof node.terminal !== 'boolean') errors.push(`Invalid terminal flag at ${id}.`);
    for (const [character, childId] of Object.entries(node.children)) {
      if ([...character].length !== 1) errors.push(`Invalid character edge ${character}.`);
      visit(childId, prefix + character);
    }
  }
  visit(trie.rootId, '');
  for (const id of ids) if (!seen.has(id)) errors.push(`Unreachable node ${id}.`);
  return {
    valid: errors.length === 0,
    errors
  };
}
export function trieOperationTrace(source = buildTrie(), query = 'car', operation = 'exact') {
  if (typeof query !== 'string') throw new TypeError('A query must be a string.');
  if (!['exact', 'prefix', 'insert', 'delete'].includes(operation)) throw new TypeError('Choose exact, prefix, insert or delete.');
  const trie = clone(source),
    trace = [],
    visitedIds = [trie.rootId],
    path = [];
  let activeId = trie.rootId,
    consumed = '';
  const save = (phase, note, fields = {}) => trace.push(clone({
    trie,
    query,
    operation,
    phase,
    note,
    activeId,
    consumed,
    visitedIds,
    matches: [],
    removedIds: [],
    result: null,
    ...fields
  }));
  save('Start at the empty prefix', 'The root represents the empty prefix. A terminal marker, including one at the root, means a complete stored string.');
  if (operation === 'insert' && buildTrie([...trieWords(trie), query]).nodes.length > TRIE_NODE_LIMIT) {
    save('Browser node limit', `Insertion would exceed ${TRIE_NODE_LIMIT} visible nodes. Rebuild a smaller word set.`, {
      result: 'limit'
    });
    return trace;
  }
  for (const character of query) {
    let parent = trieNode(trie, activeId),
      childId = parent.children[character];
    if (!childId && operation === 'insert') {
      childId = `n${trie.nextId++}`;
      parent.children[character] = childId;
      trie.nodes.push({
        id: childId,
        prefix: consumed + character,
        terminal: false,
        children: {}
      });
      save('Create one missing character edge', `Add edge “${character}” from prefix “${consumed || 'ε'}”. Reuse all existing prefix nodes.`, {
        edge: {
          fromId: activeId,
          toId: childId,
          character
        },
        createdId: childId
      });
    }
    if (!childId) {
      save('No matching character edge', `Prefix “${consumed || 'ε'}” has no “${character}” edge. The requested string or prefix is absent.`, {
        result: 'absent',
        missingCharacter: character
      });
      return trace;
    }
    path.push({
      parentId: activeId,
      character,
      childId
    });
    const fromId = activeId;
    activeId = childId;
    consumed += character;
    visitedIds.push(activeId);
    save('Consume one character', `Follow “${character}”: consumed prefix is now “${consumed}”. A node can be both terminal and have children.`, {
      edge: {
        fromId,
        toId: activeId,
        character
      }
    });
  }
  const node = trieNode(trie, activeId);
  if (operation === 'exact') {
    save('Check the terminal marker', node.terminal ? 'The entire query was consumed and this node is terminal: the word is stored.' : 'The path exists, but this node is not terminal: it is only a prefix, not a stored word.', {
      result: node.terminal ? 'found' : 'absent'
    });
  } else if (operation === 'prefix') {
    save('Enumerate below the prefix', 'Collect terminal descendants, including this node if terminal. Results are alphabetic for this bounded example, not ranked by popularity.', {
      result: 'found',
      matches: trieWords(trie, activeId)
    });
  } else if (operation === 'insert') {
    const duplicate = node.terminal;
    node.terminal = true;
    save(duplicate ? 'Keep the existing terminal marker' : 'Mark the full word terminal', duplicate ? 'This word is already stored. Set semantics do not create a duplicate.' : 'The full character path now represents a stored word; earlier nonterminal prefixes remain prefixes.', {
      result: duplicate ? 'duplicate' : 'inserted'
    });
  } else {
    if (!node.terminal) {
      save('The prefix is not a stored word', 'The path exists without a terminal marker. Nothing can be deleted.', {
        result: 'absent'
      });
      return trace;
    }
    node.terminal = false;
    save('Clear only the terminal marker', `Remove stored word “${query || 'ε'}”. Keep its children: longer words may still need this prefix.`);
    const removedIds = [];
    for (let index = path.length - 1; index >= 0; index--) {
      const item = path[index],
        child = trieNode(trie, item.childId);
      if (child.terminal || Object.keys(child.children).length) {
        save('A surviving word still needs this node', `Keep prefix “${child.prefix}”: it ${child.terminal ? 'is itself a stored word' : 'has a child needed by a longer word'}. Stop pruning.`, {
          removedIds
        });
        break;
      }
      const parent = trieNode(trie, item.parentId);
      delete parent.children[item.character];
      trie.nodes = trie.nodes.filter(candidate => candidate.id !== child.id);
      removedIds.push(child.id);
      activeId = parent.id;
      save('Prune one unused suffix node', `Prefix “${child.prefix}” is nonterminal and childless. Remove its incoming “${item.character}” edge and node; inspect its parent next.`, {
        removedIds,
        prunedId: child.id
      });
    }
    save('Deletion complete', 'Only the requested terminal word was removed. Other stored words and their shared prefix paths remain.', {
      result: 'deleted',
      removedIds
    });
  }
  return trace;
}
export function trieLayout(trie) {
  const nodes = [],
    edges = [];
  let leaf = 0,
    deepest = 0;
  function visit(id, depth) {
    const node = trieNode(trie, id),
      children = Object.entries(node.children).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0),
      ys = [];
    for (const [character, childId] of children) {
      ys.push(visit(childId, depth + 1));
      edges.push({
        fromId: id,
        toId: childId,
        character
      });
    }
    const y = ys.length ? (ys[0] + ys.at(-1)) / 2 : 45 + leaf++ * 78;
    deepest = Math.max(deepest, depth);
    nodes.push({
      ...node,
      depth,
      x: 36 + depth * 112,
      y
    });
    return y;
  }
  visit(trie.rootId, 0);
  return {
    nodes,
    edges,
    width: Math.max(390, 112 * deepest + 90),
    height: Math.max(132, leaf * 78 + 30)
  };
}
