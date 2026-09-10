export const defaultProbeProgram = 'put 1 10\nput 9 90\nput 17 170\ndel 1\nget 17\nput 17 171\nput 25 250\nrebuild 16\nget 9';
const modulo = (value, capacity) => (value % capacity + capacity) % capacity;
const copySlots = slots => slots.map(entry => entry === null ? null : {
  ...entry
});
const sortedPairs = pairs => pairs.map(pair => [...pair]).sort((left, right) => left[0] - right[0]);
const isEntry = slot => slot !== null && !slot.deleted;
export function parseProbeProgram(text) {
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if (lines.length > 24) throw new Error('Use at most 24 operation lines.');
  return lines.map((line, index) => {
    const [kind, ...tokens] = line.split(/\s+/);
    const count = kind === 'put' ? 2 : ['get', 'del', 'rebuild'].includes(kind) ? 1 : -1;
    if (tokens.length !== count || tokens.some(token => !/^-?\d+$/.test(token))) {
      throw new Error(`Line ${index + 1}: use put key value, get key, del key or rebuild capacity.`);
    }
    const values = tokens.map(Number);
    if (kind === 'rebuild') {
      if (![4, 8, 16].includes(values[0])) throw new Error('Rebuild capacity must be 4, 8 or 16.');
      return {
        kind,
        capacity: values[0],
        line
      };
    }
    if (values.some(value => !Number.isSafeInteger(value) || Math.abs(value) > 999)) {
      throw new Error('Use integer keys and values from −999 to 999.');
    }
    return {
      kind,
      key: values[0],
      value: values[1],
      line
    };
  });
}
export function probeInvariant(slots) {
  const seen = new Set();
  const issues = [];
  for (let position = 0; position < slots.length; position++) {
    const entry = slots[position];
    if (!isEntry(entry)) continue;
    if (seen.has(entry.key)) issues.push(`Key ${entry.key} occurs more than once.`);
    seen.add(entry.key);
    const home = modulo(entry.key, slots.length);
    for (let offset = 0; offset < slots.length; offset++) {
      const visited = (home + offset) % slots.length;
      if (visited === position) break;
      if (slots[visited] === null) {
        issues.push(`Key ${entry.key} in slot ${position} is hidden behind EMPTY slot ${visited}.`);
        break;
      }
    }
  }
  return issues;
}
export function probeProgramTrace(program, capacity = 8, eraseDeletion = false) {
  if (![4, 8, 16].includes(capacity) || typeof eraseDeletion !== 'boolean') throw new Error('Choose a displayed capacity and deletion rule.');
  const operations = parseProbeProgram(program);
  let slots = Array(capacity).fill(null);
  let totalProbes = 0;
  const reference = new Map();
  const states = [];
  let operationIndex = -1;
  let visited = [];
  function save(action, phase, result = null, expected = null) {
    states.push({
      slots: copySlots(slots),
      capacity: slots.length,
      live: slots.filter(isEntry).length,
      deleted: slots.filter(slot => slot?.deleted).length,
      operationIndex,
      operation: operations[operationIndex]?.line || 'Empty table',
      visited: [...visited],
      active: visited.at(-1) ?? null,
      totalProbes,
      action,
      phase,
      result,
      expected,
      agrees: expected === null || JSON.stringify(result) === JSON.stringify(expected),
      issues: probeInvariant(slots),
      logicalPairs: sortedPairs([...reference])
    });
  }
  function scan(key, trace = true) {
    let firstDeleted = null;
    for (let offset = 0; offset < slots.length; offset++) {
      const position = (modulo(key, slots.length) + offset) % slots.length;
      const entry = slots[position];
      totalProbes++;
      if (trace) visited.push(position);
      if (entry === null) {
        if (trace) save(`Slot ${position} is EMPTY: no equal key lies farther along this valid probe chain.`, 'probe');
        return {
          found: null,
          free: firstDeleted ?? position
        };
      }
      if (entry.deleted) {
        if (firstDeleted === null) firstDeleted = position;
        if (trace) save(`Slot ${position} is DELETED: remember it for insertion, but continue searching for an existing key.`, 'probe');
      } else if (entry.key === key) {
        if (trace) save(`Slot ${position} stores equal key ${key}: use this existing entry.`, 'probe');
        return {
          found: position,
          free: null
        };
      } else if (trace) {
        save(`Slot ${position} stores unequal key ${entry.key}: continue. A shared home is not equality.`, 'probe');
      }
    }
    if (trace) save(`All ${slots.length} positions were inspected once. No equal key exists; ${firstDeleted === null ? 'no slot is reusable' : `DELETED slot ${firstDeleted} is reusable`}.`, 'probe');
    return {
      found: null,
      free: firstDeleted
    };
  }
  save('Every slot starts EMPTY. No mapping is stored.', 'initial');
  for (operationIndex = 0; operationIndex < operations.length; operationIndex++) {
    const operation = operations[operationIndex];
    visited = [];
    save(`Start ${operation.line}.`, 'begin');
    if (operation.kind === 'rebuild') {
      const entries = slots.filter(isEntry);
      if (entries.length > operation.capacity) {
        save('Rebuild rejected: the requested capacity cannot hold all live entries. Table unchanged.', 'commit', 'rejected', 'rejected');
        continue;
      }
      const oldCapacity = slots.length;
      slots = Array(operation.capacity).fill(null);
      for (const entry of entries) {
        const destination = scan(entry.key, false);
        slots[destination.found ?? destination.free] = {
          ...entry
        };
      }
      const result = sortedPairs(slots.filter(isEntry).map(entry => [entry.key, entry.value]));
      save(`Rebuilt ${entries.length} entries from capacity ${oldCapacity} into ${slots.length}; recomputed homes and removed tombstones.`, 'commit', result, sortedPairs([...reference]));
      continue;
    }
    const location = scan(operation.key);
    if (operation.kind === 'get') {
      const result = location.found === null ? {
        found: false
      } : {
        found: true,
        value: slots[location.found].value
      };
      const expected = reference.has(operation.key) ? {
        found: true,
        value: reference.get(operation.key)
      } : {
        found: false
      };
      save(result.found ? `Return stored value ${result.value}.` : 'Return missing.', 'commit', result, expected);
    } else if (operation.kind === 'del') {
      const removed = location.found !== null;
      const expected = reference.delete(operation.key);
      if (removed) slots[location.found] = eraseDeletion ? null : {
        deleted: true
      };
      save(removed ? `Remove key ${operation.key}; slot ${location.found} becomes ${eraseDeletion ? 'EMPTY (faulty rule)' : 'DELETED'}.` : 'Key absent; nothing removed.', 'commit', removed, expected);
    } else {
      const destination = location.found ?? location.free;
      if (destination === null) {
        save('Insert rejected: every slot is occupied by a different key. Rebuild or remove an entry before retrying.', 'commit', 'full', 'full');
      } else {
        const inserted = !reference.has(operation.key);
        slots[destination] = {
          key: operation.key,
          value: operation.value
        };
        reference.set(operation.key, operation.value);
        save(`${location.found === null ? 'Insert' : 'Replace'} ${operation.key} → ${operation.value} in slot ${destination}.`, 'commit', location.found === null, inserted);
      }
    }
  }
  return {
    operations,
    states
  };
}
export function hashFamilyState(keysText = '1, 5, 9, 13', multiplier = 1, offset = 0, query = 9) {
  const tokens = keysText.split(',').map(token => token.trim());
  if (tokens.length > 8 || tokens.some(token => !/^\d+$/.test(token) || Number(token) > 16)) throw new Error('Use one to eight distinct integer keys from 0 to 16.');
  const keys = tokens.map(Number);
  if (new Set(keys).size !== keys.length) throw new Error('Use distinct keys so each key denotes one entry.');
  if (!Number.isInteger(multiplier) || multiplier < 1 || multiplier > 16 || !Number.isInteger(offset) || offset < 0 || offset > 16 || !keys.includes(query)) throw new Error('Choose a=1…16, b=0…16 and a query among the stored keys.');
  const home = (key, a, b) => (a * key + b) % 17 % 4;
  const buckets = Array.from({
    length: 4
  }, () => []);
  for (const key of keys) buckets[home(key, multiplier, offset)].push(key);
  const family = [];
  for (let a = 1; a <= 16; a++) {
    for (let b = 0; b <= 16; b++) {
      const colliders = keys.filter(key => home(key, a, b) === home(query, a, b));
      family.push({
        a,
        b,
        length: colliders.length,
        colliders
      });
    }
  }
  const distribution = Array.from({
    length: keys.length
  }, (_, index) => ({
    length: index + 1,
    count: family.filter(row => row.length === index + 1).length
  }));
  return {
    keys,
    query,
    multiplier,
    offset,
    buckets,
    family,
    distribution,
    selectedLength: buckets[home(query, multiplier, offset)].length,
    totalLength: family.reduce((sum, row) => sum + row.length, 0),
    theoreticalBound: 1 + (keys.length - 1) / 4,
    pairCounts: keys.filter(key => key !== query).map(key => ({
      key,
      count: family.filter(row => row.colliders.includes(key)).length
    }))
  };
}
export const resizeScenarios = {
  append: Array(24).fill('+'),
  boundary: [...Array(8).fill('+'), ...Array.from({
    length: 8
  }, () => ['+', '-']).flat()],
  drain: [...Array(16).fill('+'), ...Array(16).fill('-')]
};
export function resizeTrace(operations, policy = 'quarter', growth = 'double') {
  if (!Array.isArray(operations) || operations.length > 64 || operations.some(operation => !['+', '-'].includes(operation))) throw new Error('Use at most 64 append (+) or pop (−) operations.');
  if (!['never', 'half', 'quarter'].includes(policy) || !['double', 'one'].includes(growth)) throw new Error('Choose a displayed resizing policy.');
  let length = 0;
  let capacity = 1;
  let total = 0;
  const states = [{
    operation: 'start',
    length,
    capacity,
    copies: 0,
    cost: 0,
    total,
    credits: 0,
    potential: 0.5,
    event: 'Reserve one slot; no live values.'
  }];
  for (let index = 0; index < operations.length; index++) {
    const operation = operations[index];
    let copies = 0;
    let cost = 1;
    let event;
    const beforeCapacity = capacity;
    if (operation === '+') {
      if (length === capacity) {
        copies = length;
        capacity = growth === 'double' ? 2 * capacity : capacity + 1;
      }
      length++;
      event = copies ? `Grow ${beforeCapacity} → ${capacity}; copy ${copies}, then write the new item.` : 'Write one appended item; spare capacity is available.';
    } else if (length === 0) {
      cost = 0;
      event = 'Empty pop rejected; no counted removal or copies.';
    } else {
      length--;
      const shrink = policy === 'half' ? length <= capacity / 2 : policy === 'quarter' ? length <= capacity / 4 : false;
      if (shrink && capacity > 1) {
        capacity = Math.max(1, Math.floor(capacity / 2));
        copies = length;
      }
      event = beforeCapacity === capacity ? 'Remove the last item; retain this allocation.' : `Remove, then shrink ${beforeCapacity} → ${capacity}; copy ${copies} retained items.`;
    }
    cost += copies;
    total += cost;
    const potential = length >= capacity / 2 ? 2 * length - capacity : capacity / 2 - length;
    states.push({
      operation,
      length,
      capacity,
      copies,
      cost,
      total,
      credits: 3 * (index + 1) - total,
      potential,
      event
    });
  }
  return states;
}
export function denseSetTrace(values, removed) {
  if (!Array.isArray(values) || values.length > 8 || values.some(value => !Number.isInteger(value)) || new Set(values).size !== values.length || !Number.isInteger(removed)) throw new Error('Use up to eight distinct integers and an integer removal.');
  const items = [...values];
  const positions = new Map(items.map((value, index) => [value, index]));
  const states = [];
  const save = action => states.push({
    items: [...items],
    positions: sortedPairs([...positions]),
    action
  });
  save('Each stored value maps back to its exact dense-array position.');
  if (!positions.has(removed)) {
    save('Value absent: preserve both representations.');
    return states;
  }
  const destination = positions.get(removed);
  const last = items.at(-1);
  items[destination] = last;
  positions.set(last, destination);
  save(destination === items.length - 1 ? `Value ${removed} is already last; no different occurrence needs moving.` : `Place final value ${last} in slot ${destination} and update its index. The temporary duplicate disappears in the next step.`);
  items.pop();
  positions.delete(removed);
  save(`Remove the old final slot and the mapping for ${removed}. The remaining values again have exactly one slot each.`);
  return states;
}
