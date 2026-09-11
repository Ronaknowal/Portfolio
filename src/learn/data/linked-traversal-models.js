function boundedInteger(value, minimum, maximum, label) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be an integer from ${minimum} to ${maximum}.`);
  }
}
export function validateSuccessors(next, head = next?.length ? 0 : null) {
  if (!Array.isArray(next) || next.length > 32) throw new RangeError('Use at most 32 nodes.');
  for (let index = 0; index < next.length; index += 1) {
    if (!Object.hasOwn(next, index)) throw new TypeError('Successor arrays must contain every position.');
    if (next[index] !== null) boundedInteger(next[index], 0, next.length - 1, 'Successor');
  }
  if (head !== null) boundedInteger(head, 0, next.length - 1, 'Head');
}
export function chainSuccessors(length = 7, entry = 2) {
  boundedInteger(length, 0, 32, 'Length');
  boundedInteger(entry, -1, length - 1, 'Entry');
  const next = Array.from({
    length
  }, (_, index) => index + 1 < length ? index + 1 : null);
  if (length && entry >= 0) next[length - 1] = entry;
  return next;
}
export function linkedCycleTrace(next, head = next?.length ? 0 : null) {
  validateSuccessors(next, head);
  let slow = head;
  let fast = head;
  let rounds = 0;
  let entrySteps = 0;
  const frames = [];
  const save = (phase, note, via = null) => frames.push({
    phase,
    note,
    slow,
    fast,
    via,
    rounds,
    entrySteps
  });
  save('start', 'Both references start at head. Their initial equality is not a cycle test.');
  while (fast !== null && next[fast] !== null) {
    slow = next[slow];
    const via = next[fast];
    fast = next[via];
    rounds += 1;
    save('detect', 'Advance slow by one edge and fast by two; compare their node identities only now.', via);
    if (slow === fast) {
      const meeting = slow;
      save('meeting', 'A positive meeting proves a reachable cycle. This node need not be its entry.');
      slow = head;
      save('reset', 'Reset slow to head. Keep fast at the meeting; both now advance one edge.');
      while (slow !== fast) {
        slow = next[slow];
        fast = next[fast];
        entrySteps += 1;
        save('entry-walk', 'Take one edge with each reference; the reset reference approaches the cycle entry.');
      }
      const entry = slow;
      let cycleLength = 1;
      for (let current = next[entry]; current !== entry; current = next[current]) cycleLength += 1;
      save('entry', `Both references identify entry n${entry}. Prefix length ${entrySteps}; cycle length ${cycleLength}.`);
      return {
        next: [...next],
        head,
        frames,
        entry,
        meeting,
        prefixLength: entrySteps,
        cycleLength
      };
    }
  }
  save('acyclic', 'The two-edge guard fails at the end. No reachable cycle; return None.');
  return {
    next: [...next],
    head,
    frames,
    entry: null,
    meeting: null,
    prefixLength: null,
    cycleLength: 0
  };
}
export function middleSplitState(length, policy = 'second', cut = false) {
  boundedInteger(length, 0, 12, 'Length');
  if (!['first', 'second'].includes(policy) || typeof cut !== 'boolean') throw new TypeError('Choose a middle policy and a boolean cut state.');
  const next = chainSuccessors(length, -1);
  let slow = length ? 0 : null;
  let fast = slow;
  const frames = [{
    slow,
    fast
  }];
  const canMove = () => fast !== null && next[fast] !== null && (policy === 'second' || next[next[fast]] !== null);
  while (canMove()) {
    slow = next[slow];
    fast = next[next[fast]];
    frames.push({
      slow,
      fast
    });
  }
  const splitAfter = length ? Math.floor((length - 1) / 2) : null;
  const right = splitAfter !== null ? next[splitAfter] : null;
  if (cut && splitAfter !== null) next[splitAfter] = null;
  return {
    length,
    policy,
    frames,
    middle: slow,
    splitAfter,
    leftHead: length ? 0 : null,
    rightHead: cut ? right : null,
    next,
    leftSize: Math.ceil(length / 2),
    rightSize: Math.floor(length / 2),
    cut
  };
}
