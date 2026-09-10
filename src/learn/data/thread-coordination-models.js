export function initialRace(locked = false) {
  return { value: 0, locked, owner: null, workers: { A: { phase: 0, local: null }, B: { phase: 0, local: null } }, log: [] };
}

export function raceStep(input, id) {
  if (!['A', 'B'].includes(id)) throw new Error('Unknown worker');
  const state = structuredClone(input), worker = state.workers[id];
  if (worker.phase === 3) return state;
  if (state.locked && state.owner && state.owner !== id) {
    state.log.push(`${id} waits for the lock held by ${state.owner}; no read occurs.`); return state;
  }
  if (worker.phase === 0) { if (state.locked) state.owner = id; worker.local = state.value; state.log.push(`${id} reads shared ${state.value} into its local snapshot${state.locked ? ' while holding the lock' : ''}.`); }
  if (worker.phase === 1) { worker.local += 1; state.log.push(`${id} computes local ${worker.local}; shared value is still ${state.value}.`); }
  if (worker.phase === 2) { state.value = worker.local; state.log.push(`${id} writes shared ${state.value}${state.locked ? ' and releases the lock' : ''}.`); if (state.locked) state.owner = null; }
  worker.phase += 1;
  return state;
}

export function initialLocks(ordered = false) {
  return { owners: { L1: null, L2: null }, workers: { A: { pc: 0, needs: ['L1', 'L2'], waiting: null }, B: { pc: 0, needs: ordered ? ['L1', 'L2'] : ['L2', 'L1'], waiting: null } }, log: [] };
}

export function lockStep(input, id) {
  if (!['A', 'B'].includes(id)) throw new Error('Unknown worker');
  const state = structuredClone(input), worker = state.workers[id];
  if (worker.pc === 3) return state;
  if (worker.pc === 2) {
    for (const name of worker.needs) state.owners[name] = null;
    worker.pc = 3; worker.waiting = null; state.log.push(`${id} finishes the protected operation and releases both locks.`); return state;
  }
  const name = worker.needs[worker.pc];
  if (state.owners[name]) { worker.waiting = name; state.log.push(`${id} waits for ${name}, held by ${state.owners[name]}.`); }
  else { state.owners[name] = id; worker.pc += 1; worker.waiting = null; state.log.push(`${id} acquires ${name}.`); }
  return state;
}

export function waitEdges(state) {
  return Object.entries(state.workers).flatMap(([id, worker]) => worker.waiting && state.owners[worker.waiting] ? [[id, state.owners[worker.waiting], worker.waiting]] : []);
}

export function deadlocked(state) {
  const edges = waitEdges(state);
  return edges.some(([a,b]) => edges.some(([c,d]) => a === d && b === c));
}

export function conditionTrace(scenario = 'empty') {
  const stages = [{ action: 'Consumer holds the condition lock and checks queue', queue: [], owner: 'consumer', consumer: 'predicate false' },
    { action: 'wait() atomically releases the lock and starts waiting', queue: [], owner: 'none', consumer: 'waiting' }];
  if (scenario === 'empty') stages.push(
    { action: 'Producer notifies without adding an item', queue: [], owner: 'producer', consumer: 'notified; cannot reacquire yet' },
    { action: 'Producer releases; consumer reacquires and checks again', queue: [], owner: 'consumer', consumer: 'still false → wait again' });
  else if (scenario === 'item') stages.push(
    { action: 'Producer adds item 7 and notifies', queue: [7], owner: 'producer', consumer: 'notified; cannot reacquire yet' },
    { action: 'Producer releases; consumer reacquires and checks', queue: [7], owner: 'consumer', consumer: 'predicate true' },
    { action: 'Consumer takes 7 while still holding the lock', queue: [], owner: 'consumer', consumer: 'consumed 7, then releases' });
  else if (scenario === 'stolen') stages.push(
    { action: 'Producer adds item 7 and notifies', queue: [7], owner: 'producer', consumer: 'notified' },
    { action: 'After release, another consumer acquires first and takes 7', queue: [], owner: 'other consumer', consumer: 'waiting to reacquire' },
    { action: 'Original consumer reacquires and checks again', queue: [], owner: 'consumer', consumer: 'false again → wait again' });
  else throw new Error('Unknown condition scenario');
  return stages;
}
