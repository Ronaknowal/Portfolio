export function cursorTrace(shared = false, empty = false) {
  const values = empty ? [] : [18, 21, 24], positions = [0, 0], output = [], states = [];
  const save = note => states.push({ values, positions: [...positions], output: [...output], note });
  save(shared ? 'a and b refer to the same cursor. Its next position starts at 0.' : 'iter(values) was called twice. a and b own independent positions in the same unchanged list.');
  for (const name of ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']) {
    const c = shared ? 0 : name === 'a' ? 0 : 1, p = positions[c];
    const value = p < values.length ? values[p] : 'END';
    if (p < values.length) positions[c] += 1;
    if (shared) positions[1] = positions[0];
    output.push(`${name}: ${value}`);
    save(value === 'END' ? `${name} has reached the end. Another next call does not rewind it.` : `${name} returns index ${p}, value ${value}. ${shared ? 'Both names now observe the advanced cursor.' : 'Only this cursor advances; the other keeps its own position.'}`);
  }
  return states;
}

export function generatorTrace(action = 'exhaust') {
  const states = [], output = [];
  const save = (status, remaining, position, note) => states.push({ status, remaining, position, output: [...output], note });
  save('CREATED', null, 'Body has not started', 'Calling countdown(2) creates a generator. Even the assignment remaining = 2 has not executed.');
  if (action === 'close-created') { save('CLOSED', null, 'Body never started', 'Closing this unstarted generator does not run its body or its finally block.'); return states; }
  output.push(2); save('SUSPENDED', 2, 'Paused at yield remaining', 'The first next runs setup and reaches yield. The caller receives 2; the local remaining is still 2.');
  if (action === 'close-started') { save('CLOSED', null, 'finally runs during close()', 'close injects GeneratorExit at the suspension point. finally runs and execution ends without yielding another value.'); return states; }
  output.push(1); save('SUSPENDED', 1, 'Paused at yield remaining', 'Resuming executes remaining -= 1, tests the loop and yields 1. Receiving the last item has not yet finished the generator.');
  save('CLOSED', null, 'Loop ends; finally runs', 'One more request decrements to 0, exits the loop, runs finally, and signals StopIteration.');
  save('CLOSED', null, 'Already exhausted', 'A further next returns the chosen END default. No body statement runs again.');
  return states;
}

export function pipelineTrace(limit = 2, bad = false) {
  const lines = ['18', '', bad ? 'bad' : '24', '30'], states = [], received = [];
  let read = 0, active = 'consumer', item = '', error = null;
  const save = note => states.push({ lines, read, active, item, received: [...received], error, note });
  save('The stages exist, but no line has been requested. The final consumer controls demand.');
  while (received.length < limit && read < lines.length) {
    active = 'request'; item = ''; save(`Consumer requests result ${received.length + 1}. Demand travels upstream.`);
    do {
      active = 'source'; item = lines[read++]; save(`Source returns line ${read}: ${item === '' ? 'a blank line' : item}.`);
      active = 'filter'; save(item === '' ? 'The blank yields nothing. This same request must pull another source line.' : 'The stripped line is nonempty, so it can reach the numeric stage.');
    } while (item === '' && read < lines.length);
    active = 'parse';
    if (!Number.isFinite(Number(item))) { error = 'ValueError'; save('Parsing bad fails during consumption. This model propagates the error; it never silently skips a malformed reading.'); break; }
    item = Number(item); save(`Convert the nonblank string to numeric value ${item}.`);
    received.push(item); active = 'consumer'; save(`Deliver ${item}. The consumer has ${received.length} of its ${limit} requested results.`);
  }
  if (!error) { active = 'stopped'; save(`The consumer stops after ${limit} results. ${lines.length - read} source line(s) were never requested.`); }
  return states;
}
