export function decoratorOrderTrace(outer = 'cap', value = 8) {
  const inner = outer === 'cap' ? 'double' : 'cap', order = [outer, inner], states = [];
  const save = (active, result, note) => states.push({ order, active, result, note });
  save('caller', null, `The name points to ${outer}(${inner}(reading)). Each wrapper changes the returned result, not the input.`);
  save(outer, null, `Enter outer ${outer}; it must call the next layer to obtain a result.`);
  save(inner, null, `Enter inner ${inner}; it calls reading(${value}).`);
  save('body', value, `The original body returns ${value}. The result now travels out through wrappers.`);
  let result = inner === 'cap' ? Math.min(value, 10) : value * 2;
  save(inner, result, `${inner === 'cap' ? 'Limit the returned value to at most 10' : 'Multiply the returned value by 2'}: ${value} → ${result}.`);
  const previous = result; result = outer === 'cap' ? Math.min(result, 10) : result * 2;
  save(outer, result, `${outer === 'cap' ? 'Limit the returned value to at most 10' : 'Multiply the returned value by 2'}: ${previous} → ${result}.`);
  save('caller', result, `The caller receives ${result}. Swap the wrappers while keeping the input fixed to compare meaning.`);
  return states;
}

export function contextTrace(path = 'success', suppress = false) {
  const states = [], events = [];
  const save = (phase, resource, error, note) => states.push({ phase, resource, error, events: [...events], note });
  save('before', 'not acquired', null, 'Construct the manager. The with statement has not entered it yet.');
  events.push('enter');
  if (path === 'enter-fails') {
    save('enter', 'not acquired', 'ValueError', '__enter__ raises before acquiring anything. The as target is never assigned.');
    events.push('caught'); save('outside', 'not acquired', 'caught ValueError', 'This manager never entered successfully, so its __exit__ is not called. Its setup must clean any partial acquisition itself.');
    return states;
  }
  save('enter', 'open', null, '__enter__ returns the file. as file binds that returned resource, which differs from the manager.');
  events.push('body'); save('body', 'open', path === 'body-fails' ? 'ValueError' : null, path === 'body-fails' ? 'The first body action raises ValueError. Statements after it in this body are skipped.' : 'The body reads its input and reaches the end normally.');
  events.push('exit'); save('exit', 'closed', path === 'body-fails' ? 'ValueError' : null, `__exit__ closes the resource. It receives ${path === 'body-fails' ? 'ValueError and exception details' : 'three None values'}.`);
  const handled = path === 'body-fails' && !suppress;
  events.push(handled ? 'caught' : 'after');
  save('outside', 'closed', handled ? 'caught ValueError' : null, handled ? 'False from __exit__ leaves the exception active; the outer handler catches it.' : path === 'body-fails' ? 'True suppresses the error. Continue AFTER with; never resume the interrupted body.' : 'After normal exit, continue after with. The resource is closed.');
  return states;
}

export function exitStackTrace(fail = 'C') {
  const stack = [], events = [], states = [];
  const save = (active, note) => states.push({ stack: [...stack], events: [...events], active, note });
  save(null, 'ExitStack is empty. Only successful entries register an exit action.');
  for (const name of ['A', 'B', 'C']) {
    events.push(`acquire ${name}`);
    if (name === fail) { save(name, `${name} fails before acquisition. Its exit action is not registered; already entered resources still need cleanup.`); break; }
    stack.push(name); save(name, `${name} entered successfully. Register its exit above the previously entered resources.`);
  }
  if (fail === 'none') { events.push('body'); save(null, 'All three entered. Run the body, then leave the stack.'); }
  while (stack.length) { const name = stack.pop(); events.push(`release ${name}`); save(name, `Release ${name}, the most recently entered resource still active.`); }
  events.push(fail === 'none' ? 'after' : 'caught'); save(null, fail === 'none' ? 'All resources released; continue normally.' : 'All registered exits ran. The original acquisition failure reaches the outer handler.');
  return states;
}
