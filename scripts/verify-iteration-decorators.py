"""Execute full displayed programs and compare browser models with Python protocols.

No files, subprocesses, signals, installations, or network effects in examples.
"""
import contextlib
import inspect
import io
import itertools
import json
import platform
from pathlib import Path

fixture = json.loads(Path('scratch/iteration-decorators-review/fixtures.json').read_text(encoding='utf-8'))
namespaces = {}
for group, examples in fixture['examples'].items():
    for key, example in examples.items():
        namespace = {'__name__': '__main__'}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(example['code'], f'<{group}.{key}>', 'exec'), namespace)
        assert output.getvalue().strip() == example['output'].strip(), (group, key, output.getvalue(), example['output'])
        namespaces[group, key] = namespace

for case in fixture['cursors']:
    values = [] if case['empty'] else [18, 21, 24]
    a = iter(values)
    b = a if case['shared'] else iter(values)
    output = []
    for index, name in enumerate(['a', 'b'] * 4, start=1):
        output.append(f'{name}: {next(a if name == "a" else b, "END")}')
        assert case['states'][index]['output'] == output
        assert case['states'][index]['positions'] == [len(values)-a.__length_hint__(), len(values)-b.__length_hint__()]
    assert values == ([] if case['empty'] else [18, 21, 24])

for case in fixture['generators']:
    cleanup = []
    def countdown():
        remaining = 2
        try:
            while remaining > 0:
                yield remaining
                remaining -= 1
        finally:
            cleanup.append('cleaned')
    stream = countdown()
    yielded = []
    operations = ['close'] if case['action'] == 'close-created' else ['next', 'close'] if case['action'] == 'close-started' else ['next'] * 4
    for index, operation in enumerate(['created'] + operations):
        if operation == 'close': stream.close()
        elif operation == 'next':
            value = next(stream, 'END')
            if value != 'END': yielded.append(value)
        state = case['states'][index]
        assert inspect.getgeneratorstate(stream) == 'GEN_' + state['status']
        if state['remaining'] is not None:
            assert stream.gi_frame.f_locals['remaining'] == state['remaining']
        assert yielded == state['output']
    assert cleanup == ([] if case['action'] == 'close-created' else ['cleaned'])

for case in fixture['pipelines']:
    lines = ['18', '', 'bad' if case['bad'] else '24', '30']
    read = []
    def source():
        for number, line in enumerate(lines, 1):
            read.append(number)
            yield line
    def readings(source):
        for line in source:
            text = line.strip()
            if text: yield float(text)
    received = []
    error = None
    stream = readings(source())
    for count in range(case['limit']):
        try: received.append(next(stream))
        except ValueError: error = 'ValueError'; break
        delivered = [s for s in case['states'] if s['active'] == 'consumer' and len(s['received']) == len(received)][-1]
        assert delivered['read'] == len(read)
        assert delivered['received'] == received
    end = case['states'][-1]
    assert (end['read'], end['received'], end['error']) == (len(read), received, error)

for case in fixture['orders']:
    ns = namespaces['decorators','order']
    result = ns['cap_outside' if case['outer'] == 'cap' else 'double_outside'](case['value'])
    assert result == case['states'][-1]['result']
    # Independent arithmetic verifies every returned intermediate value.
    v = case['value']
    inner_result = v * 2 if case['outer'] == 'cap' else min(v, 10)
    assert [s['result'] for s in case['states']] == [None, None, None, v, inner_result, result, result]

for case in fixture['contexts']:
    events = []
    class Manager:
        def __enter__(self):
            events.append('enter')
            if case['path'] == 'enter-fails': raise ValueError('enter')
            self.file = io.StringIO('18')
            return self.file
        def __exit__(self, *error):
            self.file.close()
            events.append('exit')
            return case['suppress']
    manager = Manager()
    try:
        with manager as file:
            events.append('body')
            if case['path'] == 'body-fails': raise ValueError('body')
        events.append('after')
    except ValueError:
        events.append('caught')
    assert events == case['states'][-1]['events']
    assert not hasattr(manager,'file') or manager.file.closed
    for state in case['states']:
        assert events[:len(state['events'])] == state['events']

for case in fixture['stacks']:
    events = []
    @contextlib.contextmanager
    def resource(name):
        events.append('acquire ' + name)
        if name == case['fail']: raise ValueError(name)
        try: yield name
        finally: events.append('release ' + name)
    try:
        with contextlib.ExitStack() as stack:
            for name in 'ABC': stack.enter_context(resource(name))
            events.append('body')
        events.append('after')
    except ValueError: events.append('caught')
    assert events == case['states'][-1]['events']
    for state in case['states']:
        assert state['events'] == events[:len(state['events'])]

# Changed-input oracles for independent tasks, not copied expected strings.
alarm = namespaces['iteration','alarm']['first_crossing']
alarm_cases = 0
for size in range(5):
    for values in itertools.product([0, None, 18, 24], repeat=size):
        cutoff = next((i for i,v in enumerate(values) if v is None or v > 20), len(values))
        expected = (cutoff, values[cutoff]) if cutoff < len(values) and values[cutoff] is not None else None
        cursor = iter(values)
        assert alarm(cursor,20) == expected
        assert list(cursor) == list(values[min(cutoff+1,len(values)):])
        alarm_cases += 1

batches = namespaces['iteration','iterBatches']['batches']
for count in range(12):
    for width in range(1,7):
        assert list(batches(iter(range(count)),width)) == [tuple(range(count)[i:i+width]) for i in range(0,count,width)]
for width in [0,-1,True,False,1.5,'3']:
    stream = batches([],width)  # Validation is intentionally deferred.
    try: next(stream)
    except ValueError: pass
    else: raise AssertionError(('missing size validation',width))

restore = namespaces['decorators','restore']['temporary_value']
restore_cases = 0
for present in [False,True]:
    for original in [None,0,'C',[]]:
        for failure in [False,True]:
            settings = {'key':original} if present else {}
            try:
                with restore(settings,'key','F'):
                    with restore(settings,'key','K'): assert settings['key'] == 'K'
                    assert settings['key'] == 'F'
                    settings.pop('key')
                    if failure: raise ValueError('body')
            except ValueError: assert failure
            assert ('key' in settings) == present
            if present: assert settings['key'] is original
            restore_cases += 1

counted = namespaces['decorators','decoratorPractice']['counted']
@counted
def checked(value, *, multiplier=2):
    """A forwarding contract."""
    if value is None: raise ValueError('missing')
    return value * multiplier
assert checked(3,multiplier=4) == 12
try: checked(None)
except ValueError: pass
else: raise AssertionError('exception swallowed')
assert checked.calls == 2 and checked.__name__ == 'checked' and checked.__doc__ == 'A forwarding contract.'

summary = {'python':platform.python_version(),'programs':25,'modelConfigurations':27,'alarmChangedCases':alarm_cases,'batchChangedCases':78,'restoreChangedCases':restore_cases,'wrapperContract':'returned value, keywords, attempted failure count, metadata, propagated error passed'}
Path('scratch/iteration-decorators-review/native-results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
