import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

fixtures=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
examples=fixtures['examples']
with tempfile.TemporaryDirectory(prefix='thread-lesson-') as directory:
    root=Path(directory)
    for name,example in examples.items():
        script=root/f'lesson_{name}.py'
        script.write_text(example['code'],encoding='utf-8')
        run=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=20)
        assert run.returncode==0,(name,run.stderr)
        assert run.stdout.strip()==example['expected'].strip(),(name,run.stdout,example['expected'])
        assert not run.stderr,(name,run.stderr)
    # A separate native process checks changed inputs without depending on prose.
    checks=root/'changed.py'
    checks.write_text('''import math
import threading
from queue import Empty
import queue_example as lesson
before={thread.ident for thread in threading.enumerate()}
cases=[[], [0], [-3,0,2], ["bad",None,True], [float("nan"),float("inf"),1e308],list(range(25))]
checked=0
for workers in (1,2,3):
    for values in cases:
        records=lesson.run(values,workers)
        assert [r[0] for r in records]==list(range(len(values)))
        for (_,status,value),source in zip(records,values):
            valid=type(source) in (int,float) and math.isfinite(source) and math.isfinite(source*2)
            assert status==("ok" if valid else "error")
            if valid: assert value==source*2
        assert not lesson.work.unfinished_tasks
        assert lesson.work.empty() and lesson.results.empty()
        assert {thread.ident for thread in threading.enumerate()}==before
        checked+=1
for workers in (0,-1,True):
    try: lesson.run([],workers)
    except ValueError: pass
    else: raise AssertionError("invalid worker count accepted")
print("native changed queue cases:",checked,"; no leaked workers or unfinished items")
''',encoding='utf-8')
    (root/'queue_example.py').write_text(examples['queue']['code'],encoding='utf-8')
    run=subprocess.run([sys.executable,str(checks)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    print(run.stdout.strip())
print(f'PASS: all {len(examples)} displayed thread programs on Python {sys.version.split()[0]}; exact outputs, explicit errors, 18 changed queue cases, native subprocess timeouts.')
