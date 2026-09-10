"""Independent behavioural oracles; fixtures are generated from the actual lessons."""
import json, math, sys, runpy, itertools, io, subprocess, contextlib
from pathlib import Path
from fractions import Fraction

fixture = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for case in fixture['mean']:
    expected = (case['values'][0] if case['early'] else sum(case['values'])) / len(case['values'])
    assert case['states'][-1]['result'] == expected
    for state in case['states']:
        assert state['total'] == sum(case['values'][:state['consumed']])
for case in fixture['matrix']:
    functions = {'correct':lambda c:Fraction(9,5)*c+32,'offset':lambda c:Fraction(9,5)*c,'slope':lambda c:2*c+32,'constant':lambda c:32}
    data = {'zero':(0,32),'boil':(100,212),'negative':(-40,-40),'difference':(5,18)}
    for row in case['rows']:
        fn=functions[row['id']]
        for check in row['checks']:
            x,wanted=data[check['id']]
            actual=fn(x+10)-fn(x) if check['id']=='difference' else fn(x)
            assert check['pass'] == (actual==wanted)
            assert math.isclose(check['actual'],float(actual),abs_tol=1e-10)
for case in fixture['dependency']:
    assert case['common']==[v for v in range(1,5) if 2<=v<4 and (3<=v<5 if case['modern'] else 1<=v<2)]
for case in fixture['notebook']:
    namespace={};offset=2;saved=None
    for (action,*arg),state in zip(case['actions'],case['states']):
        if action=='edit':offset=arg[0]
        elif action=='restart':namespace={}
        else:
            actions=['inputs','calculate','display'] if action=='all' else [action]
            for a in actions:
                try:
                    if a=='inputs':exec(f'values=[10,20,30]\noffset={offset}',namespace)
                    if a=='calculate':exec('mean=sum(values)/len(values)-offset',namespace)
                    if a=='display':saved=eval('mean',namespace)
                except NameError:
                    if a=='display':saved='NameError'
        assert state['rate']==namespace.get('offset')
        assert state['mean']==namespace.get('mean')
        assert state['output']==saved
        if action=='restart' and saved is not None:
            assert state['counts']==[1,2,3], 'Restart alone does not clear document execution counts'
for case in fixture['random']:
    shared=iter([4,8,1,6]);split=[next(shared)]
    if case['extra']:split.append(next(shared))
    model=next(iter([7,2,9,3])) if case['separate'] else next(shared)
    assert case['split']==split and case['model']==model
for case in fixture['provenance']:
    values=json.loads(case['current']['data'])
    assert case['result']==sum(values)/len(values)-case['current']['offset']
    assert case['hit']==(case['change']=='none' if case['complete'] else True)
print('Independent mean, rational temperature, resolver, namespace, stream and provenance checks passed.')

def load_named(suffix):
    key=next(k for k in fixture['directories'] if k.endswith('/'+suffix))
    code=fixture['examples'][key]['code'];namespace={'__name__':'verification'}
    exec(compile(code,suffix+'.py','exec'),namespace)
    return namespace,Path(fixture['directories'][key])

loader,_=load_named('loader');load_scores=loader['load_scores'];p=Path('scores-oracle.json')
values={'ordinary':{'baseline':.81,'larger':.86},'zero':{'zero':0,'one':1},'equal':{'edge':.85},'invalid':{'ok':.9,'bad':-.1},'boolean':{'flag':True},'nan':{'bad':math.nan}}
for case in fixture['boundary']:
    p.write_text(json.dumps(values[case['kind']]),encoding='utf-8')
    try:result=load_scores(p,minimum=case['minimum'])
    except ValueError:assert case['states'][-1]['error'] is not None
    else:assert result==dict(case['states'][-1]['accepted']) and not case['states'][-1]['error']
for bad in ['[]','null','{"": 0.5}','{"x": "0.5"}','{"x": null}','{"x": Infinity}']:
    p.write_text(bad,encoding='utf-8')
    try:load_scores(p)
    except ValueError:pass
    else:raise AssertionError(bad)
for case in fixture['compatibility']:
    change=case['change'];raw={'edge':.85}
    if change=='rename':
        def revised(path,*,min_score=0):return raw
        try:revised(p,minimum=.85)
        except TypeError:assert case['result'] is None
        else:raise AssertionError('keyword rename accepted')
    else:
        result={k:(v*100 if change=='units' else v) for k,v in raw.items() if (v>.85 if change=='exclusive' else v>=.85)}
        assert result==case['result']
for mode in ['shared','mutate','copy']:
    case=next(c for c in fixture['ownership'] if c['mode']==mode)
    if mode=='shared':
        def add(tag,tags=[]):tags.append(tag);return tags
        a=add('first');b=add('second');assert a is b
        assert a==case['states'][-1]['objects'][0]['values']
    else:
        original=['raw'];result=list(original) if mode=='copy' else original;result.append('checked')
        assert original==case['states'][-1]['objects'][0]['values']
        assert (result is original)==(mode=='mutate')

repair,repair_dir=load_named('testingRepair');fn=repair['positive_mean']
for size in range(1,5):
    for data in itertools.product([-2,0,3,6],repeat=size):
        text='\n'.join(map(str,data));positive=[v for v in data if v>0]
        if positive:assert fn(text)==sum(positive)/len(positive)
        else:
            try:fn(text)
            except ValueError:pass
            else:raise AssertionError(data)
key=next(k for k in fixture['examples'] if k.endswith('/testingRepair'));code=fixture['examples'][key]['code']
for name,broken in [('include-zero',code.replace('if value > 0:', 'if value >= 0:')),('wrong-denominator',code.replace('/ len(positive)','/ len(text.splitlines())'))]:
    file=repair_dir/(name+'.py');file.write_text(broken,encoding='utf-8')
    r=subprocess.run([sys.executable,'-B',str(file)],capture_output=True,text=True)
    assert r.returncode==1 and 'passed: False' in r.stdout
api,api_dir=load_named('apiTransfer');fn=api['durations_ms']
for values in [[],[0],[.001,.002],[1,2,3]]:
    original=list(values);assert fn(iter(values),input_unit='s')==[v*1000 for v in values];assert values==original
for values,unit,error_type in [([True],'ms',TypeError),([-1],'ms',ValueError),([math.nan],'s',ValueError),([1e308],'s',ValueError),([10**1000],'ms',ValueError),([1],'minutes',ValueError)]:
    try:fn(values,input_unit=unit)
    except error_type:pass
    else:raise AssertionError((values,unit))
g=iter([1,-1,2])
try:fn(g)
except ValueError:assert list(g)==[2]
else:raise AssertionError('generator invalid value accepted')
source=Path('typed_duration.py');source.write_text(fixture['examples'][next(k for k in fixture['examples'] if k.endswith('/apiTransfer'))]['code'].split('if __name__')[0],encoding='utf-8')
r=subprocess.run([sys.executable,'-m','mypy','--strict',str(source)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
bad=Path('typed_bad.py');bad.write_text(fixture['examples'][next(k for k in fixture['examples'] if k.endswith('/hints'))]['code'],encoding='utf-8')
r=subprocess.run([sys.executable,'-m','mypy','--strict',str(bad)],capture_output=True,text=True);assert r.returncode==1 and 'arg-type' in r.stdout,r.stdout
print('Native API boundaries, alias identity, 340 selection cases, deliberate-bug detection and mypy acceptance/rejection passed.')

# Repeat the preserved public declaration checks, not just the new transfer API.
for name in ['contract','loader','result','optional']:
    key=next(k for k in fixture['examples'] if k.endswith('/'+name))
    code=fixture['examples'][key]['code'].split('if __name__ == "__main__":')[0]
    file=Path('typed_'+name+'.py');file.write_text(code,encoding='utf-8')
    checked=subprocess.run([sys.executable,'-m','mypy','--strict',str(file)],capture_output=True,text=True)
    assert checked.returncode==0,name+'\n'+checked.stdout+checked.stderr
suite_dir=Path(fixture['directories'][next(k for k in fixture['directories'] if k.endswith('/testingSuite'))])
discovered=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','.','-p','test_*.py'],cwd=suite_dir,capture_output=True,text=True)
assert discovered.returncode==0 and 'Ran 4 tests' in discovered.stderr,discovered.stderr
print('Preserved API declarations pass strict mypy; real unittest discovery runs four tests.')

from nbconvert.preprocessors import ExecutePreprocessor,CellExecutionError
import nbformat
directory=Path(fixture['directories'][next(k for k in fixture['directories'] if k.endswith('/execute'))])
notebook=nbformat.read(directory/'offset-analysis.ipynb',as_version=4)
runner=ExecutePreprocessor(timeout=60,kernel_name='python3',allow_errors=False)
runner.preprocess(notebook,{'metadata':{'path':str(directory)}})
assert notebook.cells[-1].outputs[0].text.strip()=='18.0'
notebook.cells[1].source=notebook.cells[1].source.replace('offset = 2','offset = 5')
try:runner.preprocess(notebook,{'metadata':{'path':str(directory)}})
except CellExecutionError:pass
else:raise AssertionError('stale assertion accepted')
notebook.cells[-1].source=notebook.cells[-1].source.replace('18.0','15.0')
runner.preprocess(notebook,{'metadata':{'path':str(directory)}})
assert notebook.cells[-1].outputs[0].text.strip()=='15.0'
broken=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print(hidden_mean)')])
try:runner.preprocess(broken,{'metadata':{'path':str(directory)}})
except CellExecutionError as error:assert 'NameError' in str(error)
else:raise AssertionError('hidden name existed in fresh kernel')
print('Real fresh kernels passed baseline and changed offset; stale assertions and missing hidden variables failed as intended.')
