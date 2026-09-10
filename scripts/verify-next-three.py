import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

directory=Path(sys.argv[1]).resolve()
project=Path.cwd().resolve()
fixture=json.loads((directory/'native-fixtures.json').read_text(encoding='utf8'))
(directory/'empty-gitconfig').write_text('',encoding='utf8')
assets=project/'public/learn-assets/plots'
assets.mkdir(parents=True,exist_ok=True)
results={'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'examples':[],'model_cases':{}}

def scratch(name):
    # Retain these isolated fixtures for evidence; no user project is used or deleted.
    return Path(tempfile.mkdtemp(prefix=name+'-',dir=directory))

def execute_example(group,name,example):
    work=scratch(group+'-'+name)
    saved=Path.cwd()
    os.chdir(work)
    namespace={'__name__':'__main__'}
    output=io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(example['code'],name+'.py','exec'),namespace)
        assert output.getvalue().rstrip()==example['output'].rstrip(),(group,name,repr(output.getvalue()))
        if 'artifact' in example:
            artifact=work/example['artifact']
            assert artifact.stat().st_size>100
            shutil.copyfile(artifact,assets/example['artifact'])
            namespace['fig'].savefig(directory/(name+'-native.png'),dpi=120)
            axes=namespace['fig'].axes
            if name in ['residual','transfer']:
                np.testing.assert_allclose(axes[1].collections[0].get_offsets()[:,1],namespace['residual'])
                assert 'Residual' in axes[1].get_ylabel() or 'predicted' in axes[1].get_ylabel()
            if name=='histogram':
                assert sum(namespace['counts'])==len(namespace['latency'])
                np.testing.assert_allclose(sum(namespace['density']*np.diff(namespace['edges'])),1)
            if name=='curves':
                np.testing.assert_array_equal(axes[0].lines[1].get_ydata(),namespace['valid'])
            if name=='bars': assert [p.get_height() for p in axes[0].patches]==[10,20] and axes[0].get_ylim()[0]==0
            if name=='scatter': np.testing.assert_allclose(axes[0].collections[0].get_offsets(),np.column_stack([namespace['size'],namespace['latency']]))
            if name=='heatmap': assert namespace['image'].get_clim()==(0,10) and namespace['image'].origin=='upper'
            if name=='scales': assert axes[1].get_yscale()=='log'
            if name=='uncertainty': np.testing.assert_allclose(axes[0].containers[-1].lines[2][0].get_segments(),[[[0,10],[0,14]],[[1,13],[1,15]]])
    finally:
        os.chdir(saved)
    results['examples'].append(group+'/'+name)
    return namespace

spaces={}
for group,examples in fixture['examples'].items():
    for name,example in examples.items(): spaces[group+'/'+name]=execute_example(group,name,example)

for case in fixture['alignment']:
    expected=case['result']
    orders=pd.DataFrame({'amount':[r['value'] for r in expected['orders']]},index=[r['label'] for r in expected['orders']])
    fees=pd.Series([f['value'] for f in expected['fees']],index=[f['label'] for f in expected['fees']])
    orders['fee']=fees if case['mode']=='labels' else fees.to_numpy()
    orders['total']=orders['amount']+orders['fee']
    for row,actual in zip(expected['rows'],orders.itertuples()):
        for field in ['fee','total']:
            value=getattr(actual,field)
            assert (pd.isna(value) and row[field] is None) or value==row[field]

for case in fixture['cleaning']:
    raw=pd.Series(['10','','bad','0'],dtype='string')
    values=pd.to_numeric(raw,errors='coerce').astype('Float64')
    mask=values.notna() if case['policy']=='known' else values>0
    for row in case['result']:
        i=row['index']
        assert (pd.isna(values[i]) and row['value'] is None) or values[i]==row['value']
        assert (pd.isna(mask[i]) and row['mask'] is None) or bool(mask[i])==row['mask']
    assert values.loc[mask].index.tolist()==[r['index'] for r in case['result'] if r['keep']]

for case in fixture['grouping']:
    frame=pd.DataFrame({'id':[101,102,103,104],'region':['North','North','South',None],'amount':[10,np.nan,30,40]})
    if case['fillZero']: frame['amount']=frame['amount'].fillna(0)
    grouped=frame.groupby('region',dropna=not case['keepMissing'])['amount']
    actual=grouped.agg(['size','count','sum','mean'])
    for group in case['result']['groups']:
        row=actual.loc[group['key']] if group['key'] is not None else actual.loc[actual.index.isna()].iloc[0]
        for key in ['size','count','sum','mean']: assert row[key]==group[key]
    transformed=grouped.transform('mean')
    for a,b in zip(transformed,case['result']['transformed']): assert (pd.isna(a) and b['mean'] is None) or a==b['mean']

def normalize_records(rows):
    def canonical(value):
        return int(value) if isinstance(value,float) and value.is_integer() else value
    return sorted(json.dumps([canonical(r.get(k)) for k in ['order','customer','region','match']]) for r in rows)
for case in fixture['joins']:
    try:
        merged=pd.DataFrame(case['left']).merge(pd.DataFrame(case['right']),on='customer',how=case['how'],validate='many_to_one' if case['validate'] else None,indicator=True).rename(columns={'_merge':'match'})
    except pd.errors.MergeError:
        assert case['result']['error']
    else:
        assert not case['result']['error']
        assert normalize_records(json.loads(merged.to_json(orient='records')))==normalize_records(case['result']['rows'])

for case in fixture['coordinates']:
    model=case['result']
    fig,ax=plt.subplots()
    ax.set_yscale(model['scale']);ax.set_ylim(model['min'],model['max']);fig.canvas.draw()
    for point in model['points']:
        pixel=ax.transData.transform((0,point['value']))
        fraction=ax.transAxes.inverted().transform(pixel)[1]
        np.testing.assert_allclose(fraction,point['fraction'])
    plt.close(fig)
for case in fixture['histograms']:
    expected=case['result'];edges=[b['lo'] for b in expected]+[expected[-1]['hi']]
    values=np.array([1,2,2,3,7,9])
    fig,ax=plt.subplots();heights,_,patches=ax.hist(values,bins=edges,density=case['density'])
    np.testing.assert_allclose(heights,[b['height'] for b in expected])
    np.testing.assert_allclose([p.get_width() for p in patches],[b['width'] for b in expected])
    for b in expected:
        count=np.histogram(values,bins=edges)[0][expected.index(b)]
        assert count==b['count']
    plt.close(fig)
for case in fixture['intervals']:
    model=case['result'];values=np.array([10,12,14]*(2 if case['repeated'] else 1))
    mean=values.mean();sd=values.std(ddof=1);half=sd if case['kind']=='sd' else sd/np.sqrt(len(values))
    np.testing.assert_allclose([mean,sd,half],[model['mean'],model['sd'],model['halfWidth']])
    fig,ax=plt.subplots();bar=ax.errorbar([0],[mean],yerr=[half])
    np.testing.assert_allclose(bar.lines[2][0].get_segments()[0],[[0,model['low']],[0,model['high']]])
    plt.close(fig)

# Independent changes to the new practice/application inputs.
source=fixture['examples']['pandas']['temporal']['code'].split('assert matched')[0]
ns={}
with contextlib.redirect_stdout(io.StringIO()): exec(source.replace('tolerance=3','tolerance=5'),ns)
assert ns['matched']['corrected'].tolist()==[9,9,12]
uptime=spaces['pandas/uptime']['samples'].copy()
uptime=pd.concat([uptime,pd.DataFrame({'sample_id':[7],'device':['C'],'watts':[6]})],ignore_index=True)
assert len(uptime.loc[uptime.device=='C'])==2 and uptime.loc[uptime.device=='C','watts'].notna().sum()==1
assert uptime.loc[(uptime.device=='C')&uptime.watts.notna(),'watts'].gt(0).mean()==1
source=fixture['examples']['pandas']['uptime']['code'].replace('"sample_id": [1, 2, 3, 4, 5, 6]','"sample_id": [1, 2, 3, 4, 5, 5]')
try: exec(source,{})
except ValueError as error: assert str(error)=='duplicate sample ID'
else: raise AssertionError('Duplicate sample accepted')
np.testing.assert_array_equal(np.array([2,4,7,10])-2*np.array([1,2,3,4]),[0,0,1,2])

def git(work,*args,allow_failure=False):
    result=subprocess.run(['git','-C',str(work),*args],text=True,capture_output=True,encoding='utf8')
    if not allow_failure: assert result.returncode==0,(args,result.stderr)
    return result
def repo(name,initial_file='report',text='version 1'):
    work=scratch(name);git(work,'init','-q','-b','main');git(work,'config','user.name','Practice Learner');git(work,'config','user.email','learner@example.invalid');git(work,'config','core.autocrlf','false');git(work,'config','merge.conflictStyle','merge')
    write(work,initial_file,text);commit(work,'Initial');return work
def write(work,name,text): (work/name).write_text(text+'\n',encoding='utf8')
def commit(work,message): git(work,'add','--all');git(work,'commit','-qm',message);return git(work,'rev-parse','HEAD').stdout.strip()
def contents(work,spec): return git(work,'show',spec).stdout.strip()

for case in fixture['staging']:
    work=repo('staging','report.txt')
    def check(i):
        versions=[int(contents(work,'HEAD:report.txt').split()[-1]),int(contents(work,':report.txt').split()[-1]),int((work/'report.txt').read_text().split()[-1])]
        state=case['trace'][i];assert versions==state['versions']
        status=git(work,'status','--short').stdout.rstrip('\r\n');assert (status[:2] if status else '  ')==state['status']
    check(0);write(work,'report.txt','version 2');check(1);git(work,'add','report.txt');check(2);write(work,'report.txt','version 3');check(3)
    if case['option']: git(work,'add','report.txt')
    check(4);git(work,'commit','-qm','Update report');check(5)

for case in fixture['branches']:
    work=repo('branch');ids={'A':git(work,'rev-parse','HEAD').stdout.strip()}
    def check(i):
        state=case['trace'][i]
        for branch in ['main','feature']:
            actual=git(work,'rev-parse','--verify',branch,allow_failure=True)
            assert (actual.stdout.strip() if actual.returncode==0 else None)==(ids[state[branch]] if state[branch] else None)
        assert git(work,'branch','--show-current').stdout.strip()==state['head']
        for node in state['nodes']:
            parents=git(work,'show','-s','--format=%P',ids[node['id']]).stdout.split()
            assert parents==[ids[p] for p in node['parents']]
            assert sorted(git(work,'ls-tree','--name-only',ids[node['id']]).stdout.split())==sorted(node['files'])
    check(0);git(work,'switch','-qc','feature');check(1);write(work,'note','explanation');ids['B']=commit(work,'Add note');check(2);git(work,'switch','-q','main');check(3)
    if case['option']: write(work,'units','ms');ids['C']=commit(work,'Add units')
    check(4);attempt=git(work,'merge','--ff-only','feature',allow_failure=True);assert (attempt.returncode!=0)==case['option'];check(5)
    if case['option']: git(work,'merge','--no-edit','feature');ids['M']=git(work,'rev-parse','HEAD').stdout.strip()
    check(6)

for case in fixture['remotes']:
    work=repo('remote');remote=scratch('bare');git(remote,'init','--bare','-q','-b','main');git(work,'remote','add','origin',str(remote));git(work,'push','-qu','origin','main');ids={'A':git(work,'rev-parse','HEAD').stdout.strip()}
    other=scratch('colleague');git(other,'clone','-q',str(remote),'.');git(other,'config','user.name','Practice Colleague');git(other,'config','user.email','colleague@example.invalid');git(other,'config','core.autocrlf','false')
    def check(i):
        state=case['trace'][i]
        assert git(work,'rev-parse','main').stdout.strip()==ids[state['local']]
        assert git(work,'rev-parse','origin/main').stdout.strip()==ids[state['tracking']]
        assert git(remote,'rev-parse','main').stdout.strip()==ids[state['shared']]
        assert (work/'report').read_text().strip()==state['working']
    check(0);write(other,'report','version 2');ids['B']=commit(other,'Colleague update');git(other,'push','-q');check(1)
    if case['option']: write(work,'report','version 1 + local note');ids['C']=commit(work,'Local note')
    check(2);git(work,'fetch','-q','origin');check(3);attempt=git(work,'merge','--ff-only','origin/main',allow_failure=True);assert (attempt.returncode!=0)==case['option'];check(4)

for case in fixture['conflicts']:
    work=repo('conflict','title.txt','Report')
    def check(i):
        state=case['trace'][i];assert (work/'title.txt').read_text().strip()==state['working']
        assert bool(git(work,'ls-files','--unmerged').stdout)==state['unmerged']
        if state['committed']: assert len(git(work,'show','-s','--format=%P','HEAD').stdout.split())==2
    check(0);git(work,'switch','-qc','feature');write(work,'title.txt','Report by region');commit(work,'Region');git(work,'switch','-q','main');write(work,'title.txt','Report by model');commit(work,'Model');assert git(work,'merge','feature',allow_failure=True).returncode!=0;check(1)
    write(work,'title.txt',case['trace'][2]['working']);check(2);git(work,'add','title.txt');check(3);git(work,'commit','-qm','Resolve title');check(4)

for key in ['alignment','cleaning','grouping','joins','coordinates','histograms','intervals','staging','branches','remotes','conflicts']: results['model_cases'][key]=len(fixture[key])
results['git_model_states']=sum(len(c['trace']) for k in ['staging','branches','remotes','conflicts'] for c in fixture[k])
results['transfer_checks']='temporal tolerance, uptime changed observation and duplicate ID, changed residual'
(directory/'native-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print('PASS:',len(results['examples']),'Python examples;',results['model_cases'],';',results['git_model_states'],'native Git states; independent transfer checks.')
