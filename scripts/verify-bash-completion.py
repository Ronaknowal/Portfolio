"""Run bounded lesson fixtures as an ordinary Linux user, only in owned temp space."""
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

if sys.platform != 'linux' or os.geteuid() == 0:
    raise SystemExit('Use native Linux as an ordinary non-root user.')
fixture=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root=Path(tempfile.mkdtemp(prefix='bash-completion-')).resolve()
env={'HOME':str(root),'TMPDIR':str(root),'PATH':'/usr/bin:/bin','LC_ALL':'C'}
checks=[]
def run(argv,cwd=None,**kwargs):
    return subprocess.run(argv,cwd=cwd or root,env=env,capture_output=True,text=True,timeout=10,**kwargs)
def bash(code,*args,cwd=None):
    return run(['/bin/bash','--noprofile','--norc','-c',code,'fixture',*map(str,args)],cwd=cwd)
def write(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8')
def assert_clean(path):
    assert not list(path.glob('.report.*')) and not list(path.glob('.collection.*')), list(path.iterdir())
try:
    version=bash('printf "%s" "$BASH_VERSION"').stdout
    for builtin in ['set','trap','read','source','local']:
        help_result=bash('help "$1"',builtin)
        assert help_result.returncode==0 and help_result.stdout
    for name,example in fixture['bashExamples'].items():
        script=root/(name+'.sh');write(script,example['code'])
        assert run(['/bin/bash','-n',str(script)]).returncode==0
        result=run(['/bin/bash','--noprofile','--norc',str(script)])
        assert result.returncode==0,(name,result.stderr)
        assert result.stdout.strip()==fixture['outputs'][name].strip(),(name,result.stdout)
        checks.append('exact '+name)
    argdir=root/'arguments';argdir.mkdir()
    for name in ['a.csv','run alpha.csv']:write(argdir/name,'')
    for case in fixture['argumentsCases']:
        code='value=$1; python3 -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' '+('"$value"' if case['quoted'] else '$value')
        result=bash(code,case['value'],cwd=argdir)
        assert result.returncode==0 and json.loads(result.stdout)==case['argv'],case
    for case in fixture['pipelineCases']:
        code='p(){ printf "partial\\n"; return "$1"; }; c(){ cat >/dev/null; return "$1"; }; if [[ $3 == yes ]]; then set -o pipefail; fi; p "$1" | c "$2"; printf "%s\\n" "$?"'
        result=bash(code,case['producer'],case['consumer'],'yes' if case['pipefail'] else 'no')
        assert result.stdout.strip()==str(case['expected']),case
    # Quoted input containing shell-looking text stays data, not a new program.
    literal='$(touch should-not-exist); echo text'
    result=bash('value=$1; printf "%s\\n" "$value"',literal)
    assert result.stdout.rstrip('\n')==literal and not (root/'should-not-exist').exists()
    # Defaults, required-argument checks and subshell directory ownership.
    assert bash('printf "<%s>|<%s>" "${1:-7}" "${1-7}"','').stdout=='<7>|<>'
    assert bash('printf "<%s>|<%s>" "${1:-7}" "${1-7}"').stdout=='<7>|<7>'
    assert bash('before=$PWD; (cd /); [[ $PWD == "$before" ]]').returncode==0
    # Caller-owned source executes in the current shell; execution as a child does not.
    write(root/'settings.sh','label=changed\n')
    assert bash('label=original; bash settings.sh; printf "%s\\n" "$label"; source settings.sh; printf "%s\\n" "$label"').stdout=='original\nchanged\n'

    project=root/'report project';project.mkdir()
    for name,code in fixture['reportFiles'].items():write(project/name,code)
    write(project/'collect-reports.sh',fixture['summaryPractice'])
    for name in ['run-report.sh','collect-reports.sh']:assert run(['/bin/bash','-n',str(project/name)]).returncode==0
    out=project/'outputs';out.mkdir()
    destination=out/'report.json'
    report_script=project/'run-report.sh'
    def report(input_name,output=destination):return run(['/bin/bash',str(report_script),str(project/input_name),str(output)])
    result=report('run alpha.csv')
    assert result.returncode==0 and result.stdout=='' and 'published:' in result.stderr
    assert json.loads(destination.read_text())=={'count':3,'mean':4.0}
    original=destination.read_bytes()
    write(project/'zero.csv','value\n0\n-2\n2\n')
    assert report('zero.csv',out/'zero.json').returncode==0
    assert json.loads((out/'zero.json').read_text())=={'count':3,'mean':0.0}
    # CSV parser, schema and number boundaries, including a parser field-size error.
    invalids={'invalid.csv':None,'header.csv':'other\n2\n','extra.csv':'value\n2,3\n','text.csv':'value\nbad\n','huge.csv':'value\n'+'2'*200000+'\n'}
    for name,contents in invalids.items():
        if contents is not None:write(project/name,contents)
        failed=report(name)
        assert failed.returncode==4 and failed.stdout=='',(name,failed.returncode,failed.stderr)
        assert destination.read_bytes()==original and 'report failed:' in failed.stderr
        assert_clean(out)
    assert report('missing.csv').returncode==2 and destination.read_bytes()==original
    assert report('run alpha.csv',out).returncode==2
    assert run(['/bin/bash',str(report_script)]).returncode==2
    assert report('empty.csv',out/'empty.json').returncode==0
    assert json.loads((out/'empty.json').read_text())=={'count':0,'mean':None}
    assert report('run alpha.csv').returncode==0 and destination.read_bytes()==original
    assert_clean(out)

    # Independent collection: order, emptiness, invalid input, stable replacement.
    collection=out/'collection.json'
    collector=project/'collect-reports.sh'
    inputs=[destination,out/'empty.json']
    def collect(paths):return run(['/bin/bash',str(collector),str(collection),*map(str,paths)])
    assert collect(inputs).returncode==0
    expected=[dict(source=str(inputs[0]),count=3,mean=4.0),dict(source=str(inputs[1]),count=0,mean=None)]
    assert json.loads(collection.read_text())==expected
    saved=collection.read_bytes()
    assert collect(inputs).returncode==0 and collection.read_bytes()==saved
    assert collect(inputs[::-1]).returncode==0 and json.loads(collection.read_text())==expected[::-1]
    saved=collection.read_bytes()
    cases=[True,{'count':True,'mean':2},{'count':0,'mean':0},{'count':2,'mean':None},{'count':2,'mean':float('inf')},{'count':-1,'mean':2},{}]
    for value in cases:
        write(out/'bad report.json',json.dumps(value))
        result=collect([destination,out/'bad report.json'])
        assert result.returncode==4 and collection.read_bytes()==saved,(value,result.stderr)
        assert_clean(out)
    write(out/'bad report.json','{unfinished')
    assert collect([out/'bad report.json']).returncode==4 and collection.read_bytes()==saved
    assert collect([out/'missing.json']).returncode==4 and collection.read_bytes()==saved
    assert collect([]).returncode==2 and collection.read_bytes()==saved

    # Publication model compared to filesystem visibility through each phase.
    for failure,states in zip([False,True],fixture['publication']):
        public=root/('public-'+str(failure));write(public,'OLD')
        staged=root/('stage-'+str(failure));assert public.read_text()=='OLD' and states[0]['staged']=='absent'
        write(staged,'PART');assert public.read_text()=='OLD' and states[1]['visible']=='previous complete report'
        if not failure:write(staged,'NEW')
        assert public.read_text()=='OLD'
        old_reader=public.open()
        if failure:staged.unlink()
        else:os.replace(staged,public)
        assert public.read_text()==('OLD' if failure else 'NEW')
        assert old_reader.read()=='OLD';old_reader.close()
        assert not staged.exists()

    # The real wrapper must discard even a maliciously incomplete worker's stdout.
    write(project/'report.py','import sys\nprint("{partial")\nraise SystemExit(4)\n')
    assert report('run alpha.csv').returncode==4 and destination.read_bytes()==original
    assert_clean(out)
    # TERM targets only the newly started fixture process group; never a user job.
    write(project/'report.py','import sys,time\nprint("{partial",flush=True)\ntime.sleep(30)\n')
    child=subprocess.Popen(['/bin/bash',str(report_script),str(project/'run alpha.csv'),str(destination)],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:
        for _ in range(200):
            candidates=list(out.glob('.report.*/result.json'))
            if candidates and candidates[0].read_text().startswith('{partial'):break
            time.sleep(.01)
        else:raise AssertionError('fixture worker did not reach staging')
        os.killpg(child.pid,signal.SIGTERM)
        stdout,stderr=child.communicate(timeout=5)
        assert child.returncode==143,(child.returncode,stderr)
        assert destination.read_bytes()==original
        assert_clean(out)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.communicate(timeout=5)
    summary={'platform':platform.platform(),'python':platform.python_version(),'bash':version,'uid':os.geteuid(),'exactPrograms':5,'modelConfigurations':16,'nativePolicies':'defaults, literal-data nonexecution, source/subshell scope, pipeline statuses, schema failure, empty/zero, repeated runs, order, partial producer rejection, TERM cleanup','reportInvalidCases':5,'collectionInvalidCases':9,'success':True}
    print(json.dumps(summary,indent=2))
finally:
    # Linux-only: delete precisely the directory created and owned by this verifier.
    assert root.parent==Path(tempfile.gettempdir()).resolve() and root.name.startswith('bash-completion-')
    shutil.rmtree(root)
