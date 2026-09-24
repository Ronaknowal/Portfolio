"""Independent native checks for the fixed examples shown by WorkflowVisuals."""
import json, os, shlex, subprocess, sys, time
from pathlib import Path
import nbformat

fixture_path=Path(sys.argv[1]).resolve()
fixture=json.loads(fixture_path.read_text(encoding='utf-8'))
root=fixture_path.parent/('native-'+str(time.time_ns()))
root.mkdir()
env={**os.environ,'GIT_CONFIG_GLOBAL':str(root/'empty-gitconfig'),'GIT_CONFIG_NOSYSTEM':'1','GIT_TERMINAL_PROMPT':'0','LC_ALL':'C'}
(root/'empty-gitconfig').write_text('',encoding='utf-8')

def run(argv,cwd=None,ok=True):
    result=subprocess.run(list(map(str,argv)),cwd=cwd or root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=20)
    if ok: assert result.returncode==0,(argv,result.stdout,result.stderr)
    return result

# The picture's three values are independently computed, then serialized.
namespace={}
exec('values=[10,20,30]\noffset=2\nmean=sum(values)/len(values)-offset',namespace)
source='values=[10,20,30]\noffset=5'
assert (namespace['offset'],namespace['mean'])==(2,18.0)
notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source),nbformat.v4.new_code_cell('print(mean)',outputs=[nbformat.v4.new_output('stream',name='stdout',text='18.0\n')])])
nbformat.write(notebook,root/'edited-with-old-output.ipynb')
saved=nbformat.read(root/'edited-with-old-output.ipynb',as_version=4)
assert 'offset=5' in saved.cells[0].source and saved.cells[1].outputs[0].text=='18.0\n'
exec(source,namespace)
assert namespace['mean']==18.0
exec('mean=sum(values)/len(values)-offset',namespace)
assert namespace['mean']==15.0

# Same source is executed and checked separately, as the diagram shows.
hint_file=root/'hint_example.py';hint_file.write_text(fixture['hints']['code'],encoding='utf-8')
assert run([sys.executable,hint_file]).stdout.rstrip()==fixture['hints']['output'].rstrip()
checked=run([sys.executable,'-m','mypy','--strict','--cache-dir',root/'mypy-cache',hint_file],ok=False)
assert checked.returncode==1 and 'incompatible type "str"' in checked.stdout,checked.stdout

# Real Bash supplies argument boundaries and exit statuses, without Python emulating parsing.
bash=Path(os.environ.get('LESSON_BASH','C:/Program Files/Git/bin/bash.exe'))
argdir=root/'arguments';argdir.mkdir()
for name in ['a.csv','run alpha.csv']:(argdir/name).write_text('',encoding='utf-8')
for case in fixture['arguments']:
    # Set the fixture inside Bash: Windows/MSYS argument conversion can expand
    # a wildcard before a -c positional argument reaches the Bash script.
    script='value='+shlex.quote(case['value'])+'; show_args(){ printf "%s\\n" "$#"; for arg in "$@"; do printf "<%s>\\n" "$arg"; done; }; show_args '+('"$value"' if case['quoted'] else '$value')
    actual=run([bash,'--noprofile','--norc','-c',script,'fixture'],cwd=argdir).stdout.splitlines()
    assert actual==[str(len(case['argv'])),*[f'<{value}>' for value in case['argv']]],(case,actual)
for case in fixture['pipelines']:
    script='p(){ printf "row 1\\nrow 2\\n"; return "$1"; }; c(){ cat > received.txt; return "$1"; }; if [[ $3 == yes ]]; then set -o pipefail; fi; p "$1" | c "$2"; printf "%s\\n" "$?"'
    actual=run([bash,'--noprofile','--norc','-c',script,'fixture',case['a'],case['b'],'yes' if case['strict'] else 'no'])
    assert actual.stdout.strip()==str(case['expected'])
    assert (root/'received.txt').read_text()=='row 1\nrow 2\n'

# Local disposable repositories reproduce both divergent/nondivergent histories.
def git(cwd,*args,ok=True):return run(['git','-c','user.name=Lesson verification','-c','user.email=lesson@example.invalid',*args],cwd=cwd,ok=ok)
for changed,states in enumerate(fixture['remote']):
    parent=root/f'git-{changed}';parent.mkdir();shared=parent/'shared.git';colleague=parent/'colleague';local=parent/'local'
    git(parent,'init','--bare','--initial-branch=main',shared)
    git(parent,'clone',shared,colleague)
    (colleague/'report.txt').write_text('version 1',encoding='utf-8');git(colleague,'add','report.txt');git(colleague,'commit','-m','A')
    a=git(colleague,'rev-parse','HEAD').stdout.strip();git(colleague,'push','origin','main');git(parent,'clone',shared,local)
    ids={a:'A'}
    def capture():
        return {'local':ids[git(local,'rev-parse','main').stdout.strip()],'tracking':ids[git(local,'rev-parse','origin/main').stdout.strip()],'shared':ids[git(shared,'rev-parse','main').stdout.strip()],'working':(local/'report.txt').read_text(encoding='utf-8')}
    observed=[capture()]
    (colleague/'report.txt').write_text('version 2',encoding='utf-8');git(colleague,'add','report.txt');git(colleague,'commit','-m','B')
    ids[git(colleague,'rev-parse','HEAD').stdout.strip()]='B';git(colleague,'push','origin','main');observed.append(capture())
    if changed:
        (local/'report.txt').write_text('version 1 + local note',encoding='utf-8');git(local,'add','report.txt');git(local,'commit','-m','C');ids[git(local,'rev-parse','HEAD').stdout.strip()]='C'
    observed.append(capture());git(local,'fetch','origin');observed.append(capture())
    merge=git(local,'merge','--ff-only','origin/main',ok=False);assert (merge.returncode!=0)==bool(changed)
    observed.append(capture())
    assert observed==[{key:state[key] for key in observed[0]} for state in states]

result={'python':sys.version.split()[0],'bash':run([bash,'--version']).stdout.splitlines()[0],'git':run(['git','--version']).stdout.strip(),'checks':['notebook source/live/stored output and recomputation','API native execution and independent mypy rejection','6 Bash argument cases','8 Bash pipeline status cases with emitted bytes retained','10 Git remote states across ordinary and divergent histories'],'artifacts':str(root)}
(fixture_path.parent/'native-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
