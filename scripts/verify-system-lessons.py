"""Run only authored lesson snippets, in disposable native-Linux fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if sys.platform != "linux" or os.geteuid() == 0:
    raise SystemExit("Run under Linux as a non-root user to check real permission semantics.")
cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in cases:
    with tempfile.TemporaryDirectory(prefix="system-lesson-") as folder:
        inherited = {key: value for key, value in os.environ.items()
                     if not key.startswith(("GIT_", "BASH_FUNC_"))}
        env = {**inherited, "HOME": folder, "XDG_CONFIG_HOME": folder,
               "TMPDIR": folder, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
               "GIT_EDITOR": "true", "GIT_PAGER": "cat", "COLUMNS": "80", "LINES": "24",
               "PATH": "/usr/bin:/bin"}
        # Avoid an inherited Git context or shell startup hook reaching another repository.
        for name in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "BASH_ENV", "ENV"]:
            env.pop(name, None)
        result = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-e", "-c", case["code"]],
                                cwd=folder, env=env, text=True, capture_output=True, timeout=90)
        if result.returncode or result.stdout.rstrip("\n") != case["output"].rstrip("\n"):
            print(json.dumps({"id": case["id"], "exit": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}, indent=2))
            raise SystemExit("Example mismatch")
        print(case["id"] + ": exact stdout verified")
print(f"{len(cases)} native-Linux examples passed; all fixtures removed.")
