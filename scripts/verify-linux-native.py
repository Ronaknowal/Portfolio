"""Verify authored examples in temporary Linux fixtures; never use project files as fixtures."""
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

if sys.platform != "linux" or os.geteuid() == 0:
    raise SystemExit("Run in Linux as an ordinary non-root user.")
cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

def environment(folder):
    return {"HOME": folder, "TMPDIR": folder, "PATH": "/usr/bin:/bin", "LC_ALL": "C"}

for case in cases["examples"]:
    with tempfile.TemporaryDirectory(prefix="linux-teaching-") as folder:
        result = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-e", "-c", case["code"]], cwd=folder, env=environment(folder), capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, (case["id"], result.stderr)
        assert result.stdout.rstrip("\n") == case["output"].rstrip("\n"), (case["id"], result.stdout, case["output"])
    print("PASS example " + case["id"])

for case in cases["streams"]:
    with tempfile.TemporaryDirectory(prefix="linux-streams-") as folder:
        result = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", case["code"]], cwd=folder, env=environment(folder), capture_output=True, text=True, timeout=10)
        assert (result.returncode, result.stdout, result.stderr) == (case["status"], case["stdout"], case["stderr"]), (case["id"], result)
        assert {p.name: p.read_text() for p in Path(folder).iterdir()} == case["files"], case["id"]
    print("PASS actual Bash routing " + case["id"])

with tempfile.TemporaryDirectory(prefix="linux-permissions-") as folder:
    directory = Path(folder) / "data"
    directory.mkdir()
    target = directory / "run.csv"
    target.write_text("latency_ms\n10\n20\n")
    try:
        for mask in range(8):
            # Exercise the same applicable permission bits as fixture owner;
            # no chown, privileged account, or filesystem-wide changes needed.
            directory.chmod(0o700)
            target.chmod(0o400 if mask & 1 else 0)
            directory.chmod((0o400 if mask & 4 else 0) | (0o100 if mask & 2 else 0))
            try:
                target.read_text()
                readable = True
            except PermissionError:
                readable = False
            try:
                os.listdir(directory)
                listable = True
            except PermissionError:
                listable = False
            assert readable == ((mask & 3) == 3), ("read", mask)
            assert listable == bool(mask & 4), ("list", mask)
        directory.chmod(0o700)
        target.chmod(0o047)
        try:
            target.read_text()
            raise AssertionError("Owner incorrectly fell back to other permissions")
        except PermissionError:
            pass
    finally:
        directory.chmod(0o700)
        target.chmod(0o600)
print("PASS 8 real permission gate states and owner-class no-fallback")

with tempfile.TemporaryDirectory(prefix="linux-paths-") as folder:
    root = Path(folder)
    current = root / "project/data/raw"
    current.mkdir(parents=True)
    (root / "project/reports").mkdir()
    (current / "run 1.csv").write_text("latency_ms\n10\n20\n")
    for path, operation, ok, expected in cases["pathCases"]:
        if path in ("/../../", ""):
            continue  # Root clamp/empty argument covered separately below.
        actual = str(root) + path if path.startswith("/") else str(current) + "/" + path
        try:
            info = os.stat(actual)
            if operation == "cd" and not __import__("stat").S_ISDIR(info.st_mode):
                raise NotADirectoryError(errno.ENOTDIR, "directory required")
            assert ok, path
            assert str(Path(actual).resolve()).removeprefix(str(root)) == expected, path
        except OSError as error:
            assert not ok, (path, error)
            assert errno.errorcode[error.errno] == expected, (path, error)
    assert os.path.samefile("/../../", "/")
    try:
        os.stat("")
        raise AssertionError("empty pathname unexpectedly resolved")
    except FileNotFoundError:
        pass
print("PASS real pathname resolution and root/empty cases")

child = subprocess.Popen(["sleep", "60"])
try:
    signal.alarm(10)
    child.send_signal(signal.SIGSTOP)
    pid, status = os.waitpid(child.pid, os.WUNTRACED)
    assert os.WIFSTOPPED(status)
    child.send_signal(signal.SIGCONT)
    pid, status = os.waitpid(child.pid, os.WCONTINUED)
    assert os.WIFCONTINUED(status)
    child.terminate()
    assert child.wait(timeout=5) == -signal.SIGTERM
finally:
    signal.alarm(0)
    if child.poll() is None:
        child.kill()
        child.wait(timeout=5)
print("PASS owned child STOP/CONT/TERM lifecycle; temporary fixtures removed")
