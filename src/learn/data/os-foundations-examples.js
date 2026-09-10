export const osExamples={
  child:{code:`import json
import os
import subprocess
import sys

value = 7
child_code = """
import json, os
value = 99
print(json.dumps({"pid": os.getpid(), "value": value}))
"""
completed = subprocess.run(
    [sys.executable, "-c", child_code],
    capture_output=True, text=True, check=True, timeout=5,
)
message = json.loads(completed.stdout)
print("different process:", message["pid"] != os.getpid())
print("child value:", message["value"])
print("parent value:", value)
print("exit status:", completed.returncode)`,output:`different process: True
child value: 99
parent value: 7
exit status: 0`},
  translation:{code:`PAGE_SIZE = 16
tables = {
    "A": {0: (0, "r"), 1: (3, "rw"), 2: (None, "rw")},
    "B": {0: (0, "r"), 1: (5, "rw"), 2: (None, "rw")},
}

def translate(process, address, write=False):
    if address < 0:
        return "unmapped"
    page, offset = divmod(address, PAGE_SIZE)
    entry = tables[process].get(page)
    if entry is None:
        return "unmapped"
    frame, rights = entry
    if write and "w" not in rights:
        return "protection fault"
    if frame is None:
        return "valid page: fault service required"
    return frame * PAGE_SIZE + offset

for process in ["A", "B"]:
    print(process, "virtual 22 ->", translate(process, 22))
print("write code:", translate("A", 6, write=True))
print("read page 2:", translate("A", 38))
print("read page 3:", translate("A", 54))
assert translate("A", 31) == 63
assert translate("A", 32) == "valid page: fault service required"`,output:`A virtual 22 -> 54
B virtual 22 -> 86
write code: protection fault
read page 2: valid page: fault service required
read page 3: unmapped`},
  mappings:{platform:'linux',code:`# Run as a standalone Python script in Linux, not in a threaded notebook.
import mmap
import os

page = os.sysconf("SC_PAGE_SIZE")
with mmap.mmap(-1, page, flags=mmap.MAP_PRIVATE,
               prot=mmap.PROT_READ | mmap.PROT_WRITE) as private, \\
     mmap.mmap(-1, page, flags=mmap.MAP_SHARED,
               prot=mmap.PROT_READ | mmap.PROT_WRITE) as shared:
    private[0:1] = b"A"
    shared[0:1] = b"A"
    pid = os.fork()
    if pid == 0:
        private[0:1] = b"P"
        shared[0:1] = b"S"
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError("child did not finish successfully")
    print("parent private:", private[0:1].decode("ascii"))
    print("parent shared:", shared[0:1].decode("ascii"))`,output:`parent private: A
parent shared: S`},
  worker:{code:`import subprocess
import sys

worker = """
import sys
try:
    value = int(sys.stdin.read())
except ValueError:
    print("invalid integer", file=sys.stderr)
    sys.exit(2)
print(value * value)
"""

def run_job(raw):
    result = subprocess.run(
        [sys.executable, "-c", worker], input=raw,
        text=True, capture_output=True, timeout=5,
    )
    if result.returncode != 0:
        return {"ok": False, "status": result.returncode,
                "error": result.stderr.strip()}
    return {"ok": True, "value": int(result.stdout)}

for raw in ["3", "bad", "0"]:
    result = run_job(raw)
    if result["ok"]:
        print(repr(raw), "->", result["value"])
    else:
        print(repr(raw), "-> failed", result["status"], result["error"])`,output:`'3' -> 9
'bad' -> failed 2 invalid integer
'0' -> 0`},
};
