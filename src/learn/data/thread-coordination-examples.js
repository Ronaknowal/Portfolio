export const threadExamples = {
 race:{title:'Force a lost update, then protect the invariant',code:`from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

total = 0
both_read = Barrier(2, timeout=5)

def broken_increment():
    global total
    snapshot = total
    both_read.wait()  # Both have read zero before either may write.
    total = snapshot + 1

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs = [pool.submit(broken_increment) for _ in range(2)]
    for job in jobs:
        job.result()  # Also surfaces any worker exception.
print("unprotected:", total)
assert total == 1

total = 0
lock = Lock()

def protected_increments():
    global total
    for _ in range(1000):
        with lock:
            snapshot = total
            total = snapshot + 1

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs = [pool.submit(protected_increments) for _ in range(2)]
    for job in jobs:
        job.result()
print("protected:", total)
assert total == 2000`,expected:'unprotected: 1\nprotected: 2000'},
 condition:{title:'Wait for stored state, including early publication',code:`from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event

def receive_one(publish_first):
    condition = Condition()
    entered = Event()
    items = [7] if publish_first else []

    def consumer():
        with condition:
            entered.set()
            if not condition.wait_for(lambda: bool(items), timeout=5):
                raise TimeoutError("no item arrived")
            return items.pop(0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(consumer)
        if not entered.wait(timeout=5):
            raise TimeoutError("consumer did not start")
        if not publish_first:
            with condition:
                items.append(7)
                condition.notify()
        return result.result(timeout=6)

print(receive_one(False))
print(receive_one(True))`,expected:'7\n7'},
 transfer:{title:'Acquire account locks in one stable order',code:`from concurrent.futures import ThreadPoolExecutor
from threading import Lock

balances = {"A": 100, "B": 100}
locks = {name: Lock() for name in balances}

def transfer(source, destination, amount):
    if source == destination or amount <= 0:
        raise ValueError("need two accounts and a positive amount")
    first, second = sorted((source, destination))
    with locks[first]:
        with locks[second]:
            if balances[source] < amount:
                return False
            balances[source] -= amount
            balances[destination] += amount
            return True

def repeat(source, destination):
    return sum(transfer(source, destination, 1) for _ in range(50))

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs = [pool.submit(repeat, "A", "B"), pool.submit(repeat, "B", "A")]
    moved = [job.result(timeout=5) for job in jobs]
assert moved == [50, 50]
assert balances == {"A": 100, "B": 100}
print(balances, "total:", sum(balances.values()))`,expected:"{'A': 100, 'B': 100} total: 200"},
 futures:{title:'Collect results and errors from finite file tasks',code:`from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from tempfile import TemporaryDirectory

def read_count(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data)

with TemporaryDirectory() as directory:
    root = Path(directory)
    for name, text in {"a.json": "[1, 2]", "b.json": "[]", "bad.json": "{"}.items():
        (root / name).write_text(text, encoding="utf-8")
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(read_count, path): path.name for path in root.iterdir()}
        for job in as_completed(jobs):
            try:
                records.append((jobs[job], job.result()))
            except json.JSONDecodeError:
                records.append((jobs[job], "invalid JSON"))
    for record in sorted(records):
        print(record)`,expected:"('a.json', 2)\n('b.json', 0)\n('bad.json', 'invalid JSON')"},
 cancellation:{title:'Request shutdown and collect its outcome',code:`from concurrent.futures import ThreadPoolExecutor
from threading import Event

ready = Event()
stop = Event()

def worker():
    ready.set()
    # Stand-in for an interruptible wait between bounded work units.
    requested = stop.wait(timeout=5)
    return "stopped cooperatively" if requested else "deadline reached"

with ThreadPoolExecutor(max_workers=1) as pool:
    result = pool.submit(worker)
    try:
        if not ready.wait(timeout=5):
            raise TimeoutError("worker did not start")
    finally:
        stop.set()
    print(result.result(timeout=6))`,expected:'stopped cooperatively'},
 queue:{title:'Independent solution: bounded sensor-record validation',code:`from queue import Queue
from threading import Thread
import math

STOP = object()
work = Queue(maxsize=2)
results = Queue()

def validate(value):
    if type(value) not in (int, float):
        raise ValueError("expected finite number")
    try:
        result = value * 2
        if not math.isfinite(value) or not math.isfinite(result):
            raise ValueError("expected finite number and result")
    except OverflowError as error:
        raise ValueError("number outside supported range") from error
    return result

def worker():
    while True:
        task = work.get()
        try:
            if task is STOP:
                return
            index, value = task
            try:
                result = validate(value)
            except ValueError as error:
                results.put((index, "error", str(error)))
            else:
                results.put((index, "ok", result))
        finally:
            work.task_done()

def run(values, workers=2):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    threads = [Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for task in enumerate(values):
        work.put(task)
    for _ in threads:
        work.put(STOP)
    work.join()  # Every get, including STOP, reaches task_done.
    for thread in threads:
        thread.join()
    return sorted(results.get_nowait() for _ in values)

if __name__ == "__main__":
    records = run([2, 0, "bad", -3])
    for record in records:
        print(record)
    assert [record[0] for record in records] == [0, 1, 2, 3]
    assert run([]) == []`,expected:"(0, 'ok', 4)\n(1, 'ok', 0)\n(2, 'error', 'expected finite number')\n(3, 'ok', -6)"},
};
