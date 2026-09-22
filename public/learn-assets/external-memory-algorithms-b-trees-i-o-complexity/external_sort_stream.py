"""Bounded fan-in sort of signed 64-bit records; output remains on disk.

The caller supplies a new output pathname. Existing destinations are rejected.
This is temporary-file resource management, not a crash-durable database commit.
"""
from contextlib import ExitStack
from heapq import merge
from itertools import islice
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
import struct


def read_records(stream, buffer_records):
    while block := stream.read(8 * buffer_records):
        if len(block) % 8:
            raise ValueError('truncated signed-64-bit record')
        for (value,) in struct.iter_unpack('<q', block):
            yield value


def write_records(stream, values, buffer_records):
    iterator = iter(values)
    while block := list(islice(iterator, buffer_records)):
        stream.write(struct.pack(f'<{len(block)}q', *block))


def sort_integer_file(source, destination, chunk_records=4096, fan_in=8, buffer_records=256):
    """O(chunk_records + fan_in*buffer_records) live record storage plus run metadata.
    At most fan_in input streams and one output stream are open during a merge.
    CPU sort/merge work is O(N log N); each merge level reads/writes all records.
    """
    if chunk_records < 1 or fan_in < 2 or buffer_records < 1:
        raise ValueError('positive chunk/buffer sizes and fan_in >= 2 required')
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('choose a new destination distinct from the input')
    count = 0
    passes = 0
    with TemporaryDirectory(prefix='sorted-runs-', dir=destination.parent) as directory:
        workspace = Path(directory)
        serial = 0

        def new_run():
            nonlocal serial
            serial += 1
            return workspace / f'run-{serial}.bin'

        runs = []
        with source.open('rb') as incoming:
            records = read_records(incoming, buffer_records)
            while chunk := list(islice(records, chunk_records)):
                count += len(chunk)
                chunk.sort()  # Sort this chunk in place; do not duplicate it with sorted().
                target = new_run()
                with target.open('wb') as outgoing:
                    write_records(outgoing, chunk, buffer_records)
                runs.append(target)
        initial_runs = len(runs)
        while len(runs) > 1:
            next_runs = []
            for start in range(0, len(runs), fan_in):
                group = runs[start:start + fan_in]
                if len(group) == 1:
                    next_runs.extend(group)  # Carry; do not rewrite an unchanged singleton.
                    continue
                target = new_run()
                with ExitStack() as resources:
                    incoming = [resources.enter_context(path.open('rb')) for path in group]
                    outgoing = resources.enter_context(target.open('wb'))
                    streams = [read_records(stream, buffer_records) for stream in incoming]
                    write_records(outgoing, merge(*streams), buffer_records)
                for path in group:
                    path.unlink()  # Consumed files are closed before reclaiming their space.
                next_runs.append(target)
            runs = next_runs
            passes += 1
        # Exclusive creation protects an existing file even if it appears mid-sort.
        # Copying in bounded blocks avoids loading the sorted result into Python.
        with destination.open('xb') as outgoing:
            if runs:
                with runs[0].open('rb') as incoming:
                    copyfileobj(incoming, outgoing, length=8 * buffer_records)
    return {'records': count, 'initial_runs': initial_runs, 'merge_levels': passes}


def main():
    values = [8, -1, 8, 2, 0, -1, 3, 12, 4, 4, -9, 7, 6]
    with TemporaryDirectory(prefix='sort-demo-') as directory:
        source, destination = Path(directory) / 'input.bin', Path(directory) / 'sorted.bin'
        with source.open('wb') as stream:
            write_records(stream, values, 3)
        print('sort:', sort_integer_file(source, destination, chunk_records=4, fan_in=2, buffer_records=2))
        with destination.open('rb') as stream:
            # Materializing is only the tiny demonstration's independent oracle.
            actual = list(read_records(stream, 3))
        assert actual == sorted(values)
        print('result:', actual)


if __name__ == '__main__':
    main()
