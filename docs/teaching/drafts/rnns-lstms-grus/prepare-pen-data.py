"""Reproduce the small, attributed UCI pen-trajectory teaching extract."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO, StringIO
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip"


def main():
    with urlopen(URL, timeout=30) as response:
        payload = response.read()
    archive = ZipFile(BytesIO(payload))
    source_arrays = {}
    report = {"url": URL, "archive_sha256": hashlib.sha256(payload).hexdigest(),
              "attribution": "E. Alpaydin and F. Alimoglu, Pen-Based Recognition of Handwritten Digits, UCI, DOI10.24432/C5MG6K",
              "license": "CC BY 4.0, current UCI dataset record", "sources": {}}
    for filename in ("pendigits.tra", "pendigits.tes"):
        data = archive.read(filename)
        array = np.loadtxt(StringIO(data.decode("ascii")), delimiter=",", dtype=int)
        assert array.shape[1] == 17 and np.isfinite(array).all()
        source_arrays[filename] = array
        report["sources"][filename] = {"rows": len(array), "sha256": hashlib.sha256(data).hexdigest(),
            "unique_coordinate_vectors": len(np.unique(array[:, :16], axis=0))}
    original_train = {tuple(row) for row in source_arrays["pendigits.tra"][:, :16]}
    original_test = {tuple(row) for row in source_arrays["pendigits.tes"][:, :16]}
    report["source_cross_split_duplicate_vectors"] = len(original_train & original_test)
    seen = set()
    rows = []
    for filename, partition, count in (("pendigits.tra", "train", 60), ("pendigits.tes", "development", 30)):
        array = source_arrays[filename]
        selected_counts = {digit: 0 for digit in range(10)}
        skipped = 0
        for row_index, row in enumerate(array):
            digit = int(row[-1])
            if selected_counts[digit] >= count:
                continue
            coordinates = tuple(row[:16])
            if coordinates in seen:
                skipped += 1
                continue
            seen.add(coordinates)
            selected_counts[digit] += 1
            rows.append([partition, f"{filename}:{row_index + 1}"] + row.tolist())
        assert all(value == count for value in selected_counts.values())
        report["sources"][filename]["selected_per_class"] = count
        report["sources"][filename]["selected_duplicate_candidates_skipped"] = skipped
    header = ["partition", "source_id"] + [f"{axis}{time + 1}" for time in range(8) for axis in ("x", "y")] + ["digit"]
    csv = ",".join(header) + "\n" + "\n".join(",".join(str(value) for value in row) for row in rows) + "\n"
    output = ROOT / "pen-trajectories.csv"
    output.write_text(csv, encoding="utf-8")
    report["extract_rows"] = len(rows)
    report["extract_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT / "data-extraction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
