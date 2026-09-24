export const measurementValidator = String.raw`import csv
import math
import re

FIELDS = ["sample_id", "temperature", "unit", "site"]

def read_measurements(path):
    records = []
    seen = set()
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != FIELDS:
            raise ValueError("unexpected header or field order")
        for record_number, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"record {record_number}: wrong field count")
            sample_id = row["sample_id"]
            if not re.fullmatch(r"[0-9]{3}", sample_id) or sample_id in seen:
                raise ValueError(f"record {record_number}: invalid or duplicate ID")
            seen.add(sample_id)
            raw = row["temperature"]
            try:
                value = None if raw == "" else float(raw)
            except ValueError as error:
                raise ValueError(f"record {record_number}: invalid numeric text {raw!r}") from error
            if value is not None and (not math.isfinite(value) or not -80 <= value <= 80):
                raise ValueError(f"record {record_number}: invalid temperature")
            if row["unit"] != "C" or not row["site"].strip():
                raise ValueError(f"record {record_number}: invalid unit or site")
            records.append({"sample_id": sample_id, "temperature": value,
                            "unit": "C", "site": row["site"]})
    if not records:
        raise ValueError("empty batch")
    return records`;

const measurementFixture = `sample_id,temperature,unit,site
001,18.5,C,"room,north"
002,,C,room south
003,0,C,room south
`;

export const fileExamples = {
  parse: {
    filename: 'parse_fields.py',
    code: String.raw`import csv
import io

text = 'sample_id,temperature,site\n001,18.5,"room,north"\n'
row = next(csv.DictReader(io.StringIO(text)))
print(row)
print(type(row["sample_id"]).__name__, type(row["temperature"]).__name__)
print(float(row["temperature"]) + 1)`,
    output: `{'sample_id': '001', 'temperature': '18.5', 'site': 'room,north'}
str str
19.5`,
  },
  roundtrip: {
    files: { 'measurement_io.py': measurementValidator },
    filename: 'roundtrip_data.py',
    code: `import json
from pathlib import Path
from tempfile import TemporaryDirectory
from measurement_io import read_measurements

source = ${JSON.stringify(measurementFixture)}
with TemporaryDirectory() as folder:
    root = Path(folder)
    raw = root / "measurements.csv"
    raw.write_text(source, encoding="utf-8")
    records = read_measurements(raw)
    envelope = {"schema_version": 1, "temperature_unit": "C",
                "records": records}
    output = root / "measurements.json"
    output.write_text(json.dumps(envelope, allow_nan=False), encoding="utf-8")
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored == envelope
    values = [row["temperature"] for row in restored["records"]
              if row["temperature"] is not None]
    print([row["sample_id"] for row in restored["records"]])
    print("rows:", len(records), "measured:", len(values))
    mean = sum(values) / len(values) if values else None
    print("mean:", mean, restored["temperature_unit"])
    print("semantic round trip:", restored == envelope)`,
    output: "['001', '002', '003']\nrows: 3 measured: 2\nmean: 9.25 C\nsemantic round trip: True",
  },
  missing: {
    filename: 'missing_batch.py',
    code: String.raw`from pathlib import Path
from tempfile import TemporaryDirectory
from measurement_io import read_measurements

source = "sample_id,temperature,unit,site\n001,,C,lab\n002,,C,lab\n"
with TemporaryDirectory() as folder:
    path = Path(folder) / "missing.csv"
    path.write_text(source, encoding="utf-8")
    rows = read_measurements(path)
    values = [row["temperature"] for row in rows if row["temperature"] is not None]
    mean = sum(values) / len(values) if values else None
    print("rows:", len(rows))
    print("mean:", "unavailable" if mean is None else mean)`,
    output: 'rows: 2\nmean: unavailable',
  },
  array: {
    filename: 'array_storage.py',
    code: String.raw`import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

values = np.array([[18.5, 20.0], [19.0, 21.5]], dtype=np.float32)
metadata = {"schema_version": 1, "axes": ["minute", "sensor"],
            "minute_values": [0, 1], "sensor_ids": ["001", "002"], "unit": "C"}
with TemporaryDirectory() as folder:
    root = Path(folder)
    np.save(root / "values.npy", values, allow_pickle=False)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    restored = np.load(root / "values.npy", allow_pickle=False)
    restored_meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert restored.dtype == values.dtype
    assert restored.shape == values.shape
    assert np.array_equal(restored, values)
    assert restored_meta == metadata
    print(restored.shape, restored.dtype)
    print(restored.tolist())
    print(restored_meta["axes"], restored_meta["unit"])`,
    output: "(2, 2) float32\n[[18.5, 20.0], [19.0, 21.5]]\n['minute', 'sensor'] C",
  },
  publish: {
    filename: 'publish_data.py',
    code: String.raw`import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as folder:
    root = Path(folder)
    destination = root / "report.json"
    temporary = root / "report.pending.json"
    destination.write_text('{"version": 1, "mean": 19.0}', encoding="utf-8")
    replacement = {"version": 2, "mean": 23.0}
    temporary.write_text(json.dumps(replacement, allow_nan=False), encoding="utf-8")
    restored = json.loads(temporary.read_text(encoding="utf-8"))
    if restored != replacement:
        raise ValueError("round-trip check failed")
    print("before:", json.loads(destination.read_text(encoding="utf-8"))["version"])
    os.replace(temporary, destination)
    print("after:", json.loads(destination.read_text(encoding="utf-8"))["version"])
    print("staging name exists:", temporary.exists())`,
    output: 'before: 1\nafter: 2\nstaging name exists: False',
  },
  practice: {
    filename: 'check_new_batch.py',
    code: String.raw`from pathlib import Path
from tempfile import TemporaryDirectory
from measurement_io import read_measurements

source = "sample_id,temperature,unit,site\n004,0,C,lab\n005,,C,lab\n006,24,C,lab\n"
with TemporaryDirectory() as folder:
    path = Path(folder) / "new.csv"
    path.write_text(source, encoding="utf-8")
    rows = read_measurements(path)
    values = [r["temperature"] for r in rows if r["temperature"] is not None]
    print("rows:", len(rows), "missing:", len(rows) - len(values))
    print("mean:", sum(values) / len(values))
    path.write_text(source.replace("006,24,C", "004,24,C"), encoding="utf-8")
    try:
        read_measurements(path)
    except ValueError:
        print("duplicate rejected")`,
    output: 'rows: 3 missing: 1\nmean: 12.0\nduplicate rejected',
  },
};
