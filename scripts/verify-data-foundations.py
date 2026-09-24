"""Execute all displayed examples and compare browser models to native behavior."""
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import numpy as np

fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = {"python": sys.version.split()[0], "numpy": np.__version__, "sqlite": sqlite3.sqlite_version,
           "examples": [], "csv_cases": 0, "join_variants": 0, "publication_traces": 0, "transaction_traces": 0}
with TemporaryDirectory() as folder:
    root = Path(folder)
    for filename, content in fixture["modules"].items():
        (root / filename).write_text(content, encoding="utf-8")
    for name, example in fixture["examples"].items():
        script = root / example["filename"]
        script.write_text(example["code"], encoding="utf-8")
        completed = subprocess.run([sys.executable, str(script)], cwd=root, text=True, encoding="utf-8", capture_output=True)
        assert completed.returncode == 0, (name, completed.stderr)
        assert completed.stdout.strip() == example["output"].strip(), (name, completed.stdout, example["output"])
        results["examples"].append(name)

    spec = importlib.util.spec_from_file_location("measurement_io", root / "measurement_io.py")
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    for case in fixture["measurements"]:
        parsed = list(csv.reader(io.StringIO(case["csv"])))
        assert parsed[1:] == case["rows"], case["id"]
        source = root / "input.csv"
        source.write_text(case["csv"], encoding="utf-8")
        try:
            records = importer.read_measurements(source)
            native_valid = True
            assert records[0]["sample_id"] == "001"
            assert records[1]["temperature"] is None
            assert records[2]["temperature"] == 0
        except ValueError:
            native_valid = False
        assert native_valid == case["valid"], case["id"]
        results["csv_cases"] += 1
    header = "sample_id,temperature,unit,site\n"
    for bad in [header, "wrong,header\n001,1\n", header+"001,1,C\n", header+"001,1,C,lab,extra\n",
                header+"001,inf,C,lab\n", header+"001,81,C,lab\n", header+"001,0,C, \n", header+"001,   ,C,lab\n",
                header+"１２３,0,C,lab\n"]:
        source.write_text(bad, encoding="utf-8")
        try:
            importer.read_measurements(source)
            raise AssertionError("Invalid schema accepted: " + bad)
        except ValueError:
            pass
        results["csv_cases"] += 1
    source.write_text(header+"001,-80,C,lab\n002,80,C,lab\n003,0,C,lab\n", encoding="utf-8")
    assert [r["temperature"] for r in importer.read_measurements(source)] == [-80,80,0]
    results["csv_cases"] += 1

    for case in fixture["publications"]:
        trace = case["trace"]
        destination, staged = root / "published.csv", root / "staged.csv"
        destination.write_text(trace[0]["destination"], encoding="utf-8")
        if staged.exists():
            staged.unlink()
        target = staged if case["strategy"] == "replace" else destination
        states = [(destination.read_text(encoding="utf-8"), None)]
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.flush()
            states.append((destination.read_text(encoding="utf-8"), staged.read_text(encoding="utf-8") if staged.exists() else None))
            handle.write("sample_id,temperature\n001,22\n")
            handle.flush()
            states.append((destination.read_text(encoding="utf-8"), staged.read_text(encoding="utf-8") if staged.exists() else None))
            if not case["fail"]:
                handle.write("002,24\n")
        states.append((destination.read_text(encoding="utf-8"), staged.read_text(encoding="utf-8") if staged.exists() else None))
        if not case["fail"] and case["strategy"] == "replace":
            os.replace(staged, destination)
            states.append((destination.read_text(encoding="utf-8"), None))
        assert states == [(s["destination"],s["temporary"]) for s in trace], case
        results["publication_traces"] += 1

    for case in fixture["joins"]:
        options = case["options"]
        con = sqlite3.connect(":memory:")
        con.executescript("CREATE TABLE sensors(sensor_id TEXT,room TEXT); CREATE TABLE readings(reading_id INTEGER,sensor_id TEXT,minute INTEGER,value REAL);")
        sensors = [("A","north"),("B","south"),("C","spare")]
        if options["duplicate"]:
            sensors.append(("A","annex"))
        con.executemany("INSERT INTO sensors VALUES (?,?)", sensors)
        con.executemany("INSERT INTO readings VALUES (?,?,?,?)", [(1,"A",0,18),(2,"A",10,22),(3,"B",0,None)])
        join = "LEFT JOIN" if options["join"] == "left" else "INNER JOIN"
        tail = f"FROM sensors s {join} readings r ON s.sensor_id=r.sensor_id"
        tail += " AND r.minute <= ?" if options["placement"] == "on" else " WHERE r.minute <= ?"
        pairs = con.execute("SELECT s.rowid,s.sensor_id,r.reading_id,r.value "+tail+" ORDER BY s.rowid,r.reading_id", (options["cutoff"],)).fetchall()
        grouped = con.execute("SELECT s.sensor_id,COUNT(*),COUNT(r.reading_id),COUNT(r.value),AVG(r.value) "+tail+" GROUP BY s.sensor_id ORDER BY s.sensor_id", (options["cutoff"],)).fetchall()
        assert [list(r) for r in pairs] == case["pairs"], options
        assert [list(r) for r in grouped] == case["grouped"], options
        con.close()
        results["join_variants"] += 1

    for index, case in enumerate(fixture["transactions"]):
        db = root / f"transaction-{index}.sqlite"
        writer = sqlite3.connect(db, isolation_level=None)
        reader = sqlite3.connect(db, isolation_level=None)
        writer.execute("CREATE TABLE credits(name TEXT PRIMARY KEY,balance INTEGER NOT NULL CHECK(balance>=0))")
        writer.executemany("INSERT INTO credits VALUES (?,?)", [("A",6),("B",4)])
        def snapshot():
            query = "SELECT balance FROM credits ORDER BY name"
            return ([row[0] for row in writer.execute(query)], [row[0] for row in reader.execute(query)])
        states = [snapshot()]
        if case["atomic"]:
            writer.execute("BEGIN")
        states.append(snapshot())
        writer.execute("UPDATE credits SET balance=balance-2 WHERE name='A'")
        states.append(snapshot())
        if case["fail"]:
            try:
                writer.execute("UPDATE credits SET balance=-1 WHERE name='B'")
                raise AssertionError("Constraint did not reject negative balance")
            except sqlite3.IntegrityError:
                pass
            states.append(snapshot())
            writer.rollback()
            states.append(snapshot())
        else:
            writer.execute("UPDATE credits SET balance=balance+2 WHERE name='B'")
            states.append(snapshot())
            writer.commit()
            states.append(snapshot())
        assert states == [(s["writer"],s["committed"]) for s in case["trace"]], case
        reader.close()
        writer.close()
        results["transaction_traces"] += 1

    spec = importlib.util.spec_from_file_location("sensor_db", root / "sensor_db.py")
    database = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database)
    connection = database.connect_demo()
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    for query, params in [
        ("INSERT INTO sensors VALUES (?,?)", ("A","duplicate")),
        ("INSERT INTO readings VALUES (?,?,?,?)", (10,"absent",0,20)),
    ]:
        try:
            connection.execute(query, params)
            raise AssertionError("Expected key violation")
        except sqlite3.IntegrityError:
            pass
    assert connection.execute("SELECT sensor_id FROM sensors WHERE sensor_id=?", ("A' OR 1=1 --",)).fetchall() == []
    connection.close()

Path(sys.argv[2]).write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print("PASS: all displayed File I/O/SQL programs, schema edge cases, native file traces, join variants, transaction visibility/rollback and key/parameter checks.")
