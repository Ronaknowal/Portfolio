export const sensorDatabase = String.raw`import sqlite3

def connect_demo():
    # Explicit SQL controls transactions; do not rely on wrapper defaults.
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE sensors (
            sensor_id TEXT NOT NULL PRIMARY KEY,
            room TEXT NOT NULL
        );
        CREATE TABLE readings (
            reading_id INTEGER PRIMARY KEY,
            sensor_id TEXT NOT NULL REFERENCES sensors(sensor_id),
            minute INTEGER NOT NULL CHECK (minute >= 0),
            value REAL CHECK (value BETWEEN -80 AND 80)
        );
        INSERT INTO sensors VALUES ('A', 'north'), ('B', 'south'), ('C', 'spare');
        INSERT INTO readings VALUES (1, 'A', 0, 18), (2, 'A', 10, 22), (3, 'B', 0, NULL);
    """)
    return connection`;

export const featureQuery = String.raw`SELECT s.sensor_id,
       COUNT(r.reading_id) AS n_readings,
       COUNT(r.value) AS n_measured,
       AVG(r.value) AS mean_temperature
FROM sensors AS s
LEFT JOIN readings AS r
  ON r.sensor_id = s.sensor_id AND r.minute <= ?
GROUP BY s.sensor_id
ORDER BY s.sensor_id`;

export const sqlExamples = {
  select: {
    files: { 'sensor_db.py': sensorDatabase },
    filename: 'query_readings.py',
    code: String.raw`from sensor_db import connect_demo

connection = connect_demo()
try:
    query = """
        SELECT reading_id, sensor_id, value
        FROM readings
        WHERE value >= ?
        ORDER BY reading_id
    """
    print(connection.execute(query, (20,)).fetchall())
    missing = "SELECT reading_id FROM readings WHERE value IS NULL ORDER BY reading_id"
    print(connection.execute(missing).fetchall())
finally:
    connection.close()`,
    output: "[(2, 'A', 22.0)]\n[(3,)]",
  },
  features: {
    filename: 'build_features.py',
    code: `from sensor_db import connect_demo

query = """${featureQuery}"""
connection = connect_demo()
try:
    for cutoff in [0, 10]:
        rows = connection.execute(query, (cutoff,)).fetchall()
        assert len(rows) == 3
        assert len({row[0] for row in rows}) == len(rows)
        print("cutoff:", cutoff)
        for row in rows:
            print(row)
finally:
    connection.close()`,
    output: "cutoff: 0\n('A', 1, 1, 18.0)\n('B', 1, 0, None)\n('C', 0, 0, None)\ncutoff: 10\n('A', 2, 2, 20.0)\n('B', 1, 0, None)\n('C', 0, 0, None)",
  },
  transaction: {
    filename: 'transfer_credits.py',
    code: String.raw`import sqlite3

connection = sqlite3.connect(":memory:", isolation_level=None)
try:
    connection.execute("CREATE TABLE credits (name TEXT PRIMARY KEY, balance INTEGER NOT NULL CHECK(balance >= 0))")
    connection.executemany("INSERT INTO credits VALUES (?, ?)", [("A", 6), ("B", 4)])
    for fail in [True, False]:
        connection.execute("BEGIN")
        try:
            connection.execute("UPDATE credits SET balance = balance - ? WHERE name = ?", (2, "A"))
            if fail:
                # Inject a bug that violates the nonnegative-balance rule.
                connection.execute("UPDATE credits SET balance = -1 WHERE name = ?", ("B",))
            else:
                connection.execute("UPDATE credits SET balance = balance + ? WHERE name = ?", (2, "B"))
            connection.execute("COMMIT")
        except sqlite3.IntegrityError:
            connection.execute("ROLLBACK")
            print("rolled back")
        print(connection.execute("SELECT name, balance FROM credits ORDER BY name").fetchall())
finally:
    connection.close()`,
    output: "rolled back\n[('A', 6), ('B', 4)]\n[('A', 4), ('B', 6)]",
  },
  practice: {
    filename: 'feature_variation.py',
    code: `from sensor_db import connect_demo

query = """${featureQuery}"""
connection = connect_demo()
try:
    connection.execute("INSERT INTO readings VALUES (?, ?, ?, ?)", (4, "B", 5, 0))
    connection.execute("INSERT INTO sensors VALUES (?, ?)", ("D", "west"))
    rows = connection.execute(query, (5,)).fetchall()
    assert len(rows) == 4 and len({row[0] for row in rows}) == 4
    assert rows == [("A", 1, 1, 18.0), ("B", 2, 1, 0.0),
                    ("C", 0, 0, None), ("D", 0, 0, None)]
    for row in rows:
        print(row)
finally:
    connection.close()`,
    output: "('A', 1, 1, 18.0)\n('B', 2, 1, 0.0)\n('C', 0, 0, None)\n('D', 0, 0, None)",
  },
};
