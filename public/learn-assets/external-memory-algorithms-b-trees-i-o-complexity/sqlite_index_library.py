"""Run beside external_memory_mechanisms.py. Uses a disposable SQLite database."""
import sqlite3
import tempfile
from pathlib import Path
from external_memory_mechanisms import BTree


def main():
    values = [10, 20, 5, 6, 12, 30, 7, 17]
    tree = BTree(2)
    for value in values:
        tree.insert(value)
    with tempfile.TemporaryDirectory(prefix='index-lesson-') as directory:
        database = Path(directory) / 'readings.sqlite3'
        # A context manager commits/rolls back; close is still explicit.
        connection = sqlite3.connect(database, autocommit=False)
        try:
            with connection:
                connection.execute('CREATE TABLE readings (id INTEGER PRIMARY KEY, reading INTEGER NOT NULL)')
                connection.executemany('INSERT INTO readings(id, reading) VALUES (?, ?)',
                                       [(i, value) for i, value in enumerate(values)])
                connection.execute('CREATE UNIQUE INDEX reading_order ON readings(reading)')
                connection.execute('INSERT OR IGNORE INTO readings(reading) VALUES (?)', (10,))
                connection.execute('DELETE FROM readings WHERE reading = ?', (6,))
            tree.remove(6)
            low, high = 7, 20
            query = 'SELECT reading FROM readings WHERE reading >= ? AND reading <= ? ORDER BY reading'
            rows = [row[0] for row in connection.execute(query, (low, high))]
            assert rows == [x for x in tree.ordered() if low <= x <= high]
            plan = list(connection.execute('EXPLAIN QUERY PLAN ' + query, (low, high)))
            # Plan text is diagnostic and version dependent, not a stable parse API.
            print('inclusive range:', rows)
            print('index named in this plan:', any('reading_order' in row[3] for row in plan))
            before = list(connection.execute('SELECT reading FROM readings ORDER BY reading'))
            try:
                with connection:
                    connection.execute('INSERT INTO readings(reading) VALUES (?)', (99,))
                    raise RuntimeError('cancel this example transaction')
            except RuntimeError:
                pass
            assert before == list(connection.execute('SELECT reading FROM readings ORDER BY reading'))
            print('exception rolls back:', True)
        finally:
            connection.close()
        reopened = sqlite3.connect(database, autocommit=False)
        try:
            print('committed rows after reopen:', reopened.execute('SELECT count(*) FROM readings').fetchone()[0])
        finally:
            reopened.close()


if __name__ == '__main__':
    main()
