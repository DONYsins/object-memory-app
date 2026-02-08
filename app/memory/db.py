import sqlite3

def init_db(db_path: str):
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object TEXT NOT NULL,
            time_iso TEXT NOT NULL,
            image_path TEXT,
            location TEXT,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
        )
    """)
    conn.commit()
    return conn

def insert_event(conn, obj, time_iso, image_path, location, bbox):
    x1, y1, x2, y2 = bbox
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events(object, time_iso, image_path, location, x1, y1, x2, y2) VALUES(?,?,?,?,?,?,?,?)",
        (obj, time_iso, image_path, location, x1, y1, x2, y2),
    )
    conn.commit()
    return cur.lastrowid

def fetch_events_by_ids(conn, ids):
    if not ids:
        return []
    q = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT id, object, time_iso, image_path, location, x1, y1, x2, y2 FROM events WHERE id IN ({q})",
        ids,
    ).fetchall()
    row_map = {r[0]: r for r in rows}
    return [row_map[i] for i in ids if i in row_map]

def fetch_last_k(conn, object_name, k=3):
    return conn.execute(
        """
        SELECT id, object, time_iso, image_path, location, x1, y1, x2, y2
        FROM events
        WHERE object = ?
        ORDER BY datetime(time_iso) DESC
        LIMIT ?
        """,
        (object_name, k),
    ).fetchall()
