import sqlite3
from pathlib import Path


class DatabaseManager:
    def __init__(self, path: str) -> None:
        self.path = str(Path(path).expanduser())
        Path(self.path).resolve().parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            connection.executescript(schema)

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def insert(self, table: str, values: dict) -> int:
        permitted = {"readings", "inspections", "cleanings", "alerts"}
        if table not in permitted or not values:
            raise ValueError("Invalid table or empty record.")
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self.connection() as connection:
            cursor = connection.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            return int(cursor.lastrowid)

    def records(self, table: str, limit: int = 1000, where: str = "",
                params: tuple = (), descending: bool = True) -> list[dict]:
        permitted = {"readings", "inspections", "cleanings", "alerts"}
        if table not in permitted:
            raise ValueError("Unknown record table.")
        if not 1 <= limit <= 100000:
            raise ValueError("Limit must be between 1 and 100000.")
        clause = f" WHERE {where}" if where else ""
        order = "timestamp DESC" if descending else "timestamp ASC"
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                f"SELECT * FROM {table}{clause} ORDER BY {order} LIMIT ?",
                (*params, limit),
            )]

    def update_alert(self, alert_id: int, **fields: int) -> None:
        allowed = {"acknowledged", "archived"}
        if not fields or set(fields) - allowed:
            raise ValueError("Unsupported alert update.")
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self.connection() as connection:
            connection.execute(
                f"UPDATE alerts SET {assignments} WHERE id = ?",
                (*fields.values(), alert_id),
            )

    def clear_demo_data(self) -> None:
        with self.connection() as connection:
            for table in ("readings", "inspections", "cleanings", "alerts"):
                connection.execute(f"DELETE FROM {table} WHERE source LIKE 'Demo%'")
