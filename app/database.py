# SQLite product catalog and analytics storage.

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.models import DB_DIR


class Database:
    def __init__(self):
        self.db_path = DB_DIR / "adcraft.db"
        print(f"    [DATABASE] Initializing SQLite database: {self.db_path}")
        self._init_db()
        print(f"    [DATABASE] Tables ready: products, generated_content, clicks")

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                price TEXT,
                category TEXT,
                brand TEXT,
                image_paths TEXT,
                enhanced_image_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_content (
                id TEXT PRIMARY KEY,
                product_id TEXT,
                content_type TEXT,
                language TEXT,
                content_json TEXT,
                pamphlet_path TEXT,
                product_image_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clicks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                platform TEXT,
                source TEXT,
                clicked_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def create_product(self, name: str, description: str = "",
                       price: str = "", category: str = "",
                       brand: str = "", image_paths: list = None) -> str:
        product_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        print(f"    [DB] Creating product: id={product_id} | name={name} | brand={brand} | category={category}")
        conn = self._conn()
        conn.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (product_id, name, description, price, category, brand,
             json.dumps(image_paths or []), None, now, now)
        )
        conn.commit()
        conn.close()
        print(f"    [DB] Product created: {product_id} | Images: {len(image_paths or [])}")
        return product_id

    def get_product(self, product_id: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["image_paths"] = json.loads(d["image_paths"])
            return d
        return None

    def list_products(self) -> List[dict]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM products ORDER BY created_at DESC").fetchall()
        conn.close()
        products = []
        for row in rows:
            d = dict(row)
            d["image_paths"] = json.loads(d["image_paths"])
            products.append(d)
        return products

    def update_product(self, product_id: str, **kwargs) -> bool:
        """Update product fields. Only updates provided fields."""
        conn = self._conn()
        product = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        if not product:
            conn.close()
            return False

        updatable = ["name", "description", "price", "category", "brand",
                      "image_paths", "enhanced_image_path"]
        sets = []
        values = []
        for key in updatable:
            if key in kwargs:
                val = kwargs[key]
                if key == "image_paths" and isinstance(val, list):
                    val = json.dumps(val)
                sets.append(f"{key} = ?")
                values.append(val)

        if not sets:
            conn.close()
            return False

        sets.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(product_id)

        conn.execute(f"UPDATE products SET {', '.join(sets)} WHERE id = ?", values)
        conn.commit()
        conn.close()
        print(f"    [DB] Product updated: {product_id} | Fields: {list(kwargs.keys())}")
        return True

    def delete_product(self, product_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM generated_content WHERE product_id = ?", (product_id,))
        conn.execute("DELETE FROM clicks WHERE product_id = ?", (product_id,))
        conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        conn.commit()
        conn.close()

    def save_generated_content(self, product_id: str, content_type: str,
                                language: str, content: dict,
                                pamphlet_path: str = None,
                                product_image_path: str = None) -> str:
        content_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        print(f"    [DB] Saving content: id={content_id} | product={product_id} | type={content_type} | lang={language}")
        conn = self._conn()
        conn.execute(
            "INSERT INTO generated_content VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (content_id, product_id, content_type, language,
             json.dumps(content, ensure_ascii=False), pamphlet_path,
             product_image_path, now)
        )
        conn.commit()
        conn.close()
        print(f"    [DB] Content saved: {content_id}")
        return content_id

    def get_product_content(self, product_id: str) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM generated_content WHERE product_id = ? ORDER BY created_at DESC",
            (product_id,)
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            d = dict(row)
            d["content_json"] = json.loads(d["content_json"])
            results.append(d)
        return results

    def track_click(self, product_id: str, platform: str, source: str = ""):
        now = datetime.now().isoformat()
        print(f"    [DB] Tracking click: product={product_id} | platform={platform} | source={source}")
        conn = self._conn()
        conn.execute(
            "INSERT INTO clicks (product_id, platform, source, clicked_at) VALUES (?, ?, ?, ?)",
            (product_id, platform, source, now)
        )
        conn.commit()
        conn.close()

    def get_analytics(self, product_id: str) -> dict:
        conn = self._conn()
        rows = conn.execute(
            "SELECT platform, COUNT(*) as count FROM clicks WHERE product_id = ? GROUP BY platform",
            (product_id,)
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM clicks WHERE product_id = ?", (product_id,)
        ).fetchone()[0]
        conn.close()
        platforms = {row["platform"]: row["count"] for row in rows}
        return {"total_clicks": total, "by_platform": platforms}
