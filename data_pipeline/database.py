"""Create normalized SQLite tables and load the cleaned CSV."""
import sqlite3
import pandas as pd

DB_PATH = "books.db"

def build_database(csv_path="books_clean.csv", db_path=DB_PATH):
    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT NOT NULL UNIQUE
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        )""")
        conn.execute("DELETE FROM books")
        for category in sorted(df["category"].dropna().unique()):
            conn.execute("INSERT OR IGNORE INTO categories(category_name) VALUES (?)", (category,))
        category_map = dict(conn.execute("SELECT category_name, category_id FROM categories").fetchall())
        for _, row in df.iterrows():
            conn.execute("""INSERT INTO books
                (title, price_gbp, price_inr, rating, in_stock, category_id)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (row["title"], row["price_gbp"], row["price_inr"], int(row["rating"]),
                 int(bool(row["in_stock"])), category_map[row["category"]]))
        conn.commit()
    print(f"Database saved to {db_path}")

if __name__ == "__main__":
    build_database()
