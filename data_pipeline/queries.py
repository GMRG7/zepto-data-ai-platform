"""Run SQL queries, save SQL/results, and verify two JOINs against pandas."""
import sqlite3
from pathlib import Path
import pandas as pd
QUERIES={
"top_rated":"SELECT title, rating FROM books ORDER BY rating DESC, title LIMIT 10",
"price_over_20":"SELECT title, price_gbp FROM books WHERE price_gbp > 20 ORDER BY price_gbp DESC",
"distinct_ratings":"SELECT DISTINCT rating FROM books ORDER BY rating",
"in_stock":"SELECT title FROM books WHERE in_stock = 1 LIMIT 20",
"category_join":"SELECT c.category_name, b.title, b.rating FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY c.category_name,b.rating DESC",
"category_counts":"SELECT c.category_name, COUNT(*) AS book_count FROM books b JOIN categories c ON b.category_id=c.category_id GROUP BY c.category_id,c.category_name ORDER BY book_count DESC"}
def main():
    out=Path("query_outputs"); out.mkdir(exist_ok=True)
    with sqlite3.connect("books.db") as conn:
        for name,sql in QUERIES.items():
            result=pd.read_sql_query(sql,conn)
            print(f"\n--- {name} ---\n{result.to_string(index=False)}")
            (out/f"{name}.sql").write_text(sql+";\n",encoding="utf-8")
            result.to_csv(out/f"{name}.csv",index=False)
        books=pd.read_sql_query("SELECT title,price_gbp,price_inr,rating,in_stock,category_id FROM books",conn)
        cats=pd.read_sql_query("SELECT category_id,category_name FROM categories",conn)
        sql_join=pd.read_sql_query(QUERIES["category_join"],conn)
        pandas_join=books.merge(cats,on="category_id")[ ["category_name","title","rating"] ].sort_values(["category_name","rating"],ascending=[True,False]).reset_index(drop=True)
        assert sql_join.reset_index(drop=True).equals(pandas_join),"category_join SQL/pandas mismatch"
        sql_counts=pd.read_sql_query(QUERIES["category_counts"],conn)
        pandas_counts=books.merge(cats,on="category_id").groupby("category_name",as_index=False).size().rename(columns={"size":"book_count"}).sort_values("book_count",ascending=False).reset_index(drop=True)
        assert sql_counts.reset_index(drop=True).equals(pandas_counts),"category_counts SQL/pandas mismatch"
        (out/"validation.txt").write_text("PASS: category_join SQL matches pandas merge.\nPASS: category_counts SQL matches pandas merge/groupby.\n",encoding="utf-8")
        print("\nSQL/pandas JOIN and category-count checks passed.")
if __name__=="__main__":main()
