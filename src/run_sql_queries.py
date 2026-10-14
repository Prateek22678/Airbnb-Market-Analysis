"""
Execute analytical SQL queries from sql/queries.sql against airbnb_austin.db.
Prints formatted tabular outputs and business interpretations.
"""
import os
import re
import sqlite3
import pandas as pd

DB_PATH = "data/processed/airbnb_austin.db"
QUERIES_PATH = "sql/queries.sql"

def parse_queries(file_path: str):
    """Parse queries from sql file split by comments or semicolons."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on named query comments
    pattern = r"-- name:\s*(\w+)\s*\n(.*?)(?=(?:-- name:|\Z))"
    matches = re.findall(pattern, text, re.DOTALL)
    queries = {}
    for name, sql in matches:
        clean_sql = sql.strip().rstrip(";")
        if clean_sql:
            queries[name] = clean_sql
    return queries

def run_all_queries():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run load_to_sqlite.py first.")

    conn = sqlite3.connect(DB_PATH)
    queries = parse_queries(QUERIES_PATH)
    print(f"Loaded {len(queries)} analytical queries from {QUERIES_PATH}.\n")

    results = {}
    for idx, (name, sql) in enumerate(queries.items(), start=1):
        print(f"====================================================================")
        print(f"[{idx}/11] EXECUTING QUERY: {name.upper()}")
        print(f"====================================================================")
        try:
            df = pd.read_sql_query(sql, conn)
            results[name] = df
            print(df.to_string(index=False))
            print(f"\n[Returned {len(df)} rows]\n")
        except Exception as e:
            print(f"[ERROR] Query {name} failed: {e}")
            raise

    conn.close()
    return results

if __name__ == "__main__":
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 120)
    run_all_queries()
