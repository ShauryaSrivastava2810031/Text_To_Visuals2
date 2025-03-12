import sqlite3
import pandas as pd


def read_sql_query(sql, db_path):
    """
    Executes an SQL query on the given SQLite database and returns the result as a DataFrame.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        # Check if the query modifies the database
        if sql.strip().lower().startswith(('insert', 'update', 'delete', 'create', 'alter', 'drop')):
            cur.execute(sql)
            conn.commit()
            return "Query executed successfully"

        # Otherwise, assume it's a SELECT query
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [description[0] for description in cur.description]
        return pd.DataFrame(rows, columns=col_names)

    except Exception as e:
        return f"Error executing query: {str(e)}"

    finally:
        conn.close()