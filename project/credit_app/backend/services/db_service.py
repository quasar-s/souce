import sqlite3
import os

DB_PATH = "db/history.db"


"""
cursor.execute('select id, name from users')
cursor.fetchone() => (1, 'alice','010-1234-5678')

sqlite3.Row 
row['id']
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    # 조회 결과를 어떻게 반환할지 설정
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # 디렉토리 생성
    # os.mkdir("db")
    os.makedirs("db", exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS transactions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   date TEXT,
                   category TEXT,
                   merchant TEXT,
                   amount INTEGER
                   )
""")
    conn.commit()
    conn.close()

    print("db 초기화")


def get_table_columns(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info({})".format(table_name))
    columns = cursor.fetchall()
    conn.close()
    return [column[1] for column in columns]
