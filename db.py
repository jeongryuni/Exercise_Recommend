# db.py
import pymysql

def get_connection():
    return pymysql.connect(
        host="3.37.90.119",
        user="root",
        password="3306",
        database="saveus",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
