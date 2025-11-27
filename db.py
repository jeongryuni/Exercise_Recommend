# db.py
import mysql.connector

def get_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="3306",
        database="saveus"
    )
    return conn
