# -*- coding: utf-8 -*-#
# filename: receive.py

import sys, os
import urllib
import sqlite3
from flask import g
from flask import Flask, jsonify
from flask import request
from flask import current_app
from flask.cli import with_appcontext

from ..core_constants import DB_PATH_SQLLITE, SCHEMA_SQL_PATH

current_directory = os.path.dirname(os.path.abspath(__file__)) # ./db
db_path = os.path.join(os.path.dirname(current_directory), DB_PATH_SQLLITE)
schema_path = os.path.join(os.path.dirname(current_directory), SCHEMA_SQL_PATH)

# print ("DEBUG: Initializing SQLLite DB directory %s" % current_directory)
# print ("DEBUG: Initializing SQLLite DB File Path %s" % db_path)
# print ("DEBUG: Initializing SQLLite DB Schema File Path %s" % schema_path)

#### Constants
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config['DATABASE'] = db_path


def connect_db():
    return sqlite3.connect(db_path)


@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db'):
        g.db.close()


def query_db_bak(query, args=(), one=False):
    cur = g.db.execute(query, args)
    rv = [dict((cur.description[idx][0], value)
               for idx, value in enumerate(row)) for row in cur.fetchall()]
    return (rv[0] if rv else None) if one else rv


def query_db(query, args=(), one=False):
    """
        return list of dict, dict key: col_name, value: col_value
    """
    cur = g.db.execute(query, args)
    results = []
    for row in cur.fetchall():
        try:
            # Process the row as key-value pairs
            processed_row = {cur.description[idx][0]: value for idx, value in enumerate(row)}
            results.append(processed_row)
        except ValueError as e:
            # Print the error and skip the row if there is a format issue
            print(f"Skipping row due to error: {e}")
            continue  # Skip the problematic row and continue with the next row
    return results

def insert_db_sql(db_path, query, args=()):
    """
        query: INSERT INTO students (name,addr,city,pin) VALUES (?,?,?,?)
        status=0: fail
        status=1: success
    """
    status = 0
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute(query, args)
            con.commit()
            if len(args) == 7:
                msg = "activity_id %s, agent_id %s, agent_name %s, role %s, content %s, created %s, ext_info %s" % args
            else:
                msg = "insert_query_db_sql query|" + query + "|args|" + str(args)
            status = 1
    except Exception as e:
        print ("DEBUG: Failed to insert_db_sql error|%s" % str(e))
        con.rollback()
        msg = "Record in insert failed with error|" + query + "|args|" + str(args) 
        status = 0
    return status

def get_db_local():
    if 'db' not in g:
        g.db = sqlite3.connect(
            # current_app.config['DATABASE'],
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def get_db(arg_db_path):
    if 'db' not in g:
        g.db = sqlite3.connect(
            # current_app.config['DATABASE'],
            arg_db_path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    db = g.pop('db', None)
 
    if db is not None:
        db.close()

def test_query_db_sql():
    cnt = 0
    for user_action in query_db('select * from user_actions'):
        cnt += 1
        print ("Cnt %d, Line|%s" % (cnt, str(user_action)))

def init_db():
    db = get_db_local()
    with app.open_resource(schema_path) as f:
        db.executescript(f.read().decode('utf8'))

def list_tables():
    db = get_db_local()
    cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    return {"tables": tables}

# @click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

def main():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    with app.app_context():
        # db = get_db()
        init_db()
        # list all tables
        tables =list_tables()
        print ("DEBUG: tables %s" % str(tables))
        # query table
        query_agent_activity_sql = 'select * from agent_activity'
        rows = query_db(query_agent_activity_sql)
        print ("DEBUG: query rows %s" % str(rows))
    return app

def run_query_db():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    query_agent_activity_sql = 'select * from agent_activity'
    with app.app_context():
        rows = query_db(query_agent_activity_sql)
        print (rows)
    return app

if __name__ == '__main__':
    main()
