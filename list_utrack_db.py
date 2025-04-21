#!/usr/bin/env python3

import psycopg2
from psycopg2 import sql
import os
import argparse
from dotenv import load_dotenv

def connect_to_postgres(host=None, port=None, database=None, user=None, password=None):
    """Connect to the PostgreSQL database server using provided or default parameters"""
    
    # Try to load environment variables from .env file if it exists
    load_dotenv('utrack-selfhost/utrack-app/utrack.env')
    
    # Get database connection parameters from arguments, environment variables, or use defaults
    host = host or os.getenv('PGHOST', 'localhost')
    database = database or os.getenv('PGDATABASE', 'utrack')
    user = user or os.getenv('POSTGRES_USER', 'utrack')
    password = password or os.getenv('POSTGRES_PASSWORD', 'utrack')
    port = port or os.getenv('POSTGRES_PORT', '5432')
    
    conn = None
    try:
        # Display connection info
        print(f'Attempting to connect to PostgreSQL database:')
        print(f'  Host: {host}')
        print(f'  Database: {database}')
        print(f'  User: {user}')
        print(f'  Port: {port}')
        
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        
        # Create a cursor
        cur = conn.cursor()
        
        # Print PostgreSQL connection properties
        print('\nPostgreSQL database version:')
        cur.execute('SELECT version()')
        db_version = cur.fetchone()
        print(db_version)
        
        # List all databases
        print('\nList of databases:')
        cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        databases = cur.fetchall()
        for db in databases:
            print(f"  - {db[0]}")
        
        # List all schemas in the current database
        print('\nList of schemas in the current database:')
        cur.execute("SELECT schema_name FROM information_schema.schemata;")
        schemas = cur.fetchall()
        for schema in schemas:
            print(f"  - {schema[0]}")
        
        # List all tables in the public schema
        print('\nList of tables in the public schema:')
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        
        if not tables:
            print("  No tables found in public schema.")
        else:
            for table in tables:
                print(f"  - {table[0]}")
                
                # Get column information for each table
                cur.execute(sql.SQL("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                """), [table[0]])
                columns = cur.fetchall()
                for column in columns:
                    print(f"      {column[0]} ({column[1]})")
                
                # Get count of rows in the table
                try:
                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table[0])))
                    row_count = cur.fetchone()[0]
                    print(f"      Total rows: {row_count}")
                except Exception as e:
                    print(f"      Could not count rows: {e}")
        
        # Close the cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()
            print('\nDatabase connection closed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connect to PostgreSQL and list tables and columns.')
    parser.add_argument('--host', help='Database host (default: localhost)')
    parser.add_argument('--port', help='Database port (default: 5432)')
    parser.add_argument('--db', help='Database name (default: utrack)')
    parser.add_argument('--user', help='Database user (default: utrack)')
    parser.add_argument('--password', help='Database password (default: utrack)')
    
    args = parser.parse_args()
    
    connect_to_postgres(
        host=args.host, 
        port=args.port, 
        database=args.db, 
        user=args.user, 
        password=args.password
    ) 