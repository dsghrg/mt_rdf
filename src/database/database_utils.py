import os
import psycopg2
import pandas as pd
import sys


def connect_to_database(db_name='cordis_temporary', schema='tmp'):
    conn = psycopg2.connect(database=db_name,
                            host=os.environ.get('DB_HOST'),
                            user=os.environ.get('DB_USER'),
                            password=os.environ.get('DB_PASSWORD'),
                            port=os.environ.get('DB_PORT'),
                            options=f'-c search_path={schema}')

    cursor = conn.cursor()

    return cursor, conn


def query_db(sql_query, db_name='cordis_temporary'):
    cur, conn = connect_to_database(db_name)
    try:
        cur.execute(sql_query)
        data = [row for row in cur]
        df = pd.DataFrame(data, columns=[col[0] for col in cur.description])
    except:
        e = sys.exc_info()[0]
        print(f'error:\t{e}')
        return None
    finally:
        cur.close()
        conn.close()

    return df

