"""
Import predictions into postgis
---

This scripts loads predictions saved inside csv tables in the format:

patch_name | prediction
----------|----------
t32..._T321|3

patch_name has to be: <table_name>_T<rid>

Into a postgres table with custom name.

The generated table will contain:

PK(r_table_name, tile_id), geom, prediction
"""
from __future__ import annotations
import os
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import psycopg2

POSTGIS_USER = os.environ["PGUSER"]
POSTGIS_PASSWORD = os.environ["PGPASSWORD"]
POSTGIS_DB = "geospatial"
POSTGIS_HOST = "home.arsbrevis.de"
POSTGIS_PORT = 31313


PARSER = ArgumentParser()
CONNECTION = psycopg2.connect(
    dbname=POSTGIS_DB, user=POSTGIS_USER, password=POSTGIS_PASSWORD, host=POSTGIS_HOST, port=POSTGIS_PORT)


TABLE_SCHEMA = """
CREATE TABLE {table_name} (
    rid INTEGER NOT NULL,
    table_name TEXT NOT NULL,
    prediction TEXT NOT NULL,
    geom GEOMETRY NOT NULL,
    sureness FLOAT,
    PRIMARY KEY(table_name, rid)
);
"""


def csv_path(string):
    path = Path(string)
    if path.suffix != ".csv":
        PARSER.error("Predictions need to be in csv format")
    return path


def list_tables():
    cursor = CONNECTION.cursor()
    cursor.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""")
    table_names = [t for t, in cursor.fetchall()]
    cursor.close()
    return table_names


def create_table(table_name: str):
    """Create a new table in postgis database."""
    schema = TABLE_SCHEMA.format(table_name=table_name)
    print("Creating table with following schema:")
    print(schema)
    print("------")
    if table_name in list_tables():
        raise RuntimeError(f"{table_name} already exists")
    cursor = CONNECTION.cursor()
    cursor.execute(schema)
    CONNECTION.commit()
    cursor.close()


def yield_row(predictions: pd.DataFrame):
    for _, row in predictions.iterrows():
        name = row["name"]
        pred = row["pred"]

        table_name, tile_id = name.rsplit("_", 1)
        tile_id = int(tile_id[1:])
        yield table_name, tile_id, pred


def insert_predictions(predictions: pd.DataFrame, dest_table: str):
    cursor = CONNECTION.cursor()
    for table_name, rid, pred in yield_row(predictions):
        cursor.execute(f"""
SELECT rast::geometry FROM {table_name} WHERE rid = {rid}
        """)
        geom, = cursor.fetchone()
        print(geom)
        cursor.execute(f"""
INSERT INTO {dest_table} (table_name, rid, prediction, geom) VALUES ('{table_name}', {rid}, '{pred}', '{geom}')
        """)
    CONNECTION.commit()
    cursor.close()
    print(f"Inserted predictions into {dest_table}")



def main(args):
    predictions = pd.read_csv(args.predictions)
    try:
        create_table(args.destination_table)
    except RuntimeError as err:
        print(err)
    insert_predictions(predictions, args.destination_table)



if __name__ == "__main__":
    PARSER.add_argument("predictions", help="csv file containing predictions", type=csv_path)
    PARSER.add_argument("destination_table", help="Destination table name in database")
    main(PARSER.parse_args())
