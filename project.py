#!/usr/bin/env python3
# load_to_sqlite.py

import pandas as pd
import sqlite3
import os

def main():
    # 1) Update this path to point at your CSV
    csv_path = 'data/Korea_Annual_Climate_ACCESS-CM2_ssp245_2021-2040.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        return

    # 2) Load the CSV
    df = pd.read_csv(csv_path)

    # 3) Drop the unwanted geometry/index columns
    #    Adjust these names if they differ
    df = df.drop(columns=[c for c in df.columns if c.startswith('system:index') or c == '.geo'], errors='ignore')

    # 4) Ensure year is integer
    df['year'] = df['year'].astype(int)

    # 5) Connect to (or create) the SQLite database
    db_path = 'climate_metrics.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 6) Create the table with year as PRIMARY KEY
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS climate_metrics (
        year INTEGER PRIMARY KEY,
        mean_temp_C REAL,
        total_precip_mm REAL
    )
    ''')

    # 7) Insert or replace each row
    for _, row in df.iterrows():
        cursor.execute('''
        INSERT OR REPLACE INTO climate_metrics (year, mean_temp_C, total_precip_mm)
        VALUES (?, ?, ?)
        ''', (row['year'], row['mean_temp_C'], row['total_precip_mm']))

    conn.commit()
    conn.close()

    print(f"Loaded {len(df)} records into '{db_path}' (table climate_metrics).")

if __name__ == '__main__':
    main()
