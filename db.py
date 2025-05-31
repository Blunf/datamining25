#!/usr/bin/env python3
# load_monthly_to_sqlite.py
#Setting up .db from csv for future monthly data
#Being ran for both data/Korea_ERA5_Daily_Monthly_1999_2019.csv and Korea_Monthly_Climate_ACCESS-CM2_ssp245_2021_2040.csv

import pandas as pd
import sqlite3
import os

def main():
    csv_path = 'data/Korea_ERA5_Daily_Monthly_1999_2019.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)

    df = df.drop(columns=[c for c in df.columns if c.startswith('system:index') or c == '.geo'], errors='ignore')

    df['year']  = df['year'].astype(int)
    df['month'] = df['month'].astype(int)

    db_dir = 'database'
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'past_climate_metrics.db')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS past_climate_metrics (
        year            INTEGER,
        month           INTEGER,
        mean_temp_C     REAL,
        mean_tasmax_C   REAL,
        mean_tasmin_C   REAL,
        total_precip_mm REAL,
        PRIMARY KEY (year, month)
    )
    ''')

    for _, row in df.iterrows():
        cursor.execute('''
        INSERT OR REPLACE INTO past_climate_metrics
        (year, month, mean_temp_C, mean_tasmax_C, mean_tasmin_C, total_precip_mm)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            row['year'],
            row['month'],
            row['mean_temp_C'],
            row['mean_tasmax_C'],
            row['mean_tasmin_C'],
            row['total_precip_mm']
        ))

    conn.commit()
    conn.close()

    print(f"Loaded {len(df)} records into '{db_path}' (table past_climate_metrics).")

if __name__ == '__main__':
    main()
