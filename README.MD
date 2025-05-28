# load\_to\_sqlite.py

Ingests annual climate metrics from CSV into SQLite with **year** as the primary key.

## Schema

```sql
CREATE TABLE IF NOT EXISTS climate_metrics (
  year INTEGER PRIMARY KEY,
  mean_temp_C REAL,
  total_precip_mm REAL
);
```

## Workflow

1. **Process CSV** (`data/Korea_Annual_Climate...csv`):

   * Dropped `system:index`, `.geo` columns
   * Converted `year` to integer
2. **Database load** (`climate_metrics.db`):

   * Creates table if needed
   * Inserts or replaces each year’s record

*Model & Data*: ACCESS‑CM2 SSP2‑4.5 (2021–2040), monthly Amon data from ORNL THREDDS, subset to Korea (lat 33–43°, lon 124–132°), annual mean temp (°C) and total precip (mm).

## Usage

```bash
pip install pandas
python load_to_sqlite.py
```

## Inspect

```bash
sqlite3 climate_metrics.db
.tables
PRAGMA table_info(climate_metrics);
SELECT * FROM climate_metrics LIMIT 5;
```
