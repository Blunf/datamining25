# DataMining25: Climate Data Pipeline for South Korea

This repository contains scripts, data, and documentation for fetching, processing, and storing monthly climate metrics (temperature and precipitation) over South Korea (33–43 °N, 124–132 °E). We build both a historical baseline (2000–2020) and a future projection (2021–2040) to support crop‐production modeling (e.g., rice, banana) under changing climate scenarios.

---

## Folder Structure
.
├── data/
│ ├── Korea_ERA5Land_Monthly_2000_2020_Simple.csv
│ └── Korea_ERA5_Daily_Monthly_2021_2040.csv
│
├── database/
│ ├── historical_climate.db # Created by load_historical_to_sqlite.py
│ └── future_climate_metrics.db # Created by load_future_to_sqlite.py
│
├── scripts/
│ ├── load_historical_to_sqlite.py
│ ├── load_future_to_sqlite.py
│ └── visualize_future_monthly_temps.py
│
└── README.md

