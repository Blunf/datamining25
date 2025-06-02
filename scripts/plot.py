#!/usr/bin/env python3
"""
Plot annual ERA5‐derived climate parameters (mean_temp_C, mean_tasmax_C,
mean_tasmin_C, total_precip_mm) for South Korea (1999–2019), each in its own subplot.
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Point this at your actual CSV file
    csv_path = '../data/Korea_NEXGDDP_ACCESS-CM2_SSP245_Annual_2021_2040.csv'
    df = pd.read_csv(csv_path)
    # Expects columns: ['year', 'mean_temp_C', 'mean_tasmax_C', 'mean_tasmin_C', 'total_precip_mm']

    years = df['year']
    params = ['mean_temp_C', 'mean_tasmax_C', 'mean_tasmin_C', 'total_precip_mm']

    # 2) Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # 3) Plot each parameter
    for idx, param in enumerate(params):
        ax = axes[idx]
        ax.plot(years, df[param], marker='o', linestyle='-')
        ax.set_title(param.replace('_', ' ').title())
        ax.set_xlabel('Year')
        ax.set_ylabel(param)
        ax.grid(alpha=0.3)

    # 4) If there are fewer than 4 params, hide the unused axes
    for j in range(len(params), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
