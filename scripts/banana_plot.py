#!/usr/bin/env python3
"""
Visualize historical banana yields (2000–2019) alongside predicted yields (2021–2040).
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load historical banana yield (2000–2019)
    hist_path = '../data/Korea_Banana_Yield_2000_2019.csv'
    df_hist = pd.read_csv(hist_path)
    # Expects columns: ['year', 'banana_yield_t_ha']

    # 2) Load predicted yields (2021–2040)
    pred_path = '../data/Predicted_Banana_Yield_2021_2040.csv'
    df_pred = pd.read_csv(pred_path)
    # Expects columns: ['year', 'predicted_banana_yield_t_ha']

    # 3) Plot both series
    plt.figure(figsize=(10, 6))

    # Historical: solid line
    plt.plot(
        df_hist['year'],
        df_hist['banana_yield_t_ha'],
        marker='o',
        linestyle='-',
        color='tab:blue',
        label='Historical Yield (2000–2019)'
    )

    # Predicted: dashed line
    plt.plot(
        df_pred['year'],
        df_pred['predicted_banana_yield_t_ha'],
        marker='o',
        linestyle='--',
        color='tab:orange',
        label='Predicted Yield (2021–2040)'
    )

    plt.xlabel('Year')
    plt.ylabel('Banana Yield (tonnes/ha)')
    plt.title('Historical vs. Predicted Banana Yield in South Korea')
    plt.xticks(list(df_hist['year']) + list(df_pred['year']), rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
