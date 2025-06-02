#!/usr/bin/env python3
"""
Train a Linear Regression model on historical monthly data (1999–2019),
aggregated to annual values (2000–2019), and use it to predict banana production
for 2021–2040 (future monthly data aggregated to annual).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def aggregate_monthly_to_annual(df_monthly):
    agg = df_monthly.groupby('year').agg({
        'mean_temp_C':     'mean',  # annual mean of monthly mean temperature
        'mean_tasmax_C':   'mean',  # annual mean of monthly max temperature
        'mean_tasmin_C':   'mean',  # annual mean of monthly min temperature
        'total_precip_mm': 'sum'    # annual sum of monthly precipitation
    }).reset_index()

    agg = agg.rename(columns={
        'mean_temp_C':     'ann_mean_temp_C',
        'mean_tasmax_C':   'ann_mean_tasmax_C',
        'mean_tasmin_C':   'ann_mean_tasmin_C',
        'total_precip_mm': 'ann_total_precip_mm'
    })
    return agg


def load_and_merge_historical_from_monthly():
    clim_monthly_path = '../data/Korea_ERA5_Daily_Monthly_1999_2019.csv'
    df_monthly = pd.read_csv(clim_monthly_path)

    df_annual_temp = aggregate_monthly_to_annual(df_monthly)

    yield_path = '../data/Korea_Banana_Yield_2000_2019.csv'
    df_yield = pd.read_csv(yield_path)

    df = pd.merge(df_annual_temp, df_yield, on='year', how='inner')
    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split(df):
    """
    Split df into train/val/test by year:
      - Train:      2000–2014
      - Validation: 2015–2016
      - Test:       2017–2019
    """
    train_df = df[df['year'] <= 2014]
    val_df   = df[(df['year'] >= 2015) & (df['year'] <= 2016)]
    test_df  = df[df['year'] >= 2017]

    FEATURES = [
        'ann_mean_temp_C',
        'ann_mean_tasmax_C',
        'ann_mean_tasmin_C',
        'ann_total_precip_mm'
    ]
    TARGET = 'banana_yield_t_ha'

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    X_val   = val_df[FEATURES].values
    y_val   = val_df[TARGET].values

    X_test  = test_df[FEATURES].values
    y_test  = test_df[TARGET].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), FEATURES


def evaluate_regression(model, X_val, y_val, X_test, y_test):
    """Compute MAE, RMSE, and R² on validation and test sets."""
    # Validation
    y_val_pred = model.predict(X_val)
    mae_val  = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val   = r2_score(y_val, y_val_pred)

    # Test
    y_test_pred = model.predict(X_test)
    mae_test  = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test   = r2_score(y_test, y_test_pred)

    print(f"Validation   → MAE: {mae_val:.3f}, RMSE: {rmse_val:.3f}, R²: {r2_val:.3f}")
    print(f"Test         → MAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")


def aggregate_future_monthly_to_annual(df_monthly):
    return aggregate_monthly_to_annual(df_monthly) 


def predict_future(model, df_future_annual):
    FEATURES = [
        'ann_mean_temp_C',
        'ann_mean_tasmax_C',
        'ann_mean_tasmin_C',
        'ann_total_precip_mm'
    ]
    X_future = df_future_annual[FEATURES].values
    preds = model.predict(X_future)
    return pd.DataFrame({
        'year': df_future_annual['year'],
        'predicted_banana_yield_t_ha': preds
    })


def main():
    # ─── Part 1: Train on historical monthly data (aggregated to annual) ─────────
    df_hist = load_and_merge_historical_from_monthly()

    print("Historical data (2000–2019), aggregated to annual:")
    print(df_hist.head(), "\n")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), FEATURES = train_test_split(df_hist)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("--- Linear Regression Performance (Historical) ---")
    evaluate_regression(lr, X_val, y_val, X_test, y_test)

    # Print coefficients
    coefs = pd.Series(lr.coef_, index=FEATURES)
    print("\nRegression Coefficients:")
    print(coefs.sort_values())

    # ─── Part 2: Retrain on 2000–2016, check 2017–2019 ─────────────────────────
    X_hist_all = np.vstack([X_train, X_val])
    y_hist_all = np.concatenate([y_train, y_val])
    lr.fit(X_hist_all, y_hist_all)

    y_test_pred = lr.predict(X_test)
    df_test_years = df_hist[df_hist['year'] >= 2017][['year']].copy()
    df_test_years['predicted_banana_yield_t_ha'] = y_test_pred
    print("\nPredictions for 2017–2019 (retrained model):")
    print(df_test_years.to_string(index=False))

    # ─── Part 3: Load future monthly data (2021–2040) and aggregate ─────────────
    future_monthly_path = '../data/Korea_Monthly_Climate_ACCESS-CM2_ssp245_2021_2040.csv'
    df_future_monthly = pd.read_csv(future_monthly_path)
    # Expects columns: ['year','month','mean_temp_C','mean_tasmax_C','mean_tasmin_C','total_precip_mm']

    df_future_annual = aggregate_future_monthly_to_annual(df_future_monthly)
    print("\nAnnual‐aggregated future data (2021–2040):")
    print(df_future_annual.head(), "\n")

    # ─── Part 4: Predict banana yield for 2021–2040 ─────────────────────────────
    df_future_preds = predict_future(lr, df_future_annual)
    print("Predicted banana yield for 2021–2040:")
    print(df_future_preds.to_string(index=False))

    out_path = '../data/Predicted_Banana_Yield_2021_2040.csv'
    df_future_preds.to_csv(out_path, index=False)
    print(f"\nSaved future predictions to '{out_path}'.")


if __name__ == "__main__":
    main()
