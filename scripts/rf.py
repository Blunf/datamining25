#!/usr/bin/env python3
"""
Train and evaluate a Random Forest Regressor to predict annual banana yield (t/ha)
using already‐aggregated annual climate data (2000–2019) and then forecast 2021–2040 yields.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_merge_historical():
    """
    Read pre‐aggregated annual climate CSV (2000–2019) and banana yield CSV (2000–2019),
    then merge on 'year'. Returns a DataFrame with columns:
      ['year','ann_mean_temp_C','ann_mean_tasmax_C','ann_mean_tasmin_C',
       'ann_total_precip_mm','banana_yield_t_ha']
    """
    clim_annual = pd.read_csv('data/Korea_Annual_Climate_2000_2019.csv')
    # Expects: ['year','ann_mean_temp_C','ann_mean_tasmax_C','ann_mean_tasmin_C','ann_total_precip_mm']

    df_yield = pd.read_csv('data/Korea_Banana_Yield_2000_2019.csv')
    # Expects: ['year','banana_yield_t_ha']

    df = pd.merge(clim_annual, df_yield, on='year', how='inner')
    return df.dropna().reset_index(drop=True)

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
        'mean_temp_C',
        'mean_tasmax_C',
        'mean_tasmin_C',
        'total_precip_mm'
    ]
    TARGET = 'banana_yield_t_ha'

    X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
    X_val,   y_val   = val_df[FEATURES].values,   val_df[TARGET].values
    X_test,  y_test  = test_df[FEATURES].values,  test_df[TARGET].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), FEATURES

def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Compute MAE, RMSE, and R² on validation and test sets."""
    y_val_pred = model.predict(X_val)
    mae_val  = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val   = r2_score(y_val, y_val_pred)

    y_test_pred = model.predict(X_test)
    mae_test  = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test   = r2_score(y_test, y_test_pred)

    print(f"Validation   → MAE: {mae_val:.3f}, RMSE: {rmse_val:.3f}, R²: {r2_val:.3f}")
    print(f"Test         → MAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

def predict_future(model, df_future_annual):
    FEATURES = [
        'mean_temp_C',
        'mean_tasmax_C',
        'mean_tasmin_C',
        'total_precip_mm'
    ]
    X_future = df_future_annual[FEATURES].values
    preds = model.predict(X_future)
    return pd.DataFrame({
        'year': df_future_annual['year'],
        'predicted_banana_yield_t_ha': preds
    })

def main():
    #── Part 1: Load historical annual climate & yield (2000–2019) ─────────────────
    df_hist = load_and_merge_historical()
    print("Historical annual data (2000–2019):")
    print(df_hist.head(), "\n")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), FEATURES = train_test_split(df_hist)

    #── Part 2: Train Random Forest on 2000–2014; validate on 2015–2016; test on 2017–2019 ──
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    print("--- Random Forest Performance (Historical) ---")
    evaluate_model(rf, X_val, y_val, X_test, y_test)

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    print("\nFeature Importances (descending):")
    print(importances.sort_values(ascending=False))

    #── Part 3: Retrain on 2000–2016, check 2017–2019 ─────────────────────────────
    X_hist_all = np.vstack([X_train, X_val])
    y_hist_all = np.concatenate([y_train, y_val])
    rf.fit(X_hist_all, y_hist_all)

    y_test_pred = rf.predict(X_test)
    df_test_years = df_hist[df_hist['year'] >= 2017][['year']].copy()
    df_test_years['predicted_banana_yield_t_ha'] = y_test_pred
    print("\nPredictions for 2017–2019 (retrained model):")
    print(df_test_years.to_string(index=False))

    #── Part 4: Load future annual climate (2021–2040) and predict ────────────────
    df_future = pd.read_csv('data/Korea_Annual_Climate_2021_2040.csv')
    # Expects: ['year','ann_mean_temp_C','ann_mean_tasmax_C','ann_mean_tasmin_C','ann_total_precip_mm']
    df_future = df_future.dropna().reset_index(drop=True)

    df_future_preds = predict_future(rf, df_future)
    print("\nPredicted banana yield for 2021–2040:")
    print(df_future_preds.to_string(index=False))

    # Save future predictions
    out_path = 'data/Predicted_Banana_Yield_RF_2021_2040.csv'
    df_future_preds.to_csv(out_path, index=False)
    print(f"\nSaved future predictions to '{out_path}'.")

if __name__ == "__main__":
    main()
