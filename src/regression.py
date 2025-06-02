#!/usr/bin/env python3
"""
Train & evaluate a Linear Regression model
to predict annual banana yield (t/ha) from annual climate features.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_merge_historical():
    # Climate data
    clim_path = '../data/Korea_ERA5_Annual_Climate_2000_2020.csv'
    df_clim = pd.read_csv(clim_path)

    # Compute annual averages/sums
    annual = df_clim.groupby('year').agg({
    'mean_temp_C':    'mean',   # annual mean of monthly means
    'mean_tasmax_C':  'mean',   # annual mean of monthly max-temp
    'mean_tasmin_C':  'mean',   # annual mean of monthly min-temp
    'total_precip_mm': 'sum'    # annual total precipitation
    }).reset_index()

    # Production data - changing the data set will effect this.
    yield_path = '../data/Korea_Banana_Yield_2000_2020.csv'
    df_yield = pd.read_csv(yield_path)
    # Expect columns: year, banana_yield_t_ha

    df = pd.merge(annual, df_yield, on='year', how='inner')

    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split(df):
    train_df = df[df['year'] <= 2015] #training
    val_df   = df[(df['year'] >= 2016) & (df['year'] <= 2017)] #validation
    test_df  = df[df['year'] >= 2018] #test

    FEATURES = [
        'mean_temp_C',
        'mean_tasmax_C',
        'mean_tasmin_C',
        'total_precip_mm'
    ]
    TARGET = 'banana_yield_t_ha'

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    X_val = val_df[FEATURES].values
    y_val = val_df[TARGET].values

    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), FEATURES


def evaluate_regression(model, X_val, y_val, X_test, y_test):
    """Compute MAE, RMSE, R² on validation and test sets."""
    #MAE - Mean of absolute error
    #RMSE - root mean squared error
    #R^2 - variation of real data
    y_val_pred = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val = r2_score(y_val, y_val_pred)

    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Validation   → MAE: {mae_val:.3f}, RMSE: {rmse_val:.3f}, R²: {r2_val:.3f}")
    print(f"Test         → MAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}, R²: {r2_test:.3f}")

def predict_future(model, features):
    """
    Given a trained model and a DataFrame 'features' with columns:
      ['year','mean_temp_C','mean_tasmax_C','mean_tasmin_C','total_precip_mm']
    predict banana yield for each year and return a DataFrame with ['year','predicted_banana_yield_t_ha'].
    """
    X_future = features[['mean_temp_C','mean_tasmax_C','mean_tasmin_C','total_precip_mm']].values
    preds = model.predict(X_future)
    return pd.DataFrame({
        'year': features['year'],
        'predicted_banana_yield_t_ha': preds
    })

def main():
    # ─── Part 1: Train on historical (2000–2020) ──────────────────────────────────
    df_hist = load_and_merge_historical()

    print("Historical merged data (first 5 rows):")
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

    # ─── Part 2: Retrain on train+val (2000–2017) and test 2018–2020 ─────────────
    X_hist = np.vstack([X_train, X_val])
    y_hist = np.concatenate([y_train, y_val])
    lr.fit(X_hist, y_hist)

    y_test_pred = lr.predict(X_test)
    df_test_years = df_hist[df_hist['year'] >= 2018][['year']].copy()
    df_test_years['predicted_banana_yield_t_ha'] = y_test_pred
    print("\nFinal Predictions for 2018–2020 (using retrained model):")
    print(df_test_years.to_string(index=False))

    # ─── Part 3: Load future climate, predict yields (2021–2040) ──────────────────
    future_clim_path = '../data/Korea_ERA5_Annual_Climate_2021_2040.csv'
    df_future = pd.read_csv(future_clim_path)
    # Expect columns: year, ann_mean_temp_C, ann_mean_tasmax_C, ann_mean_tasmin_C, ann_total_precip_mm

    df_future = df_future.dropna().reset_index(drop=True)

    df_future_preds = predict_future(lr, df_future)
    print("\nPredicted Banana Yield for 2021–2040:")
    print(df_future_preds.to_string(index=False))

    # Optionally save to CSV:
    out_path = '../data/Predicted_Banana_Yield_2021_2040.csv'
    df_future_preds.to_csv(out_path, index=False)
    print(f"\nSaved future predictions to '{out_path}'.")

if __name__ == "__main__":
    main()
