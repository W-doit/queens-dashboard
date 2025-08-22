import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Example: load your processed DataFrame (df_daily) from astro_dashboard.py
# You may want to save df_daily to a CSV in the dashboard, then load it here.

def train_sales_model(df_daily):
    # Feature engineering: one-hot encode categorical features
    features = df_daily.copy()
    features = pd.get_dummies(features, columns=['moon_phase_name', 'moon_sign', 'venus_sign', 'mercury_sign'])
    X = features.drop(columns=['sales', 'date', 'date_only'])
    y = features['sales']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    import numpy as np
    mae = mean_absolute_error(y_test, y_pred)
    # For older scikit-learn, 'squared' is not a valid kwarg, so use np.sqrt
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model, X_test, y_test, y_pred

# Example usage:
# df_daily = pd.read_csv('df_daily.csv')
# model, X_test, y_test, y_pred = train_sales_model(df_daily)
