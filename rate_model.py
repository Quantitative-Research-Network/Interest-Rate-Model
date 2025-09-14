import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def fetch_fred_series(series_id, api_key, start_date='1990-01-01'):
    """Fetch data from FRED API"""
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': datetime.now().strftime('%Y-%m-%d')
    }
    
    print(f"Fetching {series_id}...")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {series_id}: {response.status_code}")
    
    data = response.json()
    observations = data['observations']
    
    # Convert to DataFrame
    df = pd.DataFrame(observations)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[['date', 'value']].set_index('date')
    df.columns = [series_id]
    
    return df

def compute_yoy_change(series):
    """Compute year-over-year percentage change"""
    return series.pct_change(periods=12) * 100

def main():
    # Check for FRED API key
    api_key = os.getenv('FRED_API_KEY', '0a5af412d677d2f55e14ddd03b499eae')
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not found. Please export it:\nexport FRED_API_KEY=\"your_key_here\"")
    
    print("Starting Fed Rate Prediction Model")
    print("=" * 50)
    
    # Define series to fetch
    series_ids = [
        'CPIAUCSL',    # Headline CPI
        'CPILFESL',    # Core CPI
        'UNRATE',      # Unemployment rate
        'NAPM',        # ISM Manufacturing PMI
        'DGS2',        # 2Y Treasury yield
        'DGS10',       # 10Y Treasury yield
        'DFEDTARU'     # Fed Funds Target Range Upper Limit
    ]
    
    # Fetch all series
    data_frames = []
    for series_id in series_ids:
        try:
            df = fetch_fred_series(series_id, api_key)
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            continue
    
    # Combine all series
    print("\nCombining and processing data...")
    combined_data = pd.concat(data_frames, axis=1)
    
    # Convert daily yields to monthly averages with month-end timestamps
    # Create aggregation dict based on available columns
    agg_dict = {}
    if 'CPIAUCSL' in combined_data.columns:
        agg_dict['CPIAUCSL'] = 'last'
    if 'CPILFESL' in combined_data.columns:
        agg_dict['CPILFESL'] = 'last'
    if 'UNRATE' in combined_data.columns:
        agg_dict['UNRATE'] = 'last'
    if 'NAPM' in combined_data.columns:
        agg_dict['NAPM'] = 'last'
    if 'DGS2' in combined_data.columns:
        agg_dict['DGS2'] = 'mean'
    if 'DGS10' in combined_data.columns:
        agg_dict['DGS10'] = 'mean'
    if 'DFEDTARU' in combined_data.columns:
        agg_dict['DFEDTARU'] = 'last'
    
    monthly_data = combined_data.resample('M').agg(agg_dict)
    
    # Handle missing PMI data with limited forward fill (max 2 periods)
    if 'NAPM' in monthly_data.columns:
        monthly_data['NAPM'] = monthly_data['NAPM'].fillna(method='ffill', limit=2)
    
    # Create features
    print("Creating features...")
    
    # CPI year-over-year changes
    monthly_data['CPI_YoY'] = compute_yoy_change(monthly_data['CPIAUCSL'])
    monthly_data['Core_CPI_YoY'] = compute_yoy_change(monthly_data['CPILFESL'])
    
    # Term spread
    monthly_data['Term_Spread'] = monthly_data['DGS10'] - monthly_data['DGS2']
    
    # Select feature columns based on available data
    feature_cols = []
    if 'CPI_YoY' in monthly_data.columns:
        feature_cols.append('CPI_YoY')
    if 'Core_CPI_YoY' in monthly_data.columns:
        feature_cols.append('Core_CPI_YoY')
    if 'UNRATE' in monthly_data.columns:
        feature_cols.append('UNRATE')
    if 'NAPM' in monthly_data.columns:
        feature_cols.append('NAPM')
    if 'DGS2' in monthly_data.columns:
        feature_cols.append('DGS2')
    if 'DGS10' in monthly_data.columns:
        feature_cols.append('DGS10')
    if 'Term_Spread' in monthly_data.columns:
        feature_cols.append('Term_Spread')
    
    features = monthly_data[feature_cols].copy()
    
    # Apply 1-month lag to features
    features = features.shift(1)
    
    # Create target: sign of month-over-month change in fed rate
    fed_rate_diff = monthly_data['DFEDTARU'].diff()
    target = np.sign(fed_rate_diff.fillna(0)).astype(int)
    
    # Align features and target (drop first row due to diff)
    features = features.iloc[1:]
    target = target.iloc[1:]
    
    # Drop rows with missing values
    combined = pd.concat([features, target], axis=1)
    combined.columns = list(feature_cols) + ['target']
    combined = combined.dropna()
    
    print(f"Final dataset shape: {combined.shape}")
    print(f"Target distribution:\n{combined['target'].value_counts().sort_index()}")
    
    # Prepare data for modeling
    X = combined[feature_cols]
    y = combined['target']
    
    print(f"\nData range: {X.index.min().strftime('%Y-%m')} to {X.index.max().strftime('%Y-%m')}")
    
    # Time series cross-validation
    print("\nPerforming time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Base classifier with balanced class weights due to imbalanced data
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Cross-validation results
    cv_results = []
    cv_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/5...")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        
        # Store results
        accuracy = accuracy_score(y_test, y_pred)
        cv_results.append(accuracy)
        
        # Store predictions with probabilities
        test_dates = X_test.index
        for i, date in enumerate(test_dates):
            # Handle missing classes in probability output
            classes = rf.classes_
            p_cut = y_proba[i][list(classes).index(-1)] if -1 in classes else 0.0
            p_hold = y_proba[i][list(classes).index(0)] if 0 in classes else 0.0  
            p_hike = y_proba[i][list(classes).index(1)] if 1 in classes else 0.0
            
            pred_row = {
                'date': date,
                'y_true': y_test.iloc[i],
                'y_pred': y_pred[i],
                'p_cut': p_cut,
                'p_hold': p_hold,
                'p_hike': p_hike
            }
            cv_predictions.append(pred_row)
    
    # Compute overall backtest statistics
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    
    win_rate = np.mean(cv_results)
    print(f"Out-of-sample win rate (accuracy): {win_rate:.3f}")
    
    # Overall confusion matrix and classification report
    cv_df = pd.DataFrame(cv_predictions)
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(cv_df['y_true'], cv_df['y_pred'])
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(cv_df['y_true'], cv_df['y_pred'], 
                              target_names=['Cut', 'Hold', 'Hike']))
    
    # Save backtest predictions
    cv_df.to_csv('backtest_predictions.csv', index=False)
    print(f"\nBacktest predictions saved to 'backtest_predictions.csv'")
    
    # Fit final model on all available data
    print("\n" + "=" * 50)
    print("LIVE PREDICTION")
    print("=" * 50)
    
    print("Fitting final model on all available data...")
    final_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    final_model.fit(X, y)
    
    # Save the model
    joblib.dump(final_model, 'rate_model.pkl')
    print("Model saved as 'rate_model.pkl'")
    
    # Make prediction for latest available data
    latest_features = X.iloc[-1:].values
    latest_date = X.index[-1]
    
    proba = final_model.predict_proba(latest_features)[0]
    
    # Create probability dictionary
    classes = final_model.classes_
    prob_dict = {}
    prob_dict['Cut'] = proba[list(classes).index(-1)] if -1 in classes else 0.0
    prob_dict['Hold'] = proba[list(classes).index(0)] if 0 in classes else 0.0
    prob_dict['Hike'] = proba[list(classes).index(1)] if 1 in classes else 0.0
    
    print(f"\nPrediction for period following {latest_date.strftime('%Y-%m')}:")
    print("-" * 30)
    for decision, prob in prob_dict.items():
        print(f"{decision:>6}: {prob:.3f}")
    
    # Decision rule
    max_prob = max(prob_dict.values())
    if max_prob >= 0.6:
        prediction = max(prob_dict, key=prob_dict.get)
    else:
        prediction = "Uncertain"
    
    print(f"\nFinal call: {prediction}")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")

if __name__ == "__main__":
    main()