import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from feature_utils import extract_features
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
# Predict transport Mode.
# Predicts mode per window, either car, bus, train, or other (other = walking, cycle, or any other that cant be pooled into car, train, or bus.)
# inputs: path to Location.csv and Accelerometer.csv 
# try your own: use SensorLogger app (our labels are based on this app), or as long as your csv has proper labels, it should work. 

# 
model = joblib.load("model_transport_mode.pkl")  #call saved model

# Load test CSVs
def load_and_prepare(acc_path, loc_path, window_size=30):
    acc_df = pd.read_csv(acc_path)
    loc_df = pd.read_csv(loc_path)

    acc_df['timestamp'] = acc_df['seconds_elapsed']
    loc_df['timestamp'] = loc_df['seconds_elapsed']

    acc_df['acc_magnitude'] = np.sqrt(acc_df['x']**2 + acc_df['y']**2 + acc_df['z']**2)
    acc_df['window'] = (acc_df['timestamp'] // window_size).astype(int)
    loc_df['window'] = (loc_df['timestamp'] // window_size).astype(int)

    feature_rows = []
    common_windows = sorted(set(acc_df['window']).intersection(set(loc_df['window'])))

    for window in common_windows:
        acc_window = acc_df[acc_df['window'] == window]
        loc_window = loc_df[loc_df['window'] == window]
        features = extract_features(acc_window, loc_window)
        features['window'] = window
        feature_rows.append(features)

    return pd.DataFrame(feature_rows), loc_df  # Return loc_df for raw stats

#Temporal Smoothing
def smooth_predictions(predictions, window_size=3):
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window_size + 1)
        window = predictions[start:i + 1]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

# Predict mode for each window, apply threshold
def predict_modes(features_df, threshold=0.6, smooth=True, smooth_window=3):
    X = features_df.drop(columns=['window'], errors='ignore')
    proba = model.predict_proba(X)
    labels = model.classes_
    max_probs = proba.max(axis=1)
    pred_indices = proba.argmax(axis=1)
    
    # raw preds with the confidence fallback
    raw_predictions = [labels[i] if p >= threshold else 'walking' for i, p in zip(pred_indices, max_probs)]
    
    # Optional smoothing
    if smooth:
        return smooth_predictions(raw_predictions, smooth_window)
    else:
        return raw_predictions

# Show trip summary from raw GPS
def summarize_trip(loc_df):
    loc_df = loc_df[loc_df['speed'] >= 0]  # Remove invalid entries
    print("\nTrip Summary:")
    print(f"  Total Stops (<0.5 m/s): {np.sum(loc_df['speed'] < 0.5)}")
    print(f"  Average Speed: {loc_df['speed'].mean():.2f} m/s")
    print(f"  Max Speed: {loc_df['speed'].max():.2f} m/s")
    print(f"  Min Speed: {loc_df['speed'].min():.2f} m/s")

# Main entry point
def main():
    acc_path = Path('test-set\\test-bus\\Accelerometer.csv') #accelerometer.csv path
    loc_path = Path('test-set\\test-bus\\Location.csv') #location.csv path

    if not acc_path.exists() or not loc_path.exists():
        print("Error: One or both input files not found.")
        return

    features_df, loc_df = load_and_prepare(acc_path, loc_path)
    predictions = predict_modes(features_df)

    print("\nPredicted Transport Modes per Window:")
    for i, mode in enumerate(predictions):
        print(f"  Window {features_df['window'].iloc[i]}: {mode}")

    summarize_trip(loc_df)

if __name__ == '__main__':
    main()
