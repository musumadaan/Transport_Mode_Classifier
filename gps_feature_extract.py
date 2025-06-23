import pandas as pd
import numpy as np
from pathlib import Path
from feature_utils import extract_features

# this is our dataset pipeline before feeding it to the model.
# Cleaning data and making sure the format is up to the model standard is done here.

# inputs:
# Accelerometer.csv with:
#    time, seconds_elapsed, x, y, z (acceleration in 3 axes)
#
# Location.csv with:
#    time, seconds_elapsed, latitude, longitude, speed, and more

# outs: features_output.csv

# it searches through dataset/<random_folder>/ for Accelerometer.csv and Location.csv pairs
# and outputs combined feature rows from all valid folders

# label mapping per folder name
label_map = {
    "Burq_to_SFU-2025-04-03_20-19-33": "bus", #this must match the folder name inside dataset/
    "SFU_to_Burq-2025-04-03_21-56-06": "bus",
    "home_to_gym-2025-03-25_14-43-48": "car",
    "to_gym-2025-03-25_16-18-36": "car",
    "metro-2025-03-29_04-22-42": "train",
    "metro-2025-03-29_04-41-51": "train",
    "metro-2025-03-30_04-32-02": "train",
    "metro-2025-03-30_04-39-56": "train",
    "metro-2025-03-30_04-55-25": "train",
    "metro-2025-04-01_00-07-07": "train",
    "metro-2025-04-01_04-37-04": "train",
    "metro-2025-04-01_04-57-31": "train",
    "metro-2025-04-01_04-46-56": "train",
    "metro-2025-04-01_22-03-39": "train",
    "metro-2025-04-01_23-52-30": "train",
    "metro_-2025-04-10_04-35-06": "train",
    "metro-2025-04-10_04-45-51": "train",
    "bus-2025-03-29_05-18-40": "bus",
    "bus-2025-03-30_14-13-41": "bus",
    "bus-2025-03-30_14-56-02": "bus",
    "bus-2025-03-30_15-11-43": "bus",
    "bus-2025-04-07_15-02-39": "bus",
    "bus-2025-04-10_05-05-53": "bus",
    "bus-2025-04-10_05-28-04": "bus",
    "bus-2025-04-10_04-09-57": "bus",
    "620_Prairie_Ave-2025-04-05_02-49-09-20250406T060243Z-001": "car",
    "600_678_Prairie_Ave-2025-04-05_03-08-23-20250406T060244Z-001": "car",
    "600_678_Prairie_Ave-2025-04-05_02-59-22-20250406T060245Z-001": "car",
    "569_Prairie_Ave-2025-04-05_02-39-28-20250406T060245Z-001": "car",
    "to_gym-2025-03-31_23-39-42-20250406T060548Z-001": "car",
    "Car-2025-04-06_05-21-13": "car",
    "metro-2025-04-03_21-48-44": "train",
    "metro-2025-04-03_22-05-22": "train",
    "bus-2025-04-06_15-16-48": "bus",
    "bus-2025-04-05_15-08-16": "bus",
    "bus-2025-04-05_14-47-54": "bus",
    "514_Prairie_Ave-2025-04-10_19-55-48": "car",
    "514_Prairie_Ave-2025-04-10_17-17-55": "car",
    "514_Prairie_Ave-2025-04-09_23-41-33": "car",
    "1970_Oxford_Connector-2025-04-10_03-54-53": "car",
    "2370_Ottawa_St-2025-04-10_02-31-25": "car",
    "2875_Shaughnessy_St-2025-04-09_23-57-26":"car"
    
    # add more session-to-label mappings here
}

def find_file_ignore_case(folder, filename):
    for file in folder.iterdir():
        if file.name.lower() == filename.lower():
            return file
    return None

# process one folder's Accelerometer + Location in pairs
def process_single_pair(acc_file, loc_file, source_id, window_size=30):
    acc_df = pd.read_csv(acc_file)
    loc_df = pd.read_csv(loc_file)

    acc_df['timestamp'] = acc_df['seconds_elapsed']
    loc_df['timestamp'] = loc_df['seconds_elapsed']

    # Calc total acceleration magnitude from x, y, z axes
    acc_df['acc_magnitude'] = np.sqrt(acc_df['x']**2 + acc_df['y']**2 + acc_df['z']**2)

    #assign time windows (timestamp)
    acc_df['window'] = (acc_df['timestamp'] // window_size).astype(int)
    loc_df['window'] = (loc_df['timestamp'] // window_size).astype(int)

    #extract features where both sensors overlap
    feature_rows = []
    common_windows = sorted(set(acc_df['window']).intersection(set(loc_df['window'])))

    for window in common_windows:
        acc_window = acc_df[acc_df['window'] == window]
        loc_window = loc_df[loc_df['window'] == window]
        features = extract_features(acc_window, loc_window)
        features['window'] = window
        features['source_id'] = source_id
        features['label'] = label_map.get(source_id, 'unknown')  # assign label if available
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)

# main pipeline to loop through dataset/* subfolders and build full dataset
def main():
    root_dir = Path('dataset')
    all_feature_dfs = []

    for subdir in root_dir.iterdir():
        acc_file = find_file_ignore_case(subdir, 'Accelerometer.csv')
        loc_file = find_file_ignore_case(subdir, 'Location.csv')

        if acc_file is not None and loc_file is not None:
            print(f"Processing: {subdir.name}")
            try:
                df = process_single_pair(acc_file, loc_file, subdir.name)
                all_feature_dfs.append(df)
            except Exception as e:
                print(f"  Failed to process {subdir.name}: {e}")
        else:
            print(f"[!] Skipped {subdir.name}: Missing Accelerometer.csv or Location.csv")

    if all_feature_dfs:
        combined_df = pd.concat(all_feature_dfs, ignore_index=True)
        combined_df.to_csv('features_output.csv', index=False)
        print("\nAll features extracted and saved to features_output.csv")
    else:
        print("No valid Accelerometer + Location pairs found.")

if __name__ == '__main__':
    main()