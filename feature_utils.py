import numpy as np

def extract_features(acc_window, loc_window):
    return {
        # accel features
        'acc_mean': acc_window['acc_magnitude'].mean(),
        'acc_std': acc_window['acc_magnitude'].std(),
        'acc_energy': np.sum(acc_window['acc_magnitude']**2),
        'acc_spikes': np.sum(acc_window['acc_magnitude'] > 15),

        # GPS features (ignores -1 values which probably means no signal)
        'gps_mean_speed': loc_window['speed'].replace(-1.0, np.nan).mean(),
        'gps_max_speed': loc_window['speed'].replace(-1.0, np.nan).max(),
        'gps_speed_std': loc_window['speed'].replace(-1.0, np.nan).std(),
        'gps_stops': np.sum(loc_window['speed'] < 0.5)  # counts number of near-zero speed entries
    }
