import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


# Load all files and auto-label based on the filename
# file structure: 
# data/subject_activity.csv
all_data = []
data_dir = "./data"

def safe_corr(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.corrcoef(x, y)[0, 1]
        return 0 if np.isnan(r) else r

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        # Load the data and add labels
        df = pd.read_csv(os.path.join(data_dir, filename))
        # Verify columns
        required_columns = ["Timestamp", "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Label"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"File {filename} is missing required columns")
        
        # Add subject ID
        if 'Subject' not in df.columns:
            df['Subject'] = filename.split('_')[0]  # Extract from filename if needed

        all_data.append(df)

# Combine all data into a single DataFrame
combined_data = pd.concat(all_data).sort_values("Timestamp").reset_index(drop=True)
print(f"Loaded {len(combined_data)} samples with columns: {combined_data.columns.tolist()}")
print("Class balance:", combined_data["Label"].value_counts())

window_size = 10
step_size = 5

features = []
labels = []
subjects = []

# Group by subject and label
for (subject, label), group in combined_data.groupby(["Subject", "Label"]):
    # Convert timestamps to relative time (0 to max)
    timestamps = group["Timestamp"].values - group["Timestamp"].min()
    # Create sliding windows
    for start in range(0, len(group) - window_size + 1, step_size):
        end = start + window_size
        window = group.iloc[start:end]

        # Skip windows with too few samples or mixed labels
        if len(window) < window_size or window['Label'].nunique() > 1:
            #print(f"Processing window: start={start}, end={end}, len(window)={len(window)}, unique labels={window['Label'].nunique()}")
            print('first')
            continue

        # Extract features for this window
        sensor_data = window[["Sensor1", "Sensor2", "Sensor3", "Sensor4"]].values

        # Normalize the sensor data
        sensor_data = (sensor_data - 25) / (850 - 25)  # Custom range [25, 850]
        sensor_data = 1 - sensor_data  # Invert: low values = pressure

        # Possible features
        mean = np.mean(sensor_data, axis=0)
        #peak_to_peak = np.ptp(sensor_data, axis=0)
        #median = np.median(sensor_data, axis=0)
        #std_dev = np.std(sensor_data, axis=0)
        #max_val = np.max(sensor_data, axis=0)
        min_val = np.min(sensor_data, axis=0)
        #duration = timestamps[end-1] - timestamps[start]

        # Impact detection (valleys, not peaks)
        valley_counts = [
            len(find_peaks(-sensor_data[:, i], height=-0.7)[0])  # Detect dips (height=-0.7 = ~300 raw)
            for i in range(4)
        ]

        # Kick detection (A1 = arch)
        kick_detected = 1 if min_val[0] > 0.2 else 0  # Threshold at ~100 raw (0.9 scaled)

        # Pairwise differences
        correlation_heel_to_toes = safe_corr(sensor_data[:, 3], sensor_data[:, 2])
        diff_signal = sensor_data[:,3] - sensor_data[:,2]  # Heel - Toes
        # Frequency analysis (walking rhythm)
        fft = np.abs(np.fft.rfft(diff_signal))  # A3 (ball of foot)
        dominant_freq = np.argmax(fft[1:]) + 1 

        heel_valleys = len(find_peaks(-sensor_data[:,3], height=-0.5)[0])  # S4=Heel
        toe_peaks = len(find_peaks(sensor_data[:,2], height=0.3)[0])       # S3=Toes
        strike_ratio = heel_valleys / (toe_peaks + 1e-6)


        # Combine features
        #features.append(np.concatenate([mean, peak_to_peak, median, std_dev, max_val, min_val, np.array([duration])]))
        features.append(
            np.concatenate(
                [mean, np.array([correlation_heel_to_toes]), min_val, valley_counts, 
                np.array([kick_detected, dominant_freq]),
                np.array([
                heel_valleys,             # Number of heel strikes
                toe_peaks,                # Number of toe-offs
                strike_ratio,               # Should be ~1 (1 heel strike per toe-off)
                np.mean(sensor_data[:,3] - sensor_data[:,2]),  # Heel-toe pressure difference
                np.std(diff_signal)         # Variability in stride
                ])]))
        
        labels.append(label)
        subjects.append(subject)

# Convert to numpy arrays
X = np.array(features) # Shape: (n_windows, n_features)
y = np.array(labels)   # Shape: (n_windows,)
subjects = np.array(subjects)  # Shape: (n_windows,)


# Use StratifiedShuffleSplit to preserve class balance per subject
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in splitter.split(X, y, groups=subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    break  # Just take first split if you don't need cross-validation

# After splitting, check for subject overlap:
assert len(set(subjects[train_idx]) & set(subjects[test_idx])) == 0, "Data leakage!"

# Verify subject distribution
print("Train subjects:", np.unique(subjects[train_idx], return_counts=True))
print("Test subjects:", np.unique(subjects[test_idx], return_counts=True))

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={label: max(1, 1000//count) for label, count in combined_data["Label"].value_counts().items()})
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=np.unique(y)))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Feature Importance ---
feature_names = [
    "Mean_A1", "Mean_A2", "Mean_A3", "Mean_A4", "Heels_Toes_Corr",
    "Min_A1", "Min_A2", "Min_A3", "Min_A4",
    "Valley_A1", "Valley_A2", "Valley_A3", "Valley_A4",
    "Kick_Flag", "Dominant_Freq",
    "Heel_Strikes",
    "Toe_Offs",
    "Strike_Ratio",
    "HeelToe_Diff",
    "Stride_Variability"
]
importance = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
print("Feature Importance:\n", importance)

# --- Save Model with Metadata ---
joblib.dump(
    {
        "model": model,
        "feature_names": feature_names,
        "scaler": StandardScaler().fit(X_train)  # Or your custom scaler
    },
    "gesture_classifier_with_metadata.pkl"
)