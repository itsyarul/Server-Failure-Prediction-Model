
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('server_failure_dataset.csv')


df['stress_score'] = (
    (df['cpu_usage'] > 85).astype(int) +
    (df['memory_usage'] > 88).astype(int) +
    (df['temperature'] > 82).astype(int) +
    (df['network_errors'] > 18).astype(int) +
    (df['error_log_count'] > 45).astype(int) +
    (df['swap_usage'] > 78).astype(int) +
    (df['response_time_ms'] > 1400).astype(int)
)

df['cpu_x_memory'] = (df['cpu_usage'] * df['memory_usage']) / 10000
df['temp_x_disk'] = (df['temperature'] * df['disk_io']) / 10000
df['error_rate'] = df['error_log_count'] / (df['uptime_hours'] + 1)
df['latency_per_process'] = df['response_time_ms'] / df['process_count']


df = df.drop('server_id', axis=1)

X = df.drop('failure', axis=1)
y = df['failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)


param_grid = {
    'C':       [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_resampled, y_train_resampled)


final_model = grid_search.best_estimator_


y_pred = final_model.predict(X_test_scaled)
y_prob = final_model.predict_proba(X_test_scaled)[:, 1]


thresholds = np.arange(0.10, 0.91, 0.05)
best_threshold = 0.5
best_f1 = 0.0

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"Best Threshold:   {best_threshold:.2f}")
print(f"F1 Score at Best: {best_f1:.4f}")

y_pred_optimal = (y_prob >= best_threshold).astype(int)
print(f"\nClassification Report (threshold={best_threshold:.2f}):")
print(classification_report(y_test, y_pred_optimal, target_names=['No Failure', 'Failure']))

print(f"Confusion Matrix (threshold={best_threshold:.2f}):")
print(confusion_matrix(y_test, y_pred_optimal))


joblib.dump(final_model, 'failure_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel saved as: failure_model.pkl")


def predict_server_failure(server_metrics):
    """
    Predict server failure probability from a dictionary of server metrics.

    Parameters:
        server_metrics (dict): Dictionary containing server metric values.

    Returns:
        dict: Prediction results including probability, risk level,
              predicted failure, and recommended action.
    """
    loaded_model = joblib.load('failure_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')

    input_df = pd.DataFrame([server_metrics])

    # Apply same feature engineering
    input_df['stress_score'] = (
        (input_df['cpu_usage'] > 85).astype(int) +
        (input_df['memory_usage'] > 88).astype(int) +
        (input_df['temperature'] > 82).astype(int) +
        (input_df['network_errors'] > 18).astype(int) +
        (input_df['error_log_count'] > 45).astype(int) +
        (input_df['swap_usage'] > 78).astype(int) +
        (input_df['response_time_ms'] > 1400).astype(int)
    )
    input_df['cpu_x_memory'] = (input_df['cpu_usage'] * input_df['memory_usage']) / 10000
    input_df['temp_x_disk'] = (input_df['temperature'] * input_df['disk_io']) / 10000
    input_df['error_rate'] = input_df['error_log_count'] / (input_df['uptime_hours'] + 1)
    input_df['latency_per_process'] = input_df['response_time_ms'] / input_df['process_count']

    expected_columns = [
        'cpu_usage', 'memory_usage', 'disk_io', 'temperature', 'network_errors',
        'uptime_hours', 'error_log_count', 'process_count', 'swap_usage',
        'response_time_ms', 'stress_score', 'cpu_x_memory', 'temp_x_disk',
        'error_rate', 'latency_per_process'
    ]
    input_df = input_df[expected_columns]

    input_scaled = loaded_scaler.transform(input_df)
    probability = loaded_model.predict_proba(input_scaled)[0][1]
    probability = round(probability, 4)

    if probability < 0.30:
        risk_level = 'LOW'
        action = 'Monitor normally'
    elif probability < 0.50:
        risk_level = 'MEDIUM'
        action = 'Schedule maintenance check'
    elif probability < 0.70:
        risk_level = 'HIGH'
        action = 'Inspect within 24 hours'
    else:
        risk_level = 'CRITICAL'
        action = 'Immediate intervention required'

    predicted_failure = 1 if probability >= best_threshold else 0

    return {
        'failure_probability': probability,
        'risk_level': risk_level,
        'predicted_failure': predicted_failure,
        'action': action
    }

# Test with example servers
healthy_server = {
    'cpu_usage': 35.0, 'memory_usage': 45.0, 'disk_io': 30.0,
    'temperature': 55.0, 'network_errors': 3, 'uptime_hours': 720.0,
    'error_log_count': 8, 'process_count': 150, 'swap_usage': 20.0,
    'response_time_ms': 200.0
}

critical_server = {
    'cpu_usage': 96.0, 'memory_usage': 94.0, 'disk_io': 88.0,
    'temperature': 91.0, 'network_errors': 35, 'uptime_hours': 7200.0,
    'error_log_count': 72, 'process_count': 470, 'swap_usage': 88.0,
    'response_time_ms': 1750.0
}

print("\n--- Healthy Server Prediction ---")
result_healthy = predict_server_failure(healthy_server)
for key, value in result_healthy.items():
    print(f"  {key}: {value}")

print("\n--- Critical Server Prediction ---")
result_critical = predict_server_failure(critical_server)
for key, value in result_critical.items():
    print(f"  {key}: {value}")