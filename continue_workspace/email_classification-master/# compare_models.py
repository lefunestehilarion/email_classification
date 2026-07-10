# compare_models.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from zero_shot.multiclass_email_classification.models import ZeroShotModel  # Assuming this is the correct path

# Load datasets
dataset_path_v4 = '/home/shepard/Prgm/continue_workspace/email_classification/v4_classify_csv/data/email_urgent_dataset.csv'
dataset_path_zero_shot = '/home/shepard/Prgm/continue_workspace/email_classification/zero_shot/multiclass-email-classification/data/email_urgent_dataset.csv'

df_v4 = pd.read_csv(dataset_path_v4)
df_zero_shot = pd.read_csv(dataset_path_zero_shot)

# Load models
lstm_model_path = '/home/shepard/Prgm/continue_workspace/email_classification/v4_classify_csv/models/lstm_model.keras'
zero_shot_model = ZeroShotModel()

lstm_model = load_model(lstm_model_path)

# Preprocess data for LSTM model
# ...existing code...

# Preprocess data for Zero-Shot model
# ...existing code...

# Run predictions
predictions_lstm = lstm_model.predict(preprocessed_data_v4)
predictions_zero_shot = zero_shot_model.predict(preprocessed_data_zero_shot)

# Evaluate models
accuracy_lstm = accuracy_score(y_true, predictions_lstm)
precision_lstm = precision_score(y_true, predictions_lstm, average='weighted')
recall_lstm = recall_score(y_true, predictions_lstm, average='weighted')
f1_lstm = f1_score(y_true, predictions_lstm, average='weighted')

accuracy_zero_shot = accuracy_score(y_true, predictions_zero_shot)
precision_zero_shot = precision_score(y_true, predictions_zero_shot, average='weighted')
recall_zero_shot = recall_score(y_true, predictions_zero_shot, average='weighted')
f1_zero_shot = f1_score(y_true, predictions_zero_shot, average='weighted')

# Print results
print("LSTM Model Metrics:")
print(f"Accuracy: {accuracy_lstm}")
print(f"Precision: {precision_lstm}")
print(f"Recall: {recall_lstm}")
print(f"F1 Score: {f1_lstm}")

print("\nZero-Shot Model Metrics:")
print(f"Accuracy: {accuracy_zero_shot}")
print(f"Precision: {precision_zero_shot}")
print(f"Recall: {recall_zero_shot}")
print(f"F1 Score: {f1_zero_shot}")