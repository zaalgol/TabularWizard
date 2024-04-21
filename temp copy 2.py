import pandas as pd
from sklearn.model_selection import train_test_spli

from sklearn.preprocessing import StandardScaler
import joblib

from src.data_preprocessing import DataPreprocessing  # For saving the scaler object

train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
train_data = pd.read_csv(train_path)




data_preprocessing = DataPreprocessing()
numeric_columns = data_preprocessing.get_numeric_columns(train_data)

scaler = StandardScaler()

# Fit the scaler on the training data (only numeric columns)
scaler.fit(train_data[numeric_columns])  # Replace 'numeric_columns' with your actual columns

# Save the fitted scaler
joblib.dump(scaler, 'scaler.joblib')

# Now you can transform your datasets using the same scaling parameters
train_data_scaled = scaler.transform(train_data[numeric_columns])
test_data_scaled = scaler.transform(test_data[numeric_columns])
inference_data_scaled = scaler.transform(inference_data[numeric_columns])

