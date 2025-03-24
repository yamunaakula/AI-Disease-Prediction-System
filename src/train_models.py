import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Ensure models directory exists
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define dataset file paths and target columns
datasets = {
    "thyroid": ("dataset/thyroid.csv", "Thyroid_Risk_Level"),
    "heart_disease": ("dataset/heart disease.csv", "target"),
    "lung_cancer": ("dataset/lung_cancer.csv", "LUNG_CANCER"),
    "diabetes": ("dataset/diabetes.csv", "Outcome"),
    "parkinsons": ("dataset/parkinsons.csv", "status"),
}

# Function to train and save a model
def train_and_save_model(dataset_name, file_path, target_column):
    print(f"Training model for {dataset_name}...")

    # Load dataset
    df = pd.read_csv(file_path)

    # Splitting features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical columns to numeric using Label Encoding
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':  # Check if column is categorical
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le  # Store encoder for future use

    # Encode target variable if it's categorical
    if y.dtype == 'object':
        y_le = LabelEncoder()
        y = y_le.fit_transform(y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} model accuracy: {accuracy:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}\n")

# Train and save models for all datasets
for name, (path, target) in datasets.items():
    train_and_save_model(name, path, target)

print("✅ All models trained and saved successfully!")
