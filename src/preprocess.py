import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column):
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Encode categorical columns using Label Encoding for binary categories
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le  # Store encoders if needed for later decoding

    # Apply Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, target_column

if __name__ == "__main__":
    datasets = {
        "thyroid": "dataset/thyroid.csv",
        "heart disease": "dataset/heart disease.csv",
        "lung cancer": "dataset/lung_cancer.csv",
        "diabetes": "dataset/diabetes.csv",
        "parkinsons": "dataset/parkinsons.csv"
    }

    targets = {
        "thyroid": "Thyroid_Risk_Level",
        "heart disease": "target",
        "lung cancer": "LUNG_CANCER",
        "diabetes": "Outcome",
        "parkinsons": "status"
    }

    for disease, file_path in datasets.items():
        try:
            df = pd.read_csv(file_path)
            target = targets[disease]
            X_train, X_test, y_train, y_test, target = preprocess_data(df, target)
            print(f"Preprocessing complete for {file_path}. Target column: {target}. Training data shape: {X_train.shape}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
