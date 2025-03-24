import joblib
import pandas as pd
def dise():
    disease_features = {
        "heart_disease": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
                          'oldpeak', 'slope', 'ca', 'thal'],
        "diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age'],
        "lung_cancer": ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                        'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
                        'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'],
        "thyroid": ['Age', 'Gender', 'Pregnancy', 'Family_History_of_Thyroid', 'Goiter', 'Fatigue',
                    'Weight_Change', 'Hair_Loss', 'Heart_Rate_Changes', 'Sensitivity_to_Cold_or_Heat',
                    'Increased_Sweating', 'Muscle_Weakness', 'Constipation_or_More_Bowel_Movements',
                    'Depression_or_Anxiety', 'Difficulty_Concentrating_or_Memory_Problems', 'Dry_or_Itchy_Skin'],
        "parkinsons": ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
                       'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
                       'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    }
    return disease_features

def get_user_input(disease):
    """
    Collects user input for the selected disease based on its required features.
    """
    feature_inputs = {}

    disease_features=dise()

    features = disease_features.get(disease, [])
    if not features:
        print("❌ Invalid disease selection!")
        return None

    print(f"\n🔹 Enter values for {disease} prediction:")

    for feature in features:
        if feature.lower() in ['sex', 'gender']:
            value = input(f"{feature} (Enter 1 for Male, 0 for Female): ")
        elif feature.lower() in ['pregnancy', 'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
                                 'chronic disease', 'fatigue ', 'allergy ', 'wheezing', 'alcohol consuming',
                                 'coughing', 'shortness of breath', 'swallowing difficulty', 'chest pain',
                                 'family_history_of_thyroid', 'goiter']:  
            value = input(f"{feature} (Enter 1 for Yes, 0 for No): ")
        else:
            value = input(f"{feature}: ")

        feature_inputs[feature] = float(value)

    return [list(feature_inputs.values())]  # Convert to list format

def predict_disease(disease):
    """
    Loads the trained model and predicts the disease.
    """
    try:
        model = joblib.load(f"models/{disease}_model.pkl")  # Adjust path if needed
    except FileNotFoundError:
        print(f"❌ Model for {disease} not found!")
        return

    user_input = get_user_input(disease)
    if user_input is None:
        return

    # prediction = model.predict(user_input)[0]
    disease_features=dise()
    user_input_df = pd.DataFrame(user_input, columns=disease_features[disease])
    prediction = model.predict(user_input_df)[0]

    # Convert result to human-readable output
    result_text = f"⚠Suffering from {disease.replace('_', ' ')}" if prediction == 1 else f"✅Not suffering from {disease.replace('_', ' ')}"
    print(f"\n Predicted result for {disease}: {result_text}")


   

def main():
    """
    Main function to run the Disease Prediction System.
    """
    print("\n🩺 Disease Prediction System 🩺")
    print("Available diseases: thyroid, heart_disease, lung_cancer, diabetes, parkinsons\n")

    disease = input("Enter the disease you want to predict: ").strip().lower()
    predict_disease(disease)

if __name__ == "__main__":
    main()
