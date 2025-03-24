import streamlit as st
import numpy as np
import joblib
import base64
st.set_page_config(page_title="Disease Prediction System", page_icon="⚕️")  # Add a relevant icon


# Title
st.markdown("<h1 style='color: black;'>🩺 AI Disease Prediction System </h1>", unsafe_allow_html=True)

def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function and provide the path to your image
add_bg_from_local("app/assets/61802.jpg")  


st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #71c8de;  /* Change this color */
            padding: 20px;
            border-radius: 10px;
        }
        [data-testid="stSidebar"] h1, h2, h3, h4, h5, h6, p, label {
             color: #1260de; 
           
            /* Text color */
            font-weight:bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def set_custom_css():
    st.markdown(
        """
        <style>
        h3 { 
            color: black !important;  /* Targets all subheaders */
        }
        .stMarkdown h3 { 
            color: black !important; /* Ensures markdown subheaders also turn black */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call this function at the start of your Streamlit app
set_custom_css()



# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return joblib.load(file)

# Load the trained models (update paths as needed)
heart_model = load_model("models/heart_disease_model.pkl")
diabetes_model = load_model("models/diabetes_model.pkl")
lung_model = load_model("models/lung_cancer_model.pkl")
thyroid_model = load_model("models/thyroid_model.pkl")
parkinsons_model = load_model("models/parkinsons_model.pkl")

with st.sidebar:
    st.title("🩺 Disease Predictor")
    st.subheader("👨‍⚕️ Select a Disease")
    disease = st.selectbox("Choose:", ["Heart Disease", "Diabetes", "Lung Cancer", "Thyroid", "Parkinson's"])
    
    st.subheader("📄 About")
    st.write(
        "This tool predicts possible diseases based on your input.\n\n"
        "➡️ Enter the required details in the form.\n\n"
        "➡️ Click the 'Predict' button at the bottom.\n\n"
        "➡️ The system will analyze the data and provide a prediction result."
    )
    
# Input fields based on disease selection
if disease == "Heart Disease":
    st.subheader("Enter the following details for Heart Disease Prediction")
    age = st.number_input("Age", min_value=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholesterol (mg/dl)")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2)
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise")
    slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4)
    thal = st.number_input("Thalassemia (1-3)", min_value=1, max_value=3)

    # Convert categorical values to numerical
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # Model Prediction
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

elif disease == "Diabetes":
    st.subheader("Enter the following details for Diabetes Prediction")
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0)
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Blood Pressure (mm Hg)")
    SkinThickness = st.number_input("Skin Thickness (mm)")
    Insulin = st.number_input("Insulin Level")
    BMI = st.number_input("Body Mass Index (BMI)")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age", min_value=1)

    features = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)

elif disease == "Lung Cancer":
    st.subheader("Enter the following details for Lung Cancer Prediction")

    # Define input fields
    input_fields = {
        "Gender": ["Male", "Female"],
        "Smoking": ["Yes", "No"],
        "Yellow Fingers": ["Yes", "No"],
        "Anxiety": ["Yes", "No"],
        "Peer Pressure": ["Yes", "No"],
        "Chronic Disease": ["Yes", "No"],
        "Fatigue": ["Yes", "No"],
        "Allergy": ["Yes", "No"],
        "Wheezing": ["Yes", "No"],
        "Alcohol Consuming": ["Yes", "No"],
        "Coughing": ["Yes", "No"],
        "Shortness of Breath": ["Yes", "No"],
        "Swallowing Difficulty": ["Yes", "No"],
        "Chest Pain": ["Yes", "No"]
    }

    # Store user inputs dynamically
    user_inputs = {"Age": st.number_input("Age", min_value=1)}

    for field, options in input_fields.items():
        user_inputs[field] = st.selectbox(field, options)

    # Convert categorical values to numerical
    binary_fields = list(input_fields.keys())  # Fields to be converted to binary
    user_inputs = {
        key: 1 if value == "Yes" or (key == "Gender" and value == "Male") else 0
        for key, value in user_inputs.items()
    }

    # Convert inputs to numpy array
    features = np.array(list(user_inputs.values())).reshape(1, -1)


elif disease == "Thyroid":
    st.subheader("Enter the following details for Thyroid Prediction")

    # Define input fields
    input_fields = {
        "Gender": ["Male", "Female"],
        "Pregnancy": ["Yes", "No"],
        "Family History of Thyroid": ["Yes", "No"],
        "Goiter": ["Yes", "No"],
        "Fatigue": ["Yes", "No"],
        "Weight Change": ["Yes", "No"],
        "Hair Loss": ["Yes", "No"],
        "Heart Rate Changes": ["Yes", "No"],
        "Sensitivity to Cold or Heat": ["Yes", "No"],
        "Increased Sweating": ["Yes", "No"],
        "Muscle Weakness": ["Yes", "No"],
        "Constipation or More Bowel Movements": ["Yes", "No"],
        "Depression or Anxiety": ["Yes", "No"],
        "Difficulty Concentrating or Memory Problems": ["Yes", "No"],
        "Dry or Itchy Skin": ["Yes", "No"]
    }

    # Store user inputs dynamically
    user_inputs = {"Age": st.number_input("Age", min_value=1)}

    for field, options in input_fields.items():
        user_inputs[field] = st.selectbox(field, options)

    # Convert categorical values to numerical
    user_inputs = {
        key: 1 if value == "Yes" or (key == "Gender" and value == "Male") else 0
        for key, value in user_inputs.items()
    }

    # Convert inputs to numpy array
    features = np.array(list(user_inputs.values())).reshape(1, -1)

elif disease == "Parkinson's":
    st.subheader("Enter the following details for Parkinson's Disease Prediction")
    mdvp_fo = st.number_input("MDVP:Fo(Hz)")
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)")
    mdvp_flo = st.number_input("MDVP:Flo(Hz)")
    mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)")
    mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)")
    mdvp_rap = st.number_input("MDVP:RAP")
    mdvp_ppq = st.number_input("MDVP:PPQ")
    jitter_ddp = st.number_input("Jitter:DDP")
    mdvp_shimmer = st.number_input("MDVP:Shimmer")
    mdvp_shimmer_db=st.number_input("MDVP:Shimmer(dB)")
    shimmer_apq3=st.number_input("Shimmer:APQ3")
    shimmer_apq5=st.number_input("Shimmer:APQ5")
    mdvp_apq=st.number_input("MDVP:APQ")
    shimmer_dda=st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rdpe=st.number_input("RPDE")
    dfa=st.number_input("DFA")
    spread1=st.number_input("spread1")
    spread2=st.number_input("spread2")
    d2=st.number_input("D2")
    ppe=st.number_input("PPE")
    

    features = np.array([mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, 
                         mdvp_shimmer,mdvp_shimmer_db,shimmer_apq3,shimmer_apq5,mdvp_apq,shimmer_dda, nhr, hnr,rdpe,dfa,spread1,spread2,d2,ppe]).reshape(1, -1)

# Submit button
if st.button("Predict"):
    if disease == "Heart Disease":
        prediction = heart_model.predict(features)
    elif disease == "Diabetes":
        prediction = diabetes_model.predict(features)
    elif disease == "Lung Cancer":
        prediction = lung_model.predict(features)
    elif disease == "Thyroid":
        prediction = thyroid_model.predict(features)
    elif disease == "Parkinson's":
        prediction = parkinsons_model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Potential Risk Detected! Consider consulting a doctor for further evaluation.")
    else:
        st.success("✅ No Risk Detected! Keep up a healthy lifestyle. Stay well! 😊")


