import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load model and scaler
def load_model():
    with open('diabetes.pkl', 'rb') as file:
        ridge_model,lasso_model, scaler = pickle.load(file)
    return ridge_model, lasso_model, scaler

# Preprocess user input
def preprocessing_input_data(data, scaler):
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    return df_scaled

# Predict function
def predict_data(data):
    ridge_model,lasso_model, scaler = load_model()
    processed_data = preprocessing_input_data(data, scaler)
    prediction = ridge_model.predict(processed_data)
    return prediction

# MongoDB Setup
uri = "mongodb+srv://vaishnavi:vaishhh10@cluster0.yxgfopp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['diabetes']
collection = db["diabetes_prediction"]

# Main UI
def main():
    st.title("🩺 Diabetes Progression Predictor")
    st.write("Enter patient medical info to estimate disease progression score:")

    # Input fields
    age = st.number_input("Age (years)", min_value=1, max_value=100)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
    bp = st.number_input("Blood Pressure", min_value=60.0, max_value=200.0)
    s1 = st.number_input("S1 (Total Cholesterol)")
    s2 = st.number_input("S2 (LDL)")
    s3 = st.number_input("S3 (HDL)")
    s4 = st.number_input("S4 (TCH / HDL Ratio)")
    s5 = st.number_input("S5 (LTG - Triglycerides)")
    s6 = st.number_input("S6 (Blood Sugar Level)")

    if st.button("Predict Progression"):
        sex_val = 1 if sex == "Male" else 0

        user_data = {
            "age": age,
            "sex": sex_val,
            "bmi": bmi,
            "bp": bp,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6
        }

        prediction = predict_data(user_data)
        st.success(f"🎯 Predicted Disease Progression Score: {prediction[0]:.2f}")

        # Save to MongoDB
        user_data['prediction'] = float(prediction[0])
        collection.insert_one(user_data)
        st.info("✅ Prediction saved to database.")

# Run the app
if __name__ == "__main__":
    main()
