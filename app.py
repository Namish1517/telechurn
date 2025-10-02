import streamlit as st
import pandas as pd
import pickle
import random

with open("models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',  'tenure', 'PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges']

categorical_features = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV','StreamingMovies','Contract', 'PaperlessBilling','PaymentMethod']

numeric_features = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']

def generate_sample_input():
    if random.random() < 0.20:
        sample = ["Female",1,"No","No",2,"Yes","No","Fiber optic","No","No","No","No","Yes","Yes","Month-to-month","Yes","Electronic check",100,100]
    else:     
        sample = [random.choice(['Male', 'Female']),random.randint(0, 1), random.choice(['Yes','No']),random.choice(['Yes','No']),   random.randint(1, 72),  random.choice(['Yes','No']),         random.choice(['Yes','No','No phone service']),random.choice(['DSL','Fiber optic','No']),  random.choice(['Yes','No','No internet service']),  random.choice(['Yes','No','No internet service']),random.choice(['Yes','No','No internet service']), random.choice(['Yes','No','No internet service']),   random.choice(['Yes','No','No internet service']),random.choice(['Yes','No','No internet service']), random.choice(['Month-to-month','One year','Two year']),  random.choice(['Yes','No']), random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']),round(random.uniform(20,120),2),round(random.uniform(20,9000),2)]
    return ",".join(map(str, sample))


st.title("Telecom Customer Churn Predictor")
st.markdown("Paste customer data as **comma-separated values** for prediction.\n")

if st.button("Generate Random Sample Input"):
    st.session_state.sample_input = generate_sample_input()

sample_text = st.text_area(
    "Customer Data Input:",
    value=st.session_state.get("sample_input", generate_sample_input()),
    height=120
)

if st.button("Predict Churn"):
    try:
        values = [v.strip() for v in sample_text.split(",")]
        if len(values) != len(features):
            st.error(f"Expected {len(features)} values, but got {len(values)}")
        else:
            df = pd.DataFrame([dict(zip(features, values))])

            for col in numeric_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
          
            expected_features = rf_model.feature_names_in_
            df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)
           
            pred = rf_model.predict(df_encoded)[0]
            prob = rf_model.predict_proba(df_encoded)[0][1]
            st.success(f"Prediction: {'Churn' if pred==1 else 'Not Churn'} (Probability: {prob:.2f})")
    except Exception as e:
        st.error(f"Error: {e}")


