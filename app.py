# 1 -> Yes
# 0 -> No
# 1 ->Female    0->Men
# For feature scaling, scaler is exported as scaler.pkl
# model is model.pkl
# column names ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges']

  
import streamlit as st
import joblib
import numpy as np

scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")

st.title("Customer Churn Prediction")
st.divider()
Age =st.number_input("Enter your age", min_value=10, max_value=100)
Gender = st.selectbox("Select your gender", ["Male", "Female"])
Tenure= st.number_input("Enter your Tenure")
Monthly_Charges= st.number_input("Enter your Monthly_Charges", min_value=15, max_value=200,value= 40)
Total_Charges= st.number_input("Enter your TotalCharges", max_value=15000)
st.divider()

prediction= st.button("Predict")
if prediction:
    gender = 1 if Gender=="Female" else 0
    col = [Age, gender, Tenure, Monthly_Charges, Total_Charges]
    colArr = np.array(col)
    print(colArr)
    transform = scaler.transform([colArr])
    result=model.predict(transform)[0]
    resShow = "Churn" if result==1 else  "Not Churn"
    st.success(resShow)
else :
    st.error("please enter the correct number and click on the predict button")



