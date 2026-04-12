# 1 -> Yes
# 0 -> No
# 1 ->Female    0->Men
# For feature scaling, scaler is exported as scaler.pkl
# model is model.pkl
# column names ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'ContractType','TotalCharges', 'TechSupport', 'DSL', 'Fiber Optic', 'Unknown']

  
import streamlit as st
import joblib
import numpy as np

scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")
oe = joblib.load("OrdinalEncoder.pkl")

st.title("Customer Churn Prediction")
st.divider()
Age =st.number_input("Enter your Age",value=20, min_value=10, max_value=100)
Gender = st.selectbox("Select your Gender", ["Male", "Female"])
Tenure= st.number_input("Enter your Tenure", value =19)
Monthly_Charges= st.number_input("Enter the Monthly Charges (in $)", min_value=15, max_value=200,value= 40)
contractType = st.selectbox("Select your contract type", ['Month-to-Month', 'One-Year', 'Two-Year'])
Total_Charges= Tenure * Monthly_Charges
TechSupport = st.selectbox("TechSupport", ['Yes', 'No'])
Internet = st.selectbox("Select your Internet service", ['DSL', 'Fiber Optic', 'Unknown'])
st.divider()

Internet_Service = [0,0,0]
if Internet == 'DSL':
    Internet_Service[0] = 1 
elif Internet == "Fiber optic":
    Internet_Service[1] = 1
else:
    Internet_Service[2] = 1 

prediction= st.button("Predict")
if prediction:
    contractType_encoded = oe.transform([[contractType]])[0][0]
    print(contractType_encoded)
    gender = 1 if Gender=="Female" else 0
    TechSupport= 1 if TechSupport=="Yes" else 0
    col = [Age, gender, Tenure, Monthly_Charges, contractType_encoded, Total_Charges, TechSupport] 
    col.extend(Internet_Service)
    print(col)
    colArr = np.array(col)
    print(colArr)
    transform = scaler.transform([colArr])
    result=model.predict(transform)[0]
    resShow = "Churn" if result==1 else  "Not Churn"
    st.success(resShow)
else :
    st.error("please enter the correct details and click on the predict button")



