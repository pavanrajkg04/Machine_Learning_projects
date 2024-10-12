import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv('insurance.csv')

# One-hot encoding
data = pd.get_dummies(data=data, drop_first=True)

# Define features and target variable
X = data.drop(columns='charges')
y = data['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Standard scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Fit the Random Forest model
RF = RandomForestRegressor()
RF.fit(X_train, y_train)

# Function to make predictions
def predict_charges(inputs):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([inputs], columns=X.columns)
    
    # Scale the input data
    scaled_input = sc.transform(input_df)
    
    # Make a prediction
    prediction = RF.predict(scaled_input)
    
    return prediction[0]

# Streamlit UI
st.title("Insurance Charges Predictor")

# User inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker?", options=["Yes", "No"])
region = st.selectbox("Region", options=["Northeast", "Southeast", "Southwest", "Northwest"])

# Encode categorical variables for prediction
smoker_encoded = 1 if smoker == "Yes" else 0
sex_encoded = 1 if sex == "Male" else 0

# One-hot encode the region based on the input
region_encoded = [0, 0, 0]  # Northeast is omitted in drop_first encoding
if region == "Southeast":
    region_encoded[0] = 1
elif region == "Southwest":
    region_encoded[1] = 1
elif region == "Northwest":
    region_encoded[2] = 1

# Prepare the input data for prediction
inputs = [age, sex_encoded, bmi, children, smoker_encoded] + region_encoded

if st.button("Predict Charges"):
    prediction = predict_charges(inputs)
    st.write(f"Predicted Insurance Charges: ${prediction:.2f}")
