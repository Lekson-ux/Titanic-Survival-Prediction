import streamlit as st
import pandas as pd
import pickle


# Load the trained machine learning model
model = pickle.load(open("Titanic_model.pkl", "rb"))



# Create the Streamlit app
st.title("Titanic Survival Prediction")
st.subheader('Developer: Usman Oladapo :sunglasses:')
st.write("### Provide the information below to predict the survival status of passengers in 1912 Titanic ship wreck")


# Collect input from the user
age = st.number_input("Age", min_value=0, max_value=100)
fare = st.number_input('Passenger Fare(£)', min_value= 0.00, max_value = 1000.00)
gender = st.radio("Gender", ["Male", "Female"])
pclass = st.selectbox("Passenger Class", ['1st', '2nd', '3rd'])

# Preprocess the input data
input_data = pd.DataFrame({"Age": [age], 'Fare': [fare],  "Sex": [gender], "Pclass": [pclass]})
input_data["Sex"] = input_data["Sex"].map({"Male": 0, "Female": 1})
input_data["Pclass"] = input_data["Pclass"].map({"1st": 1, "2nd": 2, '3rd': 3})

# Use the model to make a prediction
prediction = model.predict(input_data)[0]

# Display the prediction to the user

if st.button('Predict survival status'):
  if prediction == 0:
    st.write("The passenger did not survive.")
  else:
    st.write("The passenger survived.")






