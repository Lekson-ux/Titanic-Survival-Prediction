import streamlit as st
import pandas as pd
import pickle
import sklearn
import streamlit
import numpy as np


load_model = pickle.load(open('Titanic_model.pkl', 'rb'))

input_data = ([[12, 30.0, 0, 1]])


prediction = load_model.predict(input_data)
print(prediction)


if prediction[0] == 0:
    print("The passenger did not survive.")
else:
    print("The passenger survived.")