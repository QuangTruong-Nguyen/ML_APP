import gradio as gr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle



with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


def proLocation(location):
    if location=='Rural':
        return 0
    elif location=='Urban':
        return 1
    else:
        return 2

def predict_loan_amount(gender, age, income, income_stability, property_age,  property_price, property_location):
    input_data = {
        
        "Gender": [1 if gender == 'M' else 0],
        "Age": [age],
        "Income (USD)": [income],
        "Income Stability": [1 if income_stability == 'Low' else 0],
        "Property Age": [property_age],
        "Property Price": [property_price],
        "Property Location": [proLocation(property_location)],
       
    }
    input_df = pd.DataFrame(input_data)
   
    prediction = model.predict(input_df.to_numpy())
    return prediction[0]

# Gradio interface
iface = gr.Interface(
    fn=predict_loan_amount,
    inputs=[
        gr.Radio(['F', 'M'], label='Gender'),
        gr.Slider(18, 70, step=1, label='Age'),
        gr.Number(label='Income (USD)'),
        gr.Radio(['Low', 'High'], label='Income Stability'),
        gr.Number(label='Property Age'),
        gr.Number(label='Property Price'),
        gr.Radio(['Rural', 'Urban', 'Semi-Urban'], label='Property Location'),
    ],
    outputs="number",
    live=True
)

iface.launch()
