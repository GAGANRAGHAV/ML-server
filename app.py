import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Your repository ID remains the same
MODEL_ID = "gaganraghav/scam_analysis"
SUBFOLDER = "scam_detection_model"

# Load Model and Tokenizer from Hugging Face Hub, specifying the subfolder where the model files reside
model = BertForSequenceClassification.from_pretrained(MODEL_ID, subfolder=SUBFOLDER)
tokenizer = BertTokenizer.from_pretrained(MODEL_ID, subfolder=SUBFOLDER)

# Function for Prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Scam" if prediction == 1 else "Not a Scam"

# Streamlit UI
st.title("AI-Powered Scam Call Detector")
st.write("Enter a message to check if it's a scam or not.")

user_input = st.text_area("Enter Text:", "")

if st.button("Check Scam"):
    if user_input.strip():
        result = predict(user_input)
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter some text to analyze.")