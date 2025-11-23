import pickle as pk
import streamlit as st 
import os
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, './Models/best_spam_model.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, './Models/vectorizer.pkl')
#load model and vectorizer
with open(MODEL_PATH, 'rb') as f:
    model = pk.load(f)
with open(VECTOR_PATH, 'rb') as f:
    vectorizer = pk.load(f)

#streamlit app
st.title("Spam Detection App")
user_input = st.text_area("Enter your message here:")
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        #vectorize input
        input_vector = vectorizer.transform([user_input])
        
        #make prediction
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]

        #display result
        if prediction == 1:
            st.error(f"The message is classified as: SPAM (Confidence: {prediction_proba[1]:.2f})")
        else:
            st.success(f"The message is classified as: HAM (Confidence: {prediction_proba[0]:.2f})")

