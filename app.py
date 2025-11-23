import pickle as pk
import streamlit as st 

#load model and vectorizer
with open('./models/best_spam_model.pkl', 'rb') as f:
    model = pk.load(f)
with open('./models/vectorizer.pkl', 'rb') as f:
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

