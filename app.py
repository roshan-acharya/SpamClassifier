import pickle as pk
import streamlit as st 

# Load the model
model=pk.load(open('spam_detection_model.pkl','rb'))
st.title("Spam Ham Classifier")
text=st.text_area("Enter the text")

if st.button("Predict"):
 
  result=model.predict([text])
  if result[0]==1:
    st.write("Spam")
  else: 
    st.write("Ham")      

