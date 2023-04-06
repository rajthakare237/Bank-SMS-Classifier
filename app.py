import streamlit as st
import pickle
!pip install scikit-learn

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Bank SMS Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    vector_input = tfidf.transform([input_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Bank SMS")
    else:
        st.header("Normal SMS")
