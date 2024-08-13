import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from tensorflow.keras.models import load_model

# Download stopwords if not already downloaded

nltk.download('punkt')
nltk.download('stopwords')

# Function to transform the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load vectorizer and model
vectorizer = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = load_model('Spam_classifier.h5')

# Streamlit UI
st.title('Email/SMS Spam Classifier')

# Single input field
input_sms = st.text_input('Enter the message')

# Prediction button
if st.button('Predict'):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms]).toarray()
        result = model.predict(vector_input)[0][0]  # Ensure to get the first element
        if result > 0.5:  # Assuming a binary classifier with sigmoid activation
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Please enter a message")
