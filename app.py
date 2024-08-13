import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from tensorflow.keras.models import load_model
import os 

nltk.download('punkt')
# Define the directory where NLTK data should be downloaded
nltk_data_dir = '/home/appuser/nltk_data'

# Ensure the directory exists
os.makedirs(nltk_data_dir, exist_ok=True)

punkt_path = os.path.join(nltk_data_dir, 'tokenizers/punkt')

# Download stopwords to the specified directory
nltk.download('stopwords', download_dir=nltk_data_dir)

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
# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the vectorizer file
vectorizer_path = os.path.join(base_dir, 'vectorizer (3).pkl')

# Load the vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the model file
model_path = os.path.join(base_dir, 'Spam_classifier.h5')

model = load_model(model_path)

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
