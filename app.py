import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from tensorflow.keras.models import load_model

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    
    y = [word for word in text if word.isalnum()]  # Remove non-alphanumeric characters
    
    # Remove stopwords and punctuation
    y = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Apply stemming
    ps = PorterStemmer()
    y = [ps.stem(word) for word in y]
    
    return " ".join(y)

# Load the vectorizer and the trained model
vectorizer = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = load_model('Spam_classifier.h5')

# Streamlit UI
st.title('Email/SMS Spam Classifier')

# Input field for message
input_sms = st.text_input('Enter the message')

# Predict button
if st.button('Predict'):
    if input_sms:
        # Preprocess the input message
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the input
        vector_input = vectorizer.transform([transformed_sms]).toarray()
        
        # Predict the class
        result = model.predict(vector_input)[0][0]  # Get the prediction
        
        # Display the result
        if result > 0.5:  # Assuming a threshold of 0.5 for spam
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Please enter a message")
