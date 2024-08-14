import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from tensorflow.keras.models import load_model

# Download necessary NLTK data
nltk.download('punkt')

nltk_data_dir = '/home/appuser/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

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
base_dir = os.path.dirname(os.path.abspath(__file__))

vectorizer_path = os.path.join(base_dir, 'vectorizer (1).pkl')
model_path = os.path.join(base_dir, 'Spam_classifier.h5')

vectorizer = pickle.load(open(vectorizer_path, 'rb'))
model = load_model(model_path)

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
