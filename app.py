import joblib

import re
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load Label Encoders
label_encoder_sentiment_path = "label_encoder_sentiment.joblib"
label_encoder_emotion_path = "label_encoder_emotion.joblib"
label_encoder_sentiment = joblib.load(label_encoder_sentiment_path)
label_encoder_emotion = joblib.load(label_encoder_emotion_path)

def data_preprocessing(text):
    # Check if 'text' is a string
    if not isinstance(text, str):
        return ""  # Return an empty string if 'text' is not a string

    # Remove all URLs
    text = re.sub(r'http\S+', '', text)

    # Remove all names starting with @
    text = re.sub(r'@\w+', '', text)

    # Remove all hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove all numeric digits
    text = re.sub(r'\d+', '', text)

    # Remove all punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove all non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]+', ' ', text)

    # Regular expression matches any string that starts with $
    text = re.sub(r'\$\w+\s*', '', text)

    # Regular expression matches any string that starts with Contract: 0x (Contract: 0x) 
    text = re.sub(r'Contract: 0x\w+\s*', '', text)

    # Regular expression matches one or more whitespace characters (\s+) and replaces them with a single space (' ')
    text = re.sub(r'\s+', ' ', text)

    # Convert the text to lower case
    text = text.lower()
    
    # Remove all single characters
    text = re.sub(r'\b\w\b', '', text)

    # Remove extra whitespaces after removing single characters
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove words with less than 3 characters
    text = ' '.join(word for word in text.split() if len(word) >= 3)

    # Remove all English Stopwords
    stop_words = stopwords.words('english')
    # Add additional stopwords
    additional_stopwords = ["also", "would", "ask", "asked"]
    stop_words.extend(additional_stopwords)
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

    return text


import streamlit as st


# Function to analyze sentiment and emotions
def analyze_text(input_text):
    # Check if the input text is empty
    if not input_text:
        return {"Sentiment": "Unknown", "Dominant_Emotion": "Unknown"}

    # Preprocess the input text
    preprocessed_text = data_preprocessing(input_text)

    # Transform the preprocessed text using the trained count_vectorizer
    text_feature_vector = count_vectorizer.transform([preprocessed_text])

    # Predict labels for the input text
    text_predictions = model.predict(text_feature_vector)

    # Customize output based on model predictions
    sentiment_mapping = {0: "Negative", 1: "Positive"}
    emotion_mapping = {
        0: "Anger",
        1: "Anticipation",
        2: "Disgust",
        3: "Fear",
        4: "Joy",
        5: "Sadness",
        6: "Surprise",
        7: "Trust",
    }

    return {
        "Sentiment": sentiment_mapping.get(text_predictions[:, 0][0], "Unknown"),
        "Dominant_Emotion": emotion_mapping.get(text_predictions[:, 1][0], "Unknown"),
    }

# Load the pre-trained model and vectorizer
# Assuming 'count_vectorizer' and 'model' are defined in your original code
# You might need to adjust the paths based on your project structure
# Load CountVectorizer
count_vectorizer_path = "count_vectorizer_model.joblib"
count_vectorizer = joblib.load(count_vectorizer_path)

# Load MultiOutputClassifier model
model_path = "multioutput_classifier_model.joblib"
model = joblib.load(model_path)


# Streamlit UI
st.title("Sentiment and Emotion Analysis")
st.text("Enter Poem:")

input_text = st.text_area("Text Entry", height=200)
result = analyze_text(input_text)

# Display the results
st.text(f"Sentiment: {result['Sentiment']}")
st.text(f"Dominant Emotion: {result['Dominant_Emotion']}")

