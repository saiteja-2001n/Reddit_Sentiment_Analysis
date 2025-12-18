import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
with open('reddit_sentimental_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('reddit_tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Text cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\s+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Sentiment prediction function
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    text_vector = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(text_vector)
    return prediction[0]

# Streamlit app
def main():
    st.title("Reddit Comment Sentiment Analysis")
    st.write("""
    This app analyzes the sentiment of Reddit comments using a machine learning model.
    It classifies comments as Positive, Negative, or Neutral.
    """)
    
    # Input text box
    user_input = st.text_area("Enter a Reddit comment to analyze:", "")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # Get prediction
            sentiment = predict_sentiment(user_input)
            
            # Display results
            st.subheader("Sentiment Analysis Result:")
            
            if sentiment == 'positive':
                st.success("Positive 😊")
            elif sentiment == 'negative':
                st.error("Negative 😠")
            else:
                st.info("Neutral 😐")
            
            # Show confidence scores (optional)
            st.write("### Detailed Analysis:")
            cleaned_text = clean_text(user_input)
            text_vector = tfidf.transform([cleaned_text]).toarray()
            proba = model.predict_proba(text_vector)[0]
            
            for i, class_ in enumerate(model.classes_):
                st.write(f"{class_.capitalize()}: {proba[i]*100:.2f}%")
        else:
            st.warning("Please enter a comment to analyze.")

if __name__ == '__main__':
    main()