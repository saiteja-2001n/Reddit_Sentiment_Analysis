# Reddit_Sentiment_Analysis

A machine learning web application built with **Streamlit** that classifies Reddit comments into three categories: **Positive**, **Neutral**, or **Negative**. This project uses Natural Language Processing (NLP) to understand the underlying sentiment of online discussions.

## 🚀 Live Demo
https://huggingface.co/spaces/saiteja2001/Reddit_Sentiment_Analysis

## ✨ Features
* **Real-time Analysis:** Enter any text or Reddit comment and get instant sentiment classification.
* **Confidence Scoring:** Displays the probability percentage for all three classes (Positive, Negative, Neutral).
* **NLP Preprocessing:** Implements advanced text cleaning including lemmatization, stop-word removal, and regex-based URL/noise filtering.
* **Interactive UI:** Clean and simple interface built with Streamlit.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **NLP:** NLTK (Natural Language Toolkit), TF-IDF Vectorization
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Pickle
  
## 📂 Project Structure
- app.py: The main Streamlit web application.
- reddit_sentimental_model.pkl: The trained Logistic Regression model.
- reddit_tfidf_vectorizer.pkl: The TF-IDF vectorizer used to transform text into numerical data.
- reddit_sentimental.ipynb: Jupyter Notebook detailing the data collection (PRAW), EDA, and model training.
- requirements.txt: Configuration file for environment dependencies.

## 📊 How it Works
- Text Cleaning: The input text is converted to lowercase, URLs are removed, and words are reduced to their root form (lemmatization).
- Vectorization: The cleaned text is converted into a numerical format using a pre-trained TF-IDF Vectorizer.
- Classification: The Logistic Regression model predicts the sentiment and provides a confidence score for each category
