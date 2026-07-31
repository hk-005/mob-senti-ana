# 📱 Mobile Phone Review Sentiment Analyzer

A machine learning web application that predicts the sentiment (Positive, Negative, or Neutral) of mobile phone customer reviews using Natural Language Processing (NLP) and Logistic Regression. Built with Python and deployed using Streamlit Community Cloud.

🚀 **Live Demo:** [Launch the App]([https://mob-senti-ana.streamlit.app) *(Replace with your actual Streamlit URL)](https://mob-senti-ana-djsmpgsw8mvahqgxnqtyvu.streamlit.app)*

---

## 📌 Features

* **Real-Time Sentiment Analysis:** Input any smartphone review text and get instant sentiment predictions.
* **Interactive Web Interface:** Clean, user-friendly UI powered by Streamlit.
* **Pre-trained ML Pipeline:** Utilizes TF-IDF vectorization paired with a trained classification model for fast and accurate predictions.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning & NLP:** `scikit-learn`, `nltk`, `pandas`, `joblib`
* **Deployment:** Streamlit Community Cloud
* **Version Control:** Git & GitHub

---

## 📁 Repository Structure

```text
mob-senti-ana/
│
├── app.py                  # Main Streamlit web application script
├── requirements.txt        # Python dependencies for deployment
├── sentiment_model.pkl     # Pre-trained sentiment classification model
└── tfidf_vectorizer.pkl    # Pre-trained TF-IDF vectorizer
