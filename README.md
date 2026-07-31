# 📱 Mobile Phone Review Sentiment Analyzer

A machine learning web application that predicts the sentiment (Positive, Negative, or Neutral) of mobile phone customer reviews using Natural Language Processing (NLP) and Logistic Regression. Built with Python and deployed using Streamlit Community Cloud.

🚀 **Live Demo:** [Launch the App](https://mob-senti-ana-djsmpgsw8mvahqgxnqtyvu.streamlit.app)

---

## 📌 Features

* **Real-Time Single Review Analysis:** Input any smartphone review text to get instant sentiment predictions with class probabilities.
* **Batch CSV Processing:** Upload bulk CSV review files, view distribution charts, metric summaries, and download classified predictions.
* **Hybrid Classification Engine:** Combines TF-IDF vectorization and Logistic Regression with rule-based heuristics for negation handling and keyword overrides.
* **Automated CI/CD Pipeline:** Includes a `pytest` test suite integrated with GitHub Actions for automated unit testing on every commit.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning & NLP:** `scikit-learn`, `nltk`, `pandas`, `joblib`
* **Testing & CI/CD:** `pytest`, GitHub Actions
* **Deployment:** Streamlit Community Cloud
* **Version Control:** Git & GitHub

---

## 📁 Repository Structure

```text
mob-senti-ana/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow configuration
├── app.py                      # Main Streamlit web application script
├── test_app.py                 # Pytest unit test suite
├── requirements.txt            # Python dependencies for deployment
├── sentiment_model.pkl         # Pre-trained sentiment classification model
└── tfidf_vectorizer.pkl        # Pre-trained TF-IDF vectorizer
