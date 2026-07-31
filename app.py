import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

@st.cache_resource
def load_assets():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, tfidf = load_assets()

lemmatizer = WordNetLemmatizer()
default_stopwords = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'cannot'}
custom_stopwords = default_stopwords - negation_words

CONTRACTIONS = {
    "dont": "do not", "don't": "do not",
    "cant": "cannot", "can't": "cannot",
    "wont": "will not", "won't": "will not"
}

def clean_text(text):
    text = text.lower()
    words = [CONTRACTIONS.get(w, w) for w in text.split()]
    text = " ".join(words)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in custom_stopwords and len(w) > 1]
    return " ".join(cleaned)

st.set_page_config(page_title="Smartphone Review Sentiment Analyzer", page_icon="📱")

st.title("📱 Smartphone Review Sentiment Analyzer")
st.write("Enter a user review below to predict sentiment (**Positive**, **Neutral**, or **Negative**).")

user_input = st.text_area("Review Text:", placeholder="e.g., Camera is not working...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned_review = clean_text(user_input)
        vectorized_text = tfidf.transform([cleaned_review])
        
        prediction = model.predict(vectorized_text)[0]
        probs = model.predict_proba(vectorized_text)[0]
        classes = model.classes_
        
        st.subheader("Result:")
        if prediction == "positive":
            st.success(f"**Predicted Sentiment:** {prediction.upper()} 🎉")
        elif prediction == "negative":
            st.error(f"**Predicted Sentiment:** {prediction.upper()} ⚠️")
        else:
            st.info(f"**Predicted Sentiment:** {prediction.upper()} 😐")
            
        st.write("### Confidence Breakdown:")
        prob_df = {classes[i].capitalize(): float(probs[i]) for i in range(len(classes))}
        st.bar_chart(prob_df)
