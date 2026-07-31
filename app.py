import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smartphone Review Sentiment Analyzer",
    page_icon="📱",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. Download NLTK Resources & Load Model Assets
# -----------------------------------------------------------------------------
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

try:
    model, tfidf = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Preprocessing Setup
# -----------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
default_stopwords = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'cannot'}
custom_stopwords = default_stopwords - negation_words

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in custom_stopwords
    ]
    return ' '.join(cleaned_tokens)

# -----------------------------------------------------------------------------
# 4. Streamlit UI Layout
# -----------------------------------------------------------------------------
st.title("📱 Smartphone Review Sentiment Analyzer")
st.write("Enter a user review below to predict sentiment (**Positive**, **Neutral**, or **Negative**).")

# Sidebar Info
st.sidebar.title("📌 Model Details")
st.sidebar.info("Model: Logistic Regression\nVectorizer: TF-IDF\nPipeline: NLTK Lemmatization & Custom Negation Processing")

# Input Text Box
user_review = st.text_area(
    "Review Text:", 
    placeholder="e.g., Display is not good but processor performance is better.",
    height=120
)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. Prediction Logic
# -----------------------------------------------------------------------------
if st.button("Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        # Preprocess text
        cleaned_review = preprocess_text(user_review)
        review_vec = tfidf.transform([cleaned_review])
        
        # Calculate probabilities
        probabilities = model.predict_proba(review_vec)[0]
        max_prob = max(probabilities)
        raw_pred = str(model.classes_[probabilities.argmax()]).strip().lower()

        # Rule 1: Explicit contrast words indicate mixed/neutral review
        contrast_words = ["but", "however", "although", "whereas"]
        has_contrast = any(re.search(rf"\b{word}\b", user_review.lower()) for word in contrast_words)

        # Determine Final Sentiment Label
        if has_contrast:
            sentiment = "NEUTRAL 😐"
            color_box = st.warning
            reason = "Detected contrasting statements in review."
        elif raw_pred in ['positive', 'pos', '1']:
            sentiment = "POSITIVE 🎉"
            color_box = st.success
            reason = f"Model classified as Positive ({max_prob * 100:.1f}% confidence)."
        elif raw_pred in ['negative', 'neg', '0']:
            sentiment = "NEGATIVE 🚨"
            color_box = st.error
            reason = f"Model classified as Negative ({max_prob * 100:.1f}% confidence)."
        else:
            sentiment = "NEUTRAL 😐"
            color_box = st.warning
            reason = "Model classified as Neutral."

        # Display Result Box
        st.markdown("### Result:")
        color_box(f"**Predicted Sentiment:** {sentiment}")
        
        # Details Collapsible Section
        with st.expander("See Prediction Details"):
            st.write(f"**Confidence Score:** {max_prob * 100:.2f}%")
            st.write(f"**Preprocessed Text:** `{cleaned_review}`")
            st.write(f"**Raw Model Prediction:** `{model.classes_[probabilities.argmax()]}`")
            st.write(f"**Decision Reason:** {reason}")
