import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 1. Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

# 2. Load Model Assets
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

# Initialize Preprocessing Tools
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

# 3. Streamlit Page Config & UI
st.set_page_config(page_title="Smartphone Review Sentiment Analyzer", page_icon="📱")

st.title("📱 Smartphone Review Sentiment Analyzer")
st.write("Enter a user review below to predict sentiment (**Positive**, **Neutral**, or **Negative**).")

user_review = st.text_area(
    "Review Text:", 
    placeholder="Display is not satisfactory but the processor performance is good",
    height=120
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        # Preprocess input using your existing pipeline
        cleaned_review = preprocess_text(user_review)
        review_vec = tfidf.transform([cleaned_review])
        
        # Get probability scores
        probabilities = model.predict_proba(review_vec)[0]
        max_prob = max(probabilities)
        predicted_class = model.classes_[probabilities.argmax()]

        # Rule 1: Check for contrast words (e.g., "but", "however")
        contrast_words = ["but", "however", "although", "whereas"]
        has_contrast = any(re.search(rf"\b{word}\b", user_review.lower()) for word in contrast_words)

        # Rule 2: Set confidence threshold for predictions
        CONFIDENCE_THRESHOLD = 0.65

        # Determine Final Sentiment
        if has_contrast and max_prob < 0.75:
            sentiment = "NEUTRAL 😐"
            color_box = st.warning
        elif max_prob < CONFIDENCE_THRESHOLD:
            sentiment = "NEUTRAL 😐"
            color_box = st.warning
        else:
            if str(predicted_class).lower() in ['positive', '1', 'pos']:
                sentiment = "POSITIVE 🎉"
                color_box = st.success
            elif str(predicted_class).lower() in ['negative', '0', 'neg']:
                sentiment = "NEGATIVE 🚨"
                color_box = st.error
            else:
                sentiment = "NEUTRAL 😐"
                color_box = st.warning

        st.markdown("### Result:")
        color_box(f"**Predicted Sentiment:** {sentiment}")
        
        with st.expander("See Prediction Details"):
            st.write(f"**Confidence Score:** {max_prob * 100:.2f}%")
            st.write(f"**Preprocessed Text:** `{cleaned_review}`")
