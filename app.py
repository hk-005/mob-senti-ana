import streamlit as st
import joblib
import re
import pandas as pd
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
    layout="wide"
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
# 3. Preprocessing & Prediction Helper Functions
# -----------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
default_stopwords = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'cannot'}
custom_stopwords = default_stopwords - negation_words

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in custom_stopwords
    ]
    return ' '.join(cleaned_tokens)

contrast_words = ["but", "however", "although", "whereas"]
negative_keywords = [
    "broke", "broken", "muffled", "terrible", "horrible", "overheats", 
    "drain", "drains", "scratched", "defective", "useless", "worst", 
    "disappointing", "sluggish", "freeze", "freezes", "crash", "crashes"
]

def predict_single_review(user_review):
    cleaned_review = preprocess_text(user_review)
    review_vec = tfidf.transform([cleaned_review])
    
    probabilities = model.predict_proba(review_vec)[0]
    max_prob = max(probabilities)
    raw_pred = str(model.classes_[probabilities.argmax()]).strip().lower()

    # Map raw model class probabilities into a neat dictionary
    class_probs = {
        str(cls).strip().upper(): round(prob * 100, 2) 
        for cls, prob in zip(model.classes_, probabilities)
    }

    has_contrast = any(re.search(rf"\b{word}\b", user_review.lower()) for word in contrast_words)
    has_negative_kw = any(re.search(rf"\b{word}\b", user_review.lower()) for word in negative_keywords)

    if has_contrast:
        sentiment = "NEUTRAL"
        reason = "Detected contrasting statements in review."
    elif has_negative_kw and raw_pred == 'neutral':
        sentiment = "NEGATIVE"
        reason = "Override: Detected strong negative keyword in neutral model prediction."
    elif raw_pred in ['positive', 'pos', '1']:
        sentiment = "POSITIVE"
        reason = f"Model classified as Positive ({max_prob * 100:.1f}% confidence)."
    elif raw_pred in ['negative', 'neg', '0']:
        sentiment = "NEGATIVE"
        reason = f"Model classified as Negative ({max_prob * 100:.1f}% confidence)."
    else:
        sentiment = "NEUTRAL"
        reason = "Model classified as Neutral."

    return sentiment, max_prob, cleaned_review, raw_pred, reason, class_probs

# -----------------------------------------------------------------------------
# 4. Streamlit UI & Navigation Tabs
# -----------------------------------------------------------------------------
st.title("📱 Smartphone Review Sentiment Analyzer")
st.write("Analyze individual reviews or perform batch analysis on large datasets.")

# Sidebar Info
st.sidebar.title("📌 Model Details")
st.sidebar.info(
    "**Model:** Logistic Regression\n"
    "**Vectorizer:** TF-IDF\n"
    "**Pipeline:** NLTK Lemmatization + Negation Handling + Heuristics"
)

tab1, tab2 = st.tabs(["📝 Single Review Analysis", "📁 Batch CSV Analysis"])

# -----------------------------------------------------------------------------
# TAB 1: Single Review Analysis
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Analyze a Single Customer Review")
    user_review = st.text_area(
        "Review Text:", 
        placeholder="e.g., The software update broke the fingerprint sensor and the speaker sound is muffled.",
        height=120
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review to analyze.")
        else:
            sentiment, max_prob, cleaned_review, raw_pred, reason, class_probs = predict_single_review(user_review)
            
            # Display Box
            if sentiment == "POSITIVE":
                color_box = st.success
                sentiment_str = "POSITIVE 🎉"
            elif sentiment == "NEGATIVE":
                color_box = st.error
                sentiment_str = "NEGATIVE 🚨"
            else:
                color_box = st.warning
                sentiment_str = "NEUTRAL 😐"

            st.markdown("### Result:")
            color_box(f"**Predicted Sentiment:** {sentiment_str}")
            
            # Probability Distribution Bar Chart
            st.write("### Class Probability Breakdown")
            prob_df = pd.DataFrame(
                list(class_probs.items()), 
                columns=["Sentiment Class", "Probability (%)"]
            ).set_index("Sentiment Class")
            
            st.bar_chart(prob_df)

            with st.expander("See Prediction Details"):
                st.write(f"**Confidence Score:** {max_prob * 100:.2f}%")
                st.write(f"**Preprocessed Text:** `{cleaned_review}`")
                st.write(f"**Raw Model Prediction:** `{raw_pred}`")
                st.write(f"**Decision Reason:** {reason}")

# -----------------------------------------------------------------------------
# TAB 2: Batch CSV Analysis
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Upload CSV for Batch Sentiment Classification")
    st.write("Upload a CSV file containing user reviews. You will be able to preview, analyze, and download the results.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview", df.head())

            # Allow user to pick the column containing text
            column_names = list(df.columns)
            selected_col = st.selectbox(
                "Select the column containing review text:", 
                column_names, 
                index=0 if "review" not in [c.lower() for c in column_names] else [c.lower() for c in column_names].index("review")
            )

            if st.button("Process Batch Predictions", type="primary"):
                with st.spinner("Analyzing reviews... Please wait."):
                    results = []
                    confidences = []
                    
                    for text in df[selected_col]:
                        sentiment, max_prob, _, _, _, _ = predict_single_review(str(text))
                        results.append(sentiment)
                        confidences.append(round(max_prob * 100, 2))

                    df["Predicted_Sentiment"] = results
                    df["Confidence_Score (%)"] = confidences

                st.success("Batch classification complete!")

                # Key Metrics Display
                col1, col2, col3, col4 = st.columns(4)
                total_count = len(df)
                pos_count = (df["Predicted_Sentiment"] == "POSITIVE").sum()
                neu_count = (df["Predicted_Sentiment"] == "NEUTRAL").sum()
                neg_count = (df["Predicted_Sentiment"] == "NEGATIVE").sum()

                col1.metric("Total Reviews", total_count)
                col2.metric("Positive 🎉", f"{pos_count} ({pos_count/total_count*100:.1f}%)")
                col3.metric("Neutral 😐", f"{neu_count} ({neu_count/total_count*100:.1f}%)")
                col4.metric("Negative 🚨", f"{neg_count} ({neg_count/total_count*100:.1f}%)")

                # Distribution Chart
                st.write("### Sentiment Distribution")
                sentiment_counts = df["Predicted_Sentiment"].value_counts()
                st.bar_chart(sentiment_counts)

                # Show Results Table
                st.write("### Processed Results", df)

                # Download Button
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                    type="secondary"
                )

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
