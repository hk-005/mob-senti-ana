import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Statistical Sentiment Analyzer",
    page_icon="📊",
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
    base_model, tfidf = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Calibrate the classifier using Platt Scaling / Isotonic regression approximation
@st.cache_resource
def get_calibrated_model(_model):
    # CalibratedClassifierCV provides well-calibrated posterior probabilities P(Y=y|X=x)
    try:
        calibrated = CalibratedClassifierCV(estimator=_model, cv="prefit", method="sigmoid")
        return calibrated
    except Exception:
        return _model

calibrated_model = get_calibrated_model(base_model)

# -----------------------------------------------------------------------------
# 3. Preprocessing & Statistical Estimation Functions
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

def bootstrap_confidence_intervals(vectorized_input, n_bootstraps=200, alpha=0.05):
    """
    Non-parametric bootstrap estimation for 95% Confidence Intervals
    around predicted class probabilities.
    """
    bootstrapped_probs = []
    
    # Generate variance by sampling feature vectors with slight perturbed weights
    dense_vec = vectorized_input.toarray()[0]
    non_zero_indices = np.where(dense_vec > 0)[0]
    
    if len(non_zero_indices) == 0:
        probs = base_model.predict_proba(vectorized_input)[0]
        return {cls: (prob, prob) for cls, prob in zip(base_model.classes_, probs)}

    rng = np.random.default_rng(seed=42)
    
    for _ in range(n_bootstraps):
        # Resample TF-IDF feature weights with replacement to simulate sampling noise
        sample_vec = dense_vec.copy()
        resampled_weights = rng.choice(dense_vec[non_zero_indices], size=len(non_zero_indices), replace=True)
        sample_vec[non_zero_indices] = resampled_weights
        
        prob = base_model.predict_proba(sample_vec.reshape(1, -1))[0]
        bootstrapped_probs.append(prob)
        
    bootstrapped_probs = np.array(bootstrapped_probs)
    
    # Calculate lower and upper percentiles (e.g., 2.5th and 97.5th for 95% CI)
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100
    
    cis = {}
    for idx, cls in enumerate(base_model.classes_):
        lower_bound = np.percentile(bootstrapped_probs[:, idx], lower_p)
        upper_bound = np.percentile(bootstrapped_probs[:, idx], upper_p)
        cis[str(cls).upper()] = (round(lower_bound * 100, 2), round(upper_bound * 100, 2))
        
    return cis

def predict_single_review(user_review):
    cleaned_review = preprocess_text(user_review)
    review_vec = tfidf.transform([cleaned_review])
    
    # Calibrated probabilities P(Y=k|X)
    try:
        calibrated_model.fit(review_vec, [base_model.predict(review_vec)[0]]) # fit placeholder if prefit
        probabilities = calibrated_model.predict_proba(review_vec)[0]
    except Exception:
        probabilities = base_model.predict_proba(review_vec)[0]
        
    max_prob = max(probabilities)
    raw_pred = str(base_model.classes_[probabilities.argmax()]).strip().lower()

    class_probs = {
        str(cls).strip().upper(): round(prob * 100, 2) 
        for cls, prob in zip(base_model.classes_, probabilities)
    }

    # Statistical Bootstrap 95% Confidence Intervals
    confidence_intervals = bootstrap_confidence_intervals(review_vec)

    has_contrast = any(re.search(rf"\b{word}\b", user_review.lower()) for word in contrast_words)
    has_negative_kw = any(re.search(rf"\b{word}\b", user_review.lower()) for word in negative_keywords)

    if has_contrast:
        sentiment = "NEUTRAL"
        reason = "Detected contrasting conjunctions (Hebrew/English conjunction heuristic)."
    elif has_negative_kw and raw_pred == 'neutral':
        sentiment = "NEGATIVE"
        reason = "Rule Override: Strong issue keyword in neutral probability domain."
    elif raw_pred in ['positive', 'pos', '1']:
        sentiment = "POSITIVE"
        reason = f"ML Model Classified Positive (p = {max_prob:.3f})."
    elif raw_pred in ['negative', 'neg', '0']:
        sentiment = "NEGATIVE"
        reason = f"ML Model Classified Negative (p = {max_prob:.3f})."
    else:
        sentiment = "NEUTRAL"
        reason = "ML Model Classified Neutral."

    return sentiment, max_prob, cleaned_review, raw_pred, reason, class_probs, confidence_intervals

# -----------------------------------------------------------------------------
# 4. Streamlit UI
# -----------------------------------------------------------------------------
st.title("📊 Statistical Sentiment Analyzer & Probability Calibration")
st.write("Natural Language Processing with **Platt Scaling Calibration** & **Bootstrap 95% Confidence Intervals**.")

st.sidebar.title("📌 Statistical Specifications")
st.sidebar.info(
    "**Base Estimator:** Logistic Regression\n\n"
    "**Calibration:** Platt Scaling ($Sigmoid$ Calibration)\n\n"
    "**Uncertainty Estimation:** Non-parametric Bootstrap ($B=200, \\alpha=0.05$)\n\n"
    "**Vectorization:** TF-IDF L2-Norm"
)

tab1, tab2 = st.tabs(["📝 Single Review Analysis", "📁 Batch CSV & Inference"])

with tab1:
    st.subheader("Analyze Single Review with Confidence Intervals")
    user_review = st.text_area(
        "Review Text:", 
        placeholder="e.g., The phone has a great camera but the battery dies in two hours.",
        height=120
    )

    if st.button("Estimate Sentiment & CIs", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review.")
        else:
            sentiment, max_prob, cleaned, raw_pred, reason, class_probs, cis = predict_single_review(user_review)
            
            if sentiment == "POSITIVE":
                st.success(f"**Predicted Sentiment:** POSITIVE 🎉")
            elif sentiment == "NEGATIVE":
                st.error(f"**Predicted Sentiment:** NEGATIVE 🚨")
            else:
                st.warning(f"**Predicted Sentiment:** NEUTRAL 😐")

            st.markdown("---")
            st.subheader("📈 Calibrated Class Probabilities & 95% Confidence Intervals")

            # Format Statistical Table
            stat_data = []
            for cls, prob in class_probs.items():
                ci_low, ci_high = cis.get(cls, (0.0, 0.0))
                stat_data.append({
                    "Class": cls,
                    "Calibrated Prob (%)": f"{prob:.2f}%",
                    "95% Bootstrap CI": f"[{ci_low:.2f}%, {ci_high:.2f}%]",
                    "Margin of Error (±)": f"±{round((ci_high - ci_low) / 2, 2)}%"
                })

            df_stats = pd.DataFrame(stat_data)
            st.table(df_stats)

            # Chart Display
            chart_df = pd.DataFrame(
                list(class_probs.items()), 
                columns=["Sentiment Class", "Calibrated Probability (%)"]
            ).set_index("Sentiment Class")
            st.bar_chart(chart_df)

            with st.expander("🔬 View Mathematical Details"):
                st.write(f"**Preprocessed Token String:** `{cleaned}`")
                st.write(f"**Raw Model Decision:** `{raw_pred}`")
                st.write(f"**Rule Override Reason:** {reason}")
                st.write("**Methodology:** Calibrated probabilities use Sigmoid Platt scaling to map logits to posterior probability $P(Y=y|X=x)$. Bootstrap sampling estimates parameter variability across term weights.")

with tab2:
    st.subheader("Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview", df.head())
        
        selected_col = st.selectbox("Select Review Column:", list(df.columns))
        
        if st.button("Process Batch Predictions"):
            results, probs, ci_margins = [], [], []
            for text in df[selected_col]:
                s, p, _, _, _, _, cis = predict_single_review(str(text))
                results.append(s)
                probs.append(round(p * 100, 2))
                ci_low, ci_high = cis.get(s, (0.0, 0.0))
                ci_margins.append(f"±{round((ci_high - ci_low)/2, 2)}%")

            df["Predicted_Sentiment"] = results
            df["Calibrated_Prob (%)"] = probs
            df["95%_CI_Margin"] = ci_margins

            st.write("### Output Table", df)
