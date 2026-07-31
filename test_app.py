import pytest
import re

# Import or copy preprocessing logic directly to test isolated functions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup NLTK dependencies for tests
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
default_stopwords = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'cannot'}
custom_stopwords = default_stopwords - negation_words

contrast_words = ["but", "however", "although", "whereas"]
negative_keywords = [
    "broke", "broken", "muffled", "terrible", "horrible", "overheats", 
    "drain", "drains", "scratched", "defective", "useless", "worst", 
    "disappointing", "sluggish", "freeze", "freezes", "crash", "crashes"
]

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

# -----------------------------------------------------------------------------
# 1. Tests for Preprocessing Pipeline
# -----------------------------------------------------------------------------
def test_preprocess_lowercasing_and_punctuation():
    input_text = "The Camera is AMAZING!!!"
    processed = preprocess_text(input_text)
    assert "amazing" in processed
    assert "!" not in processed

def test_preprocess_preserves_negation():
    input_text = "Display is not good"
    processed = preprocess_text(input_text)
    assert "not" in processed, "Negation words should not be removed as stopwords."

def test_preprocess_empty_and_invalid_input():
    assert preprocess_text("") == ""
    assert preprocess_text(123) == ""

# -----------------------------------------------------------------------------
# 2. Tests for Contrast Word Detection (Neutral Rule)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("review", [
    "Display is not good but processor performance is better.",
    "The camera is great, however the battery drains fast.",
    "Although it looks premium, the speaker is quiet."
])
def test_contrast_word_detection(review):
    has_contrast = any(re.search(rf"\b{word}\b", review.lower()) for word in contrast_words)
    assert has_contrast is True

# -----------------------------------------------------------------------------
# 3. Tests for Negative Keyword Detection (Override Rule)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("review,keyword", [
    ("The software update broke the fingerprint sensor.", "broke"),
    ("The phone overheats after 10 minutes.", "overheats"),
    ("Sound from the speaker is muffled.", "muffled")
])
def test_negative_keyword_detection(review, keyword):
    has_negative_kw = any(re.search(rf"\b{word}\b", review.lower()) for word in negative_keywords)
    assert has_negative_kw is True
