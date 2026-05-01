import streamlit as st
import joblib
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. NLTK & RESOURCE CONFIGURATION ---
# These resources are required for consistent text preprocessing
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


@st.cache_resource
def load_resources():
    """
    Loads the trained model, vectorizer, and SpaCy NER model.
    Using @st.cache_resource optimizes performance by preventing redundant reloads.
    """
    # Ensure these .pkl files are in the same directory as app.py
    model = joblib.load('best_sentiment_model.pkl')
    vectorizer = joblib.load('advanced_vectorizer.pkl')
    nlp = spacy.load("en_core_web_sm")

    return model, vectorizer, nlp


# Initialize resources
model, vectorizer, nlp = load_resources()

# Setup Stopwords and Lemmatizer to match 02_preprocessing.py exactly
stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("no")
lemmatizer = WordNetLemmatizer()


def preprocess_input(text):
    """
    Exact replica of the preprocessing logic used in the training phase.
    Ensures the model receives data in the format it was trained on.
    """
    text = text.lower()
    # Remove HTML tags and URLs
    text = re.sub(r'<.*?>|http\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    # Lemmatization and Stopword filtering
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(cleaned_words)


# --- 2. USER INTERFACE DESIGN ---
st.set_page_config(page_title="Advanced Review Insights", page_icon="📊", layout="wide")

st.title("📊 Advanced Amazon Insights Dashboard")
st.markdown("""
This dashboard leverages a **Hybrid NLP Architecture** combining **Sentiment Analysis** 
and **Named Entity Recognition (NER)** to extract deep business intelligence from customer reviews.
""")

# User Input Section
user_input = st.text_area("Enter your Amazon review in English:",
                          placeholder="Example: The new Sony headphones are great, but the shipping was late...",
                          height=150)

if st.button("Generate Deep Insights", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review to begin analysis.")
    else:
        # Create layout columns
        col1, col2 = st.columns(2)

        # --- LEFT COLUMN: SENTIMENT ANALYSIS ---
        with col1:
            st.subheader("🎯 Sentiment Prediction")

            # Preprocess and Transform Input
            cleaned_text = preprocess_input(user_input)
            transformed_input = vectorizer.transform([cleaned_text])

            # Perform Prediction
            prediction = model.predict(transformed_input)[0]

            # Calculate Confidence Score (Probability)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                confidence_score = max(probabilities) * 100
            else:
                confidence_score = None

            # BASED ON TRAINING: 1 = NEGATIVE, 2 = POSITIVE
            if prediction == 2:  # Positive Case
                st.success("### RESULT: POSITIVE 🟢")
                if confidence_score:
                    st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                    st.progress(confidence_score / 100)
                st.balloons()
            else:  # Negative Case (Prediction is 1)
                st.error("### RESULT: NEGATIVE 🔴")
                if confidence_score:
                    st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                    st.progress(confidence_score / 100)

        # --- RIGHT COLUMN: ENTITY EXTRACTION (NER) ---
        with col2:
            st.subheader("🔍 Named Entity Recognition (NER)")
            doc = nlp(user_input)

            if not doc.ents:
                st.info("No specific brands, locations, or monetary entities detected in this review.")
            else:
                st.write("Identified key entities and categories:")
                for ent in doc.ents:
                    st.markdown(f"**{ent.text}** — `{ent.label_}`")
                    st.caption(f"Category Description: {spacy.explain(ent.label_)}")

        st.divider()
        st.info(
            "System Note: Analysis is powered by a champion model trained on a balanced dataset of 200,000 Amazon reviews.")