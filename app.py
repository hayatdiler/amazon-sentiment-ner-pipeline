import streamlit as st
import joblib
import spacy
import re


# --- 1. RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    """
    Loads the trained model, vectorizer, and SpaCy NER model.
    Using @st.cache_resource to prevent reloading on every interaction.
    """
    # Loading the champion model and the vectorizer (TF-IDF)
    # Ensure these paths match your local directory structure
    model = joblib.load(r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\best_sentiment_model.pkl')
    vectorizer = joblib.load(r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\advanced_vectorizer.pkl')

    # Loading SpaCy's English Natural Language Processing model
    nlp = spacy.load("en_core_web_sm")

    return model, vectorizer, nlp


# Initializing resources
model, vectorizer, nlp = load_resources()


def preprocess_input(text):
    """
    Simple preprocessing for real-time user input.
    """
    text = text.lower()
    # Removing special characters and numbers to match training format
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# --- 2. USER INTERFACE DESIGN ---
st.set_page_config(page_title="Advanced Review Insights", page_icon="📊", layout="wide")

st.title("📊 Advanced Amazon Insights Dashboard")
st.markdown("""
This dashboard leverages a **Hybrid NLP Architecture** combining **Linear SVM/Logistic Regression** 
for Sentiment Analysis and **SpaCy NER** for Named Entity Recognition. 
It analyzes the tone of the review while identifying brands, locations, and numerical data.
""")

# User Input Section
user_input = st.text_area("Enter your Amazon review in English:",
                          placeholder="Example: I bought the new Sony headphones from London for $300...",
                          height=150)

if st.button("Generate Deep Insights", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review to begin analysis.")
    else:
        # Create two columns for side-by-side results
        col1, col2 = st.columns(2)

        # --- LEFT COLUMN: SENTIMENT ANALYSIS ---
        with col1:
            st.subheader("🎯 Sentiment Prediction")
            cleaned_text = preprocess_input(user_input)
            transformed_input = vectorizer.transform([cleaned_text])

            # Model Prediction
            # Note: Based on our training, 0 = Negative, 1 = Positive
            prediction = model.predict(transformed_input)[0]

            # Calculating Confidence Score (if the model supports probability)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                confidence_score = max(probabilities) * 100
            else:
                confidence_score = None  # LinearSVC does not natively support probabilities

            if prediction == 1:  # Positive Result
                st.success("### RESULT: POSITIVE")
                if confidence_score:
                    st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                    st.progress(confidence_score / 100)
                st.balloons()
            else:  # Negative Result
                st.error("### RESULT: NEGATIVE")
                if confidence_score:
                    st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                    st.progress(confidence_score / 100)

        # --- RIGHT COLUMN: ENTITY EXTRACTION (NER) ---
        with col2:
            st.subheader("🔍 Named Entity Recognition (NER)")
            doc = nlp(user_input)

            if not doc.ents:
                st.info("No specific brands, locations, or entities found in this review.")
            else:
                st.write("Identified key entities and categories:")
                for ent in doc.ents:
                    st.markdown(f"**{ent.text}** — `{ent.label_}`")
                    st.caption(f"Category Description: {spacy.explain(ent.label_)}")

        st.divider()
        st.info(
            "System Note: Analysis is performed using the champion model trained on a balanced dataset of 200,000 Amazon reviews.")