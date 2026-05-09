import streamlit as st
import joblib
import spacy
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. NLTK & RESOURCE CONFIGURATION ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


@st.cache_resource
def load_resources():
    model = joblib.load('best_sentiment_model.pkl')
    vectorizer = joblib.load('advanced_vectorizer.pkl')
    nlp = spacy.load("en_core_web_sm")
    return model, vectorizer, nlp


model, vectorizer, nlp = load_resources()

stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("no")
lemmatizer = WordNetLemmatizer()


def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'<.*?>|http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)


def analyze_single(user_input):
    cleaned_text = preprocess_input(user_input)
    transformed_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_input)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_input)[0]
        confidence_score = max(probabilities) * 100
    else:
        confidence_score = None
    return prediction, confidence_score


# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="Advanced Review Insights", page_icon="📊", layout="wide")
st.title("📊 Advanced Amazon Insights Dashboard")
st.markdown("""
This dashboard leverages a **Hybrid NLP Architecture** combining **Sentiment Analysis** 
and **Named Entity Recognition (NER)** to extract deep business intelligence from customer reviews.
""")

# --- 3. TABS ---
tab1, tab2 = st.tabs(["🔍 Single Review Analysis", "📂 Bulk CSV Analysis"])

# ==================== TAB 1: SINGLE REVIEW ====================
with tab1:
    user_input = st.text_area("Enter your Amazon review in English:",
                              placeholder="Example: The new Sony headphones are great, but the shipping was late...",
                              height=150)

    if st.button("Generate Deep Insights", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a review to begin analysis.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Sentiment Prediction")
                prediction, confidence_score = analyze_single(user_input)

                if prediction == 2:
                    st.success("### RESULT: POSITIVE 🟢")
                    if confidence_score:
                        st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                        st.progress(confidence_score / 100)
                    st.balloons()
                else:
                    st.error("### RESULT: NEGATIVE 🔴")
                    if confidence_score:
                        st.write(f"**Confidence Score:** {confidence_score:.2f}%")
                        st.progress(confidence_score / 100)

            with col2:
                st.subheader("🔍 Named Entity Recognition (NER)")
                doc = nlp(user_input)
                if not doc.ents:
                    st.info("No specific entities detected in this review.")
                else:
                    st.write("Identified key entities and categories:")
                    for ent in doc.ents:
                        st.markdown(f"**{ent.text}** — `{ent.label_}`")
                        st.caption(f"Category Description: {spacy.explain(ent.label_)}")

            st.divider()
            st.info(
                "System Note: Analysis is powered by a champion model trained on a balanced dataset of 200,000 Amazon reviews.")

# ==================== TAB 2: BULK CSV ANALYSIS ====================
with tab2:
    st.subheader("📂 Bulk Review Analysis")
    st.markdown("""
    Upload a CSV file containing Amazon reviews for batch analysis.  
    **Requirements:** The CSV must have a column named `review_text` containing the review texts.
    """)

    # Example CSV download
    example_df = pd.DataFrame({
        "review_text": [
            "This product is absolutely amazing, best purchase ever!",
            "Terrible quality, broke after one day. Very disappointed.",
            "Decent product for the price, nothing special but works fine."
        ]
    })
    st.download_button(
        label="📥 Download Example CSV",
        data=example_df.to_csv(index=False),
        file_name="example_reviews.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "review_text" not in df.columns:
                st.error("❌ CSV file must contain a column named 'review_text'.")
            else:
                st.success(f"✅ File uploaded successfully! {len(df)} reviews found.")

                # Limit to 500 reviews for performance
                if len(df) > 500:
                    st.warning("⚠️ More than 500 reviews detected. Only the first 500 will be analyzed.")
                    df = df.head(500)

                if st.button("🚀 Analyze All Reviews", type="primary"):
                    with st.spinner("Analyzing reviews... Please wait."):
                        results = []
                        for text in df["review_text"]:
                            pred, conf = analyze_single(str(text))
                            label = "POSITIVE 🟢" if pred == 2 else "NEGATIVE 🔴"
                            results.append({
                                "Review": str(text)[:100] + "..." if len(str(text)) > 100 else str(text),
                                "Sentiment": label,
                                "Confidence (%)": f"{conf:.2f}" if conf else "N/A"
                            })

                    results_df = pd.DataFrame(results)

                    # --- SUMMARY METRICS ---
                    st.divider()
                    st.subheader("📊 Analysis Summary")

                    positive_count = sum(1 for r in results if "POSITIVE" in r["Sentiment"])
                    negative_count = len(results) - positive_count
                    positive_pct = (positive_count / len(results)) * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Reviews", len(results))
                    col2.metric("Positive Reviews 🟢", f"{positive_count} ({positive_pct:.1f}%)")
                    col3.metric("Negative Reviews 🔴", f"{negative_count} ({100 - positive_pct:.1f}%)")

                    # --- PIE CHART ---
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie(
                        [positive_count, negative_count],
                        labels=["Positive", "Negative"],
                        colors=["#2ecc71", "#e74c3c"],
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.set_facecolor("#0e1117")
                    fig.patch.set_facecolor("#0e1117")
                    for text in ax.texts:
                        text.set_color("white")
                    st.pyplot(fig)

                    # --- RESULTS TABLE ---
                    st.divider()
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)

                    # --- CSV DOWNLOAD ---
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=results_df.to_csv(index=False),
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )

                    st.info(
                        "System Note: Analysis is powered by a champion model trained on a balanced dataset of 200,000 Amazon reviews.")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")