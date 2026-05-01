import pandas as pd
import time
import spacy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# --- 1. DATA LOADING & PRE-PROCESSING ---
dataset_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\cleaned_amazon_reviews.csv'
print("Loading preprocessed dataset...")
df = pd.read_csv(dataset_path).dropna(subset=['cleaned_text'])

# Sampling to manage computational load while maintaining statistical significance
# Target: 200,000 reviews for a robust training process
sample_size = min(200000, len(df))
df = df.sample(sample_size, random_state=42)

print("Splitting data and extracting features (N-Grams enabled)...")

# Professional 'No Data Leakage' Split
# We split the raw text first, then fit the vectorizer ONLY on training data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Advanced TF-IDF Configuration
# ngram_range=(1,2) allows the model to capture phrases like "not good" or "very happy"
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=5)

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# --- 2. ALGORITHM BENCHMARKING ---
# Defining a dictionary of candidate models for comparison
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(max_iter=2000, dual=False)
}

print("\n--- 🏆 STARTING ALGORITHM COMPETITION ---")
best_model_name = None
best_score = 0

for name, model in models.items():
    start_time = time.time()

    # Model Training
    model.fit(X_train, y_train)

    # Inference and Evaluation
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    duration = time.time() - start_time

    # Displaying Metrics
    print(f"✅ {name}: Accuracy: {score * 100:.2f}% | Training Duration: {duration:.1f}s")
    print(f"\nDetailed Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Track the winning algorithm
    if score > best_score:
        best_score = score
        best_model_name = name

print(f"\n👑 WINNING ALGORITHM: {best_model_name} ({best_score * 100:.2f}%)")

# --- 3. HYBRID ARCHITECTURE TEST: NER (NAMED ENTITY RECOGNITION) ---
print("\n--- 🔍 NER INTEGRATION TEST ---")
# Loading SpaCy's English transformer model for entity extraction
nlp = spacy.load("en_core_web_sm")

sample_review = "I bought this Sony PlayStation 5 from Amazon New York branch for $500, but the controller is broken!"
print(f"Sample Input: '{sample_review}'\n")

doc = nlp(sample_review)
print("Detected Entities:")
for ent in doc.ents:
    print(f"- {ent.text} --> [{ent.label_}] ({spacy.explain(ent.label_)})")

print("\n(Technical Note: The NER model identifies specific brands like 'Sony' or locations like 'New York'.")
print("When combined with our Sentiment model, it allows for granular market analysis.)")

# --- 4. MODEL SERIALIZATION ---
print("\n--- 💾 SAVING THE CHAMPION MODEL ---")

# Save the winning model (Logistic Regression or SVM)
model_output_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\best_sentiment_model.pkl'
joblib.dump(models[best_model_name], model_output_path)

# Save the specific vectorizer used during this benchmark session
# This ensures consistency during deployment in app.py
vectorizer_output_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\advanced_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_output_path)

print(f"✅ Deployment assets successfully saved! '{best_model_name}' is ready for production.")
print("You can now proceed to run 'app.py'.")