import pandas as pd
import time
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# --- 1. DATA PREPARATION (200,000 Samples) ---
# Loading the cleaned Amazon reviews dataset
dataset_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\cleaned_amazon_reviews.csv'
print("Loading high-volume dataset (200,000 samples)...")
df = pd.read_csv(dataset_path).dropna(subset=['cleaned_text'])

# Ensuring the dataset size is exactly 200,000 for robust benchmarking
df = df.sample(min(200000, len(df)), random_state=42)

# Label Mapping: Logistic Regression and SVM handle 1/2,
# but XGBoost strictly requires 0-indexed labels (0 and 1)
df['label'] = df['label'].replace({1: 0, 2: 1})

# Professional 'No Data Leakage' Split: Text split occurs before vectorization
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Advanced Feature Extraction using TF-IDF with N-grams
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=5)

# Vectorizer learns from training data only to maintain scientific integrity
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# --- 2. MODEL DEFINITIONS ---
# Comparing diverse architectures: Probabilistic, Linear, Ensemble, and Boosting
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(max_iter=2000, dual=False),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

performance_results = []
print("\n--- 📊 Starting Comprehensive Comparison (K-Fold & Detailed Metrics) ---")

for name, model in models.items():
    print(f"🔄 Training and evaluating {name} with 5-Fold Cross-Validation...")
    start_time = time.time()

    # K-Fold Cross-Validation: Calculating mean accuracy for scientific reliability
    k_fold_scores = cross_val_score(model, X_train, y_train, cv=5)
    k_fold_mean = k_fold_scores.mean()

    print(f"📊 K-Fold Mean Accuracy: {k_fold_mean * 100:.2f}%")

    # Standard Training for specific metric extraction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metric Calculation: Precision, Recall, and F1-Score
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    duration = time.time() - start_time

    performance_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "K-Fold Mean": round(k_fold_mean, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Time (sec)": round(duration, 1)
    })

# --- 3. LEADERBOARD GENERATION ---
# Ranking models by F1-Score to find the most balanced performer
leaderboard = pd.DataFrame(performance_results).sort_values(by="F1-Score", ascending=False)
print("\n" + "=" * 90)
print("🏆 FINAL PERFORMANCE LEADERBOARD (200K DATASET) 🏆")
print("=" * 90)
print(leaderboard.to_string(index=False))
print("=" * 90)

# Identifying the Champion Model
champion_name = leaderboard.iloc[0]['Model']
champion_model = models[champion_name]

# --- 4. MODEL SERIALIZATION ---
print(f"\n--- 💾 SERIALIZING CHAMPION: {champion_name} ---")

# Save files with standardized names for app.py deployment
joblib.dump(champion_model, r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\best_sentiment_model.pkl')
joblib.dump(vectorizer, r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\advanced_vectorizer.pkl')

print(f"✅ Champion model ({champion_name}) and vectorizer successfully packaged!")

# --- 5. VISUALIZATION: CONFUSION MATRIX ---
print("\n--- 📊 Generating Confusion Matrix for Champion Model... ---")
y_pred_champ = champion_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_champ)

classes = ['Negative', 'Positive']

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title(f'Figure 1: {champion_name} Confusion Matrix\n(Test Accuracy: {accuracy_score(y_test, y_pred_champ)*100:.2f}%)')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

# Save high-resolution visualization for GitHub documentation
plot_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\champion_confusion_matrix.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved as: {plot_path}")

plt.show()

print("\nProcess Complete! You are now ready to launch 'app.py'.")