import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. LOAD PREPROCESSED DATA ---
# Loading the balanced and cleaned dataset for sentiment analysis
# Project by: Hayat Diler
input_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\cleaned_amazon_reviews.csv'
print("Loading dataset...")
df = pd.read_csv(input_path)

# Drop any rows where 'cleaned_text' might be NaN after preprocessing
df = df.dropna(subset=['cleaned_text'])

# --- 2. DATASET SPLITTING & VECTORIZATION ---
print("Splitting dataset and extracting features (TF-IDF)...")

# Splitting into Training (80%) and Testing (20%) sets before vectorization
# This prevents data leakage by ensuring the vectorizer only 'learns' from the training set
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Initializing TfidfVectorizer with N-grams and a frequency threshold
# Standard parameters for competition: 25k features, (1,2) N-grams, min_df=5
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=5)

# Fit and transform the training data, but only transform the test data
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# --- 3. MODEL TRAINING ---
print("Training Logistic Regression model. Please wait...")
# max_iter is increased to ensure the solver reaches convergence
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# --- 4. PERFORMANCE EVALUATION ---
print("Testing model performance...")
y_pred = lr_model.predict(X_test)

# Calculate and display Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Generate detailed Classification Report (Precision, Recall, F1-Score)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative (1)', 'Positive (2)']))

# --- 5. VISUALIZATION: CONFUSION MATRIX ---
print("Generating Confusion Matrix plot...")
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Negative', 'Predicted: Positive'],
            yticklabels=['Actual: Negative', 'Actual: Positive'])

plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')

# Save the visualization as a PNG file before displaying
plot_output_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\lr_confusion_matrix.png'
plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
plt.show()

# --- 6. MODEL SERIALIZATION ---
# Saving the trained model and vectorizer for deployment in app.py
model_save_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\lr_sentiment_model.pkl'
vectorizer_save_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\tfidf_vectorizer.pkl'

joblib.dump(lr_model, model_save_path)
joblib.dump(vectorizer, vectorizer_save_path)

print(f"Model and Vectorizer successfully saved to .pkl files!")