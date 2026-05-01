import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import bz2

# --- 1. INITIALIZATION & NLTK CONFIGURATION ---
print("Initializing NLTK modules...")
# Downloading required resources for text processing
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Configuring Stop Words
# We exclude "not" and "no" as they are critical for sentiment polarity
stop_words = set(stopwords.words('english'))
stop_words.discard("not")
stop_words.discard("no")

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Cleans raw text by removing HTML, special characters, and stopwords,
    followed by lemmatization.
    """
    # Lowercase conversion
    text = text.lower()
    # Remove HTML tags and URLs
    text = re.sub(r'<.*?>|http\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    # Lemmatize words and filter out stopwords
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(cleaned_words)


# --- 2. BALANCED DATA LOADING STRATEGY ---
# Define paths for raw input and processed output
raw_data_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\Amazon Reviews for Sentiment Analysis\train.ft.txt.bz2'
balanced_data = []

# Strategy: Collect 100,000 samples per class to prevent model bias
target_per_class = 100000
negative_count = 0
positive_count = 0

print(f"Loading balanced dataset (Target: {target_per_class} Negative + {target_per_class} Positive)...")

with bz2.BZ2File(raw_data_path, 'r') as bz2_file:
    for i, line in enumerate(bz2_file):
        # Exit loop once targets for both classes are met
        if negative_count >= target_per_class and positive_count >= target_per_class:
            break

        decoded_line = line.decode('utf-8')
        label = 1 if '__label__1' in decoded_line else 2

        # Append data only if the specific class target is not yet reached
        if label == 1 and negative_count < target_per_class:
            text = decoded_line.replace('__label__1 ', '').strip()
            balanced_data.append([label, text])
            negative_count += 1
        elif label == 2 and positive_count < target_per_class:
            text = decoded_line.replace('__label__2 ', '').strip()
            balanced_data.append([label, text])
            positive_count += 1

        # Logging progress every 50,000 processed rows
        if (negative_count + positive_count) % 50000 == 0 and (negative_count + positive_count) != 0:
            print(f"Progress: {negative_count + positive_count} rows collected...")

# Convert the list to a DataFrame
df = pd.DataFrame(balanced_data, columns=['label', 'text'])

# Shuffle the dataset to ensure class interspersion
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 3. DATA PREPROCESSING EXECUTION ---
print("\nApplying text cleaning (This process may take several minutes)...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# --- 4. VERIFICATION & EXPORT ---
print("\n--- CLASS DISTRIBUTION VERIFICATION ---")
print(df['label'].value_counts())

processed_output_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\cleaned_amazon_reviews.csv'
df.to_csv(processed_output_path, index=False)

print(f"\nSuccess! Balanced and cleaned dataset saved to: {processed_output_path}")