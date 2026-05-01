import bz2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- 1. CONFIGURATION & DATA LOADING ---
# Setting the file path for the Amazon dataset
# Note: Use relative paths when uploading to GitHub for better portability
dataset_path = r'C:\Users\Asus\OneDrive\Desktop\Amazon_Pazar_Analizi\Amazon Reviews for Sentiment Analysis/train.ft.txt.bz2'
reviews_list = []

print("Loading and decompressing data, please wait...")

# Processing the BZ2 file line by line to manage memory efficiency
with bz2.BZ2File(dataset_path, 'r') as bz2_file:
    for i, line in enumerate(bz2_file):
        # Sampling the first 50,000 reviews for efficient exploratory data analysis (EDA)
        if i >= 50000:
            break

        # Decoding byte format to string
        decoded_line = line.decode('utf-8')

        # Parsing the Kaggle label format (1: Negative, 2: Positive)
        label = 1 if '__label__1' in decoded_line else 2
        review_text = decoded_line.replace('__label__1 ', '').replace('__label__2 ', '').strip()

        reviews_list.append([label, review_text])

# Converting the list to a Pandas DataFrame
df = pd.DataFrame(reviews_list, columns=['label', 'text'])
print(f"Data successfully loaded! Total rows processed: {len(df)}\n")

# --- 2. DATA VISUALIZATION ---

# Visualization 1: Sentiment Distribution
print("Generating sentiment distribution plot...")
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False)
plt.title('Distribution of Sentiments (1: Negative, 2: Positive)')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.show()

# Visualization 2: Word Cloud for Positive Reviews
print("Generating WordCloud for positive reviews...")
# Combining all positive reviews (label == 2) for word frequency analysis
positive_reviews_combined = " ".join(review for review in df[df.label == 2].text)
word_cloud = WordCloud(
    background_color="white",
    max_words=100,
    width=800,
    height=400
).generate(positive_reviews_combined)

plt.figure(figsize=(10, 5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most Frequent Words in Positive Reviews')
plt.show()