import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
#  max_features=yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
# ------------------------------------------------------------------
# 1. Load processed data
train_df = pd.read_csv('./data/interim/train_processed.csv')
test_df  = pd.read_csv('./data/interim/test_processed.csv')

# 2. Split into X (text) and y (label)
X_train_text = train_df['clean_comment'].astype(str)
y_train      = train_df['category'].values

X_test_text  = test_df['clean_comment'].astype(str)
y_test       = test_df['category'].values

# 3. Bag‑of‑Words
vectorizer       = TfidfVectorizer(max_features=50)
X_train_tfidf     = vectorizer.fit_transform(X_train_text)
X_test_tfidf       = vectorizer.transform(X_test_text)
vocab            = vectorizer.get_feature_names_out()     # column names

# 4. Convert sparse matrices to DataFrames and append label
train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vocab)
train_tfidf_df['label'] = y_train

test_tfidf_df  = pd.DataFrame(X_test_tfidf.toarray(), columns=vocab)
test_tfidf_df['label']  = y_test

# 5. Save
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

train_tfidf_df.to_csv(os.path.join(output_dir, 'train_tfidf.csv'), index=False)
test_tfidf_df.to_csv(os.path.join(output_dir, 'test_tfidf.csv'),  index=False)
