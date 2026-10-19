import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading training data...")
df = pd.read_csv("training_data.csv")

# 1. Clean Data: Fill empty text with blank strings
df['added_text'] = df['added_text'].fillna('')

# 2. Feature Engineering: Log-transform byte counts 
# (Adding 10,000 bytes isn't 10x riskier than 1,000 bytes, so we use log scale)
df['log_bytes_added'] = np.log1p(df['bytes_added'])
df['log_bytes_deleted'] = np.log1p(df['bytes_deleted'])

# 3. Setup the Feature Pipeline
print("Building the preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        # Scale our 3 dense behavioral features
        ('dense', StandardScaler(), ['is_anon', 'log_bytes_added', 'log_bytes_deleted']),
        # Extract top 2000 words/phrases from the diff text
        ('text', TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english'), 'added_text')
    ]
)

# 4. Setup the L1 Logistic Regression Model
# penalty='l1' forces useless words to have a weight of exactly 0.0
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.5, random_state=42))
])

# 5. Train & Evaluate
X = df[['is_anon', 'log_bytes_added', 'log_bytes_deleted', 'added_text']]
y = df['is_reverted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model... (this takes a few seconds)")
model.fit(X_train, y_train)
print(f"Test Accuracy: {model.score(X_test, y_test):.2%}")

# 6. Extract Top Features
clf = model.named_steps['clf']
tfidf = model.named_steps['preprocessor'].named_transformers_['text']

feature_names = ['is_anon', 'log_bytes_added', 'log_bytes_deleted'] + list(tfidf.get_feature_names_out())
coefs = clf.coef_[0]

weights = pd.DataFrame({'feature': feature_names, 'weight': coefs})

print("\n🚨 TOP 10 FEATURES THAT PREDICT A REVERT (Highest Log-Odds):")
print(weights.sort_values(by='weight', ascending=False).head(10))

print("\n✅ SAFE FEATURES (Predicts the edit will survive):")
print(weights.sort_values(by='weight', ascending=True).head(5))

# 7. Save for Phase 4
joblib.dump(model, 'wiki_revert_model.joblib')
print("\nModel saved to 'wiki_revert_model.joblib'.")