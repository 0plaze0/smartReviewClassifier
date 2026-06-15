import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

print("Step 1: Loading and sampling dataset...")
# Load the real Kaggle file and sample 50,000 rows for fast training
raw_df = pd.read_csv('Reviews.csv')
sampled_df = raw_df.sample(n=50000, random_state=42)

# Clean and format: Drop neutral 3-star reviews to keep sentiment binary
sampled_df = sampled_df[sampled_df['Score'] != 3]
X = sampled_df['Text']
# Map scores: 4-5 stars = Positive (1), 1-2 stars = Negative (0)
y = sampled_df['Score'].apply(lambda x: 1 if x > 3 else 0)

print("Step 2: Splitting data into Train/Test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Step 3: Vectorizing text data (TF-IDF)...")
# Convert text to numerical features using unigrams and bigrams
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Step 4: Training the Ensemble Classifier...")
# Initialize models with a fixed random state for 100% reproducibility
clf1 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf2 = MultinomialNB(alpha=0.1)

# Combine them into a soft-voting ensemble
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('nb', clf2)], voting='soft')
ensemble_model.fit(X_train_vec, y_train)

print("Step 5: Evaluating model performance...")
preds = ensemble_model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("\nTraining Complete!")
print(f"Final Accuracy: {acc*100:.1f}%")
print(f"Final F1-Score: {f1:.2f}")

print("\nStep 6: Saving pipeline components...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)
print("Model artifacts saved successfully!")