import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
nltk.download('stopwords')
df = pd.read_csv(
   r"C:\Users\rohit\OneDrive\Documents\vs\python\csv\spam.csv",
    encoding='latin-1',
    sep='\t',
    header=None,
    names=['label', 'message', 'unnamed_2', 'unnamed_3', 'unnamed_4']
)
df = df[['label', 'message']]
df = df.dropna(subset=['message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("Sample data:")
print(df.head())
print("\nDataset size:", df.shape)
stop_words = set(stopwords.words('english'))
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = " ".join(words)
    return text
df['message'] = df['message'].apply(preprocess)
X = df['message']
y = df['label']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("\nModel Performance")
print("------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print("\n--- Test Your Own Email ---")
while True:
    email = input("\nEnter email text (or type 'exit'): ")
    if email.lower() == "exit":
        break
    email_processed = preprocess(email)
    email_vector = vectorizer.transform([email_processed])
    prediction = model.predict(email_vector)
    if prediction[0] == 1:
        print("Result: SPAM email")
    else:
        print("Result: HAM (Not Spam)")
print("\nProgram Finished")