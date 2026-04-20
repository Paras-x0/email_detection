import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -- 1. Load Data ----------------------------------------------------------------
raw_mail_data = pd.read_csv('mail_data.csv')

# Replace null values with empty string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
print(f"Dataset shape: {mail_data.shape}")
print(mail_data.head())

# -- 2. Label Encoding -----------------------------------------------------------
label_map = {'spam': 0, 'ham': 1}
mail_data['Category'] = mail_data['Category'].map(label_map)

print("\nLabel distribution:")
print(mail_data['Category'].value_counts())

# -- 3. Split Features & Labels --------------------------------------------------
X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3, stratify=Y
)

print(f"\nTraining samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# -- 4. TF-IDF Vectorization -----------------------------------------------------
feature_extraction = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features  = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test  = Y_test.astype('int')

# -- 5. Train Model --------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, Y_train)

# -- 6. Evaluate -----------------------------------------------------------------
for split_name, X_feat, Y_true in [
    ("Training", X_train_features, Y_train),
    ("Test",     X_test_features,  Y_test)
]:
    preds = model.predict(X_feat)
    acc   = accuracy_score(Y_true, preds)
    print(f"\n-- {split_name} Results --")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(Y_true, preds, target_names=['Spam', 'Ham']))

# -- 7. Prediction Function ------------------------------------------------------
def predict_mail(text: str) -> None:
    """Predict whether a given email is spam or ham with confidence."""
    if not text.strip():
        print("No message entered. Please type something.")
        return

    features   = feature_extraction.transform([text])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]

    label = "Ham" if prediction == 1 else "Spam"
    print("\n" + "-" * 60)
    print(f"Message    : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"Prediction : {label}")
    print(f"Confidence : Ham={confidence[1]:.2%}  |  Spam={confidence[0]:.2%}")
    print("-" * 60)

# -- 8. Interactive Mail Checker -------------------------------------------------
def run_mail_checker():
    """Interactive loop for the user to check multiple emails."""
    print("\n" + "=" * 60)
    print("         Welcome to the Spam Mail Checker")
    print("=" * 60)
    print("Paste your email message across multiple lines.")
    print("When done, type 'END' on a new line and press Enter.")
    print("Type 'quit' or 'exit' on a new line to stop.\n")

    while True:
        print("\nEnter mail message (type 'END' when done):")
        lines = []

        while True:
            line = input()

            # exit conditions
            if line.strip().lower() in ('quit', 'exit', 'q'):
                print("\nExiting mail checker. Goodbye.")
                return

            # end of message signal
            if line.strip().upper() == 'END':
                break

            lines.append(line)

        # join all lines into one message
        user_input = ' '.join(lines).strip()

        # empty input guard
        if not user_input:
            print("Message cannot be empty. Please try again.")
            continue

        predict_mail(user_input)

        # ask if user wants to check another mail
        again = input("\nCheck another mail? (yes/no): ").strip().lower()
        if again not in ('yes', 'y'):
            print("\nExiting mail checker. Goodbye.")
            break

run_mail_checker()