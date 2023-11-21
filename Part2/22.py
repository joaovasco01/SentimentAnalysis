import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_json('/Users/joaovasco/Desktop/Part I/Part2/Software.json', lines=True)

# Create binary label: 1 if votes > threshold (e.g., 5), else 0
data['vote'] = pd.to_numeric(data['vote'], errors='coerce').fillna(0)
data['helpful'] = data['vote'].apply(lambda x: 1 if x > 5 else 0)

# Feature selection (using reviewText for simplicity)
X = data['reviewText']
y = data['helpful']
X = data['reviewText'].fillna('')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numeric features
vectorizer = TfidfVectorizer(max_features=3)  # Limit number of features for simplicity
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_vect, y_train)

# Predict and evaluate
predictions = model.predict(X_test_vect)
print(classification_report(y_test, predictions))
