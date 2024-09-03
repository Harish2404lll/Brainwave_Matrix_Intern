import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# Load the dataset with a different encoding
file_path = '/content/data.csv'  # Update this with your actual file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Drop columns that are mostly empty (Unnamed columns)
cleaned_data = data.drop(columns=[col for col in data.columns if "Unnamed" in col])

# Drop rows with missing values in the important columns (Headline, Body, Label)
cleaned_data = cleaned_data.dropna(subset=["Headline", "Body", "Label"])

# Identify and remove non-numeric labels in the "Label" column
non_numeric_labels = cleaned_data[pd.to_numeric(cleaned_data["Label"], errors='coerce').isna()]
cleaned_data = cleaned_data.drop(non_numeric_labels.index)

# Convert the "Label" column to integers after cleaning
cleaned_data["Label"] = cleaned_data["Label"].astype(int)

# Combine headline and body into a single feature
class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["Headline"] + " " + X["Body"]

# Define the text vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)

# Create a pipeline
pipeline = Pipeline([
    ('text_combiner', TextCombiner()),
    ('vectorizer', vectorizer),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Prepare the data for training
X = cleaned_data[["Headline", "Body"]]
y = cleaned_data["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate the model on the test set
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
