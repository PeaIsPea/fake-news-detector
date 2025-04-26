# Import necessary libraries for machine learning pipeline
from imblearn.pipeline import Pipeline  # Use imbalanced-learn's pipeline to support oversampling
from imblearn.over_sampling import RandomOverSampler  # Handle class imbalance
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Read and preprocess datasets
column = "year_month"
data_train = pd.read_csv("../dataset/train.csv").drop(column, axis=1)
data_valid = pd.read_csv("../dataset/valid.csv").drop(column, axis=1)
data_test = pd.read_csv("../dataset/test.csv").drop(column, axis=1)

# Separate features and target
target = "labels"
x_train = data_train.drop(target, axis=1)
y_train = data_train[target]
x_test = data_test.drop(target, axis=1)
y_test = data_test[target]

# Text preprocessing using Tfidf for both title and content
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("text", TfidfVectorizer(min_df=0.01, max_df=0.95, stop_words="english", ngram_range=(1, 2)), "text"),
])

# Build the pipeline: preprocess, oversample, feature select, and classify
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("oversampler", RandomOverSampler(random_state=42)),  # Balance the classes
    ("feature_selector", SelectPercentile(score_func=chi2, percentile=10)),  # Feature selection
    ("model", RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for tuning
params = {
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selector__percentile": [1, 5, 10]
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid=params, cv=4, scoring="recall_weighted", verbose=2)
grid_search.fit(x_train, y_train)

# Evaluate on test set
y_predict = grid_search.predict(x_test)
print("Best Params:", grid_search.best_params_)
print(classification_report(y_test, y_predict))

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_predict, labels=["fake", "true"], cmap="Blues")
plt.title("Confusion Matrix with Oversampling")
plt.show()

# Save trained model to .pkl
joblib.dump(grid_search.best_estimator_, "../model/fake_news_classifier.pkl")
print("✅ SAVED MODEL: model/fake_news_classifier.pkl")
