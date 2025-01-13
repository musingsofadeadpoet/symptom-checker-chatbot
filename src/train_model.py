import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from data_preprocessing import preprocess_data, split_data
from nlp_pipeline import tokenize

def load_data(filepath):
    return pd.read_csv(filepath)

def train_model(train_df):
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(train_df['symptoms'], train_df['disease'])
    return pipeline

def evaluate_model(model, test_df):
    predictions = model.predict(test_df['symptoms'])
    accuracy = accuracy_score(test_df['disease'], predictions)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    train_data_path = './data/processed/train.csv'
    test_data_path = './data/processed/test.csv'
    model_save_path = './models/symptom_checker_model.pkl'

    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)

    # Debugging step
    print(f"Train Data Columns: {train_df.columns}")
    print(f"Test Data Columns: {test_df.columns}")

    if 'disease' not in train_df.columns:
        raise KeyError("'disease' column is missing in the train dataset. Please check the preprocessing step.")

    model = train_model(train_df)
    evaluate_model(model, test_df)

    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")