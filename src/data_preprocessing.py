import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """Load the dataset from the given file path."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Perform preprocessing steps like removing missing values and combining symptoms."""
    df = df.dropna()  # Remove missing values

    # Combine all symptom columns into a single 'symptoms' column
    symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
    df['symptoms'] = df[symptom_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Normalize symptoms (convert to lowercase)
    df['symptoms'] = df['symptoms'].str.lower()

    # Retain only 'disease' and 'symptoms' columns
    df = df[['Disease', 'symptoms']].rename(columns={'Disease': 'disease'})

    # Verify the presence of the 'disease' column
    if 'disease' not in df.columns:
        raise KeyError("The dataset does not contain a column named 'disease'. Please check the file.")

    return df

def split_data(df, test_size=0.2):
    """Split the data into training and testing sets."""
    return train_test_split(df, test_size=test_size, random_state=42)

if __name__ == "__main__":
    # Define file paths
    raw_data_path = '/Users/nanguyen/symptom-checker-chatbot/data/raw/DiseaseAndSymptoms.csv'
    processed_data_path = '/Users/nanguyen/symptom-checker-chatbot/data/processed/'

    # Load, preprocess, and split the data
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    train_df, test_df = split_data(df)

    # Save the processed train and test datasets
    os.makedirs(processed_data_path, exist_ok=True)
    train_df.to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_data_path, 'test.csv'), index=False)