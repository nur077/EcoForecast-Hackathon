import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV


def load_data(file_path):
    # TODO: Load processed data from CSV file
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    X = df.drop(columns=["ID"], axis=1)
    y = df["ID"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val.to_csv('X_val.csv', index=False)
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    # TODO: Initialize your model and train it
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=20)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    # TODO: Save your trained model
    joblib.dump(model, model_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/processed_data.csv',
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='models/model.pkl',
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)
