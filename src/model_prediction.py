import pandas as pd
import numpy as np
import argparse
import joblib
import json

def load_data(file_path):
    # TODO: Load test data from CSV file
    df = pd.read_csv(file_path)
    return df

def load_model(model_path):
    # TODO: Load the trained model
    model = joblib.load(model_path)
    return model

def make_predictions(df, model):
    # TODO: Use the model to make predictions on the test data
    def rounding(number):
        number = int(number + (0.5 if number > 0 else -0.5))
        return number

    X_val = pd.read_csv('X_val.csv')
    raw_predictions = model.predict(X_val)
    predictions = np.array([rounding(predict) for predict in raw_predictions])
    return predictions

def save_predictions(predictions, predictions_file):
    # TODO: Save predictions to a JSON file
    with open(predictions_file, 'w') as json_file:
        json.dump(predictions.tolist(), json_file)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/test_data.csv',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='predictions/predictions.json',
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
