from flask import Flask, request, render_template, redirect, url_for, flash
from markupsafe import Markup
import pandas as pd
import joblib
import os
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load pre-trained models and scaler
logistic_model = joblib.load('models/logistic_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Save uploaded file directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import re

def parse_metrics(metrics_text):
    """Extracts accuracy, precision, recall, and f1-score from the metrics text."""
    lines = metrics_text.split("\n")
    
    # Extract accuracy
    accuracy_match = re.search(r"Accuracy:\s([\d\.]+)", metrics_text)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    # Find precision, recall, f1-score from the "macro avg" row
    precision, recall, f1_score = None, None, None
    for line in lines:
        if "macro avg" in line:
            parts = line.split()
            if len(parts) >= 4:  # Ensure there are enough values
                try:
                    precision = float(parts[-3])  # Third-last column
                    recall = float(parts[-2])     # Second-last column
                    f1_score = float(parts[-1])   # Last column
                except ValueError:
                    print("Error: Could not parse precision, recall, or f1-score.")
                break
    
    return accuracy, precision, recall, f1_score



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            flash('File uploaded successfully!')
            return redirect(url_for('results', file_path=file_path))

    # Load metrics for display
    logistic_metrics_text = open('models/logistic_regression_metrics.txt', 'r').read()
    rf_metrics_text = open('models/random_forest_metrics.txt', 'r').read()

    # Parse metrics
    logistic_accuracy, logistic_precision, logistic_recall, logistic_f1 = parse_metrics(logistic_metrics_text)
    rf_accuracy, rf_precision, rf_recall, rf_f1 = parse_metrics(rf_metrics_text)

    return render_template(
        'home.html',
        logistic_accuracy=logistic_accuracy, logistic_precision=logistic_precision,
        logistic_recall=logistic_recall, logistic_f1=logistic_f1,
        rf_accuracy=rf_accuracy, rf_precision=rf_precision,
        rf_recall=rf_recall, rf_f1=rf_f1
    )


@app.route('/results')
def results():
    file_path = request.args.get('file_path')
    if not file_path or not os.path.exists(file_path):
        flash('Invalid file or file not found!')
        return redirect(url_for('home'))

    # Load the test dataset
    df = pd.read_csv(file_path)

    # Drop irrelevant columns if they exist
    if 'customerID' in df.columns:
        df.drop(['customerID'], axis=1, inplace=True)

    # Ensure the target column is not included in the features
    if 'Churn' in df.columns:
        df_features = df.drop('Churn', axis=1)  # Separate features
    else:
        df_features = df

    # Handle missing values
    df_features.replace(" ", pd.NA, inplace=True)
    df_features.fillna(df_features.median(numeric_only=True), inplace=True)

    # Encode categorical variables
    categorical_cols = df_features.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df_features.columns:
            df_features[col] = df_features[col].astype('category').cat.codes

    # Scale numerical features
    X = scaler.transform(df_features)

    # Generate predictions
    logistic_predictions = logistic_model.predict(X)
    rf_predictions = rf_model.predict(X)

    # Add predictions to the DataFrame
    df['Logistic_Prediction'] = logistic_predictions
    df['RF_Prediction'] = rf_predictions

    # Save predictions to a file (optional)
    predictions_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
    df.to_csv(predictions_file, index=False)

    # Convert DataFrame to HTML and mark as safe
    table_html = Markup(df.to_html(classes='table table-bordered', index=False))

    # Render results table
    return render_template('results.html', table_html=table_html, predictions_file=predictions_file)


if __name__ == '__main__':
    app.run(debug=True)