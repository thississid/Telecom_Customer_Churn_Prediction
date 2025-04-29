from flask import Flask, request, render_template, redirect, url_for, flash
from markupsafe import Markup
import pandas as pd
import joblib
import os
import re
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load pre-trained models and scaler
logistic_model = joblib.load('models/logistic_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
deep_learning_model = tf.keras.models.load_model('models/deep_learning_model.h5')

# Save uploaded file directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    dl_metrics_text = open('models/deep_learning_model_metrics.txt', 'r').read()

    # Parse metrics
    logistic_accuracy, logistic_precision, logistic_recall, logistic_f1 = parse_metrics(logistic_metrics_text)
    rf_accuracy, rf_precision, rf_recall, rf_f1 = parse_metrics(rf_metrics_text)
    dl_accuracy, dl_precision, dl_recall, dl_f1 = parse_metrics(dl_metrics_text)

    return render_template(
        'home.html',
        logistic_accuracy=logistic_accuracy, logistic_precision=logistic_precision,
        logistic_recall=logistic_recall, logistic_f1=logistic_f1,
        rf_accuracy=rf_accuracy, rf_precision=rf_precision,
        rf_recall=rf_recall, rf_f1=rf_f1,
        dl_accuracy=dl_accuracy, dl_precision=dl_precision,
        dl_recall=dl_recall, dl_f1=dl_f1
    )

@app.route('/results')
def results():
    file_path = request.args.get('file_path')
    if not file_path or not os.path.exists(file_path):
        flash('Invalid file or file not found!')
        return redirect(url_for('home'))

    # 1) Load your test data
    df = pd.read_csv(file_path)
    if 'customerID' in df.columns:
        df.drop(['customerID'], axis=1, inplace=True)
    if 'Churn' in df.columns:
        df_features = df.drop('Churn', axis=1)
    else:
        df_features = df

    # 2) Preprocess exactly like in home()
    df_features.replace(" ", pd.NA, inplace=True)
    df_features.fillna(df_features.median(numeric_only=True), inplace=True)
    for col in df_features.select_dtypes(include=['object']).columns:
        df_features[col] = df_features[col].astype('category').cat.codes

    X = scaler.transform(df_features)

    # 3) **Generate predictions** (must come before using them!)
    logistic_predictions = logistic_model.predict(X)
    rf_predictions       = rf_model.predict(X)
    dl_predictions       = (deep_learning_model.predict(X) > 0.5).astype('int32').flatten()

    # 4) Add them back into the DataFrame
    df['Logistic_Prediction'] = logistic_predictions
    df['RF_Prediction']       = rf_predictions
    df['DL_Prediction']       = dl_predictions

    # 5) Compute correct vs wrong counts
    total = len(df)
    logistic_correct = int((df['Logistic_Prediction'] == df['Churn']).sum())
    logistic_wrong   = total - logistic_correct
    rf_correct       = int((df['RF_Prediction'] == df['Churn']).sum())
    rf_wrong         = total - rf_correct
    dl_correct       = int((df['DL_Prediction'] == df['Churn']).sum())
    dl_wrong         = total - dl_correct

    # 6) Save the CSV (optional)
    predictions_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
    df.to_csv(predictions_file, index=False)

    # 7) Convert to rows/columns for the template
    rows    = df.to_dict(orient='records')
    columns = df.columns.tolist()

    return render_template('results.html',
                           rows=rows,
                           columns=columns,
                           predictions_file=predictions_file,
                           logistic_correct=logistic_correct,
                           logistic_wrong=logistic_wrong,
                           rf_correct=rf_correct,
                           rf_wrong=rf_wrong,
                           dl_correct=dl_correct,
                           dl_wrong=dl_wrong)


if __name__ == '__main__':
    app.run(debug=True)
