from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Global variables to store models
initial_model = None
new_model = None

# Load the initial model
def load_initial_model():
    global initial_model
    if initial_model is None and os.path.exists('model_6.pkl'):
        try:
            initial_model = joblib.load('model_6.pkl')
            logging.info("Initial model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading initial model: {e}")
    return initial_model

# Load the initial model at startup
initial_model = load_initial_model()

# Preprocessing pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy='median'))
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_exclude=np.number))
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive input data from the form
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        transaction_type = request.form['transactionType']

        type_payment = 1 if transaction_type == 'PAYMENT' else 0
        type_transfer = 1 if transaction_type == 'TRANSFER' else 0

        data = {
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_PAYMENT': type_payment,
            'type_TRANSFER': type_transfer
        }

        df = pd.DataFrame([data])
        X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'type_PAYMENT', 'type_TRANSFER']]

        prob_old, fraud_status_old = None, "Model not loaded"
        if initial_model:
            prob_old = initial_model.predict_proba(X)[:, 1][0]
            fraud_status_old = "Fraud" if prob_old >= 0.65 else "Not Fraud"
            logging.info(f"Prediction with initial model: {prob_old}")

        prob_new, fraud_status_new = None, None
        if new_model:
            prob_new = new_model.predict_proba(X)[:, 1][0]
            fraud_status_new = "Fraud" if prob_new >= 0.65 else "Not Fraud"
            logging.info(f"Prediction with new model: {prob_new}")

        return render_template('result.html', prob_old=prob_old, prob_new=prob_new, fraud_status_old=fraud_status_old, fraud_status_new=fraud_status_new)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('error.html', error_message=str(e))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            logging.info("File uploaded successfully.")

            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='unsigned')

            df['type'] = df['type'].astype('category')

            features_to_remove = [0, 4, 6, 7, 8]

            X_train, y_train, _ = data_transformations_feature_removal(df, features_to_remove)

            global new_model
            new_model = BalancedRandomForestClassifier(max_depth=None, min_samples_leaf=1,
                                                       min_samples_split=5, n_estimators=100, n_jobs=-1,
                                                       random_state=42)
            new_model.fit(X_train, y_train)
            logging.info("New model trained successfully.")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return render_template('error.html', error_message=str(e))

    return render_template('index.html', message="File uploaded and new model trained successfully.")

def data_transformations_feature_removal(data, features_to_remove):
    try:
        labels = None
        if 'isFraud' in data.columns:
            labels = data['isFraud']
            data = data.drop('isFraud', axis=1)
        if "nameOrig" in data.columns and "nameDest" in data.columns:
            data.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

        logging.info(f"Columns before preprocessing: {data.columns}")

        preprocessed_data = preprocessing.fit_transform(data)

        features = preprocessing.get_feature_names_out()

        logging.info(f"Features after preprocessing: {features}")

        preprocessed_data = np.delete(preprocessed_data, features_to_remove, axis=1)

        remaining_features = np.delete(features, features_to_remove)

        if labels is not None:
            labels = labels.to_numpy()

        return preprocessed_data, labels, remaining_features

    except Exception as e:
        logging.error(f"Error during data transformation and feature removal: {e}")
        raise

# New addition: Get port from environment variable
port = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)