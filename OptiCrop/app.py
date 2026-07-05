"""
app.py — OptiCrop Flask Application
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'dataset', 'Crop_recommendation.csv')

app = Flask(__name__)

_model_cache = None

def ensure_ready():
    """Generate dataset, train models, and plots if not already done."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_PATH):
        print("[startup] Running model training pipeline...")
        from train_model import run
        _model_cache = run()
    else:
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
        print("[startup] Loaded existing model.pkl")

    return _model_cache


def load_dataframe():
    if not os.path.exists(DATA_PATH):
        from generate_data import generate_crop_dataset
        df = generate_crop_dataset()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    ensure_ready()
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            N           = float(request.form['nitrogen'])
            P           = float(request.form['phosphorous'])
            K           = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity    = float(request.form['humidity'])
            ph          = float(request.form['ph'])
            rainfall    = float(request.form['rainfall'])

            payload = ensure_ready()
            scaler  = payload['scaler']
            model   = payload['best_model']
            le      = payload['label_encoder']

            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            features_scaled = scaler.transform(features)
            pred_idx  = model.predict(features_scaled)[0]
            crop_name = le.inverse_transform([pred_idx])[0]

            return render_template(
                'result.html',
                crop=crop_name.capitalize(),
                nitrogen=N, phosphorous=P, potassium=K,
                temperature=temperature, humidity=humidity,
                ph=ph, rainfall=rainfall,
            )
        except (ValueError, KeyError) as e:
            return render_template('predict.html', error=f"Invalid input: {e}")

    return render_template('predict.html')


@app.route('/dashboard')
def dashboard():
    payload = ensure_ready()
    df      = load_dataframe()

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    stats = {}
    for feat in features:
        stats[feat] = {
            'mean':  round(df[feat].mean(), 2),
            'std':   round(df[feat].std(),  2),
            'min':   round(df[feat].min(),  2),
            'max':   round(df[feat].max(),  2),
        }

    return render_template(
        'dashboard.html',
        total_records=len(df),
        num_crops=df['label'].nunique(),
        crop_list=sorted(df['label'].unique().tolist()),
        feature_stats=stats,
        features=features,
    )


@app.route('/performance')
def performance():
    payload = ensure_ready()

    all_results = payload['all_results']
    best_name   = payload['best_model_name']
    clf_report  = payload['classification_report']

    top_crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate']
    filtered_report = {
        k: v for k, v in clf_report.items()
        if isinstance(v, dict)
    }

    return render_template(
        'performance.html',
        all_results=all_results,
        best_name=best_name,
        classification_report=filtered_report,
        accuracy_summary={k: v['accuracy'] for k, v in all_results.items()},
    )


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ensure_ready()
    app.run(host='0.0.0.0', port=5000, debug=True)
