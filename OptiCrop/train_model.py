"""
train_model.py — Train all ML models, generate visualizations, and save model.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'Crop_recommendation.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = 'viridis'
FIG_DPI  = 100

def load_or_generate_data():
    if not os.path.exists(DATA_PATH):
        from generate_data import generate_crop_dataset
        df = generate_crop_dataset()
        import os

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[data] Generated dataset → {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"[data] Loaded dataset → {len(df)} rows")
    return df

def eda_plots(df):
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Histograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        axes[i].hist(df[feat], bins=30, color='#4CAF50', edgecolor='white', alpha=0.85)
        axes[i].set_title(feat.capitalize(), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    axes[-1].axis('off')
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'histograms.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 2. Crop distribution count plot
    fig, ax = plt.subplots(figsize=(14, 6))
    crop_counts = df['label'].value_counts()
    bars = ax.bar(crop_counts.index, crop_counts.values,
                  color=plt.cm.viridis(np.linspace(0.1, 0.9, len(crop_counts))),
                  edgecolor='white', linewidth=0.5)
    ax.set_title('Crop Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Crop', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, crop_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'crop_distribution.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                mask=mask, ax=ax, linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 10})
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 4. Box plots for temperature, humidity, rainfall
    for feat, color in [('temperature','#FF7043'), ('humidity','#42A5F5'), ('rainfall','#66BB6A')]:
        fig, ax = plt.subplots(figsize=(16, 6))
        top_crops = df['label'].value_counts().head(12).index
        sub = df[df['label'].isin(top_crops)]
        data_by_crop = [sub[sub['label'] == c][feat].values for c in top_crops]
        bp = ax.boxplot(data_by_crop, patch_artist=True, notch=False,
                        medianprops=dict(color='white', linewidth=2))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_crops)))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(top_crops)+1))
        ax.set_xticklabels(top_crops, rotation=45, ha='right')
        ax.set_title(f'{feat.capitalize()} Distribution by Crop', fontsize=14, fontweight='bold')
        ax.set_ylabel(feat.capitalize(), fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{feat}_distribution.png'), dpi=FIG_DPI, bbox_inches='tight')
        plt.close()

    print("[plots] EDA visualizations saved.")

def train_models(df):
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features].values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    models = {
        'Decision Tree':      DecisionTreeClassifier(random_state=42),
        'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors':KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True),
        'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'model':    model,
            'accuracy': round(acc * 100, 2),
            'y_test':   y_test,
            'y_pred':   y_pred,
        }
        print(f"  [{name}] Accuracy: {acc*100:.2f}%")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_result = results[best_name]
    print(f"[train] Best model: {best_name} ({best_result['accuracy']}%)")

    # Confusion matrix for best model
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.3, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # Model comparison bar chart
    model_names = list(results.keys())
    accuracies  = [results[n]['accuracy'] for n in model_names]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50' if n == best_name else '#90CAF9' for n in model_names]
    bars = ax.barh(model_names, accuracies, color=colors, edgecolor='white', height=0.55)
    for bar, val in zip(bars, accuracies):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.axvline(x=90, color='red', linestyle='--', alpha=0.4, label='90% threshold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    print("[plots] Performance visualizations saved.")

    clf_report = classification_report(
        best_result['y_test'], best_result['y_pred'],
        target_names=le.classes_, output_dict=True)

    payload = {
        'best_model_name': best_name,
        'best_model':      best_result['model'],
        'scaler':          scaler,
        'label_encoder':   le,
        'feature_names':   features,
        'all_results': {n: {'accuracy': results[n]['accuracy']} for n in results},
        'classification_report': clf_report,
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[model] Saved → {MODEL_PATH}")

    return payload


def run():
    df = load_or_generate_data()

    print("\n=== Dataset Info ===")
    print(f"Shape: {df.shape}")
    print(f"Crops: {df['label'].nunique()}")
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"Missing values: {missing}")
    print(f"Duplicates: {duplicates}")
    print(df.describe().to_string())

    print("\n=== Generating EDA Plots ===")
    eda_plots(df)

    print("\n=== Training Models ===")
    payload = train_models(df)
    return payload


if __name__ == '__main__':
    run()
