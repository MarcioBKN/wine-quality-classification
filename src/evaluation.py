"""
src/evaluation.py
Funções auxiliares para avaliação e comparação de modelos de classificação.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Avalia um modelo e retorna dicionário com métricas.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'Modelo':    name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred),
        'F1-Score':  f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_prob)
    }


def compare_models(models: list, X_test, y_test) -> pd.DataFrame:
    """
    Compara múltiplos modelos. models = list of (name, fitted_pipeline).
    """
    results = [evaluate_model(name, model, X_test, y_test) for name, model in models]
    return pd.DataFrame(results).set_index('Modelo').round(4)


def plot_roc_curves(models: list, X_test, y_test, save_path: str = None):
    """
    Plota curvas ROC para múltiplos modelos.
    """
    colors = ['royalblue', 'forestgreen', 'crimson', 'darkorange']
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, model), color in zip(models, colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatório (AUC = 0.500)')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC – Comparação dos Modelos')
    ax.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(models: list, X_test, y_test, save_path: str = None):
    """
    Plota matrizes de confusão lado a lado.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Baixa/Média', 'Alta'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'Matriz de Confusão\n{name}', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def cross_validate_models(models: list, X, y, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Realiza validação cruzada estratificada para múltiplos modelos.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for name, model in models:
        f1_scores  = cross_val_score(model, X, y, cv=cv, scoring='f1')
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        results.append({
            'Modelo':        name,
            'F1 Médio':      round(f1_scores.mean(), 4),
            'F1 Std':        round(f1_scores.std(), 4),
            'ROC-AUC Médio': round(auc_scores.mean(), 4),
            'ROC-AUC Std':   round(auc_scores.std(), 4),
        })

    return pd.DataFrame(results).set_index('Modelo')
