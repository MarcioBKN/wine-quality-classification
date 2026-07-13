"""
src/preprocessing.py
Funções auxiliares de pré-processamento para o pipeline de classificação de vinhos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_wine_data(path: str = 'data/WineQT.csv') -> pd.DataFrame:
    """
    Carrega o Wine Quality Dataset (Kaggle) — WineQT.csv.
    São 1.143 amostras de vinho TINTO: a base não contém vinhos brancos.

    A coluna 'Id' é o identificador da amostra e é descartada (não é preditiva).
    """
    df = pd.read_csv(path)
    df = df.drop(columns=['Id'], errors='ignore')
    df.columns = df.columns.str.replace(' ', '_')
    return df


def create_binary_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Cria a variável target binária.
    1 = Alta Qualidade (nota >= threshold)
    0 = Baixa/Média Qualidade (nota < threshold)
    """
    df = df.copy()
    df['quality_binary'] = (df['quality'] >= threshold).astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features baseadas no conhecimento de domínio.
    """
    df = df.copy()

    # Razão entre acidez fixa e volátil
    df['acid_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 1e-6)

    # Proporção de SO2 livre no total
    df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 1e-6)

    # Relação álcool/densidade (proxy para corpo do vinho)
    df['alcohol_density'] = df['alcohol'] / df['density']

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Retorna a lista de colunas de features para o modelo.
    """
    exclude = ['quality', 'quality_binary', 'Id']
    return [c for c in df.columns if c not in exclude]


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Verifica valores faltantes."""
    missing = df.isnull().sum()
    return missing[missing > 0]


def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Detecta outliers usando o método IQR.
    Retorna um DataFrame com contagem e percentual de outliers por coluna.
    """
    results = []
    for col in columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        results.append({
            'coluna':    col,
            'outliers':  n_outliers,
            'percentual': f'{n_outliers / len(df) * 100:.1f}%'
        })
    return pd.DataFrame(results).set_index('coluna')
