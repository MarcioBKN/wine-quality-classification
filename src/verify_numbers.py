"""
src/verify_numbers.py
Auditoria independente: recalcula, a partir dos CSVs brutos, TODOS os números
publicados no README, na apresentação e em results/metricas_modelos.csv.

Não importa nada do notebook — reimplementa o pipeline do zero. Se um número
publicado divergir do recalculado, o script falha e aponta a divergência.

Uso:  python src/verify_numbers.py      (a partir da raiz do repositório)
"""

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TOL = 0.001          # tolerância para métricas (3 casas decimais)
TOL_PCT = 0.1        # tolerância para percentuais publicados (0,1 p.p.)

falhas = []


def check(rotulo, publicado, recalculado, tol):
    ok = abs(publicado - recalculado) <= tol
    status = 'OK  ' if ok else 'FALHA'
    print(f'  [{status}] {rotulo:38s} publicado={publicado:<8.3f} recalculado={recalculado:<8.3f}')
    if not ok:
        falhas.append(f'{rotulo}: publicado {publicado}, recalculado {recalculado}')


# ---------------------------------------------------------------- 1. Dados
print('\n1. DADOS BRUTOS  (data/WineQT.csv — Kaggle)')
df = pd.read_csv(ROOT / 'data/WineQT.csv').drop(columns=['Id'])
df.columns = df.columns.str.replace(' ', '_')
df['quality_binary'] = (df['quality'] >= 7).astype(int)

check('amostras', 1143, len(df), 0)
check('variáveis físico-químicas', 11, df.shape[1] - 2, 0)
check('% alta qualidade na base', 13.9, df['quality_binary'].mean() * 100, TOL_PCT)

# ------------------------------------------------- 2. Correlações (slide 4)
print('\n2. CORRELAÇÕES COM O ALVO BINÁRIO (slide 4 do PPTX)')
corr = df.corr(numeric_only=True)['quality_binary']
publicadas = {'alcohol': 0.40, 'volatile_acidity': -0.31, 'citric_acid': 0.25,
              'sulphates': 0.21, 'density': -0.15, 'fixed_acidity': 0.12,
              'chlorides': -0.10}
for var, val in publicadas.items():
    check(f'corr {var}', val, corr[var], 0.006)

# Slide 5: as três variáveis que NÃO explicam a qualidade
print('\n   Slide 5 — as que não explicam (devem ficar perto de zero):')
for var in ('residual_sugar', 'pH', 'free_sulfur_dioxide'):
    check(f'corr {var} ~ 0', 0.0, corr[var], 0.10)

# Ácido cítrico x acidez volátil: o "mérito indireto" citado no slide 6
check('corr citrico x acidez volatil', -0.54, df.citric_acid.corr(df.volatile_acidity), 0.01)

# ------------------------------------------------------ 3. Insights (slide 6)
print('\n3. INSIGHTS DO SLIDE 6')


def taxa(col, thr):
    return (df[df[col] > thr]['quality_binary'].mean() * 100,
            df[df[col] <= thr]['quality_binary'].mean() * 100)


for rotulo, col, thr, esp_hi, esp_lo in [
    ('alcool > 11', 'alcohol', 11, 35.0, 6.0),
    ('acidez volatil > 0,7', 'volatile_acidity', 0.7, 2.0, 16.0),
    ('sulfatos > 0,65', 'sulphates', 0.65, 27.0, 6.0),
    ('acido citrico > 0,3', 'citric_acid', 0.30, 26.0, 6.0),
]:
    hi, lo = taxa(col, thr)
    check(f'% alta qual. {rotulo}', esp_hi, hi, 0.6)
    check(f'% alta qual. NAO {rotulo}', esp_lo, lo, 0.6)

# -------------------------------------- 4. Modelos: pipeline reimplementado
print('\n4. MÉTRICAS DOS MODELOS (README, slide 7, metricas_modelos.csv)')
df['acid_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 1e-6)
df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 1e-6)
df['alcohol_density'] = df['alcohol'] / df['density']

feature_cols = [c for c in df.columns if c not in ('quality', 'quality_binary')]
X, y = df[feature_cols], df['quality_binary']
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

modelos = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=SEED, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=SEED,
        class_weight='balanced', n_jobs=-1),
}

recalculado = {}
for nome, clf in modelos.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', clf)]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    prob = pipe.predict_proba(X_te)[:, 1]
    recalculado[nome] = {
        'Accuracy': accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred),
        'Recall': recall_score(y_te, pred),
        'F1-Score': f1_score(y_te, pred),
        'ROC-AUC': roc_auc_score(y_te, prob),
    }

# Compara com o CSV que o notebook publicou
csv = pd.read_csv(ROOT / 'results/metricas_modelos.csv').set_index('Modelo')
for nome, metricas in recalculado.items():
    for metrica, valor in metricas.items():
        check(f'{nome[:2]} {metrica}', csv.loc[nome, metrica], valor, TOL)

# --------------------------------- 5. Os números publicados batem com o CSV?
print('\n5. TEXTO PUBLICADO x results/metricas_modelos.csv')
readme = (ROOT / 'README.md').read_text(encoding='utf-8')
pptx = zipfile.ZipFile(ROOT / 'wine_quality_eda_storytelling.pptx')
slide7 = ' '.join(re.findall(r'<a:t>(.*?)</a:t>',
                             pptx.read('ppt/slides/slide7.xml').decode('utf-8')))

for nome in modelos:
    for metrica in ('Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'):
        v = csv.loc[nome, metrica]
        no_readme = f'{v:.3f}' in readme
        no_slide = f'{v * 100:.1f}'.replace('.', ',') + '%' in slide7
        status = 'OK  ' if (no_readme and no_slide) else 'FALHA'
        print(f'  [{status}] {nome[:2]} {metrica:10s} {v:.3f} '
              f'-> README: {"sim" if no_readme else "NAO"} | slide 7: {"sim" if no_slide else "NAO"}')
        if not (no_readme and no_slide):
            falhas.append(f'{nome} {metrica} ({v:.3f}) ausente do README e/ou do slide 7')

# ------------------------------------------------------------------ Resultado
print('\n' + '=' * 72)
if falhas:
    print(f'AUDITORIA FALHOU — {len(falhas)} divergência(s):')
    for f in falhas:
        print('  -', f)
    sys.exit(1)
print('AUDITORIA OK — todos os números publicados foram reproduzidos a partir dos CSVs brutos.')
print('=' * 72)
