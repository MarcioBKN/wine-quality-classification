# 🍷 Wine Quality Classification

**Tech Challenge – Fase 2 | POSTECH DTAT**

Modelo de Machine Learning para classificação binária da qualidade de vinhos com base em características físico-químicas.

---

## 🎯 Objetivo

Desenvolver um pipeline completo de análise e modelagem para prever se um vinho é de **Alta Qualidade** (nota ≥ 7) ou **Baixa/Média Qualidade** (nota < 7), utilizando dados físico-químicos do processo produtivo.

---

## 📁 Estrutura do Repositório

```
wine-quality-classification/
│
├── data/                   # Base de dados utilizada
│   └── WineQT.csv          # Wine Quality Dataset (Kaggle)
│
├── notebooks/              # Notebook com análise e modelagem
│   └── wine_quality_classification.ipynb
│
├── src/                    # Scripts auxiliares
│   ├── preprocessing.py    # Funções de pré-processamento
│   ├── evaluation.py       # Funções de avaliação dos modelos
│   ├── verify_numbers.py   # Auditoria: recalcula todos os números publicados
│   └── build_presentation.py  # Gera o .pptx a partir dos dados e das métricas
│
├── results/                # Gráficos e métricas geradas
│   ├── 01_distribuicao_qualidade.png
│   ├── 02_balanceamento_classes.png
│   ├── 03_distribuicao_variaveis.png
│   ├── 04_correlacao.png
│   ├── 05_boxplots_top_features.png
│   ├── 06_confusion_matrix.png
│   ├── 07_roc_curves.png
│   ├── 08_feature_importance_rf.png
│   ├── 09_feature_importance_lr.png
│   └── metricas_modelos.csv
│
├── requirements.txt        # Bibliotecas utilizadas
└── README.md               # Este arquivo
```

---

## 📊 Dataset

**Fonte:** [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) (Kaggle) — arquivo `WineQT.csv`, versionado em `data/`.

| Variável | Descrição |
|---|---|
| fixed acidity | Acidez fixa (tartárico) |
| volatile acidity | Acidez volátil (acético) |
| citric acid | Ácido cítrico |
| residual sugar | Açúcar residual |
| chlorides | Cloretos |
| free sulfur dioxide | SO₂ livre |
| total sulfur dioxide | SO₂ total |
| density | Densidade |
| pH | Potencial hidrogeniônico |
| sulphates | Sulfatos |
| alcohol | Teor alcoólico |
| quality | **Target original** (3 a 8) |

**Amostras:** 1.143, todas de **vinho tinto**  
**Target binário:** Alta Qualidade (≥7) vs Baixa/Média (<7) — 13,9% contra 86,1%

> A coluna `Id` do CSV é apenas o identificador da amostra e é descartada: não é variável preditiva.
>
> **Escopo:** a base contém somente tintos. As conclusões deste projeto não podem ser estendidas a
> vinhos brancos sem nova validação — o perfil físico-químico dos brancos é bastante diferente,
> sobretudo em SO₂, açúcar residual e acidez volátil.

---

## 🤖 Modelos Treinados

Resultados no conjunto de teste (20% dos dados, split estratificado, `random_state=42`).
Valores reproduzidos por `results/metricas_modelos.csv`, gerado pelo notebook.

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.790 | 0.362 | 0.656 | 0.467 | 0.864 |
| **Random Forest** | **0.921** | **0.818** | 0.562 | **0.667** | **0.912** |

Validação cruzada (5-fold estratificado): F1 de 0.501 ± 0.028 (LR) contra 0.636 ± 0.099 (RF).

> **Melhor modelo: Random Forest** — maior ROC-AUC e maior F1-Score, por capturar relações não-lineares
> entre as variáveis físico-químicas.
>
> Os dois modelos erram de formas diferentes. A Regressão Logística tem recall maior (0.656) mas precisão
> baixa (0.362): encontra mais vinhos bons, porém erra quase 2 em cada 3 dos que aponta como bons. A Random
> Forest acerta 82% dos que aponta, ao custo de deixar escapar mais vinhos bons. Para triagem em produção,
> a Random Forest é a escolha mais segura.
>
> O recall modesto de ambos reflete a dificuldade real do problema: só 159 das 1.143 amostras são de alta
> qualidade, e o desvio-padrão alto do F1 da Random Forest na validação cruzada (± 0.099) é consequência
> direta dessa escassez.

---

## 🔍 Principais Insights

1. **Teor alcoólico** é a variável com maior influência na qualidade (correlação **+0,40** com o alvo binário,
   e 1º lugar na importância da Random Forest junto com a feature derivada `alcohol_density`). Acima de 11% de
   álcool, 35% dos vinhos são de alta qualidade — contra apenas 6% abaixo desse limite.
2. **Acidez volátil** é o principal marcador negativo (**−0,31**): é o aroma avinagrado, produzido pela bactéria
   acética. Acima de 0,7 g/L, apenas 2% dos vinhos atingem alta qualidade, contra 16% abaixo.
3. **Sulfatos** (**+0,21**) e **ácido cítrico** (**+0,25**) acompanham notas mais altas: os primeiros protegem o
   vinho da oxidação; o segundo traz frescor — embora parte do seu mérito seja indireta, já que onde há ácido
   cítrico costuma haver menos acidez volátil (correlação de −0,54 entre os dois).
4. **Açúcar residual, pH e SO₂ livre não explicam a qualidade** (correlações de +0,06, −0,07 e −0,06). Mexer na
   doçura ou corrigir o pH não move a nota — o esforço do processo deve ir para álcool e acidez volátil.
5. **Densidade** (**−0,15**) é em boa parte o espelho do álcool (etanol é menos denso que a água), e não um efeito
   independente.

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/MarcioBKN/wine-quality-classification.git
cd wine-quality-classification
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Dados
O `data/WineQT.csv` já está versionado no repositório — nenhuma ação manual é necessária.

### 4. Execute o notebook
```bash
cd notebooks
jupyter notebook wine_quality_classification.ipynb
```
Executar todas as células regenera os 9 gráficos e o `metricas_modelos.csv` em `results/`.

---

## ✅ Reprodutibilidade

As versões estão fixadas no `requirements.txt`, e todo número publicado neste README e na
apresentação pode ser reconferido a partir dos CSVs brutos:

```bash
python src/verify_numbers.py
```

O script **não importa nada do notebook** — reimplementa o pipeline do zero, retreina os dois
modelos e falha (`exit 1`) se qualquer valor divergir. Confere também se os números do README e
do slide de resultados batem com o `results/metricas_modelos.csv`.

A apresentação também é gerada por código (`python src/build_presentation.py`), lendo os dados de
`data/` e as métricas de `results/` — portanto os slides não podem divergir do modelo.

---

## 🎤 Apresentação Executiva

O storytelling da análise exploratória está em [`wine_quality_eda_storytelling.pptx`](wine_quality_eda_storytelling.pptx),
na raiz do repositório.

---

## 📦 Dependências

Ver `requirements.txt` para a lista completa.

---

## 👥 Equipe

Projeto desenvolvido para o Tech Challenge – Fase 2 do curso **Data Tech Analytics** da POSTECH.
