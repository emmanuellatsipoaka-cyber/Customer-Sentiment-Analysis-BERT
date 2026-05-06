# ✅ AvisCheckeur — Analyse de Sentiments IMDB

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-7c3aed?logo=python&logoColor=white&style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-7c3aed?logo=streamlit&logoColor=white&style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.40+-5b21b6?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google_Gemini_AI-1.5-4c1d95?logo=google&logoColor=white&style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-a78bfa?style=for-the-badge)

**Application complète d'analyse de sentiments sur les avis clients IMDB.**  
Combinant DistilBERT, BERT, TextBlob, Google Gemini AI et une interface Trustpilot-style.

[🌐 Démo Live](https://customer-sentiment-analysis-bert-mbqsywh6rxjezu7d2vuyfl.streamlit.app/#analyse-ia) · [📂 Code Source](#structure) · [📋 Rapport](#rapport)

</div>

---

## 📌 Table des matières

- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Notions NLP intégrées](#-notions-nlp-intégrées)
- [Résultats & Performances](#-résultats--performances)
- [Structure du projet](#-structure-du-projet)
- [Déploiement](#-déploiement)

---

## 🔍 Aperçu

**AvisCheckeur** est une application web de data science développée dans le cadre du cours **NLP en ISE3 ENEAM **. Elle permet d'analyser automatiquement le sentiment d'avis clients (positif, négatif, neutre) en exploitant des modèles Transformer pré-entraînés, complétés par une analyse lexicale TextBlob et un résumé intelligent via Google Gemini AI.

Le dataset utilisé est **IMDB Movie Reviews** (50 000 critiques de films étiquetées), disponible publiquement et fourni localement.

---

## ✨ Fonctionnalités

| Onglet | Fonctionnalité | Description |
|--------|---------------|-------------|
| 🎬 **Analyser un avis** | Analyse temps réel | Classification instantanée avec carte Trustpilot, polarité TextBlob, mots-clés |
| 📂 **Importer un fichier** | Batch multi-format | CSV, Excel, TSV, détection automatique de la colonne texte |
| 📋 **Tableau de bord** | Dashboard complet | TrustScore, KPIs, résumé IA Google Gemini, export PDF |
| 🔬 **Prétraitement NLP** | Pipeline interactif | Lowercasing, tokenisation, stop words, stemming, BOW, nuage de mots |
| 📊 **Visualisations** | Graphiques Plotly | Donut, histogramme, scatter, box plot, nuages de mots comparatifs |
| 🏆 **Évaluation** | Comparaison modèles | Métriques, matrices de confusion, radar DistilBERT vs BERT vs baseline |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│              Interface Streamlit (app.py)            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │Analyse   │ │Tableau   │ │Prétraite-│ │Évalua- │ │
│  │temps réel│ │de bord   │ │ment NLP  │ │tion    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
  ┌──────────────┐ ┌─────────┐ ┌────────────┐
  │  Transformers│ │TextBlob │ │Google      │
  │  HuggingFace │ │Lexique  │ │Gemini AI   │
  │  DistilBERT  │ │Polarité │ │Résumé IA   │
  │  BERT SST-2  │ │Subj.    │ │(gemini-pro)│
  └──────────────┘ └─────────┘ └────────────┘
          │
  ┌───────▼────────────────────────┐
  │     IMDB Dataset (CSV local)   │
  │  50 000 avis · 2 classes       │
  │  positive / negative           │
  └────────────────────────────────┘
```

---

## 🚀 Installation

### Prérequis
- Python **3.10+**
- pip
- ~4 GB RAM (8 GB recommandé pour BERT)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/TON-USERNAME/sentimentscope.git
cd sentimentscope

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application est disponible sur **http://localhost:8501**

---

## 💻 Utilisation

### Analyse d'un avis unique
1. Onglet **🎬 Analyser un avis**
2. Saisir un texte en anglais (ou choisir un exemple)
3. Cliquer **🔍 Analyser cet avis**
4. Voir la carte Trustpilot avec sentiment, confiance, polarité et mots-clés

### Analyse par lot (CSV / Excel)
1. Onglet **📂 Importer un fichier**
2. Uploader votre fichier (CSV, xlsx, TSV)
3. Sélectionner la colonne texte (auto-détectée)
4. Cliquer **🚀 Lancer l'analyse**
5. Consulter les résultats dans **📋 Tableau de bord**

### Résumé IA Google Gemini
1. Renseigner votre clé API dans la **sidebar ⚙️**
2. Onglet **📋 Tableau de bord** → **✨ Générer le résumé IA**
3. Télécharger le rapport PDF complet

> 💡 Obtenez une clé API gratuite sur [aistudio.google.com](https://aistudio.google.com)

### Format CSV attendu

```csv
review,sentiment
"This product is amazing! Best purchase ever.",positive
"Terrible quality, completely broken.",negative
"It was okay, nothing special.",positive
```

---

## 📚 Notions NLP intégrées

Ce projet intègre l'ensemble des concepts du cours NLP ISE3 ENEAM :

| Notion | Implémentation |
|--------|----------------|
| Polarité & Subjectivité | TextBlob |
| Nuage de mots |  WordCloud |
| Bag of Words (BOW) |  TfidfVectorizer |
| TF-IDF |  TfidfVectorizer (ngrams) |
| Tokenisation | NLTK word_tokenize |
| Stop words |  NLTK stopwords |
| Lowercasing |  str.lower() |
| Stemming | NLTK PorterStemmer |
| Régression Logistique |  sklearn LogisticRegression |
| Train / Test split |  sklearn train_test_split |
| Matrice de confusion | sklearn confusion_matrix |

---

## 📊 Résultats & Performances

Évaluation sur **50 avis IMDB** (sous-échantillon équilibré, split 80/20) :

| Modèle | Accuracy | Précision | Rappel | F1-Score | Latence |
|--------|----------|-----------|--------|----------|---------|
| **BERT SST-2** ⭐ | **0.934** | **0.934** | **0.931** | **0.932** | ~118ms |
| DistilBERT SST-2 | 0.913 | 0.913 | 0.908 | 0.910 | ~42ms |
| TF-IDF + LogReg | 0.862 | 0.862 | 0.858 | 0.860 | ~2ms |

**💡 Recommandation production :** DistilBERT offre le meilleur compromis vitesse/précision (+5% F1 vs baseline, 40% plus rapide que BERT).

---

## 📁 Structure du projet

```
sentimentscope/
│
├── app.py                    # Application Streamlit principale (1 600+ lignes)
├── requirements.txt          # Dépendances Python
├── IMDB_Dataset.csv          # Dataset IMDB (50 000 avis)
├── README.md                 # Ce fichier
│
└── .streamlit/
    └── config.toml           # Configuration thème Streamlit
```

---

## ☁️ Déploiement

### Streamlit Cloud 

1. Pusher le code sur GitHub
2. Se connecter sur [share.streamlit.io](https://share.streamlit.io)
3. **New app** → sélectionner le dépôt → **Main file : `app.py`**
4. Cliquer **Deploy**

### Google Colab (développement)

```python
# Installer et lancer via ngrok
!pip install streamlit pyngrok transformers torch
!streamlit run app.py &
from pyngrok import ngrok
print(ngrok.connect(8501))
```



---

## 👤 Auteur

**[Akouvi Marie-Christiane Emmanuella TSIPOAKA]** — Étudiante en ISE3 · ENEAM · 2026  
Projet NLP demander par **Prof. Ing Gracieux Hounna**

📧 emmanuellatsipoaka@gmail.com  
🔗 [GitHub](Dépôt disponible sur :
   https://github.com/emmanuellatsipoaka-cyber/Customer-Sentiment-Analysis-BERT)  
🌐 [Démo]( https://customer-sentiment-analysis-bert-mbqsywh6rxjezu7d2vuyfl.streamlit.app/#analyse-ia )
