# Anime-Analyse & Empfehlungssystem

Datenanalyse-Projekt auf dem [MyAnimeList Datensatz 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) mit zwei Fokusthemen, jeweils in einem eigenen Notebook:

## 📓 Notebooks

### [`notebooks/01_top_anime.ipynb`](notebooks/01_top_anime.ipynb) — Was macht einen Top-Anime aus?

Datengetriebene Analyse über 17.562 Anime-Einträge mit Score-Bewertungen.

| Teil | Inhalt |
|---|---|
| Regression | Vorhersage des Anime-Scores (1.85–9.19) aus inhaltlichen Features |
| Top-Analyse | Binärer Klassifikator: was unterscheidet Top-Animes (Score ≥ 8) vom Rest? |

**Ergebnisse:**
- **XGBoost** mit GridSearchCV: R² = 0.587 (K = 34, `n_estimators=300, max_depth=7, lr=0.05`)
- **Top-Klassifikator** (Logistic Regression balanced): ROC-AUC 0.905
- **Top-Prädiktoren:** `studio_avg_score` (+1.19), `duration_min` (+0.58), `is_TV` (+0.52), `src_Manga` (+0.41), `Drama` (+0.33)

### [`notebooks/02_empfehlung.ipynb`](notebooks/02_empfehlung.ipynb) — Item-based Empfehlungssystem

Collaborative Filtering auf 109M User-Anime-Interaktionen + 57.6M Bewertungen.

**5 Varianten verglichen** auf 500 sampled Usern (≥20 Bewertungen) mit Hold-Out-Evaluation:

| Variante | Hit@10 | Precision@10 | NDCG@10 | Diversity | Novelty |
|---|---|---|---|---|---|
| Rating-basiert | 89.8 % | 0.412 | 0.446 | 0.482 | 2.20 |
| Composite Engagement | 90.4 % | 0.408 | 0.442 | 0.490 | 2.12 |
| Hybrid (α=0.7, CF + Content) | 89.8 % | 0.351 | 0.382 | 0.345 | **2.38** |
| **MMR (λ=0.8)** | **92.4 %** | **0.411** | 0.443 | **0.608** | 2.01 |
| MMR (λ=0.4) | 88.8 % | 0.350 | 0.383 | 0.681 | 1.92 |

**Bestes Modell:** MMR-Re-Ranking bei λ=0.8 — Pareto-Verbesserung über pure CF (mehr Vielfalt OHNE Verlust bei Relevanz). Außerdem Multi-Seed-Empfehlungen über Centroid-Query.

## Datenquellen

- `anime.csv` (5.5 MB) — 17.562 Anime mit Metadaten (Genre, Studio, Score, Type, Source, Rating, Members, …)
- `rating_complete.csv` (781 MB) — 57.6M Bewertungen (nur completed+rated)
- `animelist.csv` (1.9 GB) — 109M User-Anime-Interaktionen (alle Status)
- `watching_status.csv` — Status-Code-Lookup

## Pipeline-Struktur

Alle Modelle verwenden durchgängig `sklearn.Pipeline`:
- **Regression/Klassifikation:** `SelectKBest → StandardScaler → Modell`
- **Empfehlung:** `Normalizer → NearestNeighbors`
- **MMR** sitzt **post-hoc** auf der CF-Pipeline (kein Re-Training)

## Technologien

- Python (pandas, numpy, matplotlib, seaborn, scipy)
- scikit-learn (Pipeline, SelectKBest, StandardScaler, Normalizer, NearestNeighbors, LinearRegression, RandomForestRegressor, LogisticRegression, GridSearchCV)
- xgboost (XGBRegressor, XGBClassifier)

## Projektstruktur

```
├── data/                       # nicht in Git
│   ├── anime.csv
│   ├── rating_complete.csv
│   ├── animelist.csv
│   └── watching_status.csv
├── notebooks/
│   ├── 01_top_anime.ipynb      # Regression + Top-Anime-Klassifikator
│   └── 02_empfehlung.ipynb     # Empfehlungssystem (5 Varianten + MMR + Multi-Seed)
├── presentation/               # 10-Min-Abschlusspräsentation
│   ├── summary.md
│   └── figures/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup

```bash
git clone https://github.com/Johnny-my-boi/Portfolio_Projekt.git
cd Portfolio_Projekt
uv sync
```

Daten aus dem Kaggle-Link in `data/` ablegen, dann die Notebooks in `notebooks/` ausführen.
