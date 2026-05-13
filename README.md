# Anime-Erfolg klassifizieren & Empfehlungssystem

## Projektbeschreibung

Dieses Projekt analysiert den [MyAnimeList Datensatz (2020)](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) mit über 17.500 Anime-Einträgen. Es verfolgt zwei Ziele:

1. **Klassifikation:** Welche Eigenschaften machen einen Anime "Top-Rated" (Score > 8.0)?
2. **Clustering:** Welche Anime-Gruppen gibt es, und welche ähnlichen Anime kann man basierend darauf empfehlen?

## Datensatz

- **Quelle:** Hernan4444 – Anime Recommendation Database 2020 (Kaggle)
- **Umfang:** 17.562 Anime, davon 12.421 mit Score
- **Features:** Genre, Studio, Type (TV/Movie/OVA), Source (Manga/Original/Light Novel), Rating, Episodenzahl u.a.

## Methodik

### Datenbereinigung
- "Unknown"-Werte durch NaN ersetzt
- Datentypen korrigiert (Score, Episodes, Ranked → numerisch)

### Explorative Datenanalyse (EDA)
- Score-Verteilung analysiert: Durchschnitt bei 6.5, nur 4.3% der Anime über 8.0
- TV-Anime erzielen die höchsten Durchschnittscores nach Type
- Light Novels und Manga-Adaptionen scoren am höchsten nach Source
- Comedy, Action und Fantasy sind die häufigsten Genres
- Korrelationsanalyse, Genre-Genre-Heatmap und PCA zur Feature-Exploration

### Feature Engineering

**Für die Klassifikation (`feature_cols`):**
- Genres (One-Hot, alle 43 validen Genres mit ≥ 20 Anime)
- Source (One-Hot, 13 Typen)
- Rating (One-Hot: G, PG, PG-13, R, R+, Rx)
- `is_TV` (Type-Flag)
- `studio_avg_score` (Durchschnittsscore des besten Studios eines Anime)
- Korrelations-Filter: nur Features mit |Pearson r| ≥ 0.05 mit `Top_Rated`

**Bewusst ausgeschlossen:**
- *Members* — zirkulär: viele Members, *weil* beliebt, nicht umgekehrt
- *Episodes* — zirkulär: eine Serie läuft weiter, *weil* sie erfolgreich ist

**Für Clustering & Cosine Similarity (`cluster_features`):**
- Alle validen Genres + `is_TV`, `is_Manga`, `is_LightNovel` + Rating (One-Hot)

### Klassifikation
Zielvariable: Top-Rated (Score > 8.0) — binäre Klassifikation mit stark unbalancierten Klassen (4,3% positiv).

| Modell | Precision (Top-Rated) | Recall (Top-Rated) | F1-Score |
|---|---|---|---|
| Baseline (Logistic Regression, ohne SMOTE) | 0.50 | 0.07 | 0.13 |
| Logistic Regression + SMOTE | 0.16 | 0.66 | 0.25 |
| Random Forest + SMOTE | 0.43 | 0.44 | 0.43 |
| Random Forest + SMOTE (Tuned) | 0.42 | 0.42 | 0.42 |
| **XGBoost** (scale_pos_weight = 22.3) | **0.35** | **0.58** | **0.44** |

**Zentrale Erkenntnisse:**
- XGBoost mit `scale_pos_weight` ist das beste Modell für dieses unbalancierte Problem (F1 = 0.44, Recall = 58%)
- SMOTE verbessert den Recall drastisch, senkt aber die Precision stark
- Hyperparameter-Tuning (GridSearchCV) bringt bei Random Forest keinen nennenswerten Gewinn
- Ohne Members gibt es kein dominantes Feature — `studio_avg_score`, Source und einzelne Genres sind die wichtigsten Prädiktoren

### Clustering
KMeans-Clustering auf Basis von Genres, Type, Source-Flags und Rating. Die optimale Cluster-Anzahl wird automatisch per **Silhouette Score** (k = 2–50) bestimmt. Cluster-Namen werden aus den zwei dominantesten Genres jedes Clusters abgeleitet.

Auf Basis der Cluster wurde eine Empfehlungsfunktion implementiert, die zu einem gegebenen Anime ähnliche Titel per **Cosine Similarity** vorschlägt — bei Gleichstand nach Score sortiert.

## CLI-Empfehlungssystem

```bash
uv run main.py "Fullmetal Alchemist"
# oder interaktiv:
uv run main.py
```

## Technologien

- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn (Logistic Regression, Random Forest, KMeans, PCA, SMOTE)
- imbalanced-learn (SMOTE)
- xgboost

## Projektstruktur

```
├── data/                  # Datensätze (nicht in Git)
├── notebooks/
│   └── 01_eda.ipynb       # EDA, Klassifikation & Clustering
├── main.py                # CLI-Empfehlungssystem
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
