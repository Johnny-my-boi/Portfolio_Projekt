# Anime-Erfolg klassifizieren & Empfehlungssystem

## Projektbeschreibung

Dieses Projekt analysiert den [MyAnimeList Datensatz (2020)](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) mit über 17.500 Anime-Einträgen. Es verfolgt zwei Ziele:

1. **Klassifikation:** Welche Eigenschaften machen einen Anime "Top-Rated" (Score > 8.0)?
2. **Clustering:** Welche Anime-Gruppen gibt es, und welche ähnlichen Anime kann man basierend darauf empfehlen?

## Datensatz

- **Quelle:** Hernan4444 – Anime Recommendation Database 2020 (Kaggle)
- **Umfang:** 17.562 Anime, davon 12.421 mit Score
- **Features:** Genre, Studio, Type (TV/Movie/OVA), Source (Manga/Original/Light Novel), Episodenzahl, Mitgliederzahl u.a.

## Methodik

### Datenbereinigung
- "Unknown"-Werte durch NaN ersetzt
- Datentypen korrigiert (Score, Episodes, Ranked → numerisch)
- Genres aus kommaseparierten Strings in One-Hot-Encoding überführt (Top 14 Genres)

### Explorative Datenanalyse (EDA)
- Score-Verteilung analysiert: Durchschnitt bei 6.5, nur 4.3% der Anime über 8.0
- TV-Anime erzielen die höchsten Durchschnittscores nach Type
- Light Novels und Manga-Adaptionen scoren am höchsten nach Source
- Comedy, Action und Fantasy sind die häufigsten Genres

### Klassifikation
Zielvariable: Top-Rated (Score > 8.0) – binäre Klassifikation mit stark unbalancierten Klassen (4.3% positiv).

| Modell | Precision (Top-Rated) | Recall (Top-Rated) | F1-Score |
|---|---|---|---|
| Baseline (Logistic Regression, ohne SMOTE) | 0.79 | 0.26 | 0.39 |
| Logistic Regression + SMOTE | 0.28 | 0.66 | 0.39 |
| Random Forest + SMOTE | 0.51 | 0.52 | 0.52 |
| Random Forest + SMOTE (ohne Members) | 0.12 | 0.39 | 0.18 |

**Zentrale Erkenntnis:** Die Mitgliederzahl (Members) ist mit 55% der wichtigste Prädiktor, stellt aber eine zirkuläre Abhängigkeit dar – ein Anime hat viele Members *weil* er beliebt ist, nicht umgekehrt. Ohne Members ist die Vorhersagekraft der inhaltlichen Features (Genre, Type, Source) deutlich begrenzt.

### Clustering
KMeans-Clustering mit 5 Clustern auf Basis der Genre-, Type- und Source-Features:

| Cluster | Beschreibung | Anzahl | Avg Score |
|---|---|---|---|
| 0 | Mecha & Sci-Fi | 935 | 6.61 |
| 1 | Fantasy & Magic | 906 | 6.77 |
| 2 | Allgemein / Nische | 4.652 | 6.05 |
| 3 | Romantic Comedy / Alltag | 2.811 | 6.87 |
| 4 | Battle Shounen | 3.117 | 6.77 |

Auf Basis der Cluster wurde eine Empfehlungsfunktion implementiert, die zu einem gegebenen Anime ähnliche Titel aus demselben Cluster vorschlägt.

## Ergebnisse & Fazit

- **SMOTE** verbessert den Recall für die Minderheitsklasse erheblich (26% → 66% bei Logistic Regression)
- **Random Forest** liefert das beste Gesamtergebnis bei der Klassifikation
- **Anime-Erfolg** lässt sich aus inhaltlichen Merkmalen allein schwer vorhersagen – Popularität (Members) dominiert
- **Clustering** ergibt interpretierbare Gruppen, die für ein einfaches Empfehlungssystem genutzt werden können

## Technologien

- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn (Logistic Regression, Random Forest, KMeans)
- imbalanced-learn (SMOTE)

## Projektstruktur

```
├── data/                  # Datensätze (nicht in Git)
├── notebooks/
│   └── 01_eda.ipynb       # EDA, Klassifikation & Clustering
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
