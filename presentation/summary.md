# StackFuel-Abschlusspräsentation — Zusammenfassung

Projekt: Datenanalyse + Empfehlungssystem auf dem **Anime Recommendation Database 2020** (hernan4444 auf Kaggle). Zwei Notebooks im Repo:
- `notebooks/02_empfehlung.ipynb` — **Hauptfokus** der Präsentation
- `notebooks/01_top_anime.ipynb` — Nebenanalyse "Was macht einen Top-Anime aus?", nur 1 Slide

---

## 1. Dataset-Übersicht

**Verwendete CSVs aus dem Kaggle-Datensatz:**
- `anime.csv` — 17.562 Anime mit 35 Metadaten-Spalten (5.5 MB)
- `rating_complete.csv` — 57.6M Bewertungen, nur Completed+Rated (781 MB)
- `animelist.csv` — 109.2M User-Anime-Interaktionen, alle Watching-Status (1.9 GB)
- `watching_status.csv` — Status-Code-Lookup (5 Zeilen: Currently Watching, Completed, On Hold, Dropped, Plan to Watch)

**Stand:** Datensatz-Snapshot Juli 2021, MyAnimeList-Daten

**Mengen vor / nach Cleaning:**
| Datei | Vorher | Nach Cleaning | Verlust |
|---|---|---|---|
| `anime.csv` | 17.562 Anime | 12.421 mit Score | 5.141 ohne Score (für Score-Regression unbrauchbar) |
| `rating_complete.csv` | 57.633.278 | unverändert | bereits clean (keine NaN, keine Duplikate) |
| `animelist.csv` | 109.224.747 | 109.208.207 | 16.540 Zeilen (0.015 %) |

**User:** 310.059 (rating_complete) / 325.770 (animelist)

**Features pro Anime** (`anime.csv`): MAL_ID, Name, Score (Durchschnitt), Genres (komma-sep), English/Japanese Name, Type (TV/Movie/OVA/…), Episodes, Aired (Sendedatum), Premiered (Saison+Jahr), Producers, Studios, Source (Manga/Original/Light Novel/…), Duration, Rating (Altersfreigabe), Ranked, Popularity, Members, Favorites, Watching, Completed, On-Hold, Dropped, Plan-to-Watch, Score-1…Score-10 (Score-Distribution pro Anime)

**Features pro User-Rating:**
- `rating_complete.csv`: `user_id`, `anime_id`, `rating` (1–10)
- `animelist.csv`: zusätzlich `watching_status` (1, 2, 3, 4, 6) und `watched_episodes`

**Cleaning-Schritte (animelist.csv):**
| Problem | Betroffen | Aktion |
|---|---|---|
| 1 Duplikat-Paar (user 61960 / anime 17549) | 1 | entfernt |
| Invalide `watching_status` (0, 5, 33, 55) | 540 | entfernt |
| `watched_episodes > 2× Episodes` (int16-Overflow / Datenmüll) | 16.000 | entfernt |
| `watched_episodes > Episodes` (≤ 2×, outdated `Episodes`-Feld) | 181.000 | auf `Episodes` gecapped |

**Cleaning beider Notebooks identisch** für `anime.csv`: "Unknown" → NaN, numerische Konvertierung für Score / Episodes / Ranked / Score-1…10.

**Bewusst aus Features ausgeschlossen:**
- *Members*, *Episodes* — zirkulär abhängig vom Score, den sie vorhersagen sollen
- *Members* könnte das stärkste Feature sein (|r(log Members, Score)| = 0.70), aber tautologisch ("Top-Anime = beliebter Anime")

---

## 2. EDA — drei erzählenswerte Findings

### Finding 1: User-Bewertungen sind **stark rechtsschief**
- Modal-Rating ist **8** (~25 % aller Bewertungen)
- Rating ≥ 7 macht **~75 %** aus
- Rating ≤ 4 nur ~7 %
- → User vergeben nur dann eine Bewertung, wenn ihnen der Anime gefallen hat (Selection Bias)
- → *Plot: `rating_distribution.png`*

### Finding 2: Extreme Long-Tail bei Member-Verteilung pro Anime
- Median-Anime hat ~2.000 Members
- Top-1 % der Anime haben > 200.000 Members
- Die **Top-100 Anime** sammeln mehr Member-Einträge als die *unteren 17.000 zusammen*
- → Empfehlungs-Modelle haben für 95 % der Anime nur **wenige bis keine** kollaborativen Signale
- → *Plot: `anime_popularity.png`*

### Finding 3: "Member sein" bedeutet sehr unterschiedliche Dinge
- 62 % der animelist-Einträge sind **Completed** (echte Watcher, davon 85 % bewerten — Ø Rating 7.51)
- 26 % sind **Plan-to-Watch** (Lesezeichen, nur 1 % bewertet)
- 4 % **Dropped** (46 % bewerten — Ø Rating **4.81** = klares Anti-Signal)
- → Motiviert das *Composite-Engagement-Schema* im Empfehlungs-Notebook
- → *Plot: `member_status.png`*

**Genre-Verteilung:** Comedy (6029), Action (3888), Fantasy (3285), Adventure (2957), Kids (2665) sind die häufigsten — aber unterscheiden sich kaum im Median-Score. *Plot: `genre_distribution.png`*

**Sparsity der User-Anime-Matrix:**
- `rating_complete`: 16.872 Animes × 310.059 User = 5.2 Mrd Zellen, davon nur 57.6M besetzt → **Density 1.10 %**
- `animelist`: 17.562 × 325.754 = 5.7 Mrd Zellen, 109.2M besetzt → **Density 1.91 %**
- → Sparse-Matrix-Implementierung Pflicht (scipy `csr_matrix`)

---

## 3. Top-Anime-Analyse (kurz)

*Nur 1 Slide. Sieht Hauptlimitation der reinen Regression.*

**Definition Top-Anime:** Score ≥ 8 (community-aggregierter Durchschnittsscore). Macht 4.4 % der gerateten Anime aus (548 / 12.421).

**Ansatz:**
- *Versuch 1 — Regression:* XGBoost (R² gesamt = 0.587) versagt aber auf Top-Subset (R² = **−9.3**, RMSE 0.85 statt 0.57). Klassisches "regression to the mean", systematischer Bias −0.71.
- *Versuch 2 — Binärer Klassifikator:* Logistic Regression mit `class_weight='balanced'` auf 66 Features → **ROC-AUC = 0.905**

**Top-Ergebnisse** (LogReg-Koeffizienten, standardisiert):

| Pro Top-Status | Koeffizient | | Gegen Top-Status | Koeffizient |
|---|---|---|---|---|
| `studio_avg_score` | +1.19 | | `Hentai` | −0.78 |
| `duration_min` | +0.58 | | `rating_Rx (Hentai)` | −0.77 |
| `is_TV` | +0.52 | | `rating_PG (Children)` | −0.33 |
| `src_Manga` | +0.41 | | `src_Game` | −0.28 |
| `Drama` | +0.33 | | `rating_G (All Ages)` | −0.27 |

→ *Plot: `top_anime_features.png`*

**Schwächen / ehrliche Bewertung:**
- Recall bei Precision-Trade-off problematisch: LogReg trifft 85 % der echten Top-Animes, aber nur 16 % der "Top"-Vorhersagen sind tatsächlich Top (Imbalance ~4 %)
- "Score 8.0+" ist eine willkürliche Schwelle — Wert wäre vergleichbar mit z.B. Top-10-%-Quantil
- *Studio-Score* schließt den Anime selbst ein (Mini-Leakage) — Effekt klein, aber methodisch nicht 100 % sauber

---

## 4. Recommendation Model — Hauptteil

### Ansatz: Item-based Collaborative Filtering + Erweiterungen

Eine Pipeline-Architektur, fünf Varianten:

| # | Variante | Idee |
|---|---|---|
| 1 | **Rating-basiert** | Sparse-Matrix aus `rating_complete.csv` (Anime × User → Rating 1-10) |
| 2 | **Composite Engagement** | Sparse-Matrix aus `animelist.csv`, mit Status→Engagement-Mapping: Watching=6, Completed=5, On-Hold=4, Plan-to-Watch=3, Dropped=2; falls Rating > 0 überschreibt das Rating den Default |
| 3 | **Hybrid (CF + Content)** | Engagement-Matrix mit angehängten Content-Features (Genres, Studio, Duration, Type, Source, Rating) als "virtuelle User"-Spalten; Mixing α=0.7 |
| 4 | **MMR Re-Ranking** | Post-hoc auf Composite Engagement: holt Top-50 Kandidaten, wählt sequentiell 10 unter Trade-off zwischen Relevanz (CF) und Diversität (Content-Space) |
| 5 | **Multi-Seed** | Centroid-Query: Vektoren mehrerer Seeds summiert → eine `kneighbors`-Anfrage; optional mit MMR + Expand-Horizon (Seeds in Diversity-Strafe) |

**Pipeline:**
```python
sklearn.Pipeline([
    ('normalize', Normalizer(norm='l2')),
    ('knn',       NearestNeighbors(metric='euclidean', algorithm='brute')),
])
```
Mathematisch äquivalent zu Cosine-Similarity, aber Pipeline-konform.

### Train/Test-Split + Evaluierung

**Hold-Out per User** (Leave-One-Out):
- 500 zufällig sampled User mit ≥ 20 Bewertungen aus `rating_complete`
- Pro User: höchstbewerteter Anime = **Seed**, andere mit rating ≥ 8 = **Truth**
- Top-10-Empfehlungen vom Recommender für Seed → wie viele Truth-Animes sind drin?

**Metriken:**
- **Hit@10** — mindestens 1 Truth in Top-10? (binär)
- **Precision@10** — Anteil Treffer in Top-10
- **NDCG@10** — rang-bewusst (Treffer auf Platz 1 zählt mehr als Platz 10)
- **Intra-List Diversity (ILD)** — Cosine-Distanz zwischen Empfehlungen im Content-Space
- **Novelty** — −log(Popularität) der empfohlenen Anime

### Vergleichstabelle

| Variante | Hit@10 | Precision@10 | NDCG@10 | Diversity | Novelty |
|---|---|---|---|---|---|
| Rating-basiert | 89.8 % | 0.412 | 0.446 | 0.482 | 2.20 |
| Composite Engagement | 90.4 % | 0.408 | 0.442 | 0.490 | 2.12 |
| Hybrid (α=0.7) | 89.8 % | 0.351 | 0.382 | 0.345 | **2.38** |
| **MMR (λ=0.8)** | **92.4 %** | **0.411** | **0.443** | **0.608** | 2.01 |
| MMR (λ=0.4) | 88.8 % | 0.350 | 0.383 | 0.681 | 1.92 |

→ *Plots: `model_comparison.png` und `mmr_tradeoff.png`*

### Gewähltes Modell: **MMR (λ=0.8)** — und warum

**Pareto-Verbesserung über pure CF:** Hit@10 steigt sogar (+2 pp), Precision/NDCG bleiben gleich, **Diversity +24 %**.
- Pure CF lieferte zu ähnliche Top-10 (Filter Bubble)
- Schon kleiner Diversity-Druck zieht relevante aber unterschiedlichere Items hoch
- Bei λ=0.4 würde man noch mehr Diversity gewinnen, aber Precision spürbar verlieren

### Konkrete Beispiel-Empfehlungen

**Query: Vinland Saga**
- Rating-basiert: Dr. Stone, Demon Slayer, Attack on Titan S3 P2, Promised Neverland, Attack on Titan S3, Dororo, Mob Psycho 100 II, MHA S3, OPM S2, Tower of God
- MMR (λ=0.3): + Re:ZERO S2, Your Name, Kaguya-sama, KonoSuba, Dorohedoro (deutlich diverser — Slice-of-Life, Romcom, Dark Fantasy)

**Query: K-On!**
- Rating-basiert: K-On! Season 2, K-On! The Movie, K-On!: Live House! (Franchise-Cluster)
- Composite Engagement: K-On! Season 2, Suzumiya Haruhi, Toradora!, Lucky Star, K-On! Movie (breiterer Slice-of-Life-Cluster)

**Multi-Seed: Cowboy Bebop + Samurai Champloo + Trigun (Watanabe-Vibes)**
- Centroid: Gurren Lagann, NGE, FLCL, Black Lagoon, Baccano!, FMA: Brotherhood, Death Note, Code Geass, FMA, Darker than Black
- MMR + Expand-Horizon: + Howl's Moving Castle, Mushi-Shi, Princess Mononoke, Spirited Away (Ghibli + Slice-of-Life-Thriller-Mix)

→ *Plot: `recommendation_example.png`*

---

## 5. Limitationen & Learnings

### Cold-Start-Problem
- **Neue Anime** (kein User-Rating): pure CF kann nichts vorhersagen → Hybrid-Variante liefert wenigstens content-basierte Empfehlungen (Genres, Studio etc.). Nicht ideal, aber funktionsfähig.
- **Neue User** (keine Historie): aktuell nicht adressiert. Standard-Lösung wäre eine "Welche 5 Anime hast du gemocht?"-Onboarding + Centroid-Query (analog Multi-Seed).

### Skalierbarkeit
- Sparse-Matrix passt noch in RAM (~1 GB für `rating_complete`, ~870 MB für `animelist`-Engagement)
- NearestNeighbors mit `brute force` Cosine ist OK für 17.5k Anime — bei deutlich mehr Items wäre Approximate NN (FAISS, Annoy, HNSW) nötig
- MMR-Re-Ranking O(K × Pool) — bei K=10, Pool=50 sehr schnell

### Was beim nächsten Mal anders
- **SVD/ALS Matrix Factorization** statt purem k-NN-CF — kompaktere Item-Embeddings, oft besseres NDCG. `surprise`-Lib oder `implicit`-Lib hätte sich angeboten
- **Engagement-Werte tunen** — die Defaults (2/3/4/5/6) sind heuristisch motiviert, aber nicht durch Eval optimiert
- **Popularitäts-Strafe** für mehr **Novelty** — MMR holt Diversität aber bleibt im Mainstream
- **Per-User-Personalisierung** — wir empfehlen aktuell nur "ähnlich zu Anime X", nicht "für User Y" (kein User-Profil-Vektor)
- **Cross-Validation** im K-Sweep wäre seriöser als Single-Split (aktuelles Setup: K=40 best bei Single-Split, könnte bei CV andere optimale Werte zeigen)
