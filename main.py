import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/anime.csv"
MIN_GENRE_COUNT = 20


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.replace("Unknown", np.nan)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df[df["Score"].notna()].copy()

    # Alle Genres mit >= MIN_GENRE_COUNT Anime
    all_counts = df["Genres"].dropna().str.split(", ").explode().value_counts()
    valid_genres = all_counts[all_counts >= MIN_GENRE_COUNT].index.tolist()

    for genre in valid_genres:
        df[genre] = df["Genres"].str.contains(genre, na=False).astype(int)

    df["is_TV"] = (df["Type"] == "TV").astype(int)
    df["is_Manga"] = (df["Source"] == "Manga").astype(int)
    df["is_LightNovel"] = (df["Source"] == "Light novel").astype(int)

    cluster_features = valid_genres + ["is_TV", "is_Manga", "is_LightNovel"]
    X_scaled = StandardScaler().fit_transform(df[cluster_features])

    # Optimales k per Silhouette Score (k = 2–30)
    best_k, best_score = 2, -1
    for k in range(2, 51):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    df["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

    cluster_names = {}
    for c in range(best_k):
        top2 = df[df["Cluster"] == c][valid_genres].sum().nlargest(2).index.tolist()
        cluster_names[c] = " & ".join(top2)

    df["Cluster_Name"] = df["Cluster"].map(cluster_names)
    return df, cluster_features, cluster_names


def find_anime(df, name):
    return df[df["Name"].str.contains(name, case=False, na=False)][
        ["Name", "Score", "Genres", "Cluster_Name"]
    ]


def empfehlung(df, cluster_features, cluster_names, anime_name, n=10):
    treffer = find_anime(df, anime_name)

    if treffer.empty:
        print(f"Kein Anime mit '{anime_name}' gefunden.")
        return

    if len(treffer) > 1:
        print(f"{len(treffer)} Treffer für '{anime_name}':")
        print(treffer[["Name", "Score"]].to_string(index=False))
        print(f"\n→ Verwende ersten Treffer: '{treffer.iloc[0]['Name']}'")
        print("  (Präziserer Name für eindeutigen Treffer möglich)\n")

    anime = df.loc[treffer.index[0]]
    print(f"'{anime['Name']}' → Cluster: {cluster_names[anime['Cluster']]}")

    same_cluster = df[
        (df["Cluster"] == anime["Cluster"]) & (df["Name"] != anime["Name"])
    ].copy()

    same_cluster["Similarity"] = cosine_similarity(
        anime[cluster_features].values.reshape(1, -1),
        same_cluster[cluster_features].values,
    )[0].round(2)

    top = same_cluster.sort_values(
        ["Similarity", "Score"], ascending=[False, False]
    ).head(n)[["Name", "Score", "Similarity"]]
    print(f"\nÄhnlichste Anime (Top {n} — sortiert nach Ähnlichkeit, bei Gleichstand nach Score):")
    print(top.to_string(index=False))


def main():
    anime_name = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Anime-Name: ").strip()
    if not anime_name:
        print("Kein Name eingegeben.")
        return

    print("Lade Daten und optimiere Cluster-Anzahl per Silhouette Score (k = 2–30)…")
    df, cluster_features, cluster_names = load_and_prepare()
    empfehlung(df, cluster_features, cluster_names, anime_name)


if __name__ == "__main__":
    main()
