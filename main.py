import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/anime.csv"

TOP_GENRES = [
    "Comedy", "Action", "Fantasy", "Adventure", "Sci-Fi",
    "Drama", "Shounen", "Romance", "School", "Slice of Life",
    "Supernatural", "Music", "Mecha", "Magic",
]

CLUSTER_NAMES = {
    0: "Mecha & Sci-Fi",
    1: "Fantasy & Magic",
    2: "Allgemein / Nische",
    3: "Romantic Comedy / Alltag",
    4: "Battle Shounen",
}


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.replace("Unknown", np.nan)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df[df["Score"].notna()].copy()

    for genre in TOP_GENRES:
        df[genre] = df["Genres"].str.contains(genre, na=False).astype(int)

    df["is_TV"] = (df["Type"] == "TV").astype(int)
    df["is_Manga"] = (df["Source"] == "Manga").astype(int)
    df["is_LightNovel"] = (df["Source"] == "Light novel").astype(int)

    cluster_features = TOP_GENRES + ["is_TV", "is_Manga", "is_LightNovel"]
    X_scaled = StandardScaler().fit_transform(df[cluster_features])

    df["Cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_scaled)
    df["Cluster_Name"] = df["Cluster"].map(CLUSTER_NAMES)

    return df


def find_anime(df, name):
    return df[df["Name"].str.contains(name, case=False, na=False)][
        ["Name", "Score", "Genres", "Cluster_Name"]
    ]


def empfehlung(df, anime_name, n=10):
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
    print(f"'{anime['Name']}' → Cluster: {CLUSTER_NAMES[anime['Cluster']]}")

    similar = df[(df["Cluster"] == anime["Cluster"]) & (df["Name"] != anime["Name"])]
    top = similar.nlargest(n, "Score")[["Name", "Score", "Cluster_Name"]]
    print(f"\nÄhnliche Anime (Top {n} nach Score):")
    print(top.to_string(index=False))


def main():
    anime_name = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Anime-Name: ").strip()
    if not anime_name:
        print("Kein Name eingegeben.")
        return

    print("Lade Daten und trainiere Clustering…")
    df = load_and_prepare()
    empfehlung(df, anime_name)


if __name__ == "__main__":
    main()