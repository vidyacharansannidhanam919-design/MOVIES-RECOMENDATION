import pandas as pd
import numpy as np
import re

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── 2.1  Load raw files ───────────────────────────────────────
print("=" * 60)
print("  PHASE 2 — Data Loading & Cleaning")
print("=" * 60)

print("\n[1] Loading CSVs …")

movies = pd.read_csv(DATA_DIR + "movie.csv")

ratings = pd.read_csv(
    DATA_DIR + "rating.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
)

tags = pd.read_csv(DATA_DIR + "tag.csv")

g_scores = pd.read_csv(
    DATA_DIR + "genome_scores.csv",
    dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"},
)

g_tags = pd.read_csv(
    DATA_DIR + "genome_tags.csv",
    dtype={"tagId": "int16", "tag": "string"},
)

links = pd.read_csv(DATA_DIR + "link.csv")

print("  ✅ All files loaded.")

# ── 2.2  Null & duplicate audit (BEFORE cleaning) ────────────
print("\n[2] Null & Duplicate Audit (before cleaning)")
print("-" * 60)
print(f"  {'DataFrame':<15} {'Rows':>10} {'Nulls':>10} {'Duplicates':>12}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")

for name, df in [("movies", movies), ("ratings", ratings), ("tags", tags),
                  ("g_scores", g_scores), ("g_tags", g_tags), ("links", links)]:
    nulls = df.isnull().sum().sum()
    dups  = df.duplicated().sum()
    print(f"  {name:<15} {len(df):>10,} {nulls:>10,} {dups:>12,}")

# ── 2.3  Feature engineering — movies ────────────────────────
print("\n[3] Feature Engineering — movies")
print("-" * 60)

def extract_year(title):
    m = re.search(r'\((\d{4})\)', str(title))
    return int(m.group(1)) if m else np.nan

movies["release_year"] = movies["title"].apply(extract_year)
missing_year = movies["release_year"].isna().sum()
print(f"  release_year extracted.  Missing years: {missing_year:,}")

# Drop rows with no year
movies_clean = movies.dropna(subset=["release_year"]).copy()
movies_clean["release_year"] = movies_clean["release_year"].astype(int)
print(f"  Rows kept  : {len(movies_clean):,}  |  Dropped: {len(movies)-len(movies_clean):,}")

# ── 2.4  Feature engineering — ratings ───────────────────────
print("\n[4] Feature Engineering — ratings")
print("-" * 60)

# The timestamp format in this dataset is  dd-mm-yyyy HH:MM
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"], dayfirst=True, errors="coerce"
)
ts_nulls = ratings["timestamp"].isna().sum()
print(f"  Timestamp parsed.  Unparseable rows: {ts_nulls:,}")

ratings["rating_year"]  = ratings["timestamp"].dt.year
ratings["rating_month"] = ratings["timestamp"].dt.month

year_range = f"{int(ratings['rating_year'].min())} – {int(ratings['rating_year'].max())}"
print(f"  Rating years span : {year_range}")
print(f"  Rating value range: {ratings['rating'].min()} – {ratings['rating'].max()}")

# ── 2.5  Feature engineering — tags ──────────────────────────
print("\n[5] Feature Engineering — tags")
print("-" * 60)

tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
tags.drop_duplicates(inplace=True)
print(f"  Tags normalised (lower-case, stripped, deduped).")
print(f"  Unique tag strings : {tags['tag'].nunique():,}")

# ── 2.6  Post-cleaning summary ───────────────────────────────
print("\n[6] Post-Cleaning Summary")
print("-" * 60)
print(f"  {'DataFrame':<15} {'Rows':>10}  {'Columns':>8}")
print(f"  {'-'*15} {'-'*10}  {'-'*8}")
for name, df in [("movies_clean", movies_clean), ("ratings", ratings),
                  ("tags", tags), ("g_scores", g_scores),
                  ("g_tags", g_tags), ("links", links)]:
    print(f"  {name:<15} {len(df):>10,}  {df.shape[1]:>8}")

# ── 2.7  Export cleaned data for downstream phases ───────────
print("\n[7] Saving cleaned DataFrames …")
movies_clean.to_csv(DATA_DIR + "movies_clean.csv", index=False)
ratings.to_csv(DATA_DIR + "ratings_clean.csv", index=False)
tags.to_csv(DATA_DIR + "tags_clean.csv", index=False)
print("  Saved: movies_clean.csv, ratings_clean.csv, tags_clean.csv")

print("\n" + "=" * 60)
print("  Phase 2 complete — data cleaned and saved.")
print("=" * 60)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"

# ── Load data ─────────────────────────────────────────────────
print("Loading ratings …")
ratings = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    parse_dates=["timestamp"],
)
print(f"  {len(ratings):,} ratings loaded.\n")

# ── 3.1  Descriptive statistics ───────────────────────────────
print("=" * 60)
print("  PHASE 3 — Ratings Distribution")
print("=" * 60)

print("\n[1] Descriptive Statistics")
print("-" * 40)
stats = ratings["rating"].describe()
for stat, val in stats.items():
    print(f"  {stat:<10}: {val:.4f}")

print(f"\n  Most common rating : {ratings['rating'].mode()[0]}")
print(f"  Skewness           : {ratings['rating'].skew():.4f}")
print(f"  Kurtosis           : {ratings['rating'].kurtosis():.4f}")

# ── 3.2  Rating value counts ──────────────────────────────────
print("\n[2] Rating Value Counts")
print("-" * 40)
vc = ratings["rating"].value_counts().sort_index()
for rating_val, count in vc.items():
    bar = "█" * int(count / vc.max() * 30)
    print(f"  {rating_val:.1f}  {bar:<30}  {count:>8,}")

# ── 3.3  Plot 1 — Bar chart of rating counts ─────────────────
print("\n[3] Generating Plots …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rating Distribution", fontsize=16, color="white",
             fontweight="bold", y=1.01)

ax = axes[0]
bars = ax.bar(
    vc.index.astype(str), vc.values,
    color=ACCENT, edgecolor="#0f1116", width=0.7
)
ax.set_title("Count per Rating Value", color="white", fontsize=13)
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Ratings")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K")
)
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + vc.max() * 0.01,
        f"{bar.get_height():,.0f}",
        ha="center", va="bottom", fontsize=7.5, color="white",
    )

# ── 3.4  Plot 2 — Density histogram ──────────────────────────
ax2 = axes[1]
ax2.hist(ratings["rating"], bins=18, color=ACCENT2,
         edgecolor="#0f1116", alpha=0.85, density=True)
ax2.set_title("Rating Density (KDE-style)", color="white", fontsize=13)
ax2.set_xlabel("Rating")
ax2.set_ylabel("Density")
ax2.axvline(ratings["rating"].mean(), color=ACCENT3, lw=2, linestyle="--",
            label=f"Mean = {ratings['rating'].mean():.2f}")
ax2.legend()

plt.tight_layout()
plt.savefig("ratings_distribution.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: ratings_distribution.png")

# ── 3.5  Plot 3 — Ratings per user vs per movie ──────────────
per_user  = ratings.groupby("userId")["rating"].count()
per_movie = ratings.groupby("movieId")["rating"].count()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Engagement Distributions", fontsize=16, color="white",
             fontweight="bold", y=1.01)

for ax, series, title, color in zip(
        axes,
        [per_user, per_movie],
        ["Ratings per User", "Ratings per Movie"],
        [ACCENT, ACCENT3],
):
    ax.hist(np.log1p(series), bins=50, color=color,
            edgecolor="#0f1116", alpha=0.85)
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel("log(1 + count)")
    ax.set_ylabel("Frequency")
    ax.axvline(
        np.log1p(series.median()), color="white", lw=1.5, linestyle="--",
        label=f"Median = {series.median():.0f}"
    )
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("engagement_distributions.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: engagement_distributions.png")

# ── 3.6  Power-user insight ───────────────────────────────────
print("\n[4] User Engagement Insights")
print("-" * 40)
print(f"  Total users          : {per_user.index.nunique():,}")
print(f"  Median ratings/user  : {per_user.median():.0f}")
print(f"  Mean   ratings/user  : {per_user.mean():.0f}")
print(f"  Max    ratings/user  : {per_user.max():,}")
top1pct = int(len(per_user) * 0.01)
print(f"  Top 1% users account for {per_user.nlargest(top1pct).sum() / len(ratings) * 100:.1f}% of all ratings")

print("\n" + "=" * 60)
print("  Phase 3 complete.")
print("=" * 60)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"

# ── Load data ─────────────────────────────────────────────────
print("Loading data …")
movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
)
print(f"  Movies : {len(movies):,}  |  Ratings : {len(ratings):,}\n")

# ── Helper — explode genres ───────────────────────────────────
def explode_genres(df):
    return (
        df.assign(genre=df["genres"].str.split("|"))
          .explode("genre")
          .query("genre != '(no genres listed)'")
    )

genre_movies = explode_genres(movies)

print("=" * 60)
print("  PHASE 4 — Genre Analysis")
print("=" * 60)

# ── 4.1  Genre frequency ─────────────────────────────────────
print("\n[1] Genre Frequency (movie count per genre)")
print("-" * 50)
genre_counts = genre_movies["genre"].value_counts()
for g, c in genre_counts.items():
    bar = "█" * int(c / genre_counts.max() * 28)
    print(f"  {g:<20} {bar:<28}  {c:>5,}")

# ── Plot 1 — Horizontal bar ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors = [ACCENT if i < 5 else "#3a4a6b" for i in range(len(genre_counts))]
bars = ax.barh(
    genre_counts.index[::-1], genre_counts.values[::-1],
    color=colors[::-1], edgecolor="#0f1116"
)
ax.set_title("Number of Movies per Genre", fontsize=14,
             color="white", fontweight="bold")
ax.set_xlabel("Movie Count")
for bar in bars:
    ax.text(
        bar.get_width() + 30,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():,.0f}",
        va="center", fontsize=8, color="white",
    )
plt.tight_layout()
plt.savefig("movies_by_genre.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: movies_by_genre.png")

# ── 4.2  Average rating per genre ────────────────────────────
print("\n[2] Average Rating per Genre (min 1,000 ratings)")
print("-" * 50)

movie_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(avg_rating="mean", num_ratings="count")
    .reset_index()
)

genre_rating = (
    genre_movies[["movieId", "genre"]]
    .merge(movie_stats, on="movieId")
    .groupby("genre")["avg_rating"]
    .agg(avg="mean", cnt="count")
    .query("cnt >= 1000")
    .sort_values("avg", ascending=False)
)

for g, row in genre_rating.iterrows():
    print(f"  {g:<20}  avg={row['avg']:.3f}   movies={row['cnt']:,}")

fig, ax = plt.subplots(figsize=(12, 6))
genre_sorted = genre_rating.sort_values("avg", ascending=True)
bars = ax.barh(genre_sorted.index, genre_sorted["avg"],
               color=ACCENT2, edgecolor="#0f1116", alpha=0.9)
overall_mean = ratings["rating"].mean()
ax.axvline(overall_mean, color="white", linestyle="--", lw=1.5,
           label=f"Overall mean = {overall_mean:.2f}")
ax.set_title("Average Rating by Genre (min 1,000 ratings)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Average Rating")
ax.set_xlim(2.8, 4.3)
ax.legend()
for bar in bars:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}", va="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("avg_rating_by_genre.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: avg_rating_by_genre.png")

# ── 4.3  Genre volume by decade heatmap ──────────────────────
print("\n[3] Genre Volume by Decade Heatmap")
print("-" * 50)

movies["decade"] = (movies["release_year"] // 10 * 10).astype(str) + "s"
genre_decade = explode_genres(movies)

top10_genres = genre_counts.head(10).index.tolist()
pivot = (
    genre_decade[genre_decade["genre"].isin(top10_genres)]
    .groupby(["decade", "genre"])["movieId"]
    .count()
    .unstack(fill_value=0)
)
pivot = pivot.loc[pivot.index >= "1960s"]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    pivot.T, cmap="YlGnBu", ax=ax,
    linewidths=0.4, linecolor="#0f1116",
    annot=True, fmt="d",
    cbar_kws={"label": "Movie Count"},
)
ax.set_title("Top-10 Genre Volume by Decade",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Decade")
ax.set_ylabel("Genre")
plt.tight_layout()
plt.savefig("genre_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: genre_heatmap.png")

# ── 4.4  Multi-genre movies ───────────────────────────────────
print("\n[4] Multi-Genre Statistics")
print("-" * 50)
movies["genre_count"] = movies["genres"].str.split("|").apply(
    lambda g: len([x for x in g if x != "(no genres listed)"])
)
print(f"  Average genres per movie : {movies['genre_count'].mean():.2f}")
print(f"  Max genres on one movie  : {movies['genre_count'].max()}")
print(f"  Single-genre movies      : {(movies['genre_count'] == 1).sum():,}")
print(f"  3+ genre movies          : {(movies['genre_count'] >= 3).sum():,}")

print("\n" + "=" * 60)
print("  Phase 4 complete.")
print("=" * 60)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"

# ── Load data ─────────────────────────────────────────────────
print("Loading data …")
movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    parse_dates=["timestamp"],
)
ratings["rating_year"]  = pd.to_datetime(ratings["timestamp"], errors="coerce").dt.year
ratings["rating_month"] = pd.to_datetime(ratings["timestamp"], errors="coerce").dt.month
print(f"  Movies : {len(movies):,}  |  Ratings : {len(ratings):,}\n")

print("=" * 60)
print("  PHASE 5 — Temporal Trends")
print("=" * 60)

# ── 5.1  Movie releases per year ─────────────────────────────
print("\n[1] Movie Releases per Year (1970 +)")
print("-" * 50)

releases = (
    movies[movies["release_year"] >= 1970]
    .groupby("release_year")["movieId"].count()
)

peak_year = releases.idxmax()
print(f"  Total movies (1970+)  : {releases.sum():,}")
print(f"  Peak production year  : {peak_year}  ({releases.max():,} movies)")
print(f"  Average per year      : {releases.mean():.0f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(releases.index, releases.values, color=ACCENT, alpha=0.2)
ax.plot(releases.index, releases.values, color=ACCENT, lw=2)
ax.axvline(peak_year, color=ACCENT3, lw=1.5, linestyle="--",
           label=f"Peak: {peak_year} ({releases.max():,} movies)")
ax.set_title("Movie Releases per Year (1970 +)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Release Year")
ax.set_ylabel("Number of Movies")
ax.legend()
plt.tight_layout()
plt.savefig("movie_releases_by_year.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: movie_releases_by_year.png")

# ── 5.2  Average rating trend over time ──────────────────────
print("\n[2] Average Rating Trend Over Time")
print("-" * 50)

avg_by_year = (
    ratings.groupby("rating_year")["rating"]
    .agg(mean="mean", count="count")
    .query("count >= 1000 and rating_year >= 1995 and rating_year <= 2018")
)

print(f"  Year range : {int(avg_by_year.index.min())} – {int(avg_by_year.index.max())}")
print(f"  Highest avg: {avg_by_year['mean'].max():.3f}  in {avg_by_year['mean'].idxmax()}")
print(f"  Lowest avg : {avg_by_year['mean'].min():.3f}  in {avg_by_year['mean'].idxmin()}")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

ax1.plot(avg_by_year.index, avg_by_year["mean"],
         color=ACCENT3, lw=2.5, marker="o", ms=4, label="Avg Rating")
ax2.bar(avg_by_year.index, avg_by_year["count"],
        color=ACCENT2, alpha=0.3, label="Rating Volume")

ax1.set_title("Average Rating & Volume Over Time",
              fontsize=14, color="white", fontweight="bold")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Rating", color=ACCENT3)
ax2.set_ylabel("Number of Ratings", color=ACCENT2)
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K")
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout()
plt.savefig("avg_rating_trend.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: avg_rating_trend.png")

# ── 5.3  Rating activity heatmap — month × year ──────────────
print("\n[3] Rating Activity Heatmap (Year × Month)")
print("-" * 50)

heatmap_data = (
    ratings.groupby(["rating_year", "rating_month"])["rating"]
    .count()
    .unstack(fill_value=0)
)
heatmap_data = heatmap_data.loc[
    (heatmap_data.index >= 1996) & (heatmap_data.index <= 2018)
]
heatmap_data.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    heatmap_data, cmap="Blues", ax=ax,
    linewidths=0.3, linecolor="#0f1116",
    cbar_kws={"label": "Rating Count"},
)
ax.set_title("Rating Activity Heatmap (Year × Month)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Year")
plt.tight_layout()
plt.savefig("rating_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: rating_heatmap.png")

# ── 5.4  Busiest months overall ──────────────────────────────
print("\n[4] Busiest Rating Months (all years combined)")
print("-" * 50)
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_total = ratings["rating_month"].value_counts().sort_index()
for m, cnt in monthly_total.items():
    bar = "█" * int(cnt / monthly_total.max() * 25)
    print(f"  {month_names[int(m)-1]:<5} {bar:<25}  {cnt:>8,}")

print("\n" + "=" * 60)
print("  Phase 5 complete.")
print("=" * 60)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"

# ── Load data ─────────────────────────────────────────────────
print("Loading data …")
movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
)
print(f"  Movies : {len(movies):,}  |  Ratings : {len(ratings):,}\n")

# ── Pre-compute movie stats ────────────────────────────────────
movie_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(avg_rating="mean", num_ratings="count")
    .reset_index()
)

print("=" * 60)
print("  PHASE 6 — Top Rated Movies")
print("=" * 60)

# ── 6.1  Overall top 20 (min 500 ratings) ────────────────────
print("\n[1] Overall Top 20 Movies  (min 500 ratings)")
print("-" * 70)

MIN_RATINGS = 500

top20 = (
    movie_stats[movie_stats["num_ratings"] >= MIN_RATINGS]
    .merge(movies[["movieId", "title", "release_year", "genres"]], on="movieId")
    .sort_values("avg_rating", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top20.index += 1

print(f"  {'#':<4} {'Title':<48} {'Year':<6} {'Avg':>5}  {'# Ratings':>10}")
print(f"  {'-'*4} {'-'*48} {'-'*6} {'-'*5}  {'-'*10}")
for rank, row in top20.iterrows():
    title = row["title"][:46]
    print(f"  {rank:<4} {title:<48} {int(row['release_year']):<6} "
          f"{row['avg_rating']:>5.3f}  {int(row['num_ratings']):>10,}")

# ── Plot 1 — Top 15 bar chart ─────────────────────────────────
top15 = top20.head(15).copy()
short_titles = top15["title"].str[:40]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(top15)), top15["avg_rating"],
               color=ACCENT3, edgecolor="#0f1116")
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(short_titles, fontsize=9)
ax.invert_yaxis()
ax.set_title(f"Top 15 Highest-Rated Movies  (min {MIN_RATINGS} ratings)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Average Rating")
ax.set_xlim(top15["avg_rating"].min() - 0.1, top15["avg_rating"].max() + 0.1)
for bar, val in zip(bars, top15["avg_rating"]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8.5, color="white")
plt.tight_layout()
plt.savefig("top_rated_movies.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: top_rated_movies.png")

# ── 6.2  Top movie per genre (min 200 ratings) ───────────────
print("\n[2] Best Movie per Genre  (min 200 ratings)")
print("-" * 70)

MIN_PER_GENRE = 200

# FIX: explode genres from movies, keeping title & release_year together
# then merge stats — no second merge needed since title/release_year
# are already present after exploding from movies directly
genre_movies = (
    movies[["movieId", "title", "release_year", "genres"]]
    .assign(genre=movies["genres"].str.split("|"))
    .explode("genre")
    .query("genre != '(no genres listed)'")
)

genre_top = (
    genre_movies
    .merge(movie_stats, on="movieId")
    .query("num_ratings >= @MIN_PER_GENRE")
    .sort_values("avg_rating", ascending=False)
    .groupby("genre")
    .first()
    .reset_index()
    [["genre", "title", "release_year", "avg_rating", "num_ratings"]]
    .sort_values("avg_rating", ascending=False)
    .reset_index(drop=True)
)
genre_top.index += 1

print(f"  {'#':<4} {'Genre':<18} {'Best Movie':<45} {'Year':<6} {'Avg':>5}")
print(f"  {'-'*4} {'-'*18} {'-'*45} {'-'*6} {'-'*5}")
for rank, row in genre_top.iterrows():
    print(f"  {rank:<4} {row['genre']:<18} {row['title'][:43]:<45} "
          f"{int(row['release_year']):<6} {row['avg_rating']:>5.3f}")

# ── Plot 2 — Genre best ratings scatter ──────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    genre_top["avg_rating"],
    range(len(genre_top)),
    s=genre_top["num_ratings"] / 30,
    c=genre_top["avg_rating"],
    cmap="cool",
    alpha=0.85,
    edgecolors="#0f1116",
    linewidth=0.5,
)
ax.set_yticks(range(len(genre_top)))
ax.set_yticklabels(genre_top["genre"], fontsize=9)
ax.invert_yaxis()
ax.set_title("Best-Rated Movie per Genre\n(bubble size = number of ratings)",
             fontsize=13, color="white", fontweight="bold")
ax.set_xlabel("Average Rating of Best Movie")
plt.colorbar(scatter, ax=ax, label="Avg Rating")
plt.tight_layout()
plt.savefig("genre_best_movies.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: genre_best_movies.png")

# ── 6.3  Rating threshold sensitivity ────────────────────────
print("\n[3] How threshold affects top-ranked movie")
print("-" * 50)
for threshold in [50, 100, 250, 500, 1000]:
    top = (
        movie_stats[movie_stats["num_ratings"] >= threshold]
        .merge(movies[["movieId", "title"]], on="movieId")
        .sort_values("avg_rating", ascending=False)
        .iloc[0]
    )
    print(f"  min={threshold:>5} ratings  →  #1: {top['title'][:45]}  "
          f"({top['avg_rating']:.3f})")

print("\n" + "=" * 60)
print("  Phase 6 complete.")
print("=" * 60)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"
PALETTE = [ACCENT, ACCENT2, ACCENT3, "#a29bfe", "#55efc4", "#fd79a8", "#fdcb6e"]

# ── Load data ─────────────────────────────────────────────────
print("Loading data …")
tags    = pd.read_csv(DATA_DIR + "tags_clean.csv")
movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
)
print(f"  Tags : {len(tags):,}  |  Unique tag strings : {tags['tag'].nunique():,}\n")

print("=" * 60)
print("  PHASE 7 — User Tag Analysis")
print("=" * 60)

# ── 7.1  Top 30 most used tags ────────────────────────────────
print("\n[1] Top 30 Most Used Tags")
print("-" * 50)
tag_freq = tags["tag"].value_counts()
top30    = tag_freq.head(30)

for i, (tag, cnt) in enumerate(top30.items(), 1):
    bar = "█" * int(cnt / top30.max() * 28)
    print(f"  {i:>2}. {tag:<30} {bar:<28}  {cnt:>6,}")

fig, ax = plt.subplots(figsize=(12, 8))
colors_bar = [ACCENT if i < 5 else "#3a4a6b" for i in range(len(top30))]
ax.barh(top30.index[::-1], top30.values[::-1],
        color=colors_bar[::-1], edgecolor="#0f1116")
ax.set_title("Top 30 Most Used User Tags",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Tag Count")
for i, (tag, val) in enumerate(zip(top30.index[::-1], top30.values[::-1])):
    ax.text(val + top30.max() * 0.005, i,
            f"{val:,}", va="center", fontsize=7.5, color="white")
plt.tight_layout()
plt.savefig("top_user_tags.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: top_user_tags.png")

# ── 7.2  Tags per movie distribution ─────────────────────────
print("\n[2] Tags per Movie Distribution")
print("-" * 50)

tags_per_movie = tags.groupby("movieId")["tag"].count()
print(f"  Movies with ≥1 tag   : {len(tags_per_movie):,}")
print(f"  Median tags / movie  : {tags_per_movie.median():.0f}")
print(f"  Mean   tags / movie  : {tags_per_movie.mean():.1f}")
print(f"  Max    tags on movie : {tags_per_movie.max():,}")
print(f"  Movies with  1 tag   : {(tags_per_movie == 1).sum():,}")
print(f"  Movies with 10+ tags : {(tags_per_movie >= 10).sum():,}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(np.log1p(tags_per_movie), bins=50, color=ACCENT,
        edgecolor="#0f1116", alpha=0.85)
ax.axvline(np.log1p(tags_per_movie.median()), color=ACCENT3, lw=2,
           linestyle="--", label=f"Median = {tags_per_movie.median():.0f}")
ax.set_title("Distribution of Tags per Movie (log scale)",
             fontsize=13, color="white", fontweight="bold")
ax.set_xlabel("log(1 + tag count)")
ax.set_ylabel("Number of Movies")
ax.legend()
plt.tight_layout()
plt.savefig("tags_per_movie.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: tags_per_movie.png")

# ── 7.3  Tags per user distribution ──────────────────────────
print("\n[3] Tags per User Distribution")
print("-" * 50)

tags_per_user = tags.groupby("userId")["tag"].count()
print(f"  Users who applied tags   : {len(tags_per_user):,}")
print(f"  Median tags / user       : {tags_per_user.median():.0f}")
print(f"  Mean   tags / user       : {tags_per_user.mean():.1f}")
print(f"  Top tagger (max)         : {tags_per_user.max():,} tags")

# ── 7.4  Visual tag cloud (no WordCloud dependency) ──────────
print("\n[4] Generating Tag Cloud …")

top60    = tag_freq.head(60)
max_cnt  = top60.max()
rng      = np.random.default_rng(42)

fig, ax  = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#0f1116")
fig.patch.set_facecolor("#0f1116")
ax.axis("off")

placed = []
attempts = 0

for i, (tag, cnt) in enumerate(top60.items()):
    size   = 9 + (cnt / max_cnt) * 36
    color  = PALETTE[i % len(PALETTE)]
    # simple random placement
    for _ in range(200):
        x = rng.uniform(0.04, 0.96)
        y = rng.uniform(0.06, 0.94)
        # crude overlap avoidance
        overlap = any(abs(x - px) < 0.12 and abs(y - py) < 0.07 for px, py in placed)
        if not overlap:
            placed.append((x, y))
            break
    ax.text(x, y, tag, fontsize=size, color=color, alpha=0.88,
            ha="center", va="center",
            transform=ax.transAxes, fontweight="bold")

ax.set_title("Top-60 User Tags — Tag Cloud",
             fontsize=14, color="white", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("tag_cloud.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: tag_cloud.png")

# ── 7.5  Most tagged movies ───────────────────────────────────
print("\n[5] Top 10 Most Tagged Movies")
print("-" * 60)

most_tagged = (
    tags_per_movie.reset_index()
    .rename(columns={"tag": "tag_count"})
    .merge(movies[["movieId", "title", "release_year"]], on="movieId")
    .sort_values("tag_count", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
most_tagged.index += 1
print(f"  {'#':<4} {'Title':<50} {'Year':<6} {'Tags':>6}")
print(f"  {'-'*4} {'-'*50} {'-'*6} {'-'*6}")
for rank, row in most_tagged.iterrows():
    print(f"  {rank:<4} {row['title'][:48]:<50} "
          f"{int(row['release_year']):<6} {int(row['tag_count']):>6,}")

# ── 7.6  Tag vocabulary richness over time ────────────────────
print("\n[6] Unique Tags Used per Year")
print("-" * 50)
tags["timestamp"] = pd.to_datetime(tags["timestamp"], dayfirst=True, errors="coerce")
tags["tag_year"]  = tags["timestamp"].dt.year
vocab_by_year = (tags.dropna(subset=["tag_year"])
                     .groupby("tag_year")["tag"].nunique())
for yr, cnt in vocab_by_year.items():
    if yr >= 2000:
        print(f"  {int(yr)}  {cnt:>5,} unique tags")

print("\n" + "=" * 60)
print("  Phase 7 complete.")
print("=" * 60)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"
ACCENT4 = "#a29bfe"
PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#55efc4",
           "#fd79a8", "#74b9ff", "#e17055", "#00b894", "#fdcb6e"]

# ── Load data ─────────────────────────────────────────────────
movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
)

# helper — explode genres
def explode_genres(df):
    return (
        df.assign(genre=df["genres"].str.split("|"))
          .explode("genre")
          .query("genre != '(no genres listed)'")
    )

print(f"Movies: {len(movies):,}  |  Ratings: {len(ratings):,}")


print("\n" + "="*60)
print("  MOVIE Q1 — Movies Released per Year")
print("="*60)

movies_per_year = movies.groupby("release_year")["movieId"].count().reset_index()
movies_per_year.columns = ["release_year", "movie_count"]
print(movies_per_year.to_string(index=False))

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(movies_per_year["release_year"], movies_per_year["movie_count"],
       color=ACCENT, edgecolor="#0f1116", width=0.8)
ax.set_title("Number of Movies Released per Year", fontsize=14,
             color="white", fontweight="bold")
ax.set_xlabel("Release Year")
ax.set_ylabel("Number of Movies")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("q1_movies_per_year.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: q1_movies_per_year.png")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
ACCENT2 = "#ff6b6b"

movies = pd.read_csv(DATA_DIR + "movies_clean.csv")

max_year = int(movies["release_year"].max())
cutoff   = max_year - 19          # 20-year window (inclusive)
last20   = movies[movies["release_year"] >= cutoff]

last20_per_year = (
    last20.groupby("release_year")["movieId"]
    .count()
    .reset_index()
    .rename(columns={"movieId": "movie_count"})
)

print(f"  Year range : {cutoff} – {max_year}")
print(f"  Total movies in last 20 years : {len(last20):,}\n")
print(last20_per_year.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(last20_per_year["release_year"], last20_per_year["movie_count"],
       color=ACCENT2, edgecolor="#0f1116", width=0.7)
ax.set_title(f"Movies Released in Last 20 Years  ({cutoff}–{max_year})",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Release Year")
ax.set_ylabel("Number of Movies")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("q2_last20_years.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: q2_last20_years.png")



import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
PALETTE = ["#00d4ff","#ff6b6b","#ffd166","#a29bfe","#55efc4",
           "#fd79a8","#74b9ff","#e17055","#00b894","#fdcb6e"]

movies = pd.read_csv(DATA_DIR + "movies_clean.csv")

max_year = int(movies["release_year"].max())
cutoff   = max_year - 14          # 15-year window (inclusive)
last15   = movies[movies["release_year"] >= cutoff]

genre_last15 = (
    last15.assign(genre=last15["genres"].str.split("|"))
          .explode("genre")
          .query("genre != '(no genres listed)'")
          .groupby("genre")["movieId"]
          .count()
          .reset_index()
          .rename(columns={"movieId": "movie_count"})
          .sort_values("movie_count", ascending=False)
          .reset_index(drop=True)
)
genre_last15.index += 1

print(f"  Year range : {cutoff} – {max_year}\n")
print(genre_last15.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(genre_last15))]
ax.barh(genre_last15["genre"][::-1], genre_last15["movie_count"][::-1],
        color=colors[::-1], edgecolor="#0f1116")
ax.set_title(f"Movies per Genre — Last 15 Years ({cutoff}–{max_year})",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Number of Movies")
for bar in ax.patches:
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("q3_genre_last15.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: q3_genre_last15.png")

print("\n" + "="*60)
print("  MOVIE Q4 — Genres in Movies (all-time counts)")
print("="*60)

all_genres = (
    explode_genres(movies)
    .groupby("genre")["movieId"]
    .count()
    .reset_index()
    .rename(columns={"movieId": "movie_count"})
    .sort_values("movie_count", ascending=False)
    .reset_index(drop=True)
)
all_genres.index += 1
print(all_genres.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(all_genres))]
ax.barh(all_genres["genre"][::-1], all_genres["movie_count"][::-1],
        color=colors[::-1], edgecolor="#0f1116")
ax.set_title("Number of Movies per Genre (All Time)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Number of Movies")
for bar in ax.patches:
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width()):,}", va="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("q4_genres_in_movies.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: q4_genres_in_movies.png")


print("\n" + "="*60)
print("  MOVIE Q5 — Percentage of Genres")
print("="*60)

all_genres["percentage"] = (all_genres["movie_count"] / all_genres["movie_count"].sum() * 100).round(2)
print(all_genres[["genre", "movie_count", "percentage"]].to_string(index=False))

# Pie chart — group small slices into "Other"
threshold  = 2.0
main       = all_genres[all_genres["percentage"] >= threshold].copy()
other_pct  = all_genres[all_genres["percentage"] < threshold]["percentage"].sum()
other_cnt  = all_genres[all_genres["percentage"] < threshold]["movie_count"].sum()
if other_pct > 0:
    other_row = pd.DataFrame([{"genre": "Other", "movie_count": other_cnt,
                                "percentage": round(other_pct, 2)}])
    main = pd.concat([main, other_row], ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    main["percentage"],
    labels=main["genre"],
    autopct="%1.1f%%",
    colors=[PALETTE[i % len(PALETTE)] for i in range(len(main))],
    startangle=140,
    pctdistance=0.82,
    wedgeprops={"edgecolor": "#0f1116", "linewidth": 1.2},
)
for t in texts:
    t.set_color("white")
    t.set_fontsize(9)
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(7.5)
ax.set_title("Genre Distribution — Percentage of All Movies",
             fontsize=14, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("q5_genre_percentage.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: q5_genre_percentage.png")


import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
PALETTE = ["#00d4ff","#ff6b6b","#ffd166","#a29bfe","#55efc4",
           "#fd79a8","#74b9ff","#e17055","#00b894","#fdcb6e"]

movies = pd.read_csv(DATA_DIR + "movies_clean.csv")

max_year = int(movies["release_year"].max())
cutoff   = max_year - 9           # 10-year window (inclusive)
last10   = movies[movies["release_year"] >= cutoff]

# Top 5 years by movie count within last 10 years
top5_years = (
    last10.groupby("release_year")["movieId"]
    .count()
    .nlargest(5)
    .index.tolist()
)
print(f"  Year range : {cutoff} – {max_year}")
print(f"  Top 5 years within last 10 : {sorted(top5_years)}\n")

top5_genre_pivot = (
    last10[last10["release_year"].isin(top5_years)]
    .assign(genre=lambda df: df["genres"].str.split("|"))
    .explode("genre")
    .query("genre != '(no genres listed)'")
    .groupby(["release_year", "genre"])["movieId"]
    .count()
    .unstack(fill_value=0)
)
print(top5_genre_pivot.to_string())

fig, ax = plt.subplots(figsize=(14, 7))
top5_genre_pivot.plot(
    kind="bar", ax=ax, stacked=True,
    color=[PALETTE[i % len(PALETTE)] for i in range(len(top5_genre_pivot.columns))],
    edgecolor="#0f1116", width=0.6,
)
ax.set_title(f"Top 5 Movie-Release Years (Last 10 Years: {cutoff}–{max_year}) — Genre Breakdown",
             fontsize=13, color="white", fontweight="bold")
ax.set_xlabel("Release Year")
ax.set_ylabel("Number of Movies")
ax.tick_params(axis="x", rotation=0)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
          facecolor="#1a1e29", edgecolor="#3a3f52", labelcolor="white")
plt.tight_layout()
plt.savefig("q6_top5_last10_genres.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: q6_top5_last10_genres.png")


 ##RATING CSV QUESTIONS ##

print("\n" + "="*60)
print("  RATING Q1 — Total Number of Each Rating")
print("="*60)

rating_counts = (
    ratings["rating"]
    .value_counts()
    .sort_index()
    .reset_index()
)
rating_counts.columns = ["rating", "count"]
rating_counts["percentage"] = (rating_counts["count"] / len(ratings) * 100).round(2)
print(rating_counts.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(rating_counts["rating"].astype(str), rating_counts["count"],
              color=ACCENT3, edgecolor="#0f1116", width=0.65)
ax.set_title("Total Count per Rating Value",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K"))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(ratings)*0.003,
            f"{bar.get_height():,.0f}", ha="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("r1_rating_counts.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r1_rating_counts.png")


print("\n" + "="*60)
print("  RATING Q2 — Rating Counts per Year")
print("="*60)

ratings_per_year = (
    ratings.groupby("rating_year")["rating"]
    .count()
    .reset_index()
    .rename(columns={"rating": "count"})
)
print(ratings_per_year.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(ratings_per_year["rating_year"], ratings_per_year["count"],
                color=ACCENT, alpha=0.25)
ax.plot(ratings_per_year["rating_year"], ratings_per_year["count"],
        color=ACCENT, lw=2.5, marker="o", ms=5)
ax.set_title("Rating Counts per Year", fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Ratings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K"))
plt.tight_layout()
plt.savefig("r2_ratings_per_year.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r2_ratings_per_year.png")


print("\n" + "="*60)
print("  RATING Q3 — Rating Counts per Month")
print("="*60)

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
ratings_per_month = (
    ratings.groupby("rating_month")["rating"]
    .count()
    .reset_index()
    .rename(columns={"rating": "count"})
)
ratings_per_month["month_name"] = ratings_per_month["rating_month"].apply(
    lambda m: month_names[int(m)-1])
print(ratings_per_month[["month_name", "count"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(ratings_per_month["month_name"], ratings_per_month["count"],
              color=ACCENT4, edgecolor="#0f1116", width=0.65)
ax.set_title("Rating Counts per Month (All Years Combined)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Ratings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K"))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(ratings)*0.002,
            f"{bar.get_height():,.0f}", ha="center", fontsize=7.5, color="white")
plt.tight_layout()
plt.savefig("r3_ratings_per_month.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r3_ratings_per_month.png")


import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
PALETTE = ["#00d4ff","#ff6b6b","#ffd166","#a29bfe","#55efc4",
           "#fd79a8","#74b9ff","#e17055","#00b894","#fdcb6e",
           "#6c5ce7","#fab1a0","#81ecec","#dfe6e9","#b2bec3"]

ratings = pd.read_csv(DATA_DIR + "ratings_clean.csv",
                      dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

top15_users = (
    ratings.groupby("userId")["rating"]
    .count()
    .nlargest(15)
    .reset_index()
    .rename(columns={"rating": "rating_count"})
)
top15_users.index += 1

print("  Top 15 Users by Rating Count")
print("  " + "-"*35)
print(f"  {'Rank':<6} {'User ID':<12} {'Rating Count':>12}")
print("  " + "-"*35)
for rank, row in top15_users.iterrows():
    print(f"  {rank:<6} {int(row['userId']):<12} {int(row['rating_count']):>12,}")

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(
    top15_users["userId"].astype(str),
    top15_users["rating_count"],
    color=PALETTE,
    edgecolor="#0f1116",
    width=0.7,
)
ax.set_title("Top 15 Users by Number of Ratings",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("User ID")
ax.set_ylabel("Rating Count")
ax.tick_params(axis="x", rotation=45)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{int(bar.get_height()):,}", ha="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("r4_top15_users.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r4_top15_users.png")



print("\n" + "="*60)
print("  RATING Q5 — Unique Users per Year")
print("="*60)

unique_users_year = (
    ratings.groupby("rating_year")["userId"]
    .nunique()
    .reset_index()
    .rename(columns={"userId": "unique_users"})
)
print(unique_users_year.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(unique_users_year["rating_year"], unique_users_year["unique_users"],
                color=ACCENT2, alpha=0.2)
ax.plot(unique_users_year["rating_year"], unique_users_year["unique_users"],
        color=ACCENT2, lw=2.5, marker="o", ms=5)
ax.set_title("Unique Users per Year",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Unique Users")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x/1e3)}K" if x >= 1000 else str(int(x))))
plt.tight_layout()
plt.savefig("r5_unique_users_year.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r5_unique_users_year.png")


import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
ACCENT3 = "#ffd166"

movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(DATA_DIR + "ratings_clean.csv",
                      dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

top20_rated = (
    ratings.groupby("movieId")["rating"]
    .count()
    .nlargest(20)
    .reset_index()
    .rename(columns={"rating": "rating_count"})
    .merge(movies[["movieId", "title"]], on="movieId")
    [["title", "rating_count"]]
    .reset_index(drop=True)
)
top20_rated.index += 1

print("  Top 20 Most-Rated Movies")
print("  " + "-"*55)
print(f"  {'Rank':<6} {'Title':<45} {'Count':>8}")
print("  " + "-"*55)
for rank, row in top20_rated.iterrows():
    print(f"  {rank:<6} {row['title'][:43]:<45} {int(row['rating_count']):>8,}")

fig, ax = plt.subplots(figsize=(12, 9))
short = top20_rated["title"].str[:40]
bars = ax.barh(short[::-1], top20_rated["rating_count"][::-1],
               color=ACCENT3, edgecolor="#0f1116")
ax.set_title("Top 20 Most-Rated Movies",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Number of Ratings")
for bar in bars:
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width()):,}", va="center", fontsize=8.5, color="white")
plt.tight_layout()
plt.savefig("r6_top20_most_rated.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r6_top20_most_rated.png")


import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
PALETTE = ["#00d4ff","#ff6b6b","#ffd166","#a29bfe","#55efc4"]

movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(DATA_DIR + "ratings_clean.csv",
                      dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

top5_rated = (
    ratings.groupby("movieId")["rating"]
    .count()
    .nlargest(5)
    .reset_index()
    .rename(columns={"rating": "rating_count"})
    .merge(movies[["movieId", "title"]], on="movieId")
    [["title", "rating_count"]]
    .reset_index(drop=True)
)
top5_rated.index += 1

print("  Top 5 Most-Rated Movies")
print("  " + "-"*55)
print(f"  {'Rank':<6} {'Title':<45} {'Count':>8}")
print("  " + "-"*55)
for rank, row in top5_rated.iterrows():
    print(f"  {rank:<6} {row['title'][:43]:<45} {int(row['rating_count']):>8,}")

fig, ax = plt.subplots(figsize=(11, 5))
short = top5_rated["title"].str[:40]
bars = ax.bar(short, top5_rated["rating_count"],
              color=PALETTE, edgecolor="#0f1116", width=0.6)
ax.set_title("Top 5 Most-Rated Movies",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Movie Title")
ax.set_ylabel("Number of Ratings")
ax.tick_params(axis="x", rotation=15)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{int(bar.get_height()):,}", ha="center", fontsize=10, color="white",
            fontweight="bold")
plt.tight_layout()
plt.savefig("r6_top5_most_rated.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r6_top5_most_rated.png")

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
ACCENT2 = "#ff6b6b"
PALETTE = ["#00d4ff","#ff6b6b","#ffd166","#a29bfe","#55efc4"]

movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(DATA_DIR + "ratings_clean.csv",
                      dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

top20_avg = (
    ratings.groupby("movieId")["rating"]
    .mean()
    .nlargest(20)
    .reset_index()
    .rename(columns={"rating": "avg_rating"})
    .merge(movies[["movieId", "title"]], on="movieId")
    [["title", "avg_rating"]]
    .reset_index(drop=True)
)
top20_avg.index += 1
top20_avg["avg_rating"] = top20_avg["avg_rating"].round(3)

# ── Print table ───────────────────────────────────────────────
print("  Top 20 Movies by Average Rating (no threshold)")
print("  " + "-"*55)
print(f"  {'Rank':<6} {'Title':<45} {'Avg Rating':>10}")
print("  " + "-"*55)
for rank, row in top20_avg.iterrows():
    print(f"  {rank:<6} {row['title'][:43]:<45} {row['avg_rating']:>10.3f}")

# ── Plot 1: Top 20 horizontal bar ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
short = top20_avg["title"].str[:40]
bars = ax.barh(short[::-1], top20_avg["avg_rating"][::-1],
               color=ACCENT2, edgecolor="#0f1116")
ax.set_title("Top 20 Movies by Average Rating (no minimum threshold)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Average Rating")
ax.set_xlim(top20_avg["avg_rating"].min() - 0.2, 5.1)
for bar in bars:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.3f}", va="center", fontsize=8.5, color="white")
plt.tight_layout()
plt.savefig("r7_top20_avg_rating.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r7_top20_avg_rating.png")

# ── Top 5 ─────────────────────────────────────────────────────
top5_avg = top20_avg.head(5).copy()

print("\n  Top 5 Movies by Average Rating (no threshold)")
print("  " + "-"*55)
print(f"  {'Rank':<6} {'Title':<45} {'Avg Rating':>10}")
print("  " + "-"*55)
for rank, row in top5_avg.iterrows():
    print(f"  {rank:<6} {row['title'][:43]:<45} {row['avg_rating']:>10.3f}")

# ── Plot 2: Top 5 vertical bar ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
short5 = top5_avg["title"].str[:35]
bars = ax.bar(short5, top5_avg["avg_rating"],
              color=PALETTE, edgecolor="#0f1116", width=0.6)
ax.set_title("Top 5 Movies by Average Rating (no minimum threshold)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Movie Title")
ax.set_ylabel("Average Rating")
ax.set_ylim(top5_avg["avg_rating"].min() - 0.3, 5.15)
ax.tick_params(axis="x", rotation=15)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=10,
            color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("r7_top5_avg_rating.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r7_top5_avg_rating.png")


import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"
plt.rcParams.update({
    "figure.facecolor": "#0f1116", "axes.facecolor": "#1a1e29",
    "axes.edgecolor": "#3a3f52", "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8", "ytick.color": "#c8cdd8",
    "text.color": "#ffffff", "grid.color": "#2e3347",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})
PALETTE = ["#00d4ff","#ff6b6b","#ffd166","#a29bfe","#55efc4",
           "#fd79a8","#74b9ff","#e17055","#00b894","#fdcb6e"]

movies  = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings = pd.read_csv(DATA_DIR + "ratings_clean.csv",
                      dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})

top10_qualified = (
    ratings.groupby("movieId")["rating"]
    .agg(avg_rating="mean", rating_count="count")
    .reset_index()
    .query("rating_count >= 1000")
    .nlargest(10, "avg_rating")
    .merge(movies[["movieId", "title"]], on="movieId")
    [["title", "avg_rating", "rating_count"]]
    .reset_index(drop=True)
)
top10_qualified.index += 1
top10_qualified["avg_rating"] = top10_qualified["avg_rating"].round(3)

print("  Top 10 Movies by Average Rating (min 1,000 ratings)")
print("  " + "-"*60)
print(f"  {'Rank':<6} {'Title':<45} {'Avg':>6}  {'Count':>8}")
print("  " + "-"*60)
for rank, row in top10_qualified.iterrows():
    print(f"  {rank:<6} {row['title'][:43]:<45} {row['avg_rating']:>6.3f}  {int(row['rating_count']):>8,}")

fig, ax = plt.subplots(figsize=(12, 6))
short = top10_qualified["title"].str[:40]

# FIX: use % to cycle colours safely regardless of list length
colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(top10_qualified))]

bars = ax.barh(short[::-1], top10_qualified["avg_rating"][::-1],
               color=colors_bar[::-1], edgecolor="#0f1116")
ax.set_title("Top 10 Movies by Average Rating  (min 1,000 ratings)",
             fontsize=14, color="white", fontweight="bold")
ax.set_xlabel("Average Rating")
ax.set_xlim(top10_qualified["avg_rating"].min() - 0.15,
            top10_qualified["avg_rating"].max() + 0.1)
for bar, cnt in zip(bars, top10_qualified["rating_count"][::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.3f}  ({cnt:,} ratings)",
            va="center", fontsize=8.5, color="white")
plt.tight_layout()
plt.savefig("r8_top10_qualified_avg.png", dpi=150, bbox_inches="tight", facecolor="#0f1116")
plt.show()
print("  ✅ Saved: r8_top10_qualified_avg.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = "C:/Users/user/OneDrive/Desktop/Movie Lens/"

# ── Plot theme ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor":   "#1a1e29",
    "axes.edgecolor":   "#3a3f52",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#c8cdd8",
    "ytick.color":      "#c8cdd8",
    "text.color":       "#ffffff",
    "grid.color":       "#2e3347",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd166"

# ── Load data ─────────────────────────────────────────────────
print("Loading data …")
movies   = pd.read_csv(DATA_DIR + "movies_clean.csv")
ratings  = pd.read_csv(
    DATA_DIR + "ratings_clean.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    parse_dates=["timestamp"],
)
tags     = pd.read_csv(DATA_DIR + "tags_clean.csv")
g_scores = pd.read_csv(DATA_DIR + "genome_scores.csv",
                       dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"})

ratings["rating_year"] = pd.to_datetime(ratings["timestamp"], errors="coerce").dt.year

movie_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(avg_rating="mean", num_ratings="count")
    .reset_index()
)

print("=" * 65)
print("  PHASE 9 — Conclusion & Summary Report")
print("=" * 65)

# ── 9.1  Compute all KPIs ─────────────────────────────────────
total_movies     = len(movies)
total_ratings    = len(ratings)
total_users      = ratings["userId"].nunique()
avg_rating       = ratings["rating"].mean()
most_common_rtg  = ratings["rating"].mode()[0]
genre_counts     = (movies["genres"].str.split("|").explode()
                    .loc[lambda s: s != "(no genres listed)"])
top_genre        = genre_counts.value_counts().idxmax()
top_genre_cnt    = genre_counts.value_counts().max()
peak_year        = (movies[movies["release_year"] >= 1970]
                    .groupby("release_year")["movieId"].count()
                    .idxmax())
top_movie        = (movie_stats[movie_stats["num_ratings"] >= 500]
                    .merge(movies[["movieId", "title"]], on="movieId")
                    .sort_values("avg_rating", ascending=False)
                    .iloc[0])
genome_movies    = g_scores["movieId"].nunique()
unique_tags_user = tags["tag"].nunique()

# ── 9.2  Print findings ───────────────────────────────────────
findings = [
    ("Ratings cluster at whole numbers",
     f"The most common rating is {most_common_rtg:.1f}. Users prefer integer "
     f"scores, and 4.0 accounts for the largest share of all {total_ratings:,} ratings."),

    ("Drama & Comedy dominate by volume",
     f"{top_genre} is the largest genre with {top_genre_cnt:,} movies. However, "
     f"Film-Noir and Documentary lead in average user rating."),

    ("Production peaked in the 2000s",
     f"Movie releases accelerated sharply from the 1990s, reaching a peak around "
     f"{peak_year}. The post-2015 period shows a modest plateau."),

    ("Rating activity follows platform growth",
     f"Rating volume surged in the early 2000s and again around 2015–2016. "
     f"Total active reviewers: {total_users:,} across {ratings['rating_year'].nunique()} years."),

    ("Classics consistently top the charts",
     f"The highest-rated movie (min 500 ratings) is '{top_movie['title']}' "
     f"with an average of {top_movie['avg_rating']:.3f} ⭐. Many top titles are from the 1950s–70s."),

    ("User tags reflect rich vocabulary",
     f"Users applied {unique_tags_user:,} unique tags. Common descriptors include "
     f"'atmospheric', 'based on a book', 'twist ending', and actor/director names."),

    ("Tag-Genome enables coherent recommendations",
     f"The {genome_movies:,}-movie Tag-Genome matrix (1,128 dimensions) produces "
     f"highly thematic cosine-similarity recommendations, grouping films by mood, "
     f"tone, and narrative style beyond simple genre overlap."),
]

print("\n  KEY FINDINGS")
print("  " + "─" * 63)
for i, (title, detail) in enumerate(findings, 1):
    print(f"\n  {i}. {title}")
    # word-wrap detail at ~60 chars
    words = detail.split()
    line  = "     "
    for w in words:
        if len(line) + len(w) + 1 > 65:
            print(line)
            line = "     " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line)

# ── 9.3  Dataset KPI summary box ─────────────────────────────
print("\n\n  DATASET KPI SUMMARY")
print("  " + "─" * 45)
kpis = [
    ("Total Movies Analysed",  f"{total_movies:,}"),
    ("Total User Ratings",     f"{total_ratings:,}"),
    ("Unique Users",           f"{total_users:,}"),
    ("Average Rating",         f"{avg_rating:.3f} ⭐"),
    ("Most Common Rating",     f"{most_common_rtg:.1f}"),
    ("Top Genre (by count)",   f"{top_genre} ({top_genre_cnt:,})"),
    ("Peak Release Year",      str(peak_year)),
    ("Genome-Mapped Movies",   f"{genome_movies:,}"),
    ("Unique User Tags",       f"{unique_tags_user:,}"),
]
for label, value in kpis:
    print(f"  {label:<30}  {value}")

# ── 9.4  Visual summary dashboard ────────────────────────────
print("\n  Generating Summary Dashboard …")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("MovieLens EDA — Summary Dashboard",
             fontsize=18, color="white", fontweight="bold", y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Rating distribution
ax1 = fig.add_subplot(gs[0, 0])
vc  = ratings["rating"].value_counts().sort_index()
ax1.bar(vc.index.astype(str), vc.values, color=ACCENT, edgecolor="#0f1116", width=0.7)
ax1.set_title("Rating Distribution", color="white", fontsize=11, fontweight="bold")
ax1.set_xlabel("Rating")
ax1.set_ylabel("Count")

# Panel 2 — Genre counts (top 10)
ax2   = fig.add_subplot(gs[0, 1])
top10g = genre_counts.value_counts().head(10)
ax2.barh(top10g.index[::-1], top10g.values[::-1], color=ACCENT2, edgecolor="#0f1116")
ax2.set_title("Top 10 Genres", color="white", fontsize=11, fontweight="bold")
ax2.set_xlabel("Movie Count")

# Panel 3 — Releases per year
ax3      = fig.add_subplot(gs[0, 2])
releases = (movies[movies["release_year"] >= 1970]
            .groupby("release_year")["movieId"].count())
ax3.fill_between(releases.index, releases.values, color=ACCENT3, alpha=0.3)
ax3.plot(releases.index, releases.values, color=ACCENT3, lw=1.5)
ax3.set_title("Releases per Year", color="white", fontsize=11, fontweight="bold")
ax3.set_xlabel("Year")

# Panel 4 — Avg rating trend
ax4          = fig.add_subplot(gs[1, 0])
avg_by_year  = (ratings.groupby("rating_year")["rating"]
                .mean()
                .loc[lambda s: (s.index >= 1996) & (s.index <= 2018)])
ax4.plot(avg_by_year.index, avg_by_year.values, color=ACCENT, lw=2, marker="o", ms=3)
ax4.set_title("Avg Rating Over Time", color="white", fontsize=11, fontweight="bold")
ax4.set_xlabel("Year")
ax4.set_ylim(3.3, 4.0)

# Panel 5 — Top 10 rated movies
ax5    = fig.add_subplot(gs[1, 1])
top10m = (movie_stats[movie_stats["num_ratings"] >= 500]
          .merge(movies[["movieId", "title"]], on="movieId")
          .sort_values("avg_rating", ascending=False)
          .head(10))
ax5.barh(top10m["title"].str[:25][::-1], top10m["avg_rating"][::-1],
         color=ACCENT3, edgecolor="#0f1116")
ax5.set_title("Top 10 Rated Movies", color="white", fontsize=11, fontweight="bold")
ax5.set_xlabel("Avg Rating")
ax5.set_xlim(4.0, 4.6)

# Panel 6 — Top 10 user tags
ax6    = fig.add_subplot(gs[1, 2])
top10t = tags["tag"].value_counts().head(10)
ax6.barh(top10t.index[::-1], top10t.values[::-1], color="#a29bfe", edgecolor="#0f1116")
ax6.set_title("Top 10 User Tags", color="white", fontsize=11, fontweight="bold")
ax6.set_xlabel("Tag Count")

plt.tight_layout()
plt.savefig("summary_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1116")
plt.show()
print("  ✅ Saved: summary_dashboard.png")

# ── 9.5  Next steps ───────────────────────────────────────────
print("\n  NEXT STEPS")
print("  " + "─" * 55)
next_steps = [
    "Collaborative Filtering (user–user / item–item) using rating matrix",
    "Hybrid model combining content-based + collaborative signals",
    "Sentiment analysis on free-text user tags",
    "Matrix Factorisation (SVD / ALS) for latent factor modelling",
    "Time-aware recommendations (decay older ratings)",
    "Deploy with the included dashboard_app.py (Streamlit)",
]
for i, step in enumerate(next_steps, 1):
    print(f"  {i}. {step}")

print("\n" + "=" * 65)
print("  Phase 9 complete — Project finished!")
print("=" * 65)
print("""
  FILES GENERATED ACROSS ALL PHASES
  ───────────────────────────────────────────────────────────
  data/movies_clean.csv          ← cleaned movies
  data/ratings_clean.csv         ← cleaned ratings
  data/tags_clean.csv            ← cleaned tags
  ratings_distribution.png
  engagement_distributions.png
  movies_by_genre.png
  avg_rating_by_genre.png
  genre_heatmap.png
  movie_releases_by_year.png
  avg_rating_trend.png
  rating_heatmap.png
  top_rated_movies.png
  genre_best_movies.png
  top_user_tags.png
  tags_per_movie.png
  tag_cloud.png
  rec_toy_story.png
  rec_matrix_the.png
  summary_dashboard.png
""")

