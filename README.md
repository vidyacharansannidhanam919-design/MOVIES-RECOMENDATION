# 🎬 MovieLens Movie Recommendation & Analytics

A comprehensive movie recommendation system and interactive analytics dashboard built with Python, powered by the **MovieLens dataset** (316,499 ratings · 11,112 movies · 2,161 users).

---

## 📁 Project Structure

```
Movie Lens/
├── Movie Recommendation.py      # Main analysis pipeline (Phases 2–9)
├── movielens_dashboard.py        # Interactive Plotly Dash dashboard
├── future_scope.md               # Future improvements & roadmap
├── visualizations/               # Generated visualization images
└── *.png                         # All chart outputs from each phase
```

---

## 🔬 Code Structure — Phase-by-Phase Breakdown

The `Movie Recommendation.py` script is organized into **8 sequential phases**, each building on the previous one. Below is a detailed explanation of what happens at every stage.

---

### Phase 2 — Data Loading & Cleaning *(Lines 1–112)*

**What it does:** Loads all 6 raw CSV files from the MovieLens dataset, audits them for nulls and duplicates, engineers new features, and exports cleaned data.

| Step | Description |
|------|-------------|
| **2.1 Load Raw Files** | Reads `movie.csv`, `rating.csv`, `tag.csv`, `genome_scores.csv`, `genome_tags.csv`, and `link.csv` into Pandas DataFrames with optimized dtypes (`int32`, `float32`) to reduce memory |
| **2.2 Null & Duplicate Audit** | Scans every DataFrame for missing values and duplicate rows, printing a formatted audit table |
| **2.3 Feature Engineering — Movies** | Extracts `release_year` from movie titles using regex (e.g., `"Toy Story (1995)"` → `1995`), drops rows with missing years |
| **2.4 Feature Engineering — Ratings** | Parses timestamp strings into datetime objects, derives `rating_year` and `rating_month` columns for temporal analysis |
| **2.5 Feature Engineering — Tags** | Normalizes tags to lowercase, strips whitespace, and removes duplicate tag entries |
| **2.6 Post-Cleaning Summary** | Prints a clean summary table showing final row counts and column counts for all DataFrames |
| **2.7 Export Cleaned Data** | Saves `movies_clean.csv`, `ratings_clean.csv`, and `tags_clean.csv` for all downstream phases |

**Output files:** `movies_clean.csv`, `ratings_clean.csv`, `tags_clean.csv`

---

### Phase 3 — Ratings Distribution Analysis *(Lines 114–256)*

**What it does:** Performs statistical analysis on rating values and visualizes how users rate movies.

| Step | Description |
|------|-------------|
| **3.1 Descriptive Statistics** | Computes mean, median, std, min, max, skewness, and kurtosis of all ratings |
| **3.2 Rating Value Counts** | Counts how many times each rating value (0.5 to 5.0) appears, with ASCII bar visualization in the terminal |
| **3.3 Bar Chart** | Creates a bar chart showing the count per rating value — reveals that **4.0 is the most common rating** (87,727 times) |
| **3.4 Density Histogram** | Plots a KDE-style density histogram with a mean line overlay, showing the left-skewed distribution |
| **3.5 Engagement Distributions** | Generates log-scale histograms of ratings-per-user and ratings-per-movie to identify power users and popular films |
| **3.6 Power-User Insight** | Calculates that the **top 1% of users account for a significant portion of all ratings** |

**Output files:** `ratings_distribution.png`, `engagement_distributions.png`

---

### Phase 4 — Genre Analysis *(Lines 258–426)*

**What it does:** Explodes the pipe-separated genre field and analyzes genre popularity, ratings, and trends across decades.

| Step | Description |
|------|-------------|
| **4.1 Genre Frequency** | Counts movies per genre — **Drama leads with 5,339 movies**, followed by Comedy (3,756) and Thriller (2,529) |
| **4.2 Average Rating per Genre** | Merges genre data with rating averages, filters genres with ≥1,000 ratings. Film-Noir and Documentary have the highest avg ratings |
| **4.3 Genre Volume Heatmap** | Creates a Seaborn heatmap showing how the top 10 genres grew across decades (1960s–2010s), revealing the explosion of Drama and Thriller in the 2000s |
| **4.4 Multi-Genre Statistics** | Analyzes how many genres each movie belongs to — average is ~2 genres per movie, some have up to 8 |

**Output files:** `movies_by_genre.png`, `avg_rating_by_genre.png`, `genre_heatmap.png`

---

### Phase 5 — Temporal Trends *(Lines 428–583)*

**What it does:** Analyzes how movie production and rating activity changed over time.

| Step | Description |
|------|-------------|
| **5.1 Movie Releases per Year** | Plots an area chart of annual movie releases from 1970+, identifying the peak production year |
| **5.2 Average Rating Trend** | Creates a dual-axis chart: line for average rating over time + bar for rating volume. Shows that avg rating remains stable around 3.5 despite volume changes |
| **5.3 Rating Activity Heatmap** | Builds a Year × Month heatmap using Seaborn, revealing seasonal patterns in rating activity |
| **5.4 Busiest Rating Months** | Identifies which months have the most rating activity across all years combined |

**Output files:** `movie_releases_by_year.png`, `avg_rating_trend.png`, `rating_heatmap.png`

---

### Phase 6 — Top Rated Movies *(Lines 585–749)*

**What it does:** Ranks movies by average rating with minimum-rating thresholds to avoid bias from rarely-rated films.

| Step | Description |
|------|-------------|
| **6.1 Overall Top 20** | Lists the 20 highest-rated movies with a minimum of 500 ratings, printing a formatted leaderboard table |
| **6.2 Best Movie per Genre** | Finds the single best-rated movie in each genre (min 200 ratings) and visualizes with a bubble scatter plot where bubble size = number of ratings |
| **6.3 Rating Threshold Sensitivity** | Tests how changing the minimum-rating threshold (50, 100, 250, 500, 1000) affects which movie ranks #1, demonstrating why thresholds matter |

**Output files:** `top_rated_movies.png`, `genre_best_movies.png`

---

### Phase 7 — User Tag Analysis *(Lines 751–926)*

**What it does:** Explores the free-text tags that users apply to movies to understand viewing behavior and vocabulary.

| Step | Description |
|------|-------------|
| **7.1 Top 30 Tags** | Ranks the 30 most frequently used tags (e.g., "atmospheric", "based on a book", "twist ending") |
| **7.2 Tags per Movie** | Analyzes tag distribution across movies — most movies have few tags, but some have hundreds |
| **7.3 Tags per User** | Shows how tag activity varies across users — identifies "super taggers" |
| **7.4 Tag Cloud** | Generates a visual tag cloud using random placement with collision avoidance (no WordCloud dependency needed) |
| **7.5 Most Tagged Movies** | Lists the 10 movies with the most user-applied tags |
| **7.6 Tag Vocabulary Over Time** | Tracks how many unique tags were introduced each year |

**Output files:** `top_user_tags.png`, `tags_per_movie.png`, `tag_cloud.png`

---

### Movie & Rating Questions *(Lines 928–1663)*

**What it does:** Answers specific analytical questions about the dataset through targeted queries and visualizations.

#### Movie Questions (Q1–Q6):
| Question | Analysis |
|----------|----------|
| **Q1** | Total movies released per year (all-time bar chart) |
| **Q2** | Movies in the last 15 & 20 years with annotated bar charts |
| **Q3** | Genre breakdown for movies in the last 15 years |
| **Q4** | All-time genre distribution with horizontal bar chart |
| **Q5** | Genre percentage pie chart with "Other" grouping for small slices |
| **Q6** | Top 5 movie-release years (last 10 years) with stacked genre breakdown |

#### Rating Questions (R1–R8):
| Question | Analysis |
|----------|----------|
| **R1** | Count of each rating value with percentage |
| **R2** | Rating counts per year (line + area chart) |
| **R3** | Rating counts per month across all years |
| **R4** | Top 15 most active users by rating count |
| **R5** | Unique users per year trend |
| **R6** | Top 20 and Top 5 most-rated movies |
| **R7** | Top 20 and Top 5 highest average-rated movies (no threshold) |
| **R8** | Top 10 highest-rated movies with min 1,000 ratings (qualified ranking) |

**Output files:** `q1_movies_per_year.png` through `q6_top5_last10_genres.png`, `r1_rating_counts.png` through `r8_top10_qualified_avg.png`

---

### Phase 9 — Conclusion & Summary Report *(Lines 1665–1906)*

**What it does:** Computes final KPIs, prints key findings, and generates a 6-panel summary dashboard.

| Step | Description |
|------|-------------|
| **9.1 Compute All KPIs** | Aggregates total movies, ratings, users, avg rating, top genre, peak year, top movie, genome coverage, and unique tag count |
| **9.2 Key Findings** | Prints 7 data-driven conclusions (e.g., "Ratings cluster at whole numbers", "Classics consistently top the charts") |
| **9.3 KPI Summary Box** | Displays a formatted summary table of all dataset KPIs |
| **9.4 Visual Summary Dashboard** | Creates a 2×3 matplotlib grid with: Rating Distribution, Top 10 Genres, Releases per Year, Avg Rating Trend, Top 10 Movies, Top 10 Tags |
| **9.5 Next Steps** | Lists recommended future work: collaborative filtering, hybrid models, sentiment analysis, SVD, and deployment |

**Output files:** `summary_dashboard.png`

---

## 📊 Interactive Dashboard (`movielens_dashboard.py`)

A real-time interactive web dashboard built with **Plotly Dash 4.2** that provides a polished, dark-themed analytics experience.

### Dashboard Architecture

```
movielens_dashboard.py
│
├── DATA LOADING (Lines 18-66)
│   ├── Reads movies_clean.csv and ratings_clean.csv
│   ├── Computes KPI metrics (total ratings, unique users, movies, avg rating)
│   ├── Builds rating distribution DataFrame
│   ├── Computes top movies (≥50 ratings, sorted by avg)
│   ├── Explodes and counts genre data
│   └── Aggregates yearly rating counts
│
├── DESIGN SYSTEM (Lines 68-95)
│   ├── Color tokens: BG, SURFACE, BORDER, TEXT, MUTED, ACCENT, GOLD, TEAL
│   ├── 10-color PALETTE for chart elements
│   └── PLOT_LAYOUT dict (shared Plotly config for all charts)
│
├── HELPER FUNCTIONS (Lines 97-177)
│   ├── hex_alpha()      — Converts hex + alpha to rgba() for Plotly compatibility
│   ├── card()           — Reusable dark card container with border radius
│   ├── section_label()  — Uppercase, muted section headers
│   └── kpi_card()       — KPI card with value, delta indicator, icon, and glow effect
│
├── CHART FUNCTIONS (Lines 179-314)
│   ├── make_rating_dist_chart()   — Bar chart of rating value counts
│   ├── make_top_movies_chart()    — Horizontal bar of top 8 highest-rated movies
│   ├── make_yearly_line_chart()   — Line + area chart of yearly rating activity
│   ├── make_genre_bar()           — Horizontal bar of movies per genre
│   ├── make_genre_pie()           — Donut chart of genre share (top 10)
│   ├── make_radar()               — Radar/spider chart of top 6 genres
│   └── make_yearly_bar()          — Bar chart of annual rating volume with peak highlight
│
├── TAB CONTENT BUILDERS (Lines 326-445)
│   ├── build_overview()  — 4 KPI cards + rating dist + top movies + yearly trend
│   ├── build_genres()    — Genre bar + radar + donut pie
│   └── build_trends()    — Annual volume bar + 3 stat cards (peak/quietest/span)
│
├── LAYOUT (Lines 447-521)
│   ├── Sticky header with logo, tab navigation, dataset badge
│   ├── Three pre-rendered content panels (visibility toggled by callback)
│   └── Footer with dataset info
│
└── CALLBACK (Lines 524-536)
    └── toggle_panels() — Shows/hides panels based on active tab selection
```

### Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Overview** | 4 KPI cards (Total Ratings, Unique Users, Movies Rated, Avg Rating) with delta indicators · Rating distribution bar chart · Top 8 rated movies · Yearly trend line |
| **Genres** | Movies per genre bar chart · Radar chart of top 6 genres · Donut chart of genre share |
| **Trends** | Annual rating volume bar chart with peak year highlight · 3 stat cards: Most Active Year, Quietest Year, Rating Span |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| Data Analysis | Pandas, NumPy |
| Static Visualization | Matplotlib, Seaborn |
| Interactive Dashboard | Plotly Dash 4.2, Plotly 6.3 |
| ML/Recommendations | Scikit-learn (TF-IDF, Cosine Similarity) |
| NLP | WordCloud, NLTK |

---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vidyacharansannidhanam919-design/MOVIES-RECOMENDATION.git
   cd MOVIES-RECOMENDATION
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn plotly dash scikit-learn wordcloud
   ```

3. **Download the MovieLens dataset**
   - Download from [MovieLens](https://grouplens.org/datasets/movielens/)
   - Place the raw CSV files (`movie.csv`, `rating.csv`, `tag.csv`, etc.) in the project directory
   - Run `Movie Recommendation.py` first to generate the cleaned CSVs

4. **Run the Dashboard**
   ```bash
   python movielens_dashboard.py
   ```
   Open **http://127.0.0.1:8050** in your browser.

5. **Run the Full Analysis Pipeline**
   ```bash
   python "Movie Recommendation.py"
   ```

---

## 📈 Key Insights

- **316,499** total ratings across **11,112** movies from **2,161** users
- **4.0-star** is the most common rating (87,727 ratings)
- **Drama** is the most common genre (5,339 movies), but **Film-Noir** has the highest avg rating
- Peak movie production occurred in the **mid-2000s**
- The **top 1% of users** contribute a disproportionate share of all ratings
- Classics from the 1950s–70s consistently dominate the highest-rated lists

---

## 🔮 Future Scope

- Hybrid recommendation engine (content + collaborative filtering)
- Matrix Factorization (SVD / ALS) for latent factor modeling
- Real-time streaming data integration
- Sentiment analysis on user tags
- Deployment on cloud platforms (AWS/GCP)

See [future_scope.md](future_scope.md) for the detailed roadmap.

---

## 📝 License

This project is for educational and research purposes. The MovieLens dataset is provided by [GroupLens Research](https://grouplens.org/).

---

**Built with Python, Plotly Dash & Scikit-learn**
