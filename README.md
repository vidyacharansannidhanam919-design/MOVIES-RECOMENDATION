# 🎬 MovieLens Movie Recommendation & Analytics

A comprehensive movie recommendation system and interactive analytics dashboard built with Python, powered by the MovieLens dataset.

## 📊 Project Overview

This project performs end-to-end analysis of the **MovieLens dataset** (316,499 ratings · 11,112 movies · 2,161 users) and includes:

- **Exploratory Data Analysis (EDA)** — Deep analysis of movies, ratings, genres, tags, and user behavior
- **Movie Recommendation Engine** — Content-based and collaborative filtering recommendations
- **Interactive Dashboard** — Real-time analytics dashboard built with Plotly Dash

## 🚀 Features

### Movie Recommendation System (`Movie Recommendation.py`)
- Data cleaning and preprocessing of movies, ratings, tags, and genome scores
- Movie release trend analysis by year and genre
- Rating distribution and user engagement analysis
- Genre popularity and co-occurrence heatmaps
- Tag cloud and genome-based analysis
- Content-based movie recommendations using TF-IDF and cosine similarity

### Interactive Dashboard (`movielens_dashboard.py`)
- **Overview Tab** — KPI cards (total ratings, users, movies, avg rating), rating distribution bar chart, top-rated movies, and yearly activity trend
- **Genres Tab** — Genre distribution bar chart, radar chart of top genres, and donut chart of genre share
- **Trends Tab** — Annual rating volume with peak/quietest year stats and rating span analysis

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Plotly Dash 4.2 |
| ML/Recommendations | Scikit-learn (TF-IDF, Cosine Similarity) |
| NLP | WordCloud, NLTK |

## 📁 Project Structure

```
Movie Lens/
├── Movie Recommendation.py    # Main EDA & recommendation engine
├── movielens_dashboard.py     # Interactive Plotly Dash dashboard
├── future_scope.md            # Future improvements & roadmap
├── visualizations/            # Generated visualization images
│   ├── avg_rating_by_genre.png
│   ├── avg_rating_trend.png
│   ├── movies_by_genre.png
│   ├── ratings_distribution.png
│   └── ...
├── *.png                      # Analysis output charts
└── .gitignore
```

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
   - Place `movies_clean.csv` and `ratings_clean.csv` in the project root

4. **Run the Dashboard**
   ```bash
   python movielens_dashboard.py
   ```
   Then open **http://127.0.0.1:8050** in your browser.

5. **Run the Recommendation Engine**
   ```bash
   python "Movie Recommendation.py"
   ```

## 📈 Dashboard Preview

The dashboard features a dark-themed, modern UI with:
- **4 KPI cards** showing key metrics at a glance
- **6 interactive charts** across 3 tabs (Overview, Genres, Trends)
- **Hover tooltips** for detailed data exploration
- **Responsive layout** that adapts to different screen sizes

## 📊 Key Insights

- **316,499** total ratings across **11,112** movies from **2,161** users
- **4-star** is the most popular rating (87,727 ratings)
- **Drama** is the most common genre, followed by Comedy and Thriller
- Peak rating activity occurred in **2015**
- Average rating across all movies: **3.53**

## 🔮 Future Scope

- Hybrid recommendation engine (content + collaborative filtering)
- User-based personalized recommendations
- Real-time streaming data integration
- Sentiment analysis on user tags
- Deployment on cloud platforms (AWS/GCP)

See [future_scope.md](future_scope.md) for detailed roadmap.

## 📝 License

This project is for educational and research purposes. The MovieLens dataset is provided by [GroupLens Research](https://grouplens.org/).

---

**Built with ❤️ using Python, Plotly Dash & Scikit-learn**
