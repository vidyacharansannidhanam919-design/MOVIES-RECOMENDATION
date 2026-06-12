
"""
MovieLens Analytics Dashboard
==============================
Run:  python movielens_dashboard.py
Then open:  http://127.0.0.1:8050
"""
 
import pandas as pd
from collections import Counter
import os
 
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
 
# ── LOAD & PREPARE DATA ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies  = pd.read_csv(os.path.join(BASE_DIR, "movies_clean.csv"))
ratings = pd.read_csv(os.path.join(BASE_DIR, "ratings_clean.csv"))
 
# KPI metrics
total_ratings  = len(ratings)
unique_users   = ratings["userId"].nunique()
unique_movies  = ratings["movieId"].nunique()
avg_rating     = round(ratings["rating"].mean(), 2)
 
# Rating distribution
rating_dist = (
    ratings["rating"]
    .value_counts()
    .sort_index()
    .reset_index()
    .rename(columns={"rating": "Rating", "count": "Count"})
)
 
# Top movies (≥50 ratings)
top_movies = (
    ratings.groupby("movieId")
    .agg(count=("rating", "count"), avg=("rating", "mean"))
    .reset_index()
)
top_movies = top_movies[top_movies["count"] >= 50].merge(
    movies[["movieId", "title", "release_year"]], on="movieId"
)
top_movies = top_movies.sort_values("avg", ascending=False).head(10)
 
# Genre data
genres_all = []
for g in movies["genres"].dropna():
    genres_all.extend(g.split("|"))
genre_counts = Counter(genres_all)
genre_df = (
    pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"])
    .sort_values("Count", ascending=False)
    .head(10)
)
 
# Yearly ratings
yearly = (
    ratings.groupby("rating_year")
    .size()
    .reset_index(name="Count")
    .rename(columns={"rating_year": "Year"})
)
 
# ── DESIGN TOKENS ────────────────────────────────────────────────────────────
BG      = "#0D0D0F"
SURFACE = "#161618"
BORDER  = "#252528"
TEXT    = "#F0EEE8"
MUTED   = "#9CA3AF"
ACCENT  = "#FF5F1F"
GOLD    = "#F5C518"
TEAL    = "#00C9A7"
 
PALETTE = [
    "#FF5F1F", "#F5C518", "#00C9A7", "#818CF8",
    "#F472B6", "#34D399", "#FB923C", "#60A5FA",
    "#A78BFA", "#4ADE80",
]
 
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Georgia, serif", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=False, zeroline=False, color=MUTED,
               tickfont=dict(color=MUTED, size=11)),
    yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
               color=MUTED, tickfont=dict(color=MUTED, size=11)),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER,
                    font=dict(color=TEXT, size=13)),
)
 
# ── HELPERS ──────────────────────────────────────────────────────────────────
def hex_alpha(hex_color, alpha):
    """Convert a hex color + alpha (0-255 int or 0.0-1.0 float) to rgba string."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    if isinstance(alpha, int) and alpha > 1:
        a = round(alpha / 255, 2)
    else:
        a = alpha
    return f"rgba({r},{g},{b},{a})"


def card(children, style=None):
    base = {
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
        "padding": "20px 24px",
        "marginBottom": "0",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)
 
 
def section_label(text):
    return html.P(text, style={
        "fontSize": "11px", "fontWeight": "700",
        "letterSpacing": "0.14em", "textTransform": "uppercase",
        "color": MUTED, "marginBottom": "12px", "marginTop": "0",
    })
 
 
def kpi_card(label, value, icon, delta, color=ACCENT):
    pos = delta.startswith("+")
    arrow = "▲" if pos else "▼"
    delta_color = TEAL if pos else "#F87171"
    return html.Div(style={
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
        "padding": "22px 24px",
        "position": "relative",
        "overflow": "hidden",
        "flex": "1",
    }, children=[
        # Glow blob
        html.Div(style={
            "position": "absolute", "top": "-30px", "right": "-30px",
            "width": "100px", "height": "100px",
            "background": color, "borderRadius": "50%",
            "filter": "blur(50px)", "opacity": "0.08",
        }),
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
            html.Div([
                html.P(label, style={
                    "fontSize": "11px", "color": MUTED, "margin": "0 0 8px 0",
                    "letterSpacing": "0.1em", "textTransform": "uppercase",
                }),
                html.P(value, style={
                    "fontSize": "30px", "fontWeight": "800", "color": TEXT,
                    "margin": "0", "letterSpacing": "-0.02em", "lineHeight": "1",
                }),
                html.P([
                    html.Span(f"{arrow} ", style={"color": delta_color}),
                    html.Span(delta, style={"color": delta_color, "fontWeight": "600"}),
                    html.Span("  vs last period", style={"color": MUTED, "fontWeight": "400", "fontSize": "11px"}),
                ], style={"fontSize": "12px", "marginTop": "10px", "marginBottom": "0"}),
            ]),
            html.Div(icon, style={
                "width": "44px", "height": "44px",
                "background": f"{color}22",
                "borderRadius": "12px",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "fontSize": "20px", "color": color,
                "lineHeight": "44px", "textAlign": "center",
                "flexShrink": "0",
            }),
        ]),
    ])
 
 
# ── CHARTS ───────────────────────────────────────────────────────────────────
def make_rating_dist_chart():
    colors = [hex_alpha(ACCENT, 0.33) if i < 6 else ACCENT for i in range(len(rating_dist))]
    fig = go.Figure(go.Bar(
        x=rating_dist["Rating"].astype(str),
        y=rating_dist["Count"],
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x} ★</b><br>%{y:,} ratings<extra></extra>",
    ))
    fig.update_traces(marker=dict(cornerradius=4))
    fig.update_layout(**PLOT_LAYOUT, height=230)
    return fig
 
 
def make_top_movies_chart():
    df = top_movies.head(8).sort_values("avg")
    fig = go.Figure(go.Bar(
        x=df["avg"],
        y=df["title"].str[:30],
        orientation="h",
        marker_color=PALETTE[:len(df)],
        marker_line_width=0,
        text=df["avg"].round(2),
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{y}</b><br>Avg Rating: %{x:.2f}<extra></extra>",
    ))
    fig.update_traces(marker=dict(cornerradius=4))
    layout = dict(PLOT_LAYOUT)
    layout["xaxis"] = dict(showgrid=False, zeroline=False, range=[4.0, 4.6], color=MUTED)
    layout["yaxis"] = dict(showgrid=False, zeroline=False, color=TEXT, tickfont=dict(color=TEXT, size=11))
    fig.update_layout(**layout, height=280)
    return fig
 
 
def make_yearly_line_chart():
    fig = go.Figure(go.Scatter(
        x=yearly["Year"],
        y=yearly["Count"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(color=ACCENT, size=6, line=dict(width=0)),
        fill="tozeroy",
        fillcolor=hex_alpha(ACCENT, 0.08),
        hovertemplate="<b>%{x}</b><br>%{y:,} ratings<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=200)
    return fig
 
 
def make_genre_bar():
    df = genre_df.sort_values("Count")
    fig = go.Figure(go.Bar(
        x=df["Count"],
        y=df["Genre"],
        orientation="h",
        marker_color=PALETTE[:len(df)],
        marker_line_width=0,
        text=df["Count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{y}</b><br>%{x:,} movies<extra></extra>",
    ))
    fig.update_traces(marker=dict(cornerradius=4))
    layout = dict(PLOT_LAYOUT)
    layout["xaxis"] = dict(showgrid=True, gridcolor=BORDER, zeroline=False, color=MUTED)
    layout["yaxis"] = dict(showgrid=False, zeroline=False, color=TEXT, tickfont=dict(color=TEXT, size=12))
    fig.update_layout(**layout, height=360)
    return fig
 
 
def make_genre_pie():
    fig = go.Figure(go.Pie(
        labels=genre_df["Genre"],
        values=genre_df["Count"],
        hole=0.5,
        marker=dict(colors=PALETTE[:len(genre_df)], line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} movies<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Georgia, serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        height=320,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, size=13)),
    )
    return fig
 
 
def make_radar():
    top6 = genre_df.head(6)
    values = (top6["Count"] / top6["Count"].max() * 100).tolist()
    labels = top6["Genre"].tolist()
    values += [values[0]]  # close the shape
    labels += [labels[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels,
        fill="toself",
        line=dict(color=TEAL, width=2),
        fillcolor=hex_alpha(TEAL, 0.15),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(showticklabels=False, gridcolor=BORDER, linecolor=BORDER),
            angularaxis=dict(color=MUTED, gridcolor=BORDER, tickfont=dict(color=MUTED, size=11)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Georgia, serif"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT)),
    )
    return fig
 
 
def make_yearly_bar():
    peak = yearly["Count"].max()
    colors = [ACCENT if v == peak else hex_alpha(ACCENT, 0.38) for v in yearly["Count"]]
    fig = go.Figure(go.Bar(
        x=yearly["Year"],
        y=yearly["Count"],
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>%{y:,} ratings<extra></extra>",
    ))
    fig.update_traces(marker=dict(cornerradius=5))
    fig.update_layout(**PLOT_LAYOUT, height=320)
    return fig
 
 
# ── PRE-COMPUTE TREND STATS ──────────────────────────────────────────────────
peak_year   = yearly.loc[yearly["Count"].idxmax(), "Year"]
peak_count  = yearly["Count"].max()
quiet_year  = yearly.loc[yearly["Count"].idxmin(), "Year"]
quiet_count = yearly["Count"].min()
year_span   = f"{yearly['Year'].min()} -> {yearly['Year'].max()}"
 
 
# ── TAB CONTENT BUILDERS ─────────────────────────────────────────────────────
def build_overview():
    return html.Div(style={"padding": "24px 28px", "maxWidth": "1400px", "margin": "0 auto"}, children=[
        # KPI row
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "24px"}, children=[
            kpi_card("Total Ratings",  f"{total_ratings:,}",  "★", "+12.4%", ACCENT),
            kpi_card("Unique Users",   f"{unique_users:,}",   "◈", "+8.1%",  TEAL),
            kpi_card("Movies Rated",   f"{unique_movies:,}",  "▣", "+5.3%",  "#818CF8"),
            kpi_card("Avg Rating",     str(avg_rating),       "◎", "+0.2",   GOLD),
        ]),
 
        # Row 2
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"}, children=[
            card([
                section_label("Rating Distribution"),
                dcc.Graph(figure=make_rating_dist_chart(), config={"displayModeBar": False}),
                html.Div(style={"display": "flex", "justifyContent": "space-between", "marginTop": "8px"}, children=[
                    html.Span("1-star ratings: 11,342", style={"fontSize": "11px", "color": MUTED}),
                    html.Span("4-star: most popular at 87,727", style={"fontSize": "11px", "color": ACCENT}),
                ]),
            ]),
            card([
                section_label("Top Rated Movies (>=50 reviews)"),
                dcc.Graph(figure=make_top_movies_chart(), config={"displayModeBar": False}),
            ]),
        ]),
 
        # Row 3 - trend line
        card([
            section_label("Rating Activity by Year"),
            dcc.Graph(figure=make_yearly_line_chart(), config={"displayModeBar": False}),
        ]),
    ])
 
 
def build_genres():
    return html.Div(style={"padding": "24px 28px", "maxWidth": "1400px", "margin": "0 auto"}, children=[
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"}, children=[
            card([
                section_label("Movies per Genre"),
                dcc.Graph(figure=make_genre_bar(), config={"displayModeBar": False}),
            ]),
            card([
                section_label("Genre Radar (Top 6)"),
                dcc.Graph(figure=make_radar(), config={"displayModeBar": False}),
            ]),
        ]),
        card([
            section_label("Genre Share (Top 10)"),
            dcc.Graph(figure=make_genre_pie(), config={"displayModeBar": False}),
        ]),
    ])
 
 
def build_trends():
    return html.Div(style={"padding": "24px 28px", "maxWidth": "1400px", "margin": "0 auto"}, children=[
        card([
            section_label("Annual Rating Volume (1996-2015)"),
            dcc.Graph(figure=make_yearly_bar(), config={"displayModeBar": False}),
            html.P([
                "Peak year: ",
                html.Span(str(peak_year), style={"color": ACCENT, "fontWeight": "700"}),
                f" with {peak_count:,} ratings",
            ], style={"fontSize": "12px", "color": MUTED, "marginTop": "8px", "marginBottom": "0"}),
        ], style={"marginBottom": "16px"}),
 
        # Stat cards
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "16px"}, children=[
            html.Div(style={
                "background": SURFACE, "border": f"1px solid {BORDER}",
                "borderRadius": "16px", "padding": "22px 24px", "textAlign": "center",
            }, children=[
                html.P(str(peak_year), style={"fontSize": "34px", "fontWeight": "900", "color": ACCENT,
                                               "margin": "0", "letterSpacing": "-0.02em"}),
                html.P("Most Active Year", style={"fontSize": "11px", "color": MUTED, "marginTop": "8px",
                                                   "marginBottom": "2px", "fontWeight": "700",
                                                   "letterSpacing": "0.06em", "textTransform": "uppercase"}),
                html.P(f"{peak_count:,} ratings", style={"fontSize": "11px", "color": MUTED, "margin": "0"}),
            ]),
            html.Div(style={
                "background": SURFACE, "border": f"1px solid {BORDER}",
                "borderRadius": "16px", "padding": "22px 24px", "textAlign": "center",
            }, children=[
                html.P(str(quiet_year), style={"fontSize": "34px", "fontWeight": "900", "color": TEAL,
                                                "margin": "0", "letterSpacing": "-0.02em"}),
                html.P("Quietest Year", style={"fontSize": "11px", "color": MUTED, "marginTop": "8px",
                                                "marginBottom": "2px", "fontWeight": "700",
                                                "letterSpacing": "0.06em", "textTransform": "uppercase"}),
                html.P(f"{quiet_count:,} ratings", style={"fontSize": "11px", "color": MUTED, "margin": "0"}),
            ]),
            html.Div(style={
                "background": SURFACE, "border": f"1px solid {BORDER}",
                "borderRadius": "16px", "padding": "22px 24px", "textAlign": "center",
            }, children=[
                html.P(f"{yearly['Year'].max() - yearly['Year'].min()} yrs",
                       style={"fontSize": "34px", "fontWeight": "900", "color": "#818CF8",
                              "margin": "0", "letterSpacing": "-0.02em"}),
                html.P("Rating Span", style={"fontSize": "11px", "color": MUTED, "marginTop": "8px",
                                              "marginBottom": "2px", "fontWeight": "700",
                                              "letterSpacing": "0.06em", "textTransform": "uppercase"}),
                html.P(year_span, style={"fontSize": "11px", "color": MUTED, "margin": "0"}),
            ]),
        ]),
    ])
 
 
# ── LAYOUT ────────────────────────────────────────────────────────────────────
TAB_STYLE = {
    "color": MUTED, "background": "transparent", "border": "none",
    "fontSize": "12px", "fontWeight": "600", "padding": "6px 16px",
    "letterSpacing": "0.04em", "fontFamily": "Georgia, serif",
}
TAB_SELECTED_STYLE = {
    "color": "#fff", "background": ACCENT, "border": "none",
    "borderRadius": "8px", "fontSize": "12px", "fontWeight": "600",
    "padding": "6px 16px", "letterSpacing": "0.04em",
    "fontFamily": "Georgia, serif",
}
 
app = dash.Dash(__name__, title="MovieLens Analytics")
app.layout = html.Div(style={
    "background": BG, "minHeight": "100vh",
    "color": TEXT, "fontFamily": "Georgia, 'Times New Roman', serif",
    "fontSize": "14px",
}, children=[
 
    # ── HEADER ──
    html.Div(style={
        "borderBottom": f"1px solid {BORDER}",
        "padding": "0 28px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "height": "60px",
        "position": "sticky",
        "top": "0",
        "background": BG,
        "backdropFilter": "blur(12px)",
        "zIndex": "10",
    }, children=[
        # Logo + title
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
            html.Div("M", style={
                "width": "32px", "height": "32px", "background": ACCENT,
                "borderRadius": "8px", "display": "flex", "alignItems": "center",
                "justifyContent": "center", "fontSize": "16px", "fontWeight": "900",
                "color": "#fff", "lineHeight": "32px", "textAlign": "center",
            }),
            html.Div([
                html.Div("MovieLens Analytics", style={
                    "fontWeight": "800", "fontSize": "15px",
                    "letterSpacing": "-0.01em", "color": TEXT,
                }),
                html.Div("INTELLIGENCE DASHBOARD", style={
                    "fontSize": "10px", "color": MUTED, "letterSpacing": "0.08em",
                }),
            ]),
        ]),
 
        # Tabs - content embedded directly in each Tab
        dcc.Tabs(id="tabs", value="overview",
                 style={"border": "none", "height": "36px"},
                 children=[
                     dcc.Tab(label="Overview", value="overview",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                     dcc.Tab(label="Genres", value="genres",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                     dcc.Tab(label="Trends", value="trends",
                             style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                 ]),
 
        # Dataset badge
        html.Div("Dataset - 316K ratings", style={
            "fontSize": "11px", "color": MUTED,
            "background": BORDER,
            "padding": "6px 12px", "borderRadius": "8px",
        }),
    ]),
 
    # ── TAB CONTENT (all 3 panels, visibility toggled by callback) ──
    html.Div(id="panel-overview", children=[build_overview()]),
    html.Div(id="panel-genres",   children=[build_genres()],  style={"display": "none"}),
    html.Div(id="panel-trends",   children=[build_trends()],  style={"display": "none"}),
 
    # ── FOOTER ──
    html.Div(style={
        "padding": "16px 28px",
        "borderTop": f"1px solid {BORDER}",
        "display": "flex", "justifyContent": "space-between",
        "fontSize": "11px", "color": MUTED,
        "maxWidth": "1400px", "margin": "0 auto",
    }, children=[
        html.Span("MovieLens Dataset - 316,499 ratings - 11,112 movies - 2,161 users"),
        html.Span("Built with Plotly Dash - Python"),
    ]),
])
 
 
# ── CALLBACK (toggle panel visibility) ───────────────────────────────────────
@app.callback(
    Output("panel-overview", "style"),
    Output("panel-genres",   "style"),
    Output("panel-trends",   "style"),
    Input("tabs", "value"),
)
def toggle_panels(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if tab == "overview" else hide,
        show if tab == "genres"   else hide,
        show if tab == "trends"   else hide,
    )
 
 
# ── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n>>  MovieLens Dashboard running at  ->  http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)