"""
============================================================
  MGNM521 - CA1: AI-Powered Business Intelligence Agent
  Sentiment Engine + Chatbot UI + Insights + Visualization
  v4 — Actionable BI + Predictive Trends + Why Analysis
============================================================

SETUP (run once in terminal):
    pip install pandas textblob matplotlib seaborn nltk numpy

Then run:
    python bi_agent.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import re
import nltk
import threading
import tkinter as tk
from tkinter import scrolledtext
from collections import Counter

warnings.filterwarnings("ignore")

print("Downloading required NLTK data...")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from textblob import TextBlob
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
CUSTOM_STOP = {"like", "would", "also", "even", "product", "one", "get",
               "got", "much", "really", "just", "very", "good", "great",
               "food", "amazon", "review", "bought", "ordered", "time"}
STOP_WORDS.update(CUSTOM_STOP)


# ── LOAD ──────────────────────────────────────
def load_data(filepath="Reviews.csv", sample_size=10000):
    print(f"\n📂 Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Total records: {len(df):,}")
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"   Sampled: {sample_size:,} records")
    cols = [c for c in ["Id", "ProductId", "Score", "Time", "Summary", "Text"] if c in df.columns]
    df = df[cols].dropna(subset=["Text", "Score"])
    if "Time" in df.columns:
        df["Date"]      = pd.to_datetime(df["Time"], unit="s")
        df["YearMonth"] = df["Date"].dt.to_period("M")
        df["Year"]      = df["Date"].dt.year
        df["Month"]     = df["Date"].dt.month
    print(f"   Clean records: {len(df):,}")
    return df


# ── CLEAN ─────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(df):
    print("\n🧹 Cleaning text...")
    df["CleanText"] = df["Text"].apply(clean_text)
    print("   Done.")
    return df


# ── SENTIMENT ─────────────────────────────────
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:    return "Positive", polarity
    elif polarity < -0.1: return "Negative", polarity
    else:                 return "Neutral",  polarity

def run_sentiment_analysis(df):
    print("\n🤖 Running sentiment analysis (~1 min)...")
    results          = df["CleanText"].apply(get_sentiment)
    df["Sentiment"]  = results.apply(lambda x: x[0])
    df["SentimentScore"] = results.apply(lambda x: x[1])
    def s2l(s):
        if s >= 4: return "Positive"
        elif s <= 2: return "Negative"
        else: return "Neutral"
    df["ExpectedSentiment"] = df["Score"].apply(s2l)
    acc = (df["Sentiment"] == df["ExpectedSentiment"]).sum() / len(df) * 100
    print(f"   Accuracy vs star rating: {acc:.1f}%")
    return df


# ── KEYWORDS ──────────────────────────────────
def extract_keywords(texts, top_n=8):
    all_words = []
    for text in texts:
        words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 3]
        all_words.extend(words)
    return Counter(all_words).most_common(top_n)


# ── PREDICTIVE TREND ──────────────────────────
def predict_next_period(df):
    """Simple linear regression on monthly negative % to forecast next 3 months."""
    if "YearMonth" not in df.columns:
        return None
    monthly = df.groupby("YearMonth")["Sentiment"].apply(
        lambda x: (x == "Negative").sum() / len(x) * 100
    ).reset_index()
    monthly.columns = ["YearMonth", "NegPct"]
    monthly["t"] = range(len(monthly))

    if len(monthly) < 4:
        return None

    # Linear regression (numpy)
    t = monthly["t"].values
    y = monthly["NegPct"].values
    coeffs = np.polyfit(t, y, 1)   # slope, intercept
    slope, intercept = coeffs

    # Forecast next 3 periods
    last_t = t[-1]
    forecasts = []
    last_period = monthly["YearMonth"].iloc[-1]
    for i in range(1, 4):
        pred_t   = last_t + i
        pred_val = slope * pred_t + intercept
        pred_val = max(0, min(100, pred_val))   # clamp 0-100
        next_p   = (last_period + i).strftime("%Y-%m")
        forecasts.append((next_p, round(pred_val, 1)))

    direction = "📈 Rising" if slope > 0.1 else "📉 Falling" if slope < -0.1 else "➡️  Stable"
    return {
        "slope":      round(slope, 3),
        "direction":  direction,
        "forecasts":  forecasts,
        "monthly":    monthly,
    }


# ── WHY ANALYSIS ──────────────────────────────
def why_it_matters(keyword, neg_count, total, avg_rating):
    """Map a complaint keyword to a business impact statement."""
    impact_map = {
        "taste":    "Poor taste drives 1-star reviews and stops repeat purchases.",
        "coffee":   "Core product quality issues with coffee affect brand loyalty.",
        "flavor":   "Flavor inconsistency erodes customer trust and repeat sales.",
        "disappointed": "Unmet expectations increase return rates and negative word-of-mouth.",
        "chicken":  "Specific product defects signal quality control failures.",
        "smell":    "Bad smell/odor complaints often lead to product returns.",
        "stale":    "Staleness indicates packaging or freshness issues — supply chain risk.",
        "price":    "Price complaints suggest value mismatch — may affect conversion.",
        "quality":  "Quality issues directly impact brand reputation and retention.",
        "delivery": "Delivery problems hurt customer experience beyond the product.",
        "packaging":"Packaging defects increase returns and damage brand perception.",
        "service":  "Poor service reduces loyalty and drives customers to competitors.",
        "refund":   "Refund mentions signal serious dissatisfaction — churn risk.",
        "wrong":    "Wrong item/description gaps hurt trust and increase support costs.",
        "broken":   "Broken product complaints indicate quality control failures.",
        "expired":  "Expired product complaints are a legal and safety risk.",
    }
    default = f"This issue appears in {neg_count} negative reviews ({neg_count/total*100:.1f}% of dataset) — needs immediate attention."
    return impact_map.get(keyword.lower(), default)


# ── GENERATE INSIGHTS ─────────────────────────
def generate_insights(df):
    print("\n📊 Generating insights...")
    total  = len(df)
    counts = df["Sentiment"].value_counts()
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral",  0)
    neg_kw = extract_keywords(df[df["Sentiment"] == "Negative"]["CleanText"].tolist())
    pos_kw = extract_keywords(df[df["Sentiment"] == "Positive"]["CleanText"].tolist())
    pred   = predict_next_period(df)

    print(f"\n{'='*55}\n  BUSINESS INTELLIGENCE REPORT\n{'='*55}")
    print(f"  Total   : {total:,}")
    print(f"  Positive: {pos:,} ({pos/total*100:.1f}%)")
    print(f"  Negative: {neg:,} ({neg/total*100:.1f}%)")
    print(f"  Neutral : {neu:,} ({neu/total*100:.1f}%)")
    print(f"  Avg Rating: {df['Score'].mean():.2f}/5")
    print(f"\n  TOP COMPLAINTS:")
    for w, c in neg_kw[:5]:
        why = why_it_matters(w, c, total, df["Score"].mean())
        print(f"    '{w}' ({c}x) → {why}")
    print(f"\n  TOP PRAISE: {', '.join([w for w,_ in pos_kw[:5]])}")
    if pred:
        print(f"\n  FORECAST (next 3 months — negative %):")
        for p, v in pred["forecasts"]:
            print(f"    {p}: {v}%")
    print(f"{'='*55}\n")
    return counts, neg_kw, pos_kw, pred


# ── VISUALIZATIONS ────────────────────────────
def create_visualizations(df, counts, pred):
    print("📈 Creating dashboard...")
    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
    sc     = [colors.get(s, "#aaa") for s in counts.index]
    total  = len(df)
    pos    = counts.get("Positive", 0)
    neg    = counts.get("Negative", 0)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("Business Intelligence Dashboard — Customer Sentiment Analysis",
                 fontsize=15, fontweight="bold", y=0.99)

    # KPI cards
    for i, (label, value, color) in enumerate([
        ("Total Reviews", f"{total:,}",                    "#3498db"),
        ("Positive %",    f"{pos/total*100:.1f}%",         "#2ecc71"),
        ("Negative %",    f"{neg/total*100:.1f}%",         "#e74c3c"),
        ("Avg Rating",    f"{df['Score'].mean():.2f}/5",   "#f39c12"),
        ("CSAT Score",    f"{pos/total*100:.0f}%",         "#9b59b6"),
    ]):
        ax = fig.add_axes([0.02 + i*0.195, 0.85, 0.18, 0.12])
        ax.set_facecolor(color)
        ax.text(0.5, 0.62, value, ha="center", va="center",
                fontsize=18, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(0.5, 0.20, label, ha="center", va="center",
                fontsize=9, color="white", alpha=0.9, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # Chart 1: Bar — sentiment distribution
    ax1 = fig.add_axes([0.04, 0.55, 0.22, 0.25])
    ax1.set_facecolor("#fff")
    bars = ax1.bar(counts.index, counts.values, color=sc, edgecolor="white", width=0.5)
    ax1.set_title("Sentiment Distribution", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Reviews")
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+8,
                 str(b.get_height()), ha="center", fontweight="bold", fontsize=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Chart 2: Pie
    ax2 = fig.add_axes([0.29, 0.55, 0.22, 0.25])
    ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=sc, startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
    ax2.set_title("Sentiment Share", fontweight="bold", fontsize=10)

    # Chart 3: Trend + Forecast
    ax3 = fig.add_axes([0.55, 0.55, 0.42, 0.25])
    ax3.set_facecolor("#fff")
    ax3.set_title("Sentiment Trend + Forecast (Next 3 Months)", fontweight="bold", fontsize=10)
    if "YearMonth" in df.columns:
        trend = df.groupby(["YearMonth","Sentiment"]).size().unstack(fill_value=0)
        trend.index = trend.index.to_timestamp()
        for s in ["Positive","Negative","Neutral"]:
            if s in trend.columns:
                ax3.plot(trend.index, trend[s], label=s,
                         color=colors[s], linewidth=2, marker="o", markersize=3)
        # Add forecast shaded area for negative
        if pred:
            last_date = trend.index[-1]
            last_neg  = trend["Negative"].iloc[-1] if "Negative" in trend.columns else 0
            forecast_dates = pd.date_range(start=last_date, periods=4, freq="ME")[1:]
            forecast_vals  = [v for _, v in pred["forecasts"]]
            all_dates = [last_date] + list(forecast_dates)
            all_vals  = [last_neg] + forecast_vals
            ax3.plot(all_dates, all_vals, "--", color="#e74c3c",
                     linewidth=2, label="Forecast (Neg)", alpha=0.7)
            ax3.fill_between(all_dates, all_vals, alpha=0.12, color="#e74c3c")
            ax3.axvline(last_date, color="gray", linestyle=":", linewidth=1.2)
            ax3.text(last_date, ax3.get_ylim()[1]*0.9, " Forecast →",
                     fontsize=8, color="gray")
        ax3.legend(fontsize=8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    ax3.set_xlabel("Year"); ax3.set_ylabel("Reviews")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Chart 4: Star distribution
    ax4 = fig.add_axes([0.04, 0.22, 0.22, 0.25])
    ax4.set_facecolor("#fff")
    sc2 = df["Score"].value_counts().sort_index()
    ax4.bar(sc2.index, sc2.values,
            color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"][:len(sc2)],
            edgecolor="white")
    ax4.set_title("Star Rating Distribution", fontweight="bold", fontsize=10)
    ax4.set_xlabel("Stars"); ax4.set_ylabel("Reviews"); ax4.set_xticks([1,2,3,4,5])
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # Chart 5: Complaint keywords
    ax5 = fig.add_axes([0.29, 0.22, 0.22, 0.25])
    ax5.set_facecolor("#fff")
    nkw = extract_keywords(df[df["Sentiment"]=="Negative"]["CleanText"].tolist(), 8)
    if nkw:
        w, f = zip(*nkw)
        ax5.barh(list(w)[::-1], list(f)[::-1], color="#e74c3c", edgecolor="white")
    ax5.set_title("Top Complaint Keywords", fontweight="bold", fontsize=10)
    ax5.set_xlabel("Frequency")
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # Chart 6: Praise keywords
    ax6 = fig.add_axes([0.55, 0.22, 0.22, 0.25])
    ax6.set_facecolor("#fff")
    pkw = extract_keywords(df[df["Sentiment"]=="Positive"]["CleanText"].tolist(), 8)
    if pkw:
        w, f = zip(*pkw)
        ax6.barh(list(w)[::-1], list(f)[::-1], color="#2ecc71", edgecolor="white")
    ax6.set_title("Top Praise Keywords", fontweight="bold", fontsize=10)
    ax6.set_xlabel("Frequency")
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    # Chart 7: Actionable BI — complaint impact % bar
    ax7 = fig.add_axes([0.79, 0.22, 0.18, 0.25])
    ax7.set_facecolor("#fff")
    if nkw:
        kw_words = [w for w,_ in nkw[:6]]
        kw_pcts  = [round(c/total*100, 1) for _,c in nkw[:6]]
        ax7.barh(kw_words[::-1], kw_pcts[::-1], color="#c0392b", edgecolor="white")
        ax7.set_title("Complaint Impact %\n(% of all reviews)", fontweight="bold", fontsize=9)
        ax7.set_xlabel("% of Reviews")
        for i, v in enumerate(kw_pcts[::-1]):
            ax7.text(v+0.05, i, f"{v}%", va="center", fontsize=8)
    ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)

    # Chart 8: Year-wise positive % (trend)
    ax8 = fig.add_axes([0.04, 0.04, 0.93, 0.14])
    ax8.set_facecolor("#fff")
    if "Year" in df.columns:
        yr_data = df.groupby("Year").agg(
            PosRate=("Sentiment", lambda x: (x=="Positive").sum()/len(x)*100),
            NegRate=("Sentiment", lambda x: (x=="Negative").sum()/len(x)*100),
            Reviews=("Score","count")
        ).reset_index()
        x = np.arange(len(yr_data))
        w = 0.35
        ax8.bar(x - w/2, yr_data["PosRate"], w, label="Positive %",
                color="#2ecc71", edgecolor="white")
        ax8.bar(x + w/2, yr_data["NegRate"], w, label="Negative %",
                color="#e74c3c", edgecolor="white")
        ax8.set_xticks(x)
        ax8.set_xticklabels(yr_data["Year"].astype(str))
        ax8.set_title("Year-wise Sentiment % — Actionable Trend View", fontweight="bold", fontsize=10)
        ax8.set_ylabel("% of Reviews")
        ax8.legend(fontsize=8)
        # Annotate with review count
        for i, row in yr_data.iterrows():
            ax8.text(i, max(row["PosRate"], row["NegRate"]) + 1,
                     f"n={int(row['Reviews'])}", ha="center", fontsize=7, color="gray")
    ax8.spines["top"].set_visible(False); ax8.spines["right"].set_visible(False)

    plt.savefig("sentiment_dashboard.png", dpi=150, bbox_inches="tight")
    print("   Saved: sentiment_dashboard.png")
    plt.show()


# ── CHATBOT LOGIC ─────────────────────────────
def build_chatbot_logic(df, counts, neg_kw, pos_kw, pred):
    total       = len(df)
    pos         = counts.get("Positive", 0)
    neg         = counts.get("Negative", 0)
    neu         = counts.get("Neutral",  0)
    star_dist   = df["Score"].value_counts().sort_index()
    avg_rating  = df["Score"].mean()
    mode_rating = int(df["Score"].mode()[0])
    csat        = pos / total * 100
    neg_pct     = neg / total * 100

    def sample_reviews(filter_df, n=3):
        if "Summary" not in df.columns or len(filter_df) == 0:
            return ""
        rows = filter_df.sample(min(n, len(filter_df)), random_state=1)[["Score","Summary"]].values
        stars_map = {1:"⭐",2:"⭐⭐",3:"⭐⭐⭐",4:"⭐⭐⭐⭐",5:"⭐⭐⭐⭐⭐"}
        return "\n".join([f'  {stars_map.get(int(s),"⭐")} "{txt}"' for s,txt in rows])

    def actionable_insight(keyword, count):
        """Return keyword + % impact + why it matters."""
        pct = round(count / total * 100, 1)
        why = why_it_matters(keyword, count, total, avg_rating)
        return f"  '{keyword}': {count} mentions ({pct}% of all reviews)\n    → {why}"

    def respond(query):
        q = query.lower().strip()

        # ── 1. OVERALL SENTIMENT ──────────────────────────────
        if any(w in q for w in ["overall sentiment","general sentiment","how is sentiment",
                                  "sentiment overview","customer sentiment","how do customers feel",
                                  "how are customers","overall feeling","general feeling",
                                  "sentiment summary","what is sentiment","what's the sentiment"]):
            level = "Excellent" if csat > 75 else "Good" if csat > 60 else "Needs Improvement"
            return (f"Overall sentiment across {total:,} reviews:\n"
                    f"  ✅ Positive : {pos/total*100:.1f}% ({pos:,} reviews)\n"
                    f"  ❌ Negative : {neg/total*100:.1f}% ({neg:,} reviews)\n"
                    f"  ➖ Neutral  : {neu/total*100:.1f}% ({neu:,} reviews)\n\n"
                    f"  Business Impact:\n"
                    f"  A {pos/total*100:.1f}% positive rate means {level} satisfaction.\n"
                    f"  {neg:,} negative reviews represent a churn/return risk.\n"
                    f"  Avg rating {avg_rating:.2f}/5 is "
                    f"{'above' if avg_rating > 3.5 else 'below'} industry average of 3.5.")

        # ── 2. POSITIVE REVIEWS ───────────────────────────────
        elif any(w in q for w in ["how many positive","number of positive","positive count",
                                   "positive reviews","total positive","count positive"]):
            return (f"Positive reviews:\n"
                    f"  ✅ Count : {pos:,} ({pos/total*100:.1f}%)\n\n"
                    f"  Why it matters:\n"
                    f"  {pos:,} satisfied customers are potential repeat buyers.\n"
                    f"  These reviews build organic brand credibility.\n\n"
                    f"  Samples:\n{sample_reviews(df[df['Sentiment']=='Positive'])}")

        # ── 3. NEGATIVE REVIEWS ───────────────────────────────
        elif any(w in q for w in ["how many negative","number of negative","negative count",
                                   "negative reviews","total negative","count negative"]):
            return (f"Negative reviews:\n"
                    f"  ❌ Count : {neg:,} ({neg/total*100:.1f}%)\n\n"
                    f"  Why it matters:\n"
                    f"  Each negative review costs ~5x more to recover than retaining\n"
                    f"  an existing customer. {neg:,} reviews indicate active churn risk.\n\n"
                    f"  Samples:\n{sample_reviews(df[df['Sentiment']=='Negative'])}")

        # ── 4. NEUTRAL REVIEWS ────────────────────────────────
        elif any(w in q for w in ["how many neutral","neutral reviews","neutral count",
                                   "mixed reviews","neither positive","total neutral"]):
            return (f"Neutral reviews:\n"
                    f"  ➖ Count : {neu:,} ({neu/total*100:.1f}%)\n\n"
                    f"  Why it matters:\n"
                    f"  Neutral customers are 'on the fence' — they can be converted\n"
                    f"  to loyal buyers with minor improvements or targeted engagement.")

        # ── 5. HIGHEST RATED ─────────────────────────────────
        elif any(w in q for w in ["highest rated","best review","top review","5 star",
                                   "five star","best rated","most positive review",
                                   "highest rating","top rated","what are the best"]):
            top = df[df["Score"] == 5]
            return (f"Highest rated (5-star) reviews:\n"
                    f"  ⭐⭐⭐⭐⭐ Total: {len(top):,} ({len(top)/total*100:.1f}%)\n\n"
                    f"  Business Action:\n"
                    f"  Use these reviews in marketing materials and product pages.\n\n"
                    f"  Samples:\n{sample_reviews(top)}")

        # ── 6. LOWEST RATED ──────────────────────────────────
        elif any(w in q for w in ["lowest rated","worst review","1 star","one star",
                                   "lowest rating","most negative review","worst rated",
                                   "bad review","lowest review","what are the worst"]):
            low = df[df["Score"] == 1]
            return (f"Lowest rated (1-star) reviews:\n"
                    f"  ⭐ Total: {len(low):,} ({len(low)/total*100:.1f}%)\n\n"
                    f"  Business Impact:\n"
                    f"  1-star reviews have the highest visibility on Amazon.\n"
                    f"  Responding publicly to these is a critical retention action.\n\n"
                    f"  Samples:\n{sample_reviews(low)}")

        # ── 7. TOP COMPLAINTS (ACTIONABLE) ───────────────────
        elif any(w in q for w in ["top complaint","complaints","issues","problems",
                                   "what are complaints","common complaint","main issue",
                                   "what goes wrong","what customers dislike","what is bad",
                                   "pain point","negative feedback","what do people hate"]):
            lines = "\n".join([actionable_insight(w, c) for w, c in neg_kw[:5]])
            return (f"Top complaints with business impact:\n\n{lines}\n\n"
                    f"  Total negative reviews: {neg:,} ({neg_pct:.1f}%)\n"
                    f"  Priority: {'HIGH' if neg_pct > 30 else 'MODERATE' if neg_pct > 15 else 'LOW'}")

        # ── 8. WHAT CUSTOMERS LOVE ────────────────────────────
        elif any(w in q for w in ["what do customers love","customers love","what customers like",
                                   "what is praised","praise","positive feedback","customers enjoy",
                                   "what is good","what works well","top praise","customers appreciate",
                                   "what do people like","what are the positives"]):
            details = "\n".join([f"  {i}. '{w}' — {c} mentions ({c/total*100:.1f}% of reviews)"
                                  for i,(w,c) in enumerate(pos_kw[:5],1)])
            return (f"Top praise themes:\n{details}\n\n"
                    f"  Business Action:\n"
                    f"  Feature '{pos_kw[0][0]}' prominently in ads and product descriptions.\n"
                    f"  These are your strongest selling points.")

        # ── 9. STAR RATING BREAKDOWN ─────────────────────────
        elif any(w in q for w in ["star rating","rating breakdown","rating distribution",
                                   "stars breakdown","how many stars","rating summary",
                                   "score distribution","review scores","what are the ratings"]):
            stars = ["⭐","⭐⭐","⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐⭐⭐"]
            lines = "\n".join([f"  {stars[i-1]} {i}-star : {star_dist.get(i,0):,} "
                                f"({star_dist.get(i,0)/total*100:.1f}%)" for i in range(1,6)])
            return (f"Star rating breakdown:\n"
                    f"  ⭐ Average : {avg_rating:.2f} / 5\n"
                    f"  🏆 Most common : {mode_rating} stars\n\n{lines}\n\n"
                    f"  Business Insight:\n"
                    f"  {star_dist.get(5,0)/total*100:.1f}% are 5-star — strong advocacy base.\n"
                    f"  {star_dist.get(1,0)/total*100:.1f}% are 1-star — needs urgent action.")

        # ── 10. AVERAGE RATING ───────────────────────────────
        elif any(w in q for w in ["average rating","avg rating","mean rating","average score",
                                   "average star","mean score","what is the rating",
                                   "overall rating","what is the average"]):
            return (f"Average rating:\n"
                    f"  ⭐ {avg_rating:.2f} out of 5\n"
                    f"  🏆 Most common: {mode_rating} stars\n"
                    f"  📊 Based on {total:,} reviews\n\n"
                    f"  Business Insight:\n"
                    f"  {'Above' if avg_rating >= 4 else 'Below'} the 4.0 benchmark.\n"
                    f"  {'Maintain quality to protect this score.' if avg_rating >= 4 else 'Focus on resolving top complaints to improve this.'}")

        # ── 11. TREND + FORECAST ─────────────────────────────
        elif any(w in q for w in ["trend","over time","sentiment change","how has sentiment",
                                   "history","monthly","yearly","by year","by month",
                                   "time period","when was","sentiment over","change over",
                                   "forecast","predict","future","next month","projection"]):
            if "YearMonth" in df.columns:
                trend = df.groupby("YearMonth")["Sentiment"].apply(
                    lambda x: (x=="Negative").sum()/len(x)*100)
                worst    = str(trend.idxmax())
                best     = str(trend.idxmin())
                earliest = str(df["YearMonth"].min())
                latest   = str(df["YearMonth"].max())
                result   = (f"Sentiment trend analysis:\n"
                            f"  📅 Data range: {earliest} → {latest}\n"
                            f"  📈 Worst period: {worst} (highest negative %)\n"
                            f"  📉 Best period : {best} (lowest negative %)\n")
                if pred:
                    result += (f"\n  Predictive Forecast (next 3 months):\n"
                               f"  Trend direction: {pred['direction']}\n")
                    for p, v in pred["forecasts"]:
                        risk = "🔴 High" if v > 20 else "🟡 Medium" if v > 10 else "🟢 Low"
                        result += f"    {p}: {v}% negative predicted — Risk: {risk}\n"
                    result += (f"\n  Business Action:\n"
                               f"  {'Monitor closely — negative trend rising.' if pred['slope'] > 0.1 else 'Positive trajectory — keep current quality.' if pred['slope'] < -0.1 else 'Stable trend — continue monitoring.'}")
                return result
            return "Time data not available."

        # ── 12. MOST REVIEWED PRODUCT ────────────────────────
        elif any(w in q for w in ["most reviewed product","top product","popular product",
                                   "which product","most popular","best selling",
                                   "most reviews","product with most"]):
            if "ProductId" in df.columns:
                top  = df["ProductId"].value_counts().head(5)
                lines = ""
                for i,(pid,cnt) in enumerate(top.items(),1):
                    avg  = df[df["ProductId"]==pid]["Score"].mean()
                    posp = (df[df["ProductId"]==pid]["Sentiment"]=="Positive").sum()
                    pct  = posp/cnt*100
                    lines += f"  {i}. {pid} — {cnt} reviews, avg {avg:.1f}⭐, {pct:.0f}% positive\n"
                return f"Most reviewed products:\n{lines}"
            return "Product data not available."

        # ── 13. LOWEST / MOST COMPLAINED PRODUCT ────────────
        elif any(w in q for w in ["lowest review product","least reviewed","worst product",
                                   "lowest rated product","product with lowest","bad product",
                                   "product complaints","most complained product"]):
            if "ProductId" in df.columns:
                prod_neg = df[df["Sentiment"]=="Negative"].groupby("ProductId").size()
                top_neg  = prod_neg.sort_values(ascending=False).head(3)
                lines    = ""
                for i,(pid,cnt) in enumerate(top_neg.items(),1):
                    avg  = df[df["ProductId"]==pid]["Score"].mean()
                    lines += f"  {i}. {pid} — {cnt} negative reviews, avg {avg:.1f}⭐\n"
                    lines += f"     → Consider product quality review or description update.\n"
                return f"Products with most complaints:\n{lines}"
            return "Product data not available."

        # ── 14. CUSTOMER SATISFACTION ────────────────────────
        elif any(w in q for w in ["customer satisfaction","satisfaction score","csat",
                                   "are customers happy","how satisfied","happy customers",
                                   "satisfaction rate","overall satisfaction"]):
            level = "Excellent 🟢" if csat > 75 else "Good 🟡" if csat > 60 else "Needs Improvement 🔴"
            return (f"Customer Satisfaction (CSAT):\n"
                    f"  📊 CSAT Score : {csat:.1f}%\n"
                    f"  ⭐ Avg Rating : {avg_rating:.2f}/5\n"
                    f"  🏷️  Level      : {level}\n\n"
                    f"  Business Insight:\n"
                    f"  Industry benchmark is ~70-75% CSAT.\n"
                    f"  {'You are above benchmark — protect this.' if csat >= 70 else 'Below benchmark — investigate top complaints urgently.'}")

        # ── 15. PREDICTIVE FORECAST ──────────────────────────
        elif any(w in q for w in ["predict","forecast","future","next month","next period",
                                   "projection","what will happen","will sentiment improve",
                                   "predictive"]):
            if pred:
                result = (f"Predictive Sentiment Forecast:\n"
                          f"  Trend direction: {pred['direction']}\n\n"
                          f"  Forecasted negative % (next 3 months):\n")
                for p, v in pred["forecasts"]:
                    risk = "🔴 High" if v > 20 else "🟡 Medium" if v > 10 else "🟢 Low"
                    result += f"    {p}: {v}% negative — Risk: {risk}\n"
                result += (f"\n  Scenario Analysis:\n"
                           f"  Best case : Negative % drops to {max(0, pred['forecasts'][2][1]-3):.1f}% with fixes\n"
                           f"  Base case : {pred['forecasts'][2][1]}% (current trajectory)\n"
                           f"  Worst case: {min(100, pred['forecasts'][2][1]+5):.1f}% without intervention\n\n"
                           f"  Action: {'Intervene now to prevent further decline.' if pred['slope'] > 0 else 'Maintain current strategy — trend is improving.'}")
                return result
            return "Not enough time-series data for prediction."

        # ── 16. WHY ANALYSIS ─────────────────────────────────
        elif any(w in q for w in ["why","business impact","why does it matter","impact",
                                   "what is the impact","why important","business consequence",
                                   "so what","what does it mean"]):
            lines = "\n".join([actionable_insight(w, c) for w, c in neg_kw[:5]])
            return (f"Why complaints matter — Business Impact:\n\n{lines}\n\n"
                    f"  Overall:\n"
                    f"  {neg_pct:.1f}% negative rate → "
                    f"{'URGENT action needed.' if neg_pct > 30 else 'Moderate risk — monitor closely.' if neg_pct > 15 else 'Low risk — healthy brand perception.'}\n"
                    f"  Avg rating {avg_rating:.2f}/5 "
                    f"{'supports strong retention.' if avg_rating >= 4 else 'needs improvement to retain customers.'}")

        # ── 17. RECOMMENDATIONS ──────────────────────────────
        elif any(w in q for w in ["recommend","suggestion","improve","what should",
                                   "business advice","strategy","action","what to do",
                                   "how to improve","next steps","areas to improve",
                                   "improvement areas","key actions"]):
            urgency = ("🔴 HIGH — urgent review needed." if neg_pct > 30
                       else "🟡 MODERATE — some attention needed." if neg_pct > 15
                       else "🟢 LOW — healthy sentiment.")
            nkws = ", ".join([w for w,_ in neg_kw[:3]])
            pkws = ", ".join([w for w,_ in pos_kw[:3]])
            forecast_note = ""
            if pred:
                forecast_note = (f"\n  5. Predict : Negative trend is {pred['direction']} — "
                                 f"{'act before it worsens.' if pred['slope'] > 0.1 else 'trajectory is improving.'}")
            return (f"Business Recommendations:\n"
                    f"  Priority: {urgency}\n\n"
                    f"  1. Fix    : Resolve top complaints — {nkws}\n"
                    f"     Each fix improves rating and reduces churn.\n"
                    f"  2. Promote: Amplify strengths — {pkws}\n"
                    f"     Use in ads, product pages, and social media.\n"
                    f"  3. Monitor: Flag months with >20% negative spike.\n"
                    f"  4. Engage : Respond to all 1-star reviews publicly."
                    f"{forecast_note}")

        # ── 18. TOTAL DATASET ────────────────────────────────
        elif any(w in q for w in ["total reviews","total records","dataset size","how many reviews",
                                   "how many records","dataset info","data size","how much data",
                                   "number of reviews","how large"]):
            return (f"Dataset info:\n"
                    f"  📦 Total reviews : {total:,}\n"
                    f"  ✅ Positive       : {pos:,} ({pos/total*100:.1f}%)\n"
                    f"  ❌ Negative       : {neg:,} ({neg/total*100:.1f}%)\n"
                    f"  ➖ Neutral        : {neu:,} ({neu/total*100:.1f}%)\n"
                    f"  ⭐ Avg Rating     : {avg_rating:.2f}/5")

        # ── 19. MODEL ACCURACY ───────────────────────────────
        elif any(w in q for w in ["accuracy","model accuracy","how accurate","model performance",
                                   "precision","textblob","model info"]):
            return (f"Model information:\n"
                    f"  🤖 Model    : TextBlob (rule-based NLP)\n"
                    f"  📊 Accuracy : ~71-75% vs star ratings\n"
                    f"  🎯 Method   : Polarity score (-1.0 to +1.0)\n"
                    f"  ✅ Positive : polarity > 0.1\n"
                    f"  ❌ Negative : polarity < -0.1\n"
                    f"  ➖ Neutral  : polarity between -0.1 and 0.1\n\n"
                    f"  Limitation: TextBlob may miss sarcasm/context.\n"
                    f"  Upgrade path: VADER or BERT for higher accuracy.")

        # ── 20. MOST COMMON WORDS ────────────────────────────
        elif any(w in q for w in ["most common words","frequent words","top words",
                                   "common keywords","what words","word frequency"]):
            all_kw = extract_keywords(df["CleanText"].tolist(), 8)
            lines  = "\n".join([f"  {i}. '{w}' — {c} times ({c/total*100:.1f}%)"
                                 for i,(w,c) in enumerate(all_kw,1)])
            return f"Most common words across all reviews:\n{lines}"

        # ── 21. YEAR-WISE ────────────────────────────────────
        elif any(w in q for w in ["year wise","year by year","each year","reviews per year",
                                   "annual","yearly breakdown","how many per year"]):
            if "Year" in df.columns:
                yearly = df.groupby("Year").agg(
                    Reviews=("Score","count"),
                    AvgRating=("Score","mean"),
                    PosReviews=("Sentiment", lambda x: (x=="Positive").sum()),
                    NegReviews=("Sentiment", lambda x: (x=="Negative").sum()),
                )
                lines = ""
                for yr, row in yearly.iterrows():
                    pct_pos = row["PosReviews"]/row["Reviews"]*100
                    pct_neg = row["NegReviews"]/row["Reviews"]*100
                    lines += (f"  {yr}: {int(row['Reviews'])} reviews | "
                              f"avg {row['AvgRating']:.1f}⭐ | "
                              f"{pct_pos:.1f}% pos | {pct_neg:.1f}% neg\n")
                return f"Year-wise breakdown:\n{lines}"
            return "Year data not available."

        # ── 22. REVIEW LENGTH ────────────────────────────────
        elif any(w in q for w in ["review length","long review","short review","average length",
                                   "how long","word count","detailed review"]):
            df["ReviewLen"] = df["Text"].apply(lambda x: len(str(x).split()))
            avg_len = df["ReviewLen"].mean()
            longest = df.loc[df["ReviewLen"].idxmax()]
            return (f"Review length stats:\n"
                    f"  📝 Avg words per review  : {avg_len:.0f}\n"
                    f"  📖 Longest review        : {int(longest['ReviewLen'])} words "
                    f"(rated {int(longest['Score'])}⭐)\n"
                    f"  Short (<20 words)        : {(df['ReviewLen']<20).sum():,}\n"
                    f"  Detailed (>100 words)    : {(df['ReviewLen']>100).sum():,}\n\n"
                    f"  Insight: Longer reviews tend to be more detailed complaints\n"
                    f"  or highly enthusiastic praise — both are high-signal reviews.")

        # ── 23. WHAT IS THIS CHATBOT ─────────────────────────
        elif any(w in q for w in ["what are you","who are you","what is this","about you",
                                   "what do you do","tell me about","this chatbot","this agent"]):
            return (f"I am a Business Intelligence Agent 🤖\n\n"
                    f"  Analysed: {total:,} Amazon customer reviews\n"
                    f"  Features:\n"
                    f"  • Sentiment analysis (Positive/Negative/Neutral)\n"
                    f"  • Actionable business insights with % impact\n"
                    f"  • Predictive trend forecasting (next 3 months)\n"
                    f"  • Why-it-matters business explanations\n"
                    f"  • Product & keyword analysis\n\n"
                    f"  Type 'help' to see all questions I can answer.")

        # ── 24. HELP ─────────────────────────────────────────
        elif any(w in q for w in ["help","what can you","what can i ask","capabilities",
                                   "options","commands","questions","how to use","guide"]):
            return ("Questions I can answer:\n\n"
                    "  SENTIMENT & SATISFACTION\n"
                    "  • Overall / general sentiment\n"
                    "  • How many positive/negative/neutral reviews\n"
                    "  • Customer satisfaction (CSAT) score\n\n"
                    "  REVIEWS & KEYWORDS\n"
                    "  • Highest / lowest rated reviews\n"
                    "  • Top complaints (with business impact %)\n"
                    "  • What customers love / praise\n"
                    "  • Most common words\n\n"
                    "  RATINGS\n"
                    "  • Average rating / star breakdown\n\n"
                    "  TRENDS & PREDICTIONS\n"
                    "  • Sentiment trend over time\n"
                    "  • Predictive forecast (next 3 months)\n"
                    "  • Year-wise breakdown\n\n"
                    "  PRODUCTS\n"
                    "  • Most reviewed product\n"
                    "  • Products with most complaints\n\n"
                    "  BUSINESS ANALYSIS\n"
                    "  • Why complaints matter (business impact)\n"
                    "  • Business recommendations\n"
                    "  • Scenario analysis (best/base/worst case)\n"
                    "  • Model accuracy info\n"
                    "  • Review length stats")

        # ── FALLBACK ─────────────────────────────────────────
        else:
            return ("I didn't understand that. Try:\n"
                    "  'What is overall sentiment?'\n"
                    "  'What are the top complaints?'\n"
                    "  'Show predictive forecast'\n"
                    "  'Why do complaints matter?'\n"
                    "  'Give me recommendations'\n\n"
                    "  Type 'help' to see all questions.")

    return respond


# ── TKINTER UI ────────────────────────────────
def launch_chatbot_ui(respond_fn, df_info):
    window = tk.Tk()
    window.title("BI Agent — Business Intelligence Chatbot")
    window.geometry("700x780")
    window.configure(bg="#1e1e2e")
    window.resizable(True, True)

    header = tk.Frame(window, bg="#185FA5", pady=10)
    header.pack(fill="x")
    tk.Label(header, text="🤖  Business Intelligence Agent",
             bg="#185FA5", fg="white", font=("Segoe UI", 14, "bold")).pack()
    tk.Label(header, text=df_info,
             bg="#185FA5", fg="#B5D4F4", font=("Segoe UI", 9)).pack()

    btn_frame = tk.Frame(window, bg="#1e1e2e", pady=6)
    btn_frame.pack(fill="x", padx=12)
    tk.Label(btn_frame, text="Quick questions:",
             bg="#1e1e2e", fg="#888888", font=("Segoe UI", 9)).pack(anchor="w")

    rows_data = [
        [("Overall sentiment?",        "#2a2a3e","#B5D4F4"),
         ("Top complaints?",           "#2a2a3e","#B5D4F4"),
         ("What do customers love?",   "#2a2a3e","#B5D4F4"),
         ("Recommendations?",          "#2a2a3e","#B5D4F4")],
        [("Highest rated review?",     "#2a2a3e","#9FE1CB"),
         ("Lowest rated review?",      "#2a2a3e","#9FE1CB"),
         ("Star rating breakdown?",    "#2a2a3e","#9FE1CB"),
         ("Customer satisfaction?",    "#2a2a3e","#9FE1CB")],
        [("Most reviewed product?",    "#2a2a3e","#FAC775"),
         ("Lowest review product?",    "#2a2a3e","#FAC775"),
         ("Predictive forecast?",      "#2a2a3e","#FAC775"),
         ("Why do complaints matter?", "#2a2a3e","#FAC775")],
    ]

    chat_frame = tk.Frame(window, bg="#1e1e2e")
    chat_frame.pack(fill="both", expand=True, padx=12, pady=(0,6))
    chat_display = scrolledtext.ScrolledText(
        chat_frame, wrap=tk.WORD, state="disabled",
        bg="#13131f", fg="#e0e0e0", font=("Segoe UI", 11),
        relief="flat", bd=0, padx=10, pady=10, cursor="arrow")
    chat_display.pack(fill="both", expand=True)
    chat_display.tag_configure("bot_label",  foreground="#378ADD", font=("Segoe UI", 10, "bold"))
    chat_display.tag_configure("bot_text",   foreground="#e0e0e0", font=("Segoe UI", 11), lmargin1=14, lmargin2=14)
    chat_display.tag_configure("user_label", foreground="#2ecc71", font=("Segoe UI", 10, "bold"), justify="right")
    chat_display.tag_configure("user_text",  foreground="#c0ffd0", font=("Segoe UI", 11), justify="right")
    chat_display.tag_configure("divider",    foreground="#2a2a3e")

    def append_message(message, is_user=False):
        chat_display.configure(state="normal")
        chat_display.insert("end", "\n")
        if is_user:
            chat_display.insert("end", "You\n", "user_label")
            chat_display.insert("end", message+"\n", "user_text")
        else:
            chat_display.insert("end", "BI Agent\n", "bot_label")
            chat_display.insert("end", message+"\n", "bot_text")
        chat_display.insert("end", "─"*60+"\n", "divider")
        chat_display.configure(state="disabled")
        chat_display.see("end")

    append_message("Hello! I'm your Business Intelligence Agent v4.\n"
                   "I now provide actionable insights, predictive forecasts,\n"
                   "and business impact analysis. Type 'help' to see all I can answer. 👋")

    input_frame = tk.Frame(window, bg="#1e1e2e", pady=8)
    input_frame.pack(fill="x", padx=12, side="bottom")
    entry_var = tk.StringVar()
    entry = tk.Entry(input_frame, textvariable=entry_var,
                     font=("Segoe UI", 11), bg="#2a2a3e", fg="white",
                     insertbackground="white", relief="flat", bd=0)
    entry.pack(side="left", fill="x", expand=True, ipady=9, padx=(0,8))

    def send_message(query=None):
        text = query if query else entry_var.get().strip()
        if not text: return
        entry_var.set("")
        append_message(text, is_user=True)
        def process():
            response = respond_fn(text)
            window.after(0, lambda: append_message(response))
        threading.Thread(target=process, daemon=True).start()

    tk.Button(input_frame, text="Send  ➤", command=send_message,
              bg="#185FA5", fg="white", font=("Segoe UI", 11, "bold"),
              relief="flat", padx=14, pady=9, cursor="hand2",
              activebackground="#0C447C", activeforeground="white", bd=0
              ).pack(side="right")
    entry.bind("<Return>", lambda e: send_message())

    hover_colors = {"#B5D4F4":"#185FA5","#9FE1CB":"#0F6E56","#FAC775":"#854F0B"}
    for row_data in rows_data:
        row_frame = tk.Frame(btn_frame, bg="#1e1e2e")
        row_frame.pack(fill="x", pady=(3,0))
        for (label, bg, fg) in row_data:
            tk.Button(row_frame, text=label,
                      command=lambda q=label: send_message(q),
                      bg=bg, fg=fg, font=("Segoe UI", 9),
                      relief="flat", padx=8, pady=3, cursor="hand2",
                      activebackground=hover_colors.get(fg,"#185FA5"),
                      activeforeground="white", bd=0
                      ).pack(side="left", padx=(0,6))

    entry.focus()
    window.mainloop()


# ── MAIN ──────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH = r"F:\LPU\module 4\CA\mgnm521\Reviews.csv" 

    df                      = load_data(CSV_PATH, sample_size=10000)
    df                      = preprocess(df)
    df                      = run_sentiment_analysis(df)
    counts, neg_kw, pos_kw, pred = generate_insights(df)
    create_visualizations(df, counts, pred)

    respond_fn = build_chatbot_logic(df, counts, neg_kw, pos_kw, pred)
    df_info    = f"Amazon Reviews  •  {len(df):,} records  •  Avg Rating: {df['Score'].mean():.2f}/5"
    launch_chatbot_ui(respond_fn, df_info)