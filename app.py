import textwrap, json, os, pandas as pd

app_code = r"""
import json
import os
import pandas as pd
import altair as alt
import streamlit as st
import joblib
from pathlib import Path
from nltk.tokenize import sent_tokenize

st.set_page_config(page_title="Hotel ABSA Dashboard", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_csvs():
    base = Path("/content/drive/MyDrive/Hotels/Outputs")
    review_level = base / "review_level_absa.csv"
    aspect_long  = base / "review_aspect_sentiments_long.csv"
    aspect_sum   = base / "aspect_summary.csv"

    if not review_level.exists() or not aspect_long.exists() or not aspect_sum.exists():
        return None, None, None

    df_reviews = pd.read_csv(review_level)
    df_long    = pd.read_csv(aspect_long)
    df_sum     = pd.read_csv(aspect_sum)
    return df_reviews, df_long, df_sum

df_reviews, df_long, df_sum = load_csvs()

# Load raw dataset for meta-analytics (ratings, trip type, location, date)
RAW_PATH = "/content/drive/MyDrive/Hotels/hotels_reviews.csv"
df_raw = pd.read_csv(RAW_PATH)

# --- NEW: stable key + extract reviewer + clean dates ---
df_raw = df_raw.reset_index().rename(columns={"index": "idx"})

import re

def extract_reviewer(text):
    if not isinstance(text, str):
        return None
    parts = text.split("wrote a review")
    if len(parts) >= 2:
        return parts[0].strip()
    return None

def extract_date_part(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"wrote a review\s+(.*)", text)
    if m:
        return m.group(1).strip()
    return None

# Add new columns
df_raw["Reviewer_Name"] = df_raw["Review_Date"].apply(extract_reviewer)
df_raw["Review_Date_Cleaned"] = df_raw["Review_Date"].apply(extract_date_part)

# Use cleaned date column in downstream parsing
df_raw["Cleaned_Review_Date"] = pd.to_datetime(df_raw["Review_Date_Cleaned"], errors="coerce")




import re
from datetime import datetime

_MONTHS3  = {m.lower(): i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
_MONTHSFL = {m.lower(): i for i, m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}

def _month_to_int(name: str):
    if not isinstance(name, str): return None
    n = name.strip().lower()
    return _MONTHS3.get(n) or _MONTHSFL.get(n)

def _extract_year_from_text(s: str):

    if not isinstance(s, str): return None
    m = re.search(r'\b(19|20)\d{2}\b', s)
    return int(m.group(0)) if m else None

def _extract_month_year_from_stay(s: str):

    if not isinstance(s, str): return (None, None)
    m = re.search(r'([A-Za-z]{3,9})\s+((?:19|20)\d{2})', s)
    if not m: return (None, _extract_year_from_text(s))
    mon = _month_to_int(m.group(1))
    yr  = int(m.group(2))
    return (mon, yr)

def _parse_review_date_row(row):

    rd = row.get("Review_Date", "")
    stay = row.get("Reviewer_Date_Of_Stay", "")

    # 1) Month Day [, Year]   e.g., "Mar 23" or "Mar 23, 2021"
    m1 = re.search(r'([A-Za-z]{3,9})\s+(\d{1,2})(?:,\s*((?:19|20)\d{2}))?', str(rd))
    if m1:
        mon = _month_to_int(m1.group(1))
        day = int(m1.group(2))
        yr  = int(m1.group(3)) if m1.group(3) else None
        if yr is None:
            # Try to pull year from stay
            _, yr = _extract_month_year_from_stay(stay)
        if mon and yr:
            try: return pd.Timestamp(yr, mon, day)
            except Exception: return pd.NaT

    # 2) Month Year (no day)  e.g., "Feb 2023"
    m2 = re.search(r'([A-Za-z]{3,9})\s+((?:19|20)\d{2})', str(rd))
    if m2:
        mon = _month_to_int(m2.group(1))
        yr  = int(m2.group(2))
        if mon and yr:
            try: return pd.Timestamp(yr, mon, 1)
            except Exception: return pd.NaT

    # 3) Numeric dates like 2023-03-22, 22/03/2023, 03-22-2023
    m3 = re.search(r'\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b', str(rd))  # YYYY-MM-DD
    if m3:
        try: return pd.Timestamp(int(m3.group(1)), int(m3.group(2)), int(m3.group(3)))
        except Exception: pass
    m4 = re.search(r'\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b', str(rd))  # DD-MM-YYYY or MM-DD-YYYY
    if m4:
        d1, d2, yr = int(m4.group(1)), int(m4.group(2)), int(m4.group(3))
        # Best effort: try DD-MM-YYYY first, then MM-DD-YYYY
        for day, mon in [(d1, d2), (d2, d1)]:
            try: return pd.Timestamp(yr, mon, day)
            except Exception: continue

    # 4) If only 'Mar 23' (no year) and stay has a year, use that year
    m5 = re.search(r'([A-Za-z]{3,9})\s+(\d{1,2})\b', str(rd))
    if m5:
        mon = _month_to_int(m5.group(1))
        day = int(m5.group(2))
        _, yr = _extract_month_year_from_stay(stay)
        if mon and yr:
            try: return pd.Timestamp(yr, mon, day)
            except Exception: return pd.NaT

    # 5) If only month in rd but year in stay (rare)
    m6 = re.search(r'\b([A-Za-z]{3,9})\b', str(rd))
    if m6:
        mon = _month_to_int(m6.group(1))
        smon, yr = _extract_month_year_from_stay(stay)
        if yr and (mon or smon):
            mon_final = mon or smon
            try: return pd.Timestamp(int(yr), int(mon_final), 1)
            except Exception: return pd.NaT

    return pd.NaT

# Build cleaned date column using BOTH fields
df_raw["Cleaned_Review_Date"] = df_raw.apply(_parse_review_date_row, axis=1)

# ---------- Load models ----------
def slugify(name: str):
    import re
    return re.sub(r'[^a-z0-9]+','_', name.lower()).strip('_')

aspect_detect = joblib.load("/content/drive/MyDrive/Hotels/Models/aspect_detector_ovr.joblib")
aspect_vec = aspect_detect["vec"]
aspect_clf = aspect_detect["clf"]
aspect_mlb = aspect_detect["mlb"]

sent_student = joblib.load("/content/drive/MyDrive/Hotels/Models/sentiment_sentence_student.joblib")
aspect_models = {}
ASPECTS = aspect_mlb.classes_.tolist()
for a in ASPECTS:
    p = f"/content/drive/MyDrive/Hotels/Models/sentiment_aspect_{slugify(a)}.joblib"
    if os.path.exists(p):
        aspect_models[a] = joblib.load(p)

ASPECT_PROBA_THRESHOLD = 0.50
ASPECT_TOPK = 3
SENT_MIN_LEN = 5

# ---------- Inference ----------
def predict_aspects_for_sentence(sent: str):
    X = aspect_vec.transform([sent])
    proba = aspect_clf.predict_proba(X)[0]
    labels = aspect_mlb.classes_
    pairs = [(lab, float(p)) for lab, p in zip(labels, proba) if p >= ASPECT_PROBA_THRESHOLD]
    pairs.sort(key=lambda x: -x[1])
    if ASPECT_TOPK:
        pairs = pairs[:ASPECT_TOPK]
    return pairs

def predict_sentiment_for_sentence_aspect(sent: str, aspect: str):
    model = aspect_models.get(aspect, None)
    if model is None:
        label = sent_student.predict([sent])[0]
        conf = max(sent_student.predict_proba([sent])[0])
        return label, float(conf)
    else:
        label = model.predict([sent])[0]
        conf = max(model.predict_proba([sent])[0])
        return label, float(conf)

def analyze_review(text: str):
    sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) >= SENT_MIN_LEN]
    aspect_map = {}
    for s in sents:
        asp_pairs = predict_aspects_for_sentence(s)
        if not asp_pairs:
            continue
        for (asp, _p) in asp_pairs:
            lab, conf = predict_sentiment_for_sentence_aspect(s, asp)
            prev = aspect_map.get(asp, {"votes": {"negative":0,"neutral":0,"positive":0}, "max_conf":0.0})
            prev["votes"][lab] += 1
            prev["max_conf"] = max(prev["max_conf"], conf)
            aspect_map[asp] = prev

    finalized = {}
    for asp, v in aspect_map.items():
        label = max(v["votes"].items(), key=lambda kv: kv[1])[0]
        finalized[asp] = {
            "sentiment": label,
            "hits": sum(v["votes"].values()),
            "conf": round(v["max_conf"], 3)
        }
    return finalized

# ---------- Dashboard ----------
st.title("🏨  LodgEdge ")
st.caption("Aspect-Based Sentiment Analysis + Hotel Insights")

# KPIs
if df_reviews is not None:
    total_reviews = len(df_reviews)
    total_aspect_rows = len(df_long) if df_long is not None else 0
    pos_rate = (df_long["sentiment"] == "positive").mean() if total_aspect_rows else 0
    neg_rate = (df_long["sentiment"] == "negative").mean() if total_aspect_rows else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total_reviews:,}")
    c2.metric("Aspect Mentions", f"{total_aspect_rows:,}")
    c3.metric("Positive Share", f"{pos_rate*100:.1f}%")
    c4.metric("Negative Share", f"{neg_rate*100:.1f}%")

# ⭐ Overall Rating Distribution
if "Review_Rating" in df_raw.columns:
    st.subheader("⭐ Overall Rating Distribution")

    # 🔹 Normalize ratings: convert 10–50 scale → 1–5 scale
    df_raw["Normalized_Rating"] = df_raw["Review_Rating"].apply(
        lambda x: int(x/10) if (isinstance(x, (int,float)) and x > 5) else x
    )

    chart = alt.Chart(df_raw).mark_bar().encode(
        x=alt.X("Normalized_Rating:O", title="Rating (1–5)"),
        y=alt.Y("count()", title="Count of Reviews")
    )
    st.altair_chart(chart, use_container_width=True)


# 😊 Overall Sentiment Distribution
if "overall_label" in df_reviews.columns:
    st.subheader("😊 Overall Sentiment Distribution")
    dist = df_reviews["overall_label"].value_counts().reset_index()
    dist.columns = ["sentiment","count"]
    pie = alt.Chart(dist).mark_arc().encode(
        theta="count:Q",
        color="sentiment:N"
    )
    st.altair_chart(pie, use_container_width=True)

# 🧳 Reviewer Trip Type
if "Reviewer_Trip_Type" in df_raw.columns:
    st.subheader("🧳 Reviewer Trip Type Distribution")
    trip = df_raw["Reviewer_Trip_Type"].value_counts().reset_index()
    trip.columns = ["Trip Type","count"]
    chart = alt.Chart(trip).mark_bar().encode(x="Trip Type:N", y="count:Q")
    st.altair_chart(chart, use_container_width=True)

# 🌍 Reviewer Location
if "Reviewer_Location" in df_raw.columns:
    st.subheader("🌍 Reviewer Location Distribution (Top 15)")
    loc = df_raw["Reviewer_Location"].value_counts().reset_index().head(15)
    loc.columns = ["Location","count"]
    chart = alt.Chart(loc).mark_bar().encode(
        y=alt.Y("Location:N", sort="-x"), x="count:Q"
    )
    st.altair_chart(chart, use_container_width=True)

# 📊 Aspect Frequency
if df_long is not None:
    st.subheader("📊 Aspect Frequency Across Reviews")
    freq = df_long["aspect"].value_counts().reset_index()
    freq.columns = ["aspect","count"]
    chart = alt.Chart(freq).mark_bar().encode(
        y=alt.Y("aspect:N", sort="-x"), x="count:Q"
    )
    st.altair_chart(chart, use_container_width=True)

# 🔍 Sentiment per Aspect
if df_sum is not None:
    st.subheader("🔍 Sentiment Breakdown per Aspect")
    df_sum_melt = df_sum.melt(id_vars="aspect", value_vars=["negative","neutral","positive"],
                              var_name="sentiment", value_name="count")
    chart = alt.Chart(df_sum_melt).mark_bar().encode(
        y=alt.Y("aspect:N", sort="-x"), x="count:Q", color="sentiment:N"
    )
    st.altair_chart(chart, use_container_width=True)

# 💪 Strengths & Weaknesses (no overlap)
if df_sum is not None:
    st.subheader("💪 Strengths and Weaknesses")
    top_pos = df_sum.sort_values("positive", ascending=False).head(5)
    top_neg = df_sum[~df_sum["aspect"].isin(top_pos["aspect"])].sort_values("negative", ascending=False).head(5)

    c1, c2 = st.columns(2)
    c1.markdown("**Top 5 Strengths (Most Positive Aspects)**")
    pos_chart = alt.Chart(top_pos).mark_bar().encode(
        y=alt.Y("aspect:N", sort="-x"), x="positive:Q"
    )
    c1.altair_chart(pos_chart, use_container_width=True)

    c2.markdown("**Top 5 Weaknesses (Most Negative Aspects)**")
    neg_chart = alt.Chart(top_neg).mark_bar().encode(
        y=alt.Y("aspect:N", sort="-x"), x="negative:Q"
    )
    c2.altair_chart(neg_chart, use_container_width=True)

# 📈 Sentiment Over Time (robust: join on idx + cleaned dates + fallback sentiment if needed)
if ("Cleaned_Review_Date" in df_raw.columns) and ("idx" in df_raw.columns) and ("idx" in df_reviews.columns):
    st.subheader("📈 Sentiment Over Time")

    df_plot = df_reviews.copy()
    # Fallback if overall_label missing: derive from Review_Rating (10–50 or 1–5)
    if "overall_label" not in df_plot.columns or df_plot["overall_label"].isna().all():
        if "Review_Rating" in df_raw.columns:
            ratings = df_raw[["idx","Review_Rating"]].copy()
            df_plot = df_plot.merge(ratings, on="idx", how="left")
            def rate_to_label(x):
                try:
                    val = float(x)
                except Exception:
                    return None
                # normalize 10–50 to 1–5 if needed
                if val > 5:
                    val = val / 10.0
                if val <= 2:  return "negative"
                if val >= 4:  return "positive"
                return "neutral"
            df_plot["overall_label"] = df_plot["Review_Rating"].apply(rate_to_label)

    dates = df_raw[["idx","Cleaned_Review_Date"]].dropna(subset=["Cleaned_Review_Date"])
    merged = df_plot.merge(dates, on="idx", how="inner")

    if merged.empty or "overall_label" not in merged.columns:
        st.info("No valid dates or sentiments to plot.")
    else:
        merged["month"] = merged["Cleaned_Review_Date"].dt.to_period("M").dt.to_timestamp()
        timeline = (merged.dropna(subset=["month","overall_label"])
                          .groupby(["month","overall_label"])
                          .size().reset_index(name="count"))
        if timeline.empty:
            st.info("No data after cleaning dates. Check your Review_Date values.")
        else:
            chart = alt.Chart(timeline, width=4000, height=400).mark_line(point=True).encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("count:Q", title="# Reviews"),
                color=alt.Color("overall_label:N", title="Sentiment",
                                sort=["negative","neutral","positive"])
            )
            st.altair_chart(chart, use_container_width=True)

# ---------- Try a New Review ----------
st.subheader("✍️ Try a New Review")
user_text = st.text_area("Paste a hotel review here:", height=120)

if st.button("Analyze Review") and user_text.strip():
    result = analyze_review(user_text)
    if not result:
        st.warning("No aspects detected in this review.")
    else:
        df_res = pd.DataFrame([
            {"aspect": a, "sentiment": v["sentiment"], "hits": v["hits"], "conf": v["conf"]}
            for a, v in result.items()
        ])
        st.dataframe(df_res, use_container_width=True)

        chart = alt.Chart(df_res).mark_bar().encode(
            y="aspect:N", x="hits:Q", color="sentiment:N"
        )
        st.altair_chart(chart, use_container_width=True)

# =========================
# 💼 Business Impact Insights
# (Append-only: does not change existing code)
# =========================
import numpy as np
import re

st.header("💼 Business Impact Insights")

# ---------- Helpers ----------
def _pick_col(df, patterns):
    pats = [re.compile(p, re.I) for p in patterns]
    for c in df.columns:
        for p in pats:
            if p.search(c):
                return c
    return None

def _safe_monthly_overall(df_reviews, df_raw):
    # Build monthly overall sentiment timeline (independent of earlier block scope)
    if not all(x in df_raw.columns for x in ["idx", "Cleaned_Review_Date"]):
        return None
    if "idx" not in df_reviews.columns:
        return None
    if "overall_label" not in df_reviews.columns:
        return None

    r = df_reviews[["idx", "overall_label"]].copy()
    d = df_raw[["idx", "Cleaned_Review_Date"]].dropna().copy()
    m = r.merge(d, on="idx", how="inner")
    if m.empty:
        return None
    m["month"] = m["Cleaned_Review_Date"].dt.to_period("M").dt.to_timestamp()
    return m

def _aspect_score_row(pos, neu, neg):
    tot = (pos or 0) + (neu or 0) + (neg or 0)
    if tot == 0:
        return 0.0
    # Score 0..100: weight neutral half
    return 100.0 * (pos + 0.5 * neu) / tot

def _verdict(pct):
    if pct >= 85: return "excellent"
    if pct >= 70: return "good"
    if pct >= 50: return "needs work"
    return "bad"

def _find_reviewer_col(df):
    return _pick_col(df, [
        r"^reviewer.*name$", r"reviewer$",
        r"^guest.*name$", r"^author$", r"user(name)?$", r"customer.*name"
    ])

def _find_hotel_col(df):
    return _pick_col(df, [r"hotel", r"property", r"resort"])

# ---------- Tabs ----------
tabs = st.tabs([
    "Aspect Performance",
    "Per-Aspect Trends",
    "Best Months",
    "Year Overview",
    "Aspect–Revenue Correlation (Synthetic)",
    "Guest Retention & Loyalty",
    "Upsell & Cross-Sell",
    "Early-Warning Alerts",
    "Revenue Impact Forecasting (Synthetic)",
    "Competitor Benchmarking"
])

# -------------------------
# 1) Aspect Performance
# -------------------------
with tabs[0]:
    st.subheader("Aspect Performance Score")
    if df_sum is not None and set(["aspect","positive","neutral","negative"]).issubset(df_sum.columns):
        perf = df_sum.copy()
        perf["score_pct"] = perf.apply(lambda r: _aspect_score_row(r.get("positive",0), r.get("neutral",0), r.get("negative",0)), axis=1)
        perf["verdict"] = perf["score_pct"].round(1).apply(_verdict)
        perf = perf.sort_values("score_pct", ascending=False).reset_index(drop=True)

        st.dataframe(perf[["aspect","positive","neutral","negative","score_pct","verdict"]], use_container_width=True)

        bar = alt.Chart(perf).mark_bar().encode(
            y=alt.Y("aspect:N", sort="-x", title="Aspect"),
            x=alt.X("score_pct:Q", title="Performance Score (0–100)"),
            color=alt.Color("verdict:N", sort=["bad","needs work","good","excellent"])
        )
        st.altair_chart(bar, use_container_width=True)
        st.caption("Score = (Positive + 0.5×Neutral) / Total × 100. Verdict thresholds: ≥85 excellent; 70–84 good; 50–69 needs work; <50 bad.")
    else:
        st.info("`df_sum` with columns ['aspect','positive','neutral','negative'] not found.")

# -------------------------
# 2) Per-Aspect Trends
# -------------------------
with tabs[1]:
    st.subheader("Time-Aware Sentiment per Aspect (Monthly)")
    # We need df_long with aspect+sentiment and idx to join dates
    if df_long is not None and "sentiment" in df_long.columns and "aspect" in df_long.columns:
        if ("idx" in df_long.columns) and ("idx" in df_raw.columns) and ("Cleaned_Review_Date" in df_raw.columns):
            dl = df_long[["idx","aspect","sentiment"]].merge(
                df_raw[["idx","Cleaned_Review_Date"]].dropna(), on="idx", how="inner"
            )
        elif "idx" in df_raw.columns and "idx" not in df_long.columns:
            # Best-effort positional align (fallback only)
            dl = pd.concat([df_long.reset_index(drop=True), df_raw[["idx","Cleaned_Review_Date"]].dropna().reset_index(drop=True)], axis=1)
            if "Cleaned_Review_Date" not in dl.columns:
                st.info("Unable to align aspects to dates; `idx` missing in df_long.")
                dl = None
        else:
            dl = None

        if dl is not None and not dl.empty and "Cleaned_Review_Date" in dl.columns:
            dl["month"] = pd.to_datetime(dl["Cleaned_Review_Date"]).dt.to_period("M").dt.to_timestamp()
            aspects = sorted(dl["aspect"].dropna().unique().tolist())
            pick = st.multiselect("Choose aspects to plot", options=aspects, default=aspects[:5])
            if pick:
                sub = dl[dl["aspect"].isin(pick)].copy()
                timeline = (sub.groupby(["month","aspect","sentiment"]).size().reset_index(name="count"))
                chart = alt.Chart(timeline).mark_line(point=True).encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("count:Q", title="# Mentions"),
                    color="sentiment:N",
                    facet=alt.Facet("aspect:N", columns=2),
                    tooltip=["month:T","aspect:N","sentiment:N","count:Q"]
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Select at least one aspect.")
        else:
            st.info("Could not build per-aspect timeline (missing dates or idx).")
    else:
        st.info("`df_long` with ['aspect','sentiment'] not found.")

# -------------------------
# 3) Best Performing Months
# -------------------------
with tabs[2]:
    st.subheader("Best Performing Months (Overall Positivity)")
    m = _safe_monthly_overall(df_reviews, df_raw)
    if m is not None:
        agg = (m.groupby("month")["overall_label"]
                 .value_counts()
                 .unstack(fill_value=0)
                 .reset_index())
        for c in ["negative","neutral","positive"]:
            if c not in agg.columns: agg[c] = 0
        agg["total"] = agg[["negative","neutral","positive"]].sum(axis=1)
        agg["positivity_rate"] = (agg["positive"] / agg["total"]).round(4)
        topn = agg.sort_values("positivity_rate", ascending=False).head(10)
        st.dataframe(topn[["month","positive","neutral","negative","total","positivity_rate"]], use_container_width=True)

        chart = alt.Chart(agg).mark_line(point=True).encode(
            x="month:T", y=alt.Y("positivity_rate:Q", title="Positivity Rate"),
            tooltip=["month:T","positivity_rate:Q"]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Monthly overall sentiment not available.")

# -------------------------
# 4) Year Overview
# -------------------------
with tabs[3]:
    st.subheader("Year Overview & Positivity Rate")
    m = _safe_monthly_overall(df_reviews, df_raw)
    if m is not None:
        m["year"] = m["month"].dt.year
        yr = (m.groupby("year")["overall_label"].value_counts()
                .unstack(fill_value=0).reset_index())
        for c in ["negative","neutral","positive"]:
            if c not in yr.columns: yr[c] = 0
        yr["total"] = yr[["negative","neutral","positive"]].sum(axis=1)
        yr["positivity_rate"] = (yr["positive"] / yr["total"]).round(4)
        st.dataframe(yr[["year","positive","neutral","negative","total","positivity_rate"]],
                     use_container_width=True)

        chart = alt.Chart(yr).mark_bar().encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("positive:Q", title="# Positive"),
            tooltip=["year:O","positive:Q","neutral:Q","negative:Q","positivity_rate:Q"]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Year overview not available.")

# -----------------------------------------
# 5) Aspect–Revenue Correlation (Synthetic)
# -----------------------------------------
with tabs[4]:
    st.subheader("Aspect–Revenue Correlation (Synthetic KPIs)")
    # Build monthly per-aspect positivity
    if df_long is not None and "sentiment" in df_long.columns and "aspect" in df_long.columns and "idx" in df_long.columns:
        join = df_long[["idx","aspect","sentiment"]].merge(
            df_raw[["idx","Cleaned_Review_Date"]].dropna(), on="idx", how="inner"
        )
        if not join.empty:
            join["month"] = pd.to_datetime(join["Cleaned_Review_Date"]).dt.to_period("M").dt.to_timestamp()
            aspect_list = sorted(join["aspect"].dropna().unique().tolist())
            a_sel = st.selectbox("Aspect", options=aspect_list, index=0)

            sub = join[join["aspect"] == a_sel]
            month_counts = (sub.groupby(["month","sentiment"]).size().unstack(fill_value=0).reset_index())
            for c in ["negative","neutral","positive"]:
                if c not in month_counts.columns: month_counts[c] = 0
            month_counts["total"] = month_counts[["negative","neutral","positive"]].sum(axis=1)
            month_counts = month_counts[month_counts["total"] > 0].copy()
            month_counts["pos_share"] = month_counts["positive"] / month_counts["total"]

            # Synthetic KPIs correlated with pos_share
            rng = np.random.default_rng(42)
            base_ADR = 80 + 70 * month_counts["pos_share"].values + rng.normal(0, 5, size=len(month_counts))
            conv_rate = 0.10 + 0.25 * month_counts["pos_share"].values + rng.normal(0, 0.01, size=len(month_counts))
            month_counts["synthetic_ADR"] = base_ADR.round(2)
            month_counts["synthetic_Conversion"] = np.clip(conv_rate, 0, 1).round(3)

            # Correlations
            corr_adr = float(np.corrcoef(month_counts["pos_share"], month_counts["synthetic_ADR"])[0,1])
            corr_conv = float(np.corrcoef(month_counts["pos_share"], month_counts["synthetic_Conversion"])[0,1])

            st.caption(f"Correlation with ADR: **{corr_adr:.2f}**, with Conversion: **{corr_conv:.2f}** (synthetic).")

            sc1 = alt.Chart(month_counts).mark_circle(size=80).encode(
                x=alt.X("pos_share:Q", title="Positive Share (Aspect)"),
                y=alt.Y("synthetic_ADR:Q", title="ADR (Synthetic)"),
                tooltip=["month:T","pos_share:Q","synthetic_ADR:Q"]
            )
            sc1 = sc1 + sc1.transform_regression("pos_share","synthetic_ADR").mark_line()
            st.altair_chart(sc1, use_container_width=True)

            sc2 = alt.Chart(month_counts).mark_circle(size=80).encode(
                x=alt.X("pos_share:Q", title="Positive Share (Aspect)"),
                y=alt.Y("synthetic_Conversion:Q", title="Conversion Rate (Synthetic)"),
                tooltip=["month:T","pos_share:Q","synthetic_Conversion:Q"]
            )
            sc2 = sc2 + sc2.transform_regression("pos_share","synthetic_Conversion").mark_line()
            st.altair_chart(sc2, use_container_width=True)
        else:
            st.info("Could not compute per-aspect monthly sentiment (missing idx/date join).")
    else:
        st.info("`df_long` with ['idx','aspect','sentiment'] required.")

# -------------------------------
# 6) Guest Retention & Loyalty
# -------------------------------
with tabs[5]:
    st.subheader("Guest Retention & Loyalty Metrics")

    # --- try auto-detect first ---
    rid = _find_reviewer_col(df_raw)

    # --- interactive fallback if auto-detect failed ---
    if rid is None:
        st.info("Pick the column that identifies the reviewer (name, username, profile id, etc.).")
        # candidates: text-like columns with some repetition (to capture repeat guests)
        obj_cols = [c for c in df_raw.columns if df_raw[c].dtype == "object"]
        candidates = []
        for c in obj_cols:
            s = df_raw[c].dropna().astype(str)
            if len(s) >= 20:
                uniq = s.nunique()
                # we want some duplicates (repeat guests), but not all unique
                if 0 < uniq < len(s):
                    candidates.append((c, len(s) - uniq))
        # sort by number of duplicates (more repeats first)
        candidates = [c for c, _ in sorted(candidates, key=lambda x: -x[1])] or obj_cols

        rid = st.selectbox("Reviewer identifier column", options=candidates)

    if rid is None:
        st.info("No suitable text column found for reviewer id.")
    else:
        base = df_raw[[rid, "idx"]].dropna()
        if "idx" in df_reviews.columns and "overall_label" in df_reviews.columns:
            guests = base.merge(df_reviews[["idx", "overall_label"]], on="idx", how="left")

            # Repeat guests
            counts = guests.groupby(rid).size().rename("reviews").reset_index()
            repeat = counts[counts["reviews"] >= 2]
            st.metric("Repeat Guests", f"{len(repeat):,}")

            # Positivity per guest + dummy rebooking likelihood
            gpos = (guests.groupby(rid)["overall_label"]
                        .value_counts().unstack(fill_value=0).reset_index())
            for c in ["negative", "neutral", "positive"]:
                if c not in gpos.columns: gpos[c] = 0
            gpos["total"] = gpos[["negative","neutral","positive"]].sum(axis=1)
            gpos = gpos[gpos["total"] > 0].copy()
            gpos["pos_rate"] = gpos["positive"] / gpos["total"]

            def rebook_prob(p):
                return float(np.clip(0.2 + 0.7 * p, 0.0, 0.95))

            gpos["rebook_likelihood"] = gpos["pos_rate"].apply(rebook_prob)

            st.dataframe(
                gpos.sort_values("rebook_likelihood", ascending=False)
                    [[rid, "positive", "neutral", "negative", "total", "pos_rate", "rebook_likelihood"]]
                    .head(50),
                use_container_width=True
            )

            chart = alt.Chart(gpos).mark_bar().encode(
                x=alt.X("rebook_likelihood:Q", bin=alt.Bin(maxbins=20), title="Rebooking Likelihood (dummy)"),
                y="count()"
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Need df_reviews with ['idx','overall_label'] to compute loyalty metrics.")


# -------------------------------
# 7) Upsell & Cross-Sell Ideas
# -------------------------------
with tabs[6]:
    st.subheader("Upsell & Cross-Sell Opportunities")
    if df_sum is not None and {"aspect","positive"}.issubset(df_sum.columns):
        top_pos = df_sum.sort_values("positive", ascending=False).head(10)["aspect"].tolist()
        st.write("Top positively-mentioned aspects:", ", ".join(top_pos))
        # Simple suggestions
        suggestions = {
            "Breakfast": "Promote premium breakfast/brunch packages.",
            "Food": "Chef’s tasting menu add-on at booking.",
            "Spa": "Spa credit bundles with stay.",
            "Pool": "Poolside cabana/day-pass upsell.",
            "Staff": "VIP check-in / concierge upsell.",
            "Cleanliness": "Premium housekeeping / late checkout add-on.",
            "Location": "Local tour packages & transfers.",
            "Price": "Bundle offers (stay + meal) to lift perceived value."
        }
        for a in top_pos:
            idea = suggestions.get(a, "Create an add-on or bundle highlighting this strength.")
            st.markdown(f"- **{a}** → {idea}")
    else:
        st.info("`df_sum` with aspect sentiment totals is required.")

# --------------------------------
# 8) Early-Warning Alerts (Risk)
# --------------------------------
with tabs[7]:
    st.subheader("Early-Warning Alerts → Revenue Risk")
    # Use per-aspect month over month change
    if df_long is not None and "idx" in df_long.columns and "sentiment" in df_long.columns and "aspect" in df_long.columns:
        j = df_long[["idx","aspect","sentiment"]].merge(
            df_raw[["idx","Cleaned_Review_Date"]].dropna(), on="idx", how="inner"
        )
        if not j.empty:
            j["month"] = pd.to_datetime(j["Cleaned_Review_Date"]).dt.to_period("M").dt.to_timestamp()
            # positive rate per aspect per month
            g = (j.groupby(["aspect","month"])["sentiment"]
                   .value_counts().unstack(fill_value=0).reset_index())
            for c in ["negative","neutral","positive"]:
                if c not in g.columns: g[c] = 0
            g["total"] = g[["negative","neutral","positive"]].sum(axis=1)
            g = g[g["total"]>0].copy()
            g["pos_rate"] = g["positive"]/g["total"]

            # Compare last 2 months that exist
            last_months = sorted(g["month"].unique())
            if len(last_months) >= 2:
                m1, m0 = last_months[-1], last_months[-2]
                cur = g[g["month"]==m1][["aspect","pos_rate"]].set_index("aspect")
                prev = g[g["month"]==m0][["aspect","pos_rate"]].set_index("aspect")
                chg = (cur["pos_rate"] - prev["pos_rate"]).sort_values()
                threshold = st.slider("Alert threshold (percentage points drop)", 5, 30, 12)
                for a, delta in chg.items():
                    drop_pp = delta * 100
                    if drop_pp <= -threshold:
                        est_risk = abs(drop_pp) * 0.6  # dummy mapping
                        st.error(f"**{a}** dropped {abs(drop_pp):.1f}pp (from {prev.loc[a,'pos_rate']*100:.1f}% to {cur.loc[a,'pos_rate']*100:.1f}%). Potential revenue risk ≈ {est_risk:.1f}%.")
                if (chg*100 >= -threshold).all():
                    st.success("No significant aspect drops detected.")
            else:
                st.info("Need at least two months of aspect data to compute alerts.")
        else:
            st.info("Could not align aspects to dates for alerts.")
    else:
        st.info("`df_long` with ['idx','aspect','sentiment'] required.")

# --------------------------------
# 9) Revenue Impact Forecasting (Synthetic)
# --------------------------------
with tabs[8]:
    st.subheader("Revenue Impact Forecasting (Synthetic)")
    if df_sum is not None and {"aspect","positive","neutral","negative"}.issubset(df_sum.columns):
        base = df_sum.copy()
        base["score_pct"] = base.apply(lambda r: _aspect_score_row(r.get("positive",0), r.get("neutral",0), r.get("negative",0)), axis=1)
        aspects = base["aspect"].tolist()
        a_sel = st.selectbox("Aspect to improve", options=aspects, index=0)
        improve = st.slider("Improvement in aspect score (pp)", 0, 40, 20)

        cur = float(base.loc[base["aspect"]==a_sel, "score_pct"].values[0])
        new = np.clip(cur + improve, 0, 100)

        # Simple synthetic elasticity: every +10pp → +4% RevPAR
        uplift_pct = (new - cur) / 10.0 * 4.0
        # Synthetic current RevPAR ($)
        rng = np.random.default_rng(7)
        cur_revpar = float(100 + rng.normal(0, 5))
        horizon = st.selectbox("Horizon", ["3 months","6 months","12 months"], index=1)
        mult = {"3 months": 0.5, "6 months": 1.0, "12 months": 2.0}[horizon]
        proj_rev_gain = cur_revpar * (uplift_pct/100.0) * mult

        c1, c2, c3 = st.columns(3)
        c1.metric("Current score", f"{cur:.1f}%")
        c2.metric("New score", f"{new:.1f}%")
        c3.metric("Projected RevPAR Gain", f"${proj_rev_gain:,.0f}")

        line = pd.DataFrame({
            "Scenario": ["Current","Improved"],
            "Score": [cur, new]
        })
        ch = alt.Chart(line).mark_bar().encode(
            x="Scenario:N", y=alt.Y("Score:Q", title="Aspect Score (%)")
        )
        st.altair_chart(ch, use_container_width=True)
        st.caption("Synthetic model: +10pp in aspect score → ~+4% RevPAR; horizon scales impact. Replace with real KPIs when available.")
    else:
        st.info("`df_sum` with aspect totals required for forecasting baseline.")

# --------------------------------
# 10) Competitor Benchmarking
# --------------------------------
with tabs[9]:
    st.subheader("Competitor Benchmarking (Across Hotels)")
    hcol = _find_hotel_col(df_raw)
    if hcol is None:
        st.info("Could not find a hotel/property column in df_raw.")
    else:
        # overall positivity by hotel
        m = _safe_monthly_overall(df_reviews, df_raw)
        if m is not None:
            hotels = df_raw[["idx", hcol]]
            mh = m.merge(hotels, on="idx", how="left").dropna(subset=[hcol])
            agg = (mh.groupby(hcol)["overall_label"].value_counts()
                     .unstack(fill_value=0).reset_index())
            for c in ["negative","neutral","positive"]:
                if c not in agg.columns: agg[c] = 0
            agg["total"] = agg[["negative","neutral","positive"]].sum(axis=1)
            agg = agg[agg["total"]>0].copy()
            agg["positivity_rate"] = (agg["positive"]/agg["total"]).round(4)

            # Choose primary hotel (from page title if present)
            default_hotel = None
            for v in df_raw[hcol].value_counts().index:
                if "savoy" in str(v).lower(): default_hotel = v; break

            target = st.selectbox("Your hotel", options=agg[hcol].tolist(),
                                  index=(agg[hcol].tolist().index(default_hotel) if default_hotel in agg[hcol].tolist() else 0))
            topn = st.slider("Show top N hotels by positivity", 5, 30, 10)

            rank = agg.sort_values("positivity_rate", ascending=False).head(topn)
            chart = alt.Chart(rank).mark_bar().encode(
                x=alt.X("positivity_rate:Q", title="Positivity Rate"),
                y=alt.Y(f"{hcol}:N", sort="-x", title="Hotel"),
                tooltip=[f"{hcol}:N","positivity_rate:Q","positive:Q","neutral:Q","negative:Q","total:Q"]
            )
            st.altair_chart(chart, use_container_width=True)

            # Highlight your hotel
            me = agg[agg[hcol]==target]
            if not me.empty:
                pr = float(me["positivity_rate"].values[0])*100
                st.metric(f"Positivity rate — {target}", f"{pr:.1f}%")
        else:
            st.info("Could not compute overall sentiment per hotel (missing idx/date join).")

"""

with open("app.py","w",encoding="utf-8") as f:
    f.write(app_code)

print("Wrote updated app.py ✅ (timeline fixed: idx join + date cleaning)")
