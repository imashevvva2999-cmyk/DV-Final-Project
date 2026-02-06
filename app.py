import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_PATH = "product_info.csv"

st.set_page_config(page_title="Makeup & Beauty Dashboard", layout="wide")


def clean_products(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    num_cols = [
        "price_usd",
        "sale_price_usd",
        "value_price_usd",
        "child_min_price",
        "child_max_price",
        "rating",
        "reviews",
        "loves_count",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price_usd" in df.columns:
        for fallback in ["sale_price_usd", "value_price_usd", "child_min_price"]:
            if fallback in df.columns:
                df["price_usd"] = df["price_usd"].fillna(df[fallback])

    if "rating" in df.columns:
        df["rating"] = df["rating"].clip(lower=0, upper=5)

    if "reviews" in df.columns:
        df["reviews"] = df["reviews"].fillna(0)
    if "loves_count" in df.columns:
        df["loves_count"] = df["loves_count"].fillna(0)

    critical = [c for c in ["product_id", "product_name", "brand_name", "primary_category"] if c in df.columns]
    if critical:
        df = df.dropna(subset=critical)

    if "product_id" in df.columns:
        df = df.drop_duplicates(subset=["product_id"])
    else:
        df = df.drop_duplicates()

    return df

def add_features(df):
    df = df.copy()

    if "price_usd" in df.columns:
        bins = [0, 15, 30, 60, 100, np.inf]
        labels = ["$0-15", "$15-30", "$30-60", "$60-100", "$100+"]
        df["price_bin"] = pd.cut(df["price_usd"], bins=bins, labels=labels, include_lowest=True)

    if "rating" in df.columns:
        r_bins = [0, 2, 3, 4, 4.5, 5.01]
        r_labels = ["0-2", "2-3", "3-4", "4-4.5", "4.5-5"]
        df["rating_bin"] = pd.cut(df["rating"], bins=r_bins, labels=r_labels, include_lowest=True)

    if "reviews" in df.columns:
        df["review_bucket"] = pd.cut(
            df["reviews"],
            bins=[-1, 9, 50, 200, np.inf],
            labels=["<10", "10-50", "50-200", "200+"]
        )

    if "new" in df.columns:
        df["is_new"] = df["new"].fillna(0).astype(int) == 1
    else:
        df["is_new"] = False

    if "sephora_exclusive" in df.columns:
        df["is_exclusive"] = df["sephora_exclusive"].fillna(0).astype(int) == 1
    else:
        df["is_exclusive"] = False

    return df

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = clean_products(df)
    df = add_features(df)
    return df


def has_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0, missing


st.title("Makeup & Beauty Product Dashboard")

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("Missing product_info.csv. Place the CSV in the same folder as app.py.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
filtered = df.copy()

if "brand_name" in filtered.columns:
    brands = sorted(filtered["brand_name"].dropna().unique())
    select_all_brands = st.sidebar.checkbox("Select all brands", value=True)
    selected_brands = st.sidebar.multiselect("Brand", brands, default=brands if select_all_brands else [])
    if selected_brands:
        filtered = filtered[filtered["brand_name"].isin(selected_brands)]

if "primary_category" in filtered.columns:
    categories = sorted(filtered["primary_category"].dropna().unique())
    select_all_cats = st.sidebar.checkbox("Select all categories", value=True)
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories if select_all_cats else [])
    if selected_categories:
        filtered = filtered[filtered["primary_category"].isin(selected_categories)]

if "price_usd" in filtered.columns:
    price_series = filtered["price_usd"].dropna()
    if len(price_series) > 0:
        min_price = float(price_series.min())
        max_price = float(price_series.max())
        price_range = st.sidebar.slider(
            "Price range (USD)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )
        filtered = filtered[(filtered["price_usd"] >= price_range[0]) & (filtered["price_usd"] <= price_range[1])]

if "rating" in filtered.columns:
    rating_series = filtered["rating"].dropna()
    if len(rating_series) > 0:
        min_rating = float(rating_series.min())
        max_rating = float(rating_series.max())
        rating_range = st.sidebar.slider(
            "Rating range",
            min_value=min_rating,
            max_value=max_rating,
            value=(min_rating, max_rating)
        )
        filtered = filtered[(filtered["rating"] >= rating_range[0]) & (filtered["rating"] <= rating_range[1])]

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Products", f"{len(filtered):,}")
kpi2.metric("Avg Rating", f"{filtered['rating'].mean():.2f}" 
            if "rating" in filtered.columns 
            else "N/A")
kpi3.metric("Median Price", f"${filtered['price_usd'].median():.2f}" 
            if "price_usd" in filtered.columns 
            else "N/A")
kpi4.metric("Total Reviews", f"{filtered['reviews'].sum():,.0f}" 
            if "reviews" in filtered.columns 
            else "N/A")

st.markdown("---")

tabs = st.tabs([
    "Q1 Category Rating",
    "Q2 Top Brands (N reviews)",
    "Q3 Price Distribution",
    "Q4 Price by Category",
    "Q5 Price vs Rating",
    "Q6 Reviews vs Rating",
    "Q7 Popular Brands (Loves)",
    "Q8 Most Reviewed Category",
    "Q9 Exclusive vs Not",
    "Q10 New vs Non-New"
])

# Charts
with tabs[0]:
    st.subheader("Q1: Which primary_category has the highest average rating?")
    ok, miss = has_columns(filtered, ["primary_category", "rating"])
    if ok:
        data = filtered.groupby("primary_category")["rating"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=data.values, y=data.index, ax=ax, color="#3bf6c7")
        ax.set_xlabel("Average Rating")
        ax.set_ylabel("Primary Category")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[1]:
    st.subheader("Q2: Among brands with at least N total reviews, which has highest average rating?")
    ok, miss = has_columns(filtered, ["brand_name", "rating", "reviews"])
    if ok:
        max_total_reviews = int(filtered.groupby("brand_name")["reviews"].sum().max())
        N = st.slider(
            "N (minimum total reviews per brand)",
            min_value=0,
            max_value=max_total_reviews if max_total_reviews > 0 else 0,
            value=min(100, max_total_reviews), 
            step=max(10, max_total_reviews // 50) if max_total_reviews > 0 else 1
        )
        brand_stats = filtered.groupby("brand_name").agg(
            avg_rating=("rating", "mean"),
            total_reviews=("reviews", "sum")
        )
        brand_filtered = brand_stats[brand_stats["total_reviews"] >= N].sort_values("avg_rating", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=brand_filtered["avg_rating"], y=brand_filtered.index, ax=ax, color="#103db9")
        ax.set_xlabel("Average Rating")
        ax.set_ylabel("Brand")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[2]:
    st.subheader("Q3: What is the overall distribution of price_usd?")
    ok, miss = has_columns(filtered, ["price_usd"])
    if ok:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered["price_usd"].dropna(), bins=30, ax=ax, color="#f59e0b")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Count of Products")
        plt.xlim(0, 500)
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[3]:
    st.subheader("Q4: How does price_usd vary by primary_category?")
    ok, miss = has_columns(filtered, ["primary_category", "price_usd"])
    if ok:
        top_cats = (filtered["primary_category"].value_counts().head(12).index)
        df_plot = filtered[filtered["primary_category"].isin(top_cats)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_plot, x="price_usd", y="primary_category", ax=ax, color="#8b5cf6")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Primary Category (top 12 by count)")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[4]:
    st.subheader("Q5: Does price_usd relate to rating?")
    ok, miss = has_columns(filtered, ["price_usd", "rating"])
    if ok:
        sample = filtered[["price_usd", "rating"]].dropna()
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=sample, x="price_usd", y="rating", alpha=0.35, ax=ax, color="#ef4444")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Rating")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[5]:
    st.subheader("Q6: Do products with more reviews have higher ratings?")
    ok, miss = has_columns(filtered, ["reviews", "rating"])
    if ok:
        sample = filtered[["reviews", "rating"]].dropna()
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=sample, x="reviews", y="rating", alpha=0.35, ax=ax, color="#22c55e")
        ax.set_xlabel("Reviews (count)")
        ax.set_ylabel("Rating")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[6]:
    st.subheader("Q7: Which brand is most popular?")
    ok, miss = has_columns(filtered, ["brand_name", "loves_count"])
    if ok:
        pop = filtered.groupby("brand_name")["loves_count"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=pop.values, y=pop.index, ax=ax, color="#0ea5e9")
        ax.set_xlabel("Total Loves")
        ax.set_ylabel("Brand")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[7]:
    st.subheader("Q8: Which primary_category has the most reviews overall?")
    ok, miss = has_columns(filtered, ["primary_category", "reviews"])
    if ok:
        cat_reviews = filtered.groupby("primary_category")["reviews"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=cat_reviews.values, y=cat_reviews.index, ax=ax, color="#14b8a6")
        ax.set_xlabel("Total Reviews")
        ax.set_ylabel("Primary Category")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[8]:
    st.subheader("Q9: Do Sephora exclusive products have different ratings than non-exclusive?")
    ok, miss = has_columns(filtered, ["is_exclusive", "rating"])
    if ok:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=filtered, x="is_exclusive", y="rating", ax=ax, palette="Set2")
        ax.set_xlabel("Is Sephora Exclusive?")
        ax.set_ylabel("Rating")
        st.pyplot(fig)
    else:
        st.warning(f"Missing: {', '.join(miss)}")

with tabs[9]:
    st.subheader("Q10: Do new products have different prices/ratings than non-new products?")
    ok1, miss1 = has_columns(filtered, ["is_new", "price_usd"])
    ok2, miss2 = has_columns(filtered, ["is_new", "rating"])
    colA, colB = st.columns(2)

    with colA:
        if ok1:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=filtered, x="is_new", y="price_usd", ax=ax, palette="Pastel1")
            ax.set_xlabel("Is New?")
            ax.set_ylabel("Price (USD)")
            st.pyplot(fig)
        else:
            st.warning(f"Missing for price comparison: {', '.join(miss1)}")

    with colB:
        if ok2:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=filtered, x="is_new", y="rating", ax=ax, palette="Pastel2")
            ax.set_xlabel("Is New?")
            ax.set_ylabel("Rating")
            st.pyplot(fig)
        else:
            st.warning(f"Missing for rating comparison: {', '.join(miss2)}")
