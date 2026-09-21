"""
Nexariza AI Analytics Brain — Capstone Dashboard
Week 6 deliverable | Nexariza AI AI/ML Internship

A unified Business Intelligence dashboard combining:
- Engagement forecasting (Prophet)
- Anomaly detection & alerts (rolling z-score)
- Lead scoring (XGBoost, interactive lookup)
- Sentiment analysis of mentions/reviews (VADER)

Run with:  streamlit run dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Nexariza AI Analytics Brain", page_icon="📊", layout="wide")

st.title("📊 Nexariza AI Analytics Brain")
st.caption("Capstone dashboard — engagement forecasting, anomaly alerts, lead scoring, and sentiment analysis in one view.")


@st.cache_data
def load_engagement():
    df = pd.read_csv("engagement_timeseries.csv", parse_dates=["date"])
    df["rolling_mean"] = df["engagement"].rolling(14, min_periods=7).mean()
    df["rolling_std"] = df["engagement"].rolling(14, min_periods=7).std()
    df["z_score"] = (df["engagement"] - df["rolling_mean"]) / df["rolling_std"]
    return df


@st.cache_data
def load_leads():
    return pd.read_csv("leads_dataset.csv")


@st.cache_data
def load_mentions():
    return pd.read_csv("mentions_dataset.csv", parse_dates=["date"])


@st.cache_resource
def load_lead_model():
    model = joblib.load("models/lead_scoring_model.pkl")
    features = joblib.load("models/lead_feature_names.pkl")
    return model, features


@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()


@st.cache_data
def run_forecast(df, periods=30):
    from prophet import Prophet
    prophet_df = df[["date", "engagement"]].rename(columns={"date": "ds", "engagement": "y"})
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast


tab_forecast, tab_leads, tab_sentiment, tab_overview = st.tabs(
    ["📈 Forecast & Alerts", "🎯 Lead Scoring", "💬 Sentiment", "🏠 Overview"]
)

eng_df = load_engagement()
leads_df = load_leads()
mentions_df = load_mentions()

# ==========================================================================
with tab_forecast:
    st.subheader("Engagement Forecast (Prophet, 30-day horizon)")
    with st.spinner("Fitting forecast model..."):
        forecast = run_forecast(eng_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eng_df["date"], y=eng_df["engagement"], name="Actual", line=dict(color="#333")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#3a86ff")))
    fig.add_trace(go.Scatter(
        x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
        y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
        fill="toself", fillcolor="rgba(58,134,255,0.15)", line=dict(color="rgba(255,255,255,0)"),
        name="Confidence interval", showlegend=True,
    ))
    fig.update_layout(title="Daily Engagement — History + 30-Day Forecast", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔔 Anomaly Alerts")
    threshold = st.slider("Alert sensitivity (z-score threshold)", 1.5, 4.0, 2.5, 0.1)
    anomalies = eng_df[eng_df["z_score"].abs() > threshold].copy()
    anomalies["alert_type"] = np.where(anomalies["z_score"] > 0, "📈 Spike", "📉 Drop")

    if anomalies.empty:
        st.success("No anomalies detected at this sensitivity level.")
    else:
        for _, row in anomalies.sort_values("date", ascending=False).iterrows():
            severity = "🔴" if abs(row["z_score"]) > 3 else "🟠"
            st.warning(
                f"{severity} **{row['alert_type']}** on {row['date'].date()} — "
                f"engagement was {row['engagement']:.0f} (z-score: {row['z_score']:.2f})"
            )

# ==========================================================================
with tab_leads:
    st.subheader("Lead Conversion Scoring")

    model, feature_names = load_lead_model()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Score a new lead")
        company_size = st.selectbox("Company size", ["1-10", "11-50", "51-200", "201-1000", "1000+"])
        industry = st.selectbox("Industry", ["SaaS", "E-commerce", "Healthcare", "Finance", "Manufacturing", "Other"])
        lead_source = st.selectbox("Lead source", ["Organic Search", "Paid Ads", "Referral", "Content/Webinar", "Cold Outreach"])
        website_visits = st.slider("Website visits", 0, 30, 6)
        content_downloads = st.slider("Content downloads", 0, 10, 1)
        demo_requested = st.checkbox("Demo requested", value=False)
        email_opens = st.slider("Email opens (last 30d)", 0, 20, 4)
        days_since = st.slider("Days since first touch", 1, 180, 30)
        budget_confirmed = st.checkbox("Budget confirmed", value=False)
        dm_contact = st.checkbox("Decision-maker contacted", value=False)

        if st.button("🎯 Score This Lead", type="primary"):
            row = {
                "website_visits": website_visits, "content_downloads": content_downloads,
                "demo_requested": int(demo_requested), "email_opens_last_30d": email_opens,
                "days_since_first_touch": days_since, "budget_confirmed": int(budget_confirmed),
                "decision_maker_contact": int(dm_contact),
            }
            for cs in ["1-10", "11-50", "51-200", "201-1000", "1000+"]:
                row[f"company_size_{cs}"] = int(company_size == cs)
            for ind in ["SaaS", "E-commerce", "Healthcare", "Finance", "Manufacturing", "Other"]:
                row[f"industry_{ind}"] = int(industry == ind)
            for src in ["Organic Search", "Paid Ads", "Referral", "Content/Webinar", "Cold Outreach"]:
                row[f"lead_source_{src}"] = int(lead_source == src)

            X_row = pd.DataFrame([row])
            for col in feature_names:
                if col not in X_row.columns:
                    X_row[col] = 0
            X_row = X_row[feature_names]

            prob = model.predict_proba(X_row)[0, 1]
            st.session_state["lead_score"] = prob

        if "lead_score" in st.session_state:
            prob = st.session_state["lead_score"]
            tier = "🟢 High" if prob >= 0.6 else ("🟡 Medium" if prob >= 0.3 else "🔴 Low")
            st.metric("Conversion Probability", f"{prob:.1%}")
            st.markdown(f"### Priority: {tier}")

    with col2:
        st.markdown("#### Top Conversion Drivers (model-wide)")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
        fig_imp = px.bar(x=importances.values, y=importances.index, orientation="h",
                          title="Feature Importance", labels={"x": "Importance", "y": ""})
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Conversion Rate by Source")
        conv_by_source = leads_df.groupby("lead_source")["converted"].mean().sort_values()
        fig_src = px.bar(x=conv_by_source.values, y=conv_by_source.index, orientation="h",
                          title="", labels={"x": "Conversion rate", "y": ""})
        st.plotly_chart(fig_src, use_container_width=True)

# ==========================================================================
with tab_sentiment:
    st.subheader("Customer Sentiment Monitor")

    analyzer = get_sentiment_analyzer()

    @st.cache_data
    def score_sentiment(df):
        df = df.copy()
        scores = df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
        df["sentiment_score"] = scores
        df["sentiment"] = pd.cut(scores, bins=[-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"])
        return df

    scored = score_sentiment(mentions_df)

    sources = st.multiselect("Filter by source", scored["source"].unique().tolist(), default=scored["source"].unique().tolist())
    filtered = scored[scored["source"].isin(sources)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Mentions", len(filtered))
    c2.metric("Avg. Sentiment Score", f"{filtered['sentiment_score'].mean():.2f}")
    pct_negative = (filtered["sentiment"] == "Negative").mean()
    c3.metric("% Negative", f"{pct_negative:.1%}", delta=None)

    col1, col2 = st.columns(2)
    with col1:
        dist = filtered["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
        fig_dist = px.pie(values=dist.values, names=dist.index, title="Sentiment Distribution",
                           color=dist.index, color_discrete_map={"Positive": "#2b9348", "Neutral": "#fb8500", "Negative": "#d90429"})
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        by_source = filtered.groupby("source")["sentiment_score"].mean().sort_values()
        fig_src = px.bar(x=by_source.values, y=by_source.index, orientation="h", title="Avg. Sentiment by Source")
        st.plotly_chart(fig_src, use_container_width=True)

    st.markdown("#### Sentiment Over Time")
    daily_sentiment = filtered.groupby(filtered["date"].dt.to_period("W"))["sentiment_score"].mean()
    daily_sentiment.index = daily_sentiment.index.to_timestamp()
    fig_time = px.line(x=daily_sentiment.index, y=daily_sentiment.values, title="Weekly Average Sentiment Score")
    fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("#### Recent Mentions")
    st.dataframe(
        filtered[["date", "source", "text", "sentiment", "sentiment_score"]].sort_values("date", ascending=False).head(30),
        use_container_width=True,
    )

# ==========================================================================
with tab_overview:
    st.subheader("Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. Daily Engagement (30d)", f"{eng_df['engagement'].tail(30).mean():.0f}")
    c2.metric("Active Anomaly Alerts", len(eng_df[eng_df["z_score"].abs() > 2.5]))
    c3.metric("Lead Conversion Rate", f"{leads_df['converted'].mean():.1%}")

    scored_overview = mentions_df.copy()
    analyzer2 = get_sentiment_analyzer()
    scored_overview["score"] = scored_overview["text"].apply(lambda t: analyzer2.polarity_scores(t)["compound"])
    c4.metric("Avg. Sentiment Score", f"{scored_overview['score'].mean():.2f}")

    st.markdown("""
    This capstone system brings together four ML/analytics components built throughout the internship:
    - **Forecasting** (Prophet + ARIMA) for planning content/campaign capacity
    - **Anomaly detection** (rolling z-score) for early warning on engagement issues
    - **Lead scoring** (XGBoost) for sales prioritization
    - **Sentiment analysis** (VADER) for brand health monitoring

    Explore each in the tabs above.
    """)
