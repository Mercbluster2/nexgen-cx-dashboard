import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NexGen: Customer Experience Dashboard", layout="wide")

ORDER_KEYS = ["OrderID","order_id","Order_Id","Order_ID","orderId"]
CF_DATE = ["FeedbackDate","feedback_date","Feedback_Date","Date","date","OrderDate","order_date"]
CF_TEXT = ["FeedbackText","feedback_text","Feedback_Text","Comment","comment","Review","review","Feedback"]
CF_RATING = ["Rating","rating","Stars","stars","Customer_Rating","User_Rating"]
CF_ISSUE = ["IssueCategory","issue_category","Issue_Category","Issue","issue"]
OD_SEGMENT = ["CustomerSegment","customer_segment","Customer_Segment","Segment","segment"]
OD_PRIORITY = ["Priority","priority"]
OD_PRODUCT = ["ProductCategory","product_category","Product_Category","Category","category"]
DP_PROMISED = ["PromisedDeliveryDate","promised_delivery_date","PromisedDate","Promised_Date"]
DP_ACTUAL = ["ActualDeliveryDate","actual_delivery_date","DeliveredDate","Actual_Date"]
DP_DELAY = ["DelayDays","Delay_Days","Delay","delay","DelayInDays"]

def find_first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

@st.cache_data(show_spinner=False)
def load_table(f):
    try:
        return pd.read_csv(f)
    except Exception:
        try:
            f.seek(0)
        except Exception:
            pass
        try:
            return pd.read_excel(f)
        except Exception:
            return None

def parse_dt(s):
    x = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if x.isna().all():
        x = pd.to_datetime(s.astype(str), errors="coerce", dayfirst=True, infer_datetime_format=True)
    return x

def classify(df, fname):
    cols = set(df.columns)
    if cols & set(CF_TEXT) or cols & set(CF_RATING): return "customer_feedback"
    if cols & set(DP_PROMISED) or cols & set(DP_ACTUAL) or cols & set(DP_DELAY): return "delivery_performance"
    if cols & set(OD_SEGMENT) or cols & set(OD_PRODUCT) or cols & set(OD_PRIORITY): return "orders"
    low = fname.lower()
    if "feedback" in low: return "customer_feedback"
    if "delivery" in low: return "delivery_performance"
    if "order" in low: return "orders"
    return None

def simple_sentiment(series):
    pos = {"great","good","fast","excellent","timely","smooth","happy","recommend","satisfied","perfect","on time"}
    neg = {"bad","poor","slow","late","delay","delayed","damaged","broken","worst","unhappy","wrong","not good","not happy"}
    labels, scores = [], []
    for t in series.fillna("").astype(str):
        tl = t.lower()
        p = sum(w in tl for w in pos)
        n = sum(w in tl for w in neg)
        sc = (p-n)/max(1,(p+n))
        scores.append(sc)
        labels.append("Positive" if sc>0.1 else "Negative" if sc<-0.1 else "Neutral")
    return pd.Series(labels), pd.Series(scores)

uploaded = st.sidebar.file_uploader("Upload CSV/XLSX files", type=["csv","xlsx"], accept_multiple_files=True)
dfs = {"customer_feedback": None, "delivery_performance": None, "orders": None}
if uploaded:
    for f in uploaded:
        df = load_table(f)
        if df is not None and not df.empty:
            kind = classify(df, f.name)
            if kind and dfs[kind] is None:
                dfs[kind] = df

if not any(v is not None for v in dfs.values()) and os.path.isdir("data"):
    for key, name in [("customer_feedback","customer_feedback.csv"),("delivery_performance","delivery_performance.csv"),("orders","orders.csv")]:
        p = os.path.join("data", name)
        if os.path.exists(p):
            try:
                dfs[key] = pd.read_csv(p)
            except Exception:
                try:
                    dfs[key] = pd.read_excel(p)
                except Exception:
                    pass

cf, dp, od = dfs["customer_feedback"], dfs["delivery_performance"], dfs["orders"]

order_key_cf = cf_date_col = cf_text_col = cf_rating_col = cf_issue_col = None
order_key_dp = promised_col = actual_col = delay_col = None
order_key_od = seg_col = prio_col = prod_col = None

if cf is not None:
    order_key_cf = find_first(cf, ORDER_KEYS)
    cf_date_col = find_first(cf, CF_DATE)
    cf_text_col = find_first(cf, CF_TEXT)
    cf_rating_col = find_first(cf, CF_RATING)
    cf_issue_col = find_first(cf, CF_ISSUE)
    if cf_text_col:
        lab, sc = simple_sentiment(cf[cf_text_col])
        cf["Sentiment"] = lab
        cf["SentimentScore"] = sc
    if cf_rating_col:
        cf[cf_rating_col] = pd.to_numeric(cf[cf_rating_col], errors="coerce")

if dp is not None:
    order_key_dp = find_first(dp, ORDER_KEYS)
    promised_col = find_first(dp, DP_PROMISED)
    actual_col = find_first(dp, DP_ACTUAL)
    delay_col = find_first(dp, DP_DELAY)
    if promised_col and actual_col:
        dp[promised_col] = parse_dt(dp[promised_col])
        dp[actual_col] = parse_dt(dp[actual_col])
        dp["DelayDays"] = (dp[actual_col] - dp[promised_col]).dt.days
    elif delay_col:
        dp["DelayDays"] = pd.to_numeric(dp[delay_col], errors="coerce")
    else:
        dp["DelayDays"] = np.nan

if od is not None:
    order_key_od = find_first(od, ORDER_KEYS)
    seg_col = find_first(od, OD_SEGMENT)
    prio_col = find_first(od, OD_PRIORITY)
    prod_col = find_first(od, OD_PRODUCT)

merged = None
if cf is not None:
    merged = cf.copy()
    for tdf, tkey in [(dp, order_key_dp), (od, order_key_od)]:
        if tdf is not None and order_key_cf and tkey:
            merged = merged.merge(tdf, left_on=order_key_cf, right_on=tkey, how="left", suffixes=("","_r"))

st.title("💬 Customer Experience Dashboard — NexGen Logistics")
st.caption("Identify at-risk customers, analyze sentiment trends, and link delivery performance to feedback.")

st.sidebar.markdown("**Loaded:** " + (", ".join([k for k,v in dfs.items() if v is not None]) if any(v is not None for v in dfs.values()) else "_none_"))

filtered = None
if merged is not None and len(merged):
    if cf_date_col in merged:
        mmin = pd.to_datetime(merged[cf_date_col], errors="coerce").min()
        mmax = pd.to_datetime(merged[cf_date_col], errors="coerce").max()
        if pd.isna(mmin) or pd.isna(mmax):
            from datetime import datetime as _dt
            mmin, mmax = _dt(2024,1,1), _dt.today()
        dr = st.sidebar.date_input("Feedback Date Range", value=(mmin, mmax))
    else:
        dr = None
    filtered = merged.copy()
    mask = pd.Series(True, index=filtered.index)
    if dr and cf_date_col in filtered:
        dts = pd.to_datetime(filtered[cf_date_col], errors="coerce")
        mask &= (dts >= pd.to_datetime(dr[0])) & (dts <= pd.to_datetime(dr[1]))
    if seg_col in filtered:
        seg_opts = sorted(filtered[seg_col].dropna().unique())
        seg_sel = st.sidebar.multiselect("Customer Segment", seg_opts, default=seg_opts)
        if seg_sel:
            mask &= filtered[seg_col].isin(seg_sel)
    if prio_col in filtered:
        prio_opts = sorted(filtered[prio_col].dropna().unique())
        prio_sel = st.sidebar.multiselect("Delivery Priority", prio_opts, default=prio_opts)
        if prio_sel:
            mask &= filtered[prio_col].isin(prio_sel)
    if prod_col in filtered:
        prod_opts = sorted(filtered[prod_col].dropna().unique())
        prod_sel = st.sidebar.multiselect("Product Category", prod_opts, default=prod_opts)
        if prod_sel:
            mask &= filtered[prod_col].isin(prod_sel)
    filtered = filtered[mask]

if filtered is None or not len(filtered):
    st.warning("Upload your files or place them in ./data to begin.")
else:
    c1, c2, c3, c4 = st.columns(4)
    if cf_rating_col in filtered and filtered[cf_rating_col].notna().any():
        c1.metric("Average Rating", f"{filtered[cf_rating_col].mean():.2f}")
    else:
        c1.metric("Average Rating", "—")
    if "Sentiment" in filtered:
        pos = (filtered["Sentiment"]=="Positive").mean()*100
        c2.metric("% Positive Sentiment", f"{pos:.1f}%")
    else:
        c2.metric("% Positive Sentiment","—")
    if "DelayDays" in filtered and pd.to_numeric(filtered["DelayDays"], errors="coerce").notna().any():
        c3.metric("Average Delivery Delay (days)", f"{pd.to_numeric(filtered['DelayDays'], errors='coerce').mean():.2f}")
    else:
        c3.metric("Average Delivery Delay (days)", "—")
    if cf_rating_col in filtered:
        if cf_issue_col in filtered:
            comp = ((filtered[cf_issue_col].notna()) & (filtered[cf_rating_col] <= 2)).mean()*100
        else:
            comp = (filtered[cf_rating_col] <= 2).mean()*100
        c4.metric("Complaint Rate", f"{comp:.1f}%")
    else:
        c4.metric("Complaint Rate","—")

    st.subheader("📈 Trends & Distributions")
    if cf_rating_col in filtered and cf_date_col in filtered:
        tmp = filtered[[cf_date_col, cf_rating_col]].dropna()
        if len(tmp):
            tmp[cf_date_col] = pd.to_datetime(tmp[cf_date_col], errors="coerce")
            trend = tmp.groupby(pd.Grouper(key=cf_date_col, freq="W")).mean().reset_index()
            st.plotly_chart(px.line(trend, x=cf_date_col, y=cf_rating_col, title="Average Rating Over Time (Weekly)"), use_container_width=True)
    if "Sentiment" in filtered:
        sent = filtered["Sentiment"].value_counts(dropna=False).reset_index()
        sent.columns = ["Sentiment","Count"]
        st.plotly_chart(px.pie(sent, names="Sentiment", values="Count", title="Sentiment Distribution"), use_container_width=True)
    if cf_issue_col in filtered:
        issue = filtered[cf_issue_col].fillna("Unspecified").value_counts().reset_index()
        issue.columns = ["Issue","Count"]
        st.plotly_chart(px.bar(issue, x="Issue", y="Count", title="Issue Frequency"), use_container_width=True)
    if "DelayDays" in filtered and cf_rating_col in filtered:
        tmp = filtered[[cf_rating_col,"DelayDays"]].dropna()
        if len(tmp):
            tmp["DelayDays"] = pd.to_numeric(tmp["DelayDays"], errors="coerce")
            tmp = tmp.dropna()
            if len(tmp):
                st.plotly_chart(px.scatter(tmp, x="DelayDays", y=cf_rating_col, trendline="ols", title="Delay vs Rating"), use_container_width=True)

    
    tabs = st.tabs(["By Product Category","By Customer Segment"])
    with tabs[0]:
        if prod_col in filtered and cf_rating_col in filtered:
            cat = filtered.groupby(prod_col)[cf_rating_col].mean().reset_index().sort_values(cf_rating_col, ascending=False)
            st.plotly_chart(px.bar(cat, x=prod_col, y=cf_rating_col, title="Avg Rating by Product Category"), use_container_width=True)
    with tabs[1]:
        if seg_col in filtered and cf_rating_col in filtered:
            seg = filtered.groupby(seg_col)[cf_rating_col].mean().reset_index().sort_values(cf_rating_col, ascending=False)
            st.plotly_chart(px.bar(seg, x=seg_col, y=cf_rating_col, title="Avg Rating by Customer Segment"), use_container_width=True)

    st.header("🧾 Negative Feedback Word Cloud")
    if "Sentiment" in filtered and cf_text_col in filtered:
        neg = filtered.loc[filtered["Sentiment"]=="Negative", cf_text_col].dropna().astype(str)
        if len(neg)==0:
            neg = filtered[cf_text_col].dropna().astype(str)
        try:
            from wordcloud import WordCloud
            if len(neg)>0:
                wc = WordCloud(width=1000, height=400, background_color="white").generate(" ".join(neg))
                st.image(wc.to_array(), use_container_width=True, caption="Negative Feedback Word Cloud")
            else:
                st.info("No feedback text available.")
        except Exception:
            st.info("Install wordcloud to view this section.")

    st.subheader("🚨 At-Risk Customers")
    if cf_rating_col in filtered:
        ar = filtered.copy()
        cond = ar[cf_rating_col] <= 2
        if "Sentiment" in ar: cond = cond | (ar["Sentiment"]=="Negative")
        if "DelayDays" in ar: cond = cond | (pd.to_numeric(ar["DelayDays"], errors="coerce") > 2)
        ar = ar.loc[cond]
        cols = []
        for c in [find_first(cf, ORDER_KEYS) if cf is not None else None, cf_rating_col, "Sentiment", "DelayDays", cf_issue_col, seg_col, prio_col, prod_col, cf_text_col]:
            if c and c in ar.columns and c not in cols:
                cols.append(c)
        st.dataframe(ar[cols] if cols else ar, use_container_width=True)
        st.download_button("Download At-Risk Customers CSV", data=ar.to_csv(index=False).encode("utf-8"), file_name="at_risk_customers.csv", mime="text/csv")

st.caption("Built with Streamlit — Option 4: Customer Experience Dashboard")
