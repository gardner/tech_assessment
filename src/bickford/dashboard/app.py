"""Streamlit dashboard for promptfoo evaluation results.

This dashboard visualizes RAG retrieval accuracy metrics from promptfoo evaluations,
comparing naive vs BM25 retrieval approaches across latency and accuracy dimensions.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

"""
# 🔍 RAG Retrieval Accuracy Dashboard

Comparing **Naive** vs **BM25** retrieval approaches across 141 test queries.
"""

""  # Add some space

# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_evaluation_results():
    """Load and parse promptfoo evaluation results."""
    results_path = Path("evaluations/results.json")

    if not results_path.exists():
        st.error(f"Results file not found: {results_path}")
        st.info("Run `./eval.sh` to generate evaluation results.")
        st.stop()

    with open(results_path) as f:
        data = json.load(f)

    return data

data = load_evaluation_results()

# ============================================================================
# Data Processing
# ===========================================================================

def extract_provider_name(provider_id):
    """Extract clean provider name from ID."""
    if "naive" in provider_id:
        return "Naive"
    elif "bm25" in provider_id:
        return "BM25"
    return provider_id

# Extract summary metrics from prompts
prompts_df = pd.DataFrame(data["results"]["prompts"])
prompts_df["provider_name"] = prompts_df["provider"].apply(extract_provider_name)

# Extract individual test results
results_list = []
for result in data["results"]["results"]:
    results_list.append({
        "provider": extract_provider_name(result["provider"]["id"]),
        "latency_ms": result["latencyMs"],
        "pass": result["gradingResult"]["pass"],
        "score": result["score"],
        "query": result["prompt"]["raw"][:60] + "...",  # Truncate for display
        "test_idx": result["testIdx"],
    })

results_df = pd.DataFrame(results_list)

# ============================================================================
# Summary Metrics
# ============================================================================

st.markdown("## 📊 Summary Metrics")

cols = st.columns(4)

for idx, (provider, group) in enumerate(prompts_df.groupby("provider_name")):
    with cols[idx]:
        metrics = group.iloc[0]["metrics"]

        pass_rate = (metrics["testPassCount"] /
                    (metrics["testPassCount"] + metrics["testFailCount"])) * 100

        st.container(border=True).metric(
            label=f"{provider} Pass Rate",
            value=f"{pass_rate:.1f}%",
            delta=f"{metrics['testPassCount']}/{metrics['testPassCount'] + metrics['testFailCount']}"
        )

for idx, (provider, group) in enumerate(prompts_df.groupby("provider_name")):
    with cols[idx + 2]:
        metrics = group.iloc[0]["metrics"]
        avg_latency = metrics["totalLatencyMs"] / metrics["tokenUsage"]["numRequests"]

        st.container(border=True).metric(
            label=f"{provider} Avg Latency",
            value=f"{avg_latency:.0f}ms",
            delta=f"{metrics['totalLatencyMs']:,}ms total"
        )


# ============================================================================
# Pass/Fail Comparison
# ============================================================================

st.markdown("## ✅ Pass/Fail Comparison")

pass_fail_data = results_df.groupby(["provider", "pass"]).size().reset_index(name="count")
pass_fail_data["status"] = pass_fail_data["pass"].map({True: "Pass", False: "Fail"})

chart = alt.Chart(pass_fail_data).mark_bar().encode(
    x=alt.X("provider:N", title="Provider"),
    y=alt.Y("count:Q", title="Number of Tests"),
    color=alt.Color("status:N",
                    scale=alt.Scale(domain=["Pass", "Fail"],
                                   range=["#2ecc71", "#e74c3c"]),
                    legend=alt.Legend(title="Result")),
    tooltip=["provider", "status", "count"]
).properties(
    height=300,
    title="Pass/Fail Distribution by Provider"
)

st.altair_chart(chart, width='stretch')

# ============================================================================
# Latency Analysis
# ============================================================================

st.markdown("## ⏱️ Latency Analysis")

# Calculate statistics per provider
latency_stats = results_df.groupby("provider")["latency_ms"].agg([
    ("mean", "mean"),
    ("std", "std"),
    ("median", "median"),
    ("min", "min"),
    ("max", "max"),
    ("p95", lambda x: x.quantile(0.95))
]).reset_index()

# Display statistics table
st.markdown("### Latency Statistics (milliseconds)")
st.dataframe(
    latency_stats.style.format({
        "mean": "{:.1f}",
        "std": "{:.1f}",
        "median": "{:.1f}",
        "min": "{:.0f}",
        "max": "{:.0f}",
        "p95": "{:.1f}"
    }),
    width='stretch',
    hide_index=True
)

# ============================================================================
# Latency Distribution Charts
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Latency Distribution (Box Plot)")

    box_chart = alt.Chart(results_df).mark_boxplot().encode(
        x=alt.X("provider:N", title="Provider"),
        y=alt.Y("latency_ms:Q", title="Latency (ms)", scale=alt.Scale(zero=False)),
        color=alt.Color("provider:N", legend=None)
    ).properties(height=400)

    st.altair_chart(box_chart, width='stretch')

with col2:
    st.markdown("### Mean Latency with Std Dev")

    # Create error bar chart
    error_chart = alt.Chart(latency_stats).mark_errorbar().encode(
        x=alt.X("provider:N", title="Provider"),
        y=alt.Y("mean:Q", title="Latency (ms)"),
        yError="std:Q",
    )

    points_chart = alt.Chart(latency_stats).mark_point(
        size=100,
        filled=True
    ).encode(
        x=alt.X("provider:N"),
        y=alt.Y("mean:Q"),
        color=alt.Color("provider:N", legend=None),
        tooltip =["provider", alt.Tooltip("mean:Q", format=".1f"),
                  alt.Tooltip("std:Q", format=".1f")]
    )

    combined_chart = (error_chart + points_chart).properties(height=400)

    st.altair_chart(combined_chart, width='stretch')

# ============================================================================
# Latency Histogram
# ============================================================================

st.markdown("### Latency Distribution (Histogram)")

hist_chart = alt.Chart(results_df).mark_bar(opacity=0.7).encode(
    x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=30), title="Latency (ms)"),
    y=alt.Y("count():Q", title="Frequency"),
    color=alt.Color("provider:N"),
    tooltip=["provider", alt.Tooltip("latency_ms:Q", aggregate="mean", format=".1f", title="Avg Latency")]
).properties(height=300)

st.altair_chart(hist_chart, width='stretch')

# ============================================================================
# Scatter Plot: Latency vs Test Index
# ============================================================================

st.markdown("### Latency Over Time (Test Sequence)")

scatter_chart = alt.Chart(results_df).mark_circle(size=60).encode(
    x=alt.X("test_idx:Q", title="Test Index"),
    y=alt.Y("latency_ms:Q", title="Latency (ms)", scale=alt.Scale(type='log')),
    color=alt.Color("provider:N"),
    shape=alt.Shape("pass:N", scale=alt.Scale(range=["circle", "cross"]),
                    legend=alt.Legend(title="Pass/Fail")),
    tooltip=["provider", "test_idx", "latency_ms", "pass", "query"]
).properties(height=400)

st.altair_chart(scatter_chart, width='stretch')

# ============================================================================
# Pass Rate by Latency Bucket
# ============================================================================

st.markdown("## 📈 Pass Rate by Latency Bucket")

# Create latency buckets
results_df["latency_bucket"] = pd.cut(
    results_df["latency_ms"],
    bins=[0, 100, 500, 1000, 2000, 5000],
    labels=["<100ms", "100-500ms", "500ms-1s", "1-2s", ">2s"]
)

bucket_stats = results_df.groupby(["provider", "latency_bucket"], observed=False).agg({
    "pass": ["sum", "count"]
}).reset_index()
bucket_stats.columns = ["provider", "latency_bucket", "pass_count", "total_count"]
bucket_stats["pass_rate"] = (bucket_stats["pass_count"] / bucket_stats["total_count"]) * 100

bucket_chart = alt.Chart(bucket_stats).mark_bar().encode(
    x=alt.X("latency_bucket:N", title="Latency Bucket"),
    y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("provider:N"),
    column=alt.Column("provider:N", title=None),
    tooltip=["provider", "latency_bucket", alt.Tooltip("pass_rate:Q", format=".1f"),
             "total_count"]
).properties(height=300, width=250)

st.altair_chart(bucket_chart)

# ============================================================================
# Detailed Results Table
# ============================================================================

st.markdown("## 📋 Detailed Results")

# Filter controls
col1, col2 = st.columns([1, 3])

with col1:
    provider_filter = st.multiselect(
        "Filter by Provider",
        options=results_df["provider"].unique(),
        default=results_df["provider"].unique()
    )

with col2:
    status_filter = st.multiselect(
        "Filter by Status",
        options=["Pass", "Fail"],
        default=["Pass", "Fail"]
    )

# Apply filters
filtered_df = results_df[
    (results_df["provider"].isin(provider_filter)) &
    (results_df["pass"].map({True: "Pass", False: "Fail"}).isin(status_filter))
]

# Display filtered results
st.dataframe(
    filtered_df[["provider", "query", "latency_ms", "pass", "score"]].style.format({
        "latency_ms": "{:.0f}",
        "score": "{:.2f}"
    }).map(lambda x: "background-color: #d4edda" if x == True else
                    ("background-color: #f8d7da" if x == False else ""), subset=["pass"]),
    width='stretch',
    hide_index=True
)

# ============================================================================
# Key Insights
# ============================================================================

st.markdown("## 💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance Summary")

    for provider in results_df["provider"].unique():
        provider_data = results_df[results_df["provider"] == provider]
        pass_rate = (provider_data["pass"].sum() / len(provider_data)) * 100
        avg_latency = provider_data["latency_ms"].mean()

        st.markdown(f"**{provider}:**")
        st.markdown(f"- Pass Rate: `{pass_rate:.1f}%`")
        st.markdown(f"- Average Latency: `{avg_latency:.0f}ms`")
        st.markdown(f"- Median Latency: `{provider_data['latency_ms'].median():.0f}ms`")
        st.markdown("")

with col2:
    st.markdown("### Recommendations")

    # Determine which provider is better
    naive_pass_rate = (results_df[results_df["provider"] == "Naive"]["pass"].sum() /
                      len(results_df[results_df["provider"] == "Naive"])) * 100
    bm25_pass_rate = (results_df[results_df["provider"] == "BM25"]["pass"].sum() /
                     len(results_df[results_df["provider"] == "BM25"])) * 100

    naive_latency = results_df[results_df["provider"] == "Naive"]["latency_ms"].mean()
    bm25_latency = results_df[results_df["provider"] == "BM25"]["latency_ms"].mean()

    if bm25_pass_rate > naive_pass_rate:
        st.success(f"✅ BM25 has {bm25_pass_rate - naive_pass_rate:.1f}% higher accuracy")
    else:
        st.info(f"ℹ️ Naive has {naive_pass_rate - bm25_pass_rate:.1f}% higher accuracy")

    if naive_latency < bm25_latency:
        st.success(f"⚡ Naive is {((bm25_latency - naive_latency) / bm25_latency * 100):.1f}% faster")
    else:
        st.info(f"⚡ BM25 is {((naive_latency - bm25_latency) / naive_latency * 100):.1f}% faster")

    # Trade-off analysis
    if bm25_pass_rate > naive_pass_rate and bm25_latency > naive_latency:
        st.warning("⚖️ **Trade-off:** BM25 offers better accuracy at the cost of increased latency")
    elif bm25_pass_rate < naive_pass_rate and bm25_latency < naive_latency:
        st.warning("⚖️ **Trade-off:** Naive offers better accuracy at the cost of increased latency")