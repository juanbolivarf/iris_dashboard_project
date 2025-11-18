import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Dashboard", layout="wide")
st.title("🌸 Iris Dataset EDA Dashboard")

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("Iris.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------
# Filters
# -------------------------
species_filter = st.multiselect(
    "Select Species",
    options=df["Species"].unique(),
    default=df["Species"].unique()
)

df_filtered = df[df["Species"].isin(species_filter)]

# -------------------------
# KPI cards
# -------------------------
col1, col2 = st.columns(2)
col1.metric("Total Flowers", len(df_filtered))
col2.metric("Selected Species Count", df_filtered["Species"].nunique())

# -------------------------
# Scatter plot
# -------------------------
st.subheader("Sepal Length vs Petal Length")
fig_scatter = px.scatter(
    df_filtered,
    x="SepalLengthCm",
    y="PetalLengthCm",
    color="Species",
    title="Sepal Length vs Petal Length",
    height=450
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------
# Pairplot (dynamic – no PNG needed)
# -------------------------
st.subheader("Pairplot")

with st.expander("Show pairplot (may take a few seconds)"):
    # seaborn pairplot returns a PairGrid; we take its underlying figure
    g = sns.pairplot(df_filtered, hue="Species")
    st.pyplot(g.fig)
    plt.close("all")  # clean up the figure in memory

# -------------------------
# Correlation heatmap
# -------------------------
st.subheader("Correlation Heatmap")
numeric_df = df_filtered.drop(columns=["Species"])
corr = numeric_df.corr()

fig_heat = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap",
    color_continuous_scale="Blues",
    height=500
)
st.plotly_chart(fig_heat, use_container_width=True)

