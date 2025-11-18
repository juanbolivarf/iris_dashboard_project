import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

st.title("🌸 Iris Dataset EDA Dashboard")

df = pd.read_csv("Iris.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Filters
species_filter = st.multiselect("Select Species", df["Species"].unique(), default=df["Species"].unique())
df_filtered = df[df["Species"].isin(species_filter)]

# KPI cards
col1, col2 = st.columns(2)
col1.metric("Total Flowers", len(df_filtered))
col2.metric("Selected Species Count", df_filtered["Species"].nunique())

# Scatter plot
fig = px.scatter(
    df_filtered,
    x="SepalLengthCm", y="PetalLengthCm",
    color="Species",
    title="Sepal Length vs Petal Length"
)
st.plotly_chart(fig)

# Pairplot
st.subheader("Pairplot")
st.image("pairplot.png")  # If you want, I can generate it dynamically

# Heatmap
st.subheader("Correlation Heatmap")
fig2 = px.imshow(df_filtered.drop("Species", axis=1).corr(), text_auto=True, title="Correlation Heatmap")
st.plotly_chart(fig2)
