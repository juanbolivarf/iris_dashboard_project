import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df

df = load_data()

# -------------------------------------------------
# UI Header
# -------------------------------------------------
st.set_page_config(page_title="Iris Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🌸 Iris Dataset EDA Dashboard - by Juan Bolívar Ferrer</h1>",
    unsafe_allow_html=True
)

st.write("### Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Filters
# -------------------------------------------------
species_list = sorted(df["Species"].unique().tolist())

selected_species = st.multiselect(
    "Select Species",
    species_list,
    default=species_list
)

filtered_df = df[df["Species"].isin(selected_species)]

# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
total_flowers = filtered_df.shape[0]
species_count = filtered_df["Species"].nunique()

col1, col2 = st.columns(2)
col1.metric("Total Flowers", total_flowers)
col2.metric("Selected Species Count", species_count)

# -------------------------------------------------
# Scatterplot (Sepal Length vs Petal Length)
# -------------------------------------------------
st.write("## Sepal Length vs Petal Length")

fig_scatter = px.scatter(
    filtered_df,
    x="SepalLengthCm",
    y="PetalLengthCm",
    color="Species",
    title="Sepal Length vs Petal Length",
    height=500
)

st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------
# Pairplot Section (FIXED)
# -------------------------------------------------
st.write("## Pairplot")

with st.expander("Show pairplot (may take a few seconds)"):
    st.write("Rendering pairplot...")

    fig = sns.pairplot(filtered_df, hue="Species")
    st.pyplot(fig)      # ✅ Dynamic seaborn rendering
    plt.close()

# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------
st.write("## Correlation Heatmap")

numeric_df = filtered_df.drop(columns=["Species"])
corr = numeric_df.corr()

plt.figure(figsize=(8, 5))
fig_corr = sns.heatmap(corr, annot=True, cmap="Blues")
st.pyplot(fig_corr.get_figure())
plt.close()
