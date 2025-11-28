import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Dataset EDA Dashboard",
    page_icon="🌸",
    layout="wide",
)

# -------------------------------------------------------------------
# 1. Data loading
# -------------------------------------------------------------------
@st.cache_data
def load_data(csv_path: str = "Iris.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop Id if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    return df


# -------------------------------------------------------------------
# 2. Train simple model using only petal features
# -------------------------------------------------------------------
@st.cache_resource
def train_petal_model(df: pd.DataFrame):
    """
    Trains a multinomial Logistic Regression model using only
    PetalLengthCm and PetalWidthCm to predict Species.
    """
    from sklearn.model_selection import train_test_split

    X = df[["PetalLengthCm", "PetalWidthCm"]]
    y = df["Species"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        multi_class="multinomial", max_iter=200, random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    df = load_data()

    st.title("🌸 Iris Dataset EDA Dashboard - by Juan Bolívar Ferrer")

    # ---------------- Sidebar filters ----------------
    with st.sidebar:
        st.header("Filters")

        species_options = sorted(df["Species"].unique())
        selected_species = st.multiselect(
            "Select Species",
            options=species_options,
            default=species_options,
        )

    if not selected_species:
        st.warning("Please select at least one species from the sidebar.")
        return

    filtered_df = df[df["Species"].isin(selected_species)]

    # ---------------- Dataset preview ----------------
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)

    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric("Total Flowers (filtered)", len(filtered_df))
    with kpi_col2:
        st.metric("Selected Species Count", len(selected_species))

    # ---------------- Sepal vs Petal scatter ----------------
    st.subheader("Sepal Length vs Petal Length")
    fig_scatter = px.scatter(
        filtered_df,
        x="SepalLengthCm",
        y="PetalLengthCm",
        color="Species",
        labels={
            "SepalLengthCm": "Sepal Length (cm)",
            "PetalLengthCm": "Petal Length (cm)",
        },
        height=450,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------------- Pairplot (no PNG file!) ----------------
    st.subheader("Pairplot")
    with st.expander("Show pairplot (may take a few seconds)"):
        sns.set(style="ticks")
        g = sns.pairplot(filtered_df, hue="Species", diag_kind="kde")
        st.pyplot(g.fig)
        plt.close("all")

    # ---------------- Correlation heatmap ----------------
    st.subheader("Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig_corr, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # ---------------- Prediction section ----------------
    st.subheader("🔮 Predict Species from Petal Measurements")
    st.write(
        "Use the petal length and width to predict the iris species with a "
        "Logistic Regression model trained only on petal features."
    )

    model, scaler = train_petal_model(df)

    c1, c2 = st.columns(2)
    with c1:
        petal_length = st.slider(
            "Petal length (cm)",
            float(df["PetalLengthCm"].min()),
            float(df["PetalLengthCm"].max()),
            float(df["PetalLengthCm"].mean()),
        )
    with c2:
        petal_width = st.slider(
            "Petal width (cm)",
            float(df["PetalWidthCm"].min()),
            float(df["PetalWidthCm"].max()),
            float(df["PetalWidthCm"].mean()),
        )

    if st.button("Predict Species"):
        x_new = np.array([[petal_length, petal_width]])
        x_new_scaled = scaler.transform(x_new)
        pred_species = model.predict(x_new_scaled)[0]
        proba = model.predict_proba(x_new_scaled)[0]

        st.success(f"Predicted species: **{pred_species}**")

        proba_df = pd.DataFrame(
            {"Species": model.classes_, "Probability": proba}
        )
        proba_df = proba_df.set_index("Species")
        st.bar_chart(proba_df)


if __name__ == "__main__":
    main()

