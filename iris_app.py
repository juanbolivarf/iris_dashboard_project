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
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="Iris Species Dashboard",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Species Analytics & Prediction Dashboard")

st.markdown(
    """
This dashboard allows you to:

- Explore the **Iris dataset** (Setosa, Versicolor, Virginica)  
- Visualize feature relationships and species clusters  
- **Predict the species** of a flower **based only on its petal measurements**
    """
)

# ------------------------------------------------
# Load data
# ------------------------------------------------
@st.cache_data
def load_data(path: str = "Iris.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df

df = load_data()

# ------------------------------------------------
# Train model using only petal features
# ------------------------------------------------
FEATURE_COLS = ["PetalLengthCm", "PetalWidthCm"]
TARGET_COL = "Species"

@st.cache_resource
def train_petal_model(data: pd.DataFrame):
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline: scaler + KNN classifier
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc

model, test_accuracy = train_petal_model(df)

# ------------------------------------------------
# Sidebar filters
# ------------------------------------------------
st.sidebar.header("🔎 Filters")

species_options = df["Species"].unique().tolist()
species_selected = st.sidebar.multiselect(
    "Filter by Species",
    options=species_options,
    default=species_options
)

df_filtered = df[df["Species"].isin(species_selected)].copy()

if df_filtered.empty:
    st.error("No data available for the selected filters.")
    st.stop()

# ------------------------------------------------
# KPI cards
# ------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples (filtered)", len(df_filtered))

with col2:
    st.metric("Number of Species (filtered)", df_filtered["Species"].nunique())

with col3:
    st.metric("Petal-based Model Accuracy", f"{test_accuracy:.2%}")

st.markdown("---")

# ------------------------------------------------
# Scatter: Petal Length vs Petal Width
# ------------------------------------------------
st.subheader("📊 Petal Length vs Petal Width")

fig_scatter = px.scatter(
    df_filtered,
    x="PetalLengthCm",
    y="PetalWidthCm",
    color="Species",
    title="Petal Length vs Petal Width by Species",
    height=450
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------
# Pairplot (optional, inside expander)
# ------------------------------------------------
st.subheader("📈 Pairplot (All Features)")

with st.expander("Show pairplot (may take a few seconds)"):
    fig_pair = sns.pairplot(df_filtered, hue="Species")
    st.pyplot(fig_pair.fig)
    plt.close("all")

# ------------------------------------------------
# Correlation heatmap
# ------------------------------------------------
st.subheader("🔗 Feature Correlation Heatmap")

corr = df_filtered.drop(columns=["Species"]).corr()
fig_heat = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap — Iris Features",
    color_continuous_scale="Blues",
    height=500
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ------------------------------------------------
# Species prediction from petal characteristics
# ------------------------------------------------
st.header("🔮 Predict Species from Petal Measurements")

st.markdown(
    """
In this section, you can **simulate a new flower** by choosing its
**petal length** and **petal width**.  
The model was trained using only these two features.
"""
)

c1, c2 = st.columns(2)

with c1:
    petal_length = st.slider(
        "Petal Length (cm)",
        float(df["PetalLengthCm"].min()),
        float(df["PetalLengthCm"].max()),
        float(df["PetalLengthCm"].mean()),
        step=0.1,
    )

with c2:
    petal_width = st.slider(
        "Petal Width (cm)",
        float(df["PetalWidthCm"].min()),
        float(df["PetalWidthCm"].max()),
        float(df["PetalWidthCm"].mean()),
        step=0.1,
    )

if st.button("Predict species"):
    X_new = np.array([[petal_length, petal_width]])
    pred_species = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    st.success(f"**Predicted species:** {pred_species}")

    proba_df = pd.DataFrame(
        {"Species": model.classes_, "Probability": np.round(proba, 3)}
    )
    st.write("Prediction probabilities:")
    st.table(proba_df)

st.caption(
    "The predictive model uses a KNN classifier trained only with "
    "`PetalLengthCm` and `PetalWidthCm`, which are the most discriminative features."
)
git