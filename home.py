import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Healthcare EDA Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("healthcare_dataset.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Overview", "EDA Charts", "Conclusion"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("Healthcare Dataset – EDA Dashboard")
    st.markdown("""
    ### Exploratory Data Analysis Project
    This dashboard presents a structured exploratory data analysis (EDA)
    of the healthcare dataset using Streamlit.

    **Sections included:**
    - Dataset Overview
    - Statistical Summary
    - Visual Analysis
    - Key Observations
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ---------------- OVERVIEW ----------------
elif page == "Overview":
    st.title("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")

    with col2:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

# ---------------- EDA CHARTS ----------------
elif page == "EDA Charts":
    st.title("Exploratory Data Analysis Charts")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.subheader("Histogram")
    selected_col = st.selectbox("Select a numeric column", numeric_cols)

    fig, ax = plt.subplots()
    ax.hist(df[selected_col], bins=20)
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Bar Chart (Categorical)")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        cat_col = st.selectbox("Select a categorical column", cat_cols)
        fig, ax = plt.subplots()
        df[cat_col].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel(cat_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No categorical columns found.")

# ---------------- CONCLUSION ----------------
elif page == "Conclusion":
    st.title("Conclusion")

    st.markdown("""
    ### Key Findings
    - The dataset contains both numerical and categorical healthcare attributes.
    - Some variables show strong correlations, useful for predictive modeling.
    - Data distribution varies across features, indicating the presence of skewness.
    - Visual analysis helps in understanding trends and potential outliers.

    ### Final Note
    This EDA helps in preparing the dataset for further tasks such as
    machine learning, prediction, or decision-making analysis.
    """)

