import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 🔑 VERY IMPORTANT
import matplotlib.pyplot as plt
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Healthcare EDA Dashboard",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "healthcare_dataset.csv")
    return pd.read_csv(file_path)

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🏥 Healthcare Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Dataset", "EDA Charts", "Conclusion"]
)

# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":
    st.title("🏥 Healthcare EDA Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.success("Dataset loaded successfully")

# =========================================================
# DATASET
# =========================================================
elif page == "Dataset":
    st.title("📁 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# =========================================================
# EDA CHARTS
# =========================================================
elif page == "EDA Charts":
    st.title("📊 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    chart = st.selectbox(
        "Select Chart",
        [
            "Numeric Distribution",
            "Categorical Distribution",
            "Top Categories",
            "Correlation Heatmap"
        ]
    )

    # -------------------------------
    # Numeric Distribution
    # -------------------------------
    if chart == "Numeric Distribution" and len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------
    # Categorical Distribution
    # -------------------------------
    elif chart == "Categorical Distribution" and len(categorical_cols) > 0:
        col = categorical_cols[0]
        counts = df[col].value_counts()

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Distribution of {col}")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------
    # Top Categories
    # -------------------------------
    elif chart == "Top Categories" and len(categorical_cols) > 0:
        col = categorical_cols[0]
        top_vals = df[col].value_counts().head(10)

        fig, ax = plt.subplots()
        ax.barh(top_vals.index.astype(str), top_vals.values)
        ax.set_title(f"Top Categories in {col}")
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    elif chart == "Correlation Heatmap" and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots()
        cax = ax.imshow(corr, cmap="coolwarm")
        fig.colorbar(cax)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Required columns not available for this chart.")

# =========================================================
# CONCLUSION
# =========================================================
elif page == "Conclusion":
    st.title("💡 Key Insights & Conclusion")

    st.markdown("""
    - The dataset contains both numerical and categorical healthcare features.
    - Distribution analysis highlights variation across attributes.
    - Category-wise analysis shows dominant groups.
    - Correlation analysis helps identify related features.

    **This EDA helps in understanding data behavior before further analysis.**
    """)

    st.success("EDA completed successfully 🎉")
