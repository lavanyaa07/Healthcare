import streamlit as st
import pandas as pd
import seaborn as sns
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
# LOAD DATA (CLOUD SAFE)
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
st.sidebar.markdown("Exploratory Data Analysis")

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "Overview",
        "Dataset",
        "Exploratory Data Analysis",
        "Key Insights & Conclusion"
    ]
)

# =========================================================
# OVERVIEW PAGE
# =========================================================
if page == "Overview":
    st.title("🏥 Healthcare Exploratory Data Analysis Dashboard")

    st.markdown("""
    ### 📘 Project Objective
    This dashboard performs **Exploratory Data Analysis (EDA)** on a healthcare dataset
    to identify trends, distributions, and patterns in patient-related data.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Total Missing Values", int(df.isnull().sum().sum()))

    st.success("✅ Dataset loaded successfully")

# =========================================================
# DATASET PAGE
# =========================================================
elif page == "Dataset":
    st.title("📁 Dataset Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

    with col2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

# =========================================================
# EDA PAGE
# =========================================================
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    chart = st.selectbox(
        "Select Analysis",
        [
            "Numeric Distribution",
            "Categorical Distribution",
            "Top Categories",
            "Value Distribution",
            "Correlation Heatmap"
        ]
    )

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # -------------------------------
    # Numeric Distribution
    # -------------------------------
    if chart == "Numeric Distribution":
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available.")

    # -------------------------------
    # Categorical Distribution
    # -------------------------------
    elif chart == "Categorical Distribution":
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            counts = df[col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available.")

    # -------------------------------
    # Top Categories
    # -------------------------------
    elif chart == "Top Categories":
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            top_vals = df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax)
            ax.set_title(f"Top Categories in {col}")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available.")

    # -------------------------------
    # Value Distribution
    # -------------------------------
    elif chart == "Value Distribution":
        if len(numeric_cols) > 0:
            col = numeric_cols[-1]
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[col].dropna(), bins=40, kde=False, ax=ax)
            ax.set_title(f"Value Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available.")

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    elif chart == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                df[numeric_cols].corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation.")

# =========================================================
# CONCLUSION PAGE
# =========================================================
elif page == "Key Insights & Conclusion":
    st.title("💡 Key Insights & Conclusion")

    st.markdown("""
    ### 🔍 Key Insights
    - Numerical features show varying distributions and ranges.
    - Certain categories dominate patient records.
    - Missing values exist and may require preprocessing.
    - Correlation analysis helps identify related healthcare attributes.

    ### ✅ Final Conclusion
    This EDA provides a strong foundation for understanding healthcare data
    and supports further statistical analysis or machine learning tasks.
    """)

    st.success("🎉 EDA completed successfully")
