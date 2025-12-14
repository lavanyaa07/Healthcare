import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Healthcare EDA Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset (Cloud-safe)
# --------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "healthcare_dataset.csv")
    return pd.read_csv(file_path)

df = load_data()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Overview", "EDA Charts", "Conclusion"]
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":
    st.title("Healthcare Dataset – Exploratory Data Analysis")

    st.markdown("""
    ### Project Overview
    This Streamlit dashboard presents a structured Exploratory Data Analysis (EDA)
    on a healthcare dataset. The analysis focuses on understanding data distribution,
    relationships between variables, and key patterns useful for decision-making
    and predictive modeling.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# ==================================================
# OVERVIEW PAGE
# ==================================================
elif page == "Overview":
    st.title("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        st.write(f"**Total Rows:** {df.shape[0]}")
        st.write(f"**Total Columns:** {df.shape[1]}")

    with col2:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

# ==================================================
# EDA CHARTS PAGE
# ==================================================
elif page == "EDA Charts":
    st.title("Exploratory Data Analysis Charts")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # -------------------------------
    # Histogram
    # -------------------------------
    st.subheader("Univariate Analysis – Histogram")

    selected_num = st.selectbox("Select Numeric Column", num_cols)

    fig, ax = plt.subplots()
    ax.hist(df[selected_num], bins=25, edgecolor="black")
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {selected_num}")
    st.pyplot(fig)

    # -------------------------------
    # Box Plot
    # -------------------------------
    st.subheader("Outlier Detection – Box Plot")

    fig, ax = plt.subplots()
    sns.boxplot(x=df[selected_num], ax=ax)
    ax.set_title(f"Box Plot of {selected_num}")
    st.pyplot(fig)

    # -------------------------------
    # Scatter Plot
    # -------------------------------
    st.subheader("Bivariate Analysis – Scatter Plot")

    col_x = st.selectbox("X-axis", num_cols, key="x_axis")
    col_y = st.selectbox("Y-axis", num_cols, key="y_axis")

    fig, ax = plt.subplots()
    ax.scatter(df[col_x], df[col_y], alpha=0.6)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"{col_x} vs {col_y}")
    st.pyplot(fig)

    # -------------------------------
    # Categorical vs Numeric
    # -------------------------------
    st.subheader("Categorical vs Numeric Analysis")

    if cat_cols:
        cat_col = st.selectbox("Categorical Column", cat_cols)
        val_col = st.selectbox("Numeric Column", num_cols, key="val")

        grouped = df.groupby(cat_col)[val_col].mean()

        fig, ax = plt.subplots()
        grouped.plot(kind="bar", ax=ax)
        ax.set_xlabel(cat_col)
        ax.set_ylabel(f"Average {val_col}")
        ax.set_title(f"Average {val_col} by {cat_col}")
        st.pyplot(fig)

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    st.subheader("Multivariate Analysis – Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Correlation Between Numeric Features")
    st.pyplot(fig)

# ==================================================
# CONCLUSION PAGE
# ==================================================
elif page == "Conclusion":
    st.title("Conclusion")

    st.markdown("""
    ### Key Observations
    - Numerical features show varied distributions with the presence of outliers.
    - Certain healthcare attributes exhibit moderate to strong correlations.
    - Categorical analysis highlights differences in average values across groups.
    - Visualization-based EDA provides clarity before applying machine learning models.

    ### Final Outcome
    This exploratory analysis helps in understanding healthcare data behavior,
    identifying important features, and preparing the dataset for further
    predictive or statistical analysis.
    """)

