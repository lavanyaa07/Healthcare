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
# LOAD DATA (Cloud Safe)
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
st.sidebar.markdown("Analyze healthcare data using **EDA**")

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
    to understand patient trends, distributions, and patterns across medical attributes.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.success("✅ Dataset loaded successfully")

# =========================================================
# DATASET PAGE
# =========================================================
elif page == "Dataset":
    st.title("📁 Dataset Overview")

    st.subheader("Preview of Dataset")
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
            "Age Distribution",
            "Gender Distribution",
            "Patients by Medical Condition",
            "Billing Amount Distribution",
            "Admission Type Distribution"
        ]
    )

    # -------------------------------
    # Age Distribution
    # -------------------------------
    if chart == "Age Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution of Patients")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # -------------------------------
    # Gender Distribution
    # -------------------------------
    elif chart == "Gender Distribution":
        gender_counts = df["Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax)
        ax.set_title("Gender Distribution")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Patients")
        st.pyplot(fig)

    # -------------------------------
    # Medical Condition Count
    # -------------------------------
    elif chart == "Patients by Medical Condition":
        condition_counts = df["Medical_Condition"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=condition_counts.values, y=condition_counts.index, ax=ax)
        ax.set_title("Top Medical Conditions")
        ax.set_xlabel("Number of Patients")
        st.pyplot(fig)

    # -------------------------------
    # Billing Amount Distribution
    # -------------------------------
    elif chart == "Billing Amount Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["Billing_Amount"], bins=40, kde=True, ax=ax)
        ax.set_title("Billing Amount Distribution")
        ax.set_xlabel("Billing Amount")
        st.pyplot(fig)

    # -------------------------------
    # Admission Type Distribution
    # -------------------------------
    elif chart == "Admission Type Distribution":
        admission_counts = df["Admission_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=admission_counts.index, y=admission_counts.values, ax=ax)
        ax.set_title("Admission Type Distribution")
        ax.set_xlabel("Admission Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# =========================================================
# INSIGHTS PAGE
# =========================================================
elif page == "Key Insights & Conclusion":
    st.title("💡 Key Insights & Conclusion")

    st.markdown("""
    ### 🔍 Key Insights
    - Majority of patients fall within the adult and middle-age groups.
    - Certain medical conditions dominate patient admissions.
    - Billing amounts show wide variation, indicating diverse treatment costs.
    - Emergency and routine admissions form major admission categories.

    ### ✅ Final Conclusion
    This EDA provides a clear understanding of healthcare data patterns and
    prepares the dataset for further statistical analysis or predictive modeling.
    """)

    st.success("🎉 EDA completed successfully!")
