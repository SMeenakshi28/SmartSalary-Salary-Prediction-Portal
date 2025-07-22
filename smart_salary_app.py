import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Salary Predictor",
    page_icon="💼",  # Path to your logo file
    layout="centered"
)

# Custom CSS for enhanced UI styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css?family=Rancho&effect=shadow-multiple|3d-float');
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;700&display=swap');
    @import url('https://i.pinimg.com/1200x/ab/73/6b/ab736b14ce7847df079a34741a59bf3a.jpg');
    :root {
        --primary-color: #2eaaa2;
        --secondary-color: #C56C86;
        --background-color: #f0f2f6;
        --font-color: #3b3739;
        --accent-color: #FF7582;
        --base: #FFFFFF;
        --font: 'Cormorant Garamond', serif;
    }
    body {
        background-color: var(--background-color);
        color: var(--font-color);
        font-family: var(--font);
        font-weight: italic;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stslider>div>div>stslider {
        background-color: var(--secondary-color);
        color: var(--font-color);
        border-radius: 5px;
    }
    .stselectbox{
        background-color: var(--secondary-color);
        color: var(--font-color);
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)
# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Salary Data.csv")
    df = df.dropna(subset=["Salary"])  # Remove rows without salary
    return df

data = load_data()

# Features and target
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Preprocessing
categorical_cols = ["Gender", "Education Level", "Job Title"]
numerical_cols = ["Age", "Years of Experience"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_cols),
    ("num", numerical_transformer, numerical_cols)
])

# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# UI
st.markdown("<h1 class='font-effect-3d-float' style='text-align: center; color: #2eaaa2; font-family: 'Rancho', cursive; font-weight: bold;'>💼 SmartSalary: Salary Prediction Portal</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### 🧾 Enter your information below:")

# Input fields in columns
col1, col2 = st.columns(2)
with col1:
    age = st.slider("📅 Age", 18, 65, 30)
    gender = st.selectbox("⚥ Gender", sorted(data["Gender"].dropna().unique()))
    education = st.selectbox("🎓 Education Level", sorted(data["Education Level"].dropna().unique()))
with col2:
    job_title = st.selectbox("💼 Job Title", sorted(data["Job Title"].dropna().unique()))
    experience = st.slider("🕓 Years of Experience", 0, 40, 3)

# Prepare input
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education Level": education,
    "Job Title": job_title,
    "Years of Experience": experience
}])

# Predict Salary
if st.button("🚀 Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 **Estimated Salary: ₹{int(prediction):,}/month**")

    # Bar chart for prediction
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.bar(["Predicted Salary"], [prediction], color="#00b4d8")
    ax.set_ylabel("Salary (INR)")
    ax.set_title("Predicted Salary")
    #st.pyplot(fig2)

# Dataset preview
with st.expander("📄 View Sample Dataset"):
    st.dataframe(data.head())

# Model Performance Metrics
with st.expander("📊 Model Performance Metrics"):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.markdown("##### 🔧 Regression Evaluation")
    st.write(f"✅ **R² Score**: `{r2:.2f}`")
    st.write(f"📉 **Mean Squared Error (MSE)**: `{mse:,.0f}`")
    st.write(f"📐 **Root Mean Squared Error (RMSE)**: `{rmse:,.0f}`")

    # Predicted vs Actual plot
    st.markdown("##### 📈 Predicted vs Actual Salaries")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax2, color="#0077b6")
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Ideal line
    ax2.set_xlabel("Actual Salary")
    ax2.set_ylabel("Predicted Salary")
    ax2.set_title("Actual vs Predicted Salary")
    st.pyplot(fig2)


# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>📘 Created using Streamlit + ML</div>", unsafe_allow_html=True)
# Add a footer with links to GitHub and LinkedIn
st.markdown(
    """    <div style='text-align: center; margin-top: 20px;'>
    <a href='https://github.com/SMeenakshi28' target='_blank' style='color: #00b4d8; text-decoration: none; margin-right: 20px;'>GitHub</a>
    <a href='https://www.linkedin.com/in/meenakshi-sakala-a72407306/' target='_blank' style='color: #00b4d8; text-decoration: none;'>LinkedIn</a>
    </div>""",
    unsafe_allow_html=True
)
# Add a footer with copyright information
st.markdown(
    """<div style='text-align: center; color: gray; margin-top: 10px;'>
    © 2025 SmartSalary. All rights reserved.
    </div>""",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)
