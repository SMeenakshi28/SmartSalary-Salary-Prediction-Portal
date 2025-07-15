# 💼 SmartSalary: Salary Prediction Portal

SmartSalary is a **machine learning-powered web app** built using **Streamlit** that predicts a user's **monthly salary** based on factors like age, gender, education level, job title, and years of experience. It combines an elegant UI with real-time salary estimation and performance visualization.

---

## 🚀 Features

- 🎯 **Accurate Salary Prediction** using a trained `RandomForestRegressor`.
- 📋 Interactive UI for inputting personal details.
- 📊 Real-time prediction visualizations (bar charts, scatter plots).
- 📈 Displays model performance (R² score, MSE, RMSE).
- 🌐 Styled using custom fonts, colors, and layout via CSS.
- 📂 View the sample dataset directly from the app.
- 🧠 Uses pipelines and column transformers for data preprocessing.


## 🧠 Machine Learning Model

- **Model Used**: `RandomForestRegressor` from `sklearn.ensemble`
- **Preprocessing**:
  - Missing values handled using `SimpleImputer`
  - Categorical encoding via `OneHotEncoder`
- **Feature Columns**:
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience
- **Target Column**: Salary

---

## 🛠️ Tech Stack

| Category       | Tools/Libraries                          |
|----------------|-------------------------------------------|
| Frontend       | Streamlit, HTML/CSS, Google Fonts         |
| Backend        | Python, Pandas, NumPy                     |
| ML Framework   | Scikit-learn (RandomForest, Pipelines)   |
| Visualization  | Matplotlib, Seaborn                       |
| Styling        | Custom CSS with Google Fonts integration |
