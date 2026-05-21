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

# Model Selection and Evaluation

## Why I Used Random Forest Regressor

The salary prediction problem is a regression problem because the target variable (salary) contains continuous numerical values. For this project, I selected the **Random Forest Regressor** algorithm because it provides high prediction accuracy and handles real-world datasets effectively.

### Reasons for Choosing Random Forest Regressor

- Handles complex and non-linear relationships between features and salary.
- Provides better accuracy compared to simple linear models.
- Reduces overfitting by using multiple decision trees.
- Works efficiently with both numerical and categorical data.
- Performs well on real-world datasets with large feature variations.
- Produces stable and reliable predictions.

---

# Model Evaluation Metrics

## 1. R² Score

R² Score measures how well the model predicts salary values and explains data variance.

### Formula

```math
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
```

- Higher R² value indicates better model performance.
- R² = 1 means perfect prediction.
- R² = 0 means the model cannot explain the data variance.

---

## 2. Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted salary values.

### Formula

```math
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
```

- Penalizes large prediction errors.
- Helps evaluate overall model accuracy.
- Lower MSE indicates better performance.

---

## 3. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and represents prediction error in salary units.

### Formula

```math
RMSE = \sqrt{MSE}
```

- Easier to interpret than MSE.
- Shows how close predictions are to actual salary values.
- Lower RMSE indicates more accurate predictions.

---

# Data Visualization

## Why Bar Charts Were Used

Bar charts were used to compare salary distributions across different categories such as job roles, education levels, and experience ranges.

- Easy comparison between categories.
- Improves data readability.
- Helps identify salary trends visually.

---

## Why Scatter Plots Were Used

Scatter plots were used to visualize relationships between actual and predicted salary values.

- Helps analyze prediction accuracy.
- Shows trends and correlations between variables.
- Helps identify outliers and prediction deviations.
- Points closer to the diagonal line indicate better predictions.

## Deployment
- Deployed in Streamlit Cloud
- Link
https://smartsalary-salary-prediction-app.streamlit.app/
