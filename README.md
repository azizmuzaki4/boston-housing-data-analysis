# 🏠 Boston Housing Dashboard
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-blue)
![Made with ❤️ by Aziz Muzaki](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F%20by%20Aziz%20Muzaki-orange)

The **Boston Housing Dashboard** is an interactive web application built with **Streamlit** that allows users to explore, visualize, model, and predict housing prices using the Boston Housing dataset.  
It provides a clean, modern UI with customizable visualizations, multiple regression models, and downloadable results.

---

## 🌐 Live Demo

👉 [View Live Dashboard](https://boston-housing-data-analysis.streamlit.app/)

---

## 🖼️ Dashboard Preview

| | |
|:--:|:--:|
| ![DDA](https://github.com/azizmuzaki4/boston-housing-data-analysis/blob/main/descriptive_data_analysis.png) | ![Visualization](https://github.com/azizmuzaki4/boston-housing-data-analysis/blob/main/visualization_page.png) |
| ![Modeling](https://github.com/azizmuzaki4/boston-housing-data-analysis/blob/main/modeling_page.png) | ![Prediction](https://github.com/azizmuzaki4/boston-housing-data-analysis/blob/main/prediction_page.png) |

---

## 🚀 Features
- **Data Source Options**
  - Upload your own Excel file.
  - Use the provided local Boston Housing dataset file.

- **Data Exploration**
  - Search and filter data interactively.
  - View descriptive statistics, missing values, and correlation with `MEDV`.

- **Visualization**
  - Correlation heatmap.
  - Scatter plots, histograms, and box plots with customizable axes and styles.
  - Sidebar preview of selected visualizations.

- **Modeling**
  - Train and evaluate multiple regression models:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
  - View model performance metrics (MSE, RMSE, MAE, R²).
  - Feature importance visualization.
  - Download model evaluation results as CSV.

- **Prediction**
  - Select a trained model and input custom feature values.
  - Predict `MEDV` (Median value of owner-occupied homes).
  - View evaluation metrics and residual analysis plots.
  - Download prediction results as CSV.
  - Highlight best and worst predictions in the results table.

---

## 🛠️ Technologies Used
- **Python 3.x**
- **Streamlit** – Web app framework
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Seaborn & Matplotlib** – Data visualization
- **Scikit-learn** – Machine learning models and metrics

---

## 📂 Project Structure
```
boston_housing_data_analysis.py   # Main Streamlit application
README.md             # Project documentation
boston_house_price.xlsx  # Local dataset (optional)
requirements.txt      # Python dependencies
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/azizmuzaki4/boston-housing-data-analysis.git
cd boston-housing-data-analysis
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
streamlit run boston_housing_data_analysis.py
```

---

## 📊 Usage
1. **Select Data Source** – Upload a Excel or use the provided local file.  
2. **Navigate via Sidebar** – Choose between:
   - **DDA** (Descriptive Data Analysis)
   - **Visualization**
   - **Modeling**
   - **Prediction**
3. **Customize Visualizations** – Change plot styles and color palettes.  
4. **Train Models** – Compare performance metrics and download results.  
5. **Make Predictions** – Input custom values and analyze prediction accuracy.

---

## 📈 Example Visualizations
- **Correlation Heatmap** – Shows relationships between numerical features.
- **Scatter Plot** – Explore feature-target relationships.
- **Histogram & Box Plot** – Understand data distribution and outliers.

---

## ⚠️ Notes
- Ensure the dataset contains the `MEDV` column for modeling and prediction.
- If using the local dataset option, update the file path in the script to match your environment.
- Large datasets may impact performance in Streamlit.

---

## 📜 License
This project is licensed under the MIT License – feel free to use, modify, and distribute.

---

## 👨‍💻 Author
Developed by **[Aziz Muzaki]** – Data Science & Machine Learning Enthusiast.

---

