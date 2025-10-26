# boston_dashboard.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Boston Housing Dashboard", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f8ff 0%, #d6eaff 50%, #b3d9ff 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 { color: #003366; text-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    h2, h3 { color: #004080; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003366 0%, #0059b3 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p { color: white; }
    .stButton>button {
        background-color: #0059b3;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #003366;
        color: #ffcc00;
    }
    .css-1d391kg, .css-1l3j5n4 {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- Color theme for Seaborn & Matplotlib ---
custom_palette = ["#003366", "#0059b3", "#0073e6", "#3399ff", "#66b3ff", "#99ccff"]
sns.set_palette(custom_palette)
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.facecolor": "#f0f8ff",
    "figure.facecolor": "#f0f8ff",
    "axes.edgecolor": "#003366",
    "axes.labelcolor": "#003366",
    "xtick.color": "#003366",
    "ytick.color": "#003366",
    "grid.color": "#99ccff"
})

# --- Title ---
st.title("🏠 Boston Housing Data Analysis & Prediction")

# --- Data source selection ---
data_source = st.radio("Select data source:", ["Upload Excel", "Use Our Boston Housing File"])
if data_source == "Upload Excel":
    uploaded_file = st.file_uploader("Upload Boston Housing Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        st.info("Please upload an Excel file first.")
        st.stop()
elif data_source == "Use Our Boston Housing File":
    try:
        url = "https://github.com/azizmuzaki4/boston-housing-data-analysis/raw/refs/heads/main/boston_house_price.xlsx"
        df = pd.read_excel(url)
    except FileNotFoundError:
        st.error("Local file 'boston.xlsx' not found. Please make sure the file exists in the project folder.")
        st.stop()

# --- Data Preview ---
st.write("## 📊 Data Preview with Search & Filter")
with st.expander("🔍 View All Raw Data"):
    search_term = st.text_input("Search word/number in table:")
    selected_columns = st.multiselect("Select columns to display:", options=df.columns, default=df.columns)
    filtered_df = df[selected_columns]
    if search_term:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        ]
    st.dataframe(filtered_df, use_container_width=True, height=500)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["DDA", "Visualization", "Modeling", "Prediction"])

# Sidebar Visualization Settings
st.sidebar.header("Visualization Settings")
plot_style = st.sidebar.selectbox(
    "Select Plot Style", 
    plt.style.available, 
    index=plt.style.available.index("tableau-colorblind10") if "tableau-colorblind10" in plt.style.available else 0
)
sns_palette = st.sidebar.selectbox(
    "Select Seaborn Color Palette", 
    ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
)

plt.style.use(plot_style)
sns.set_palette(sns_palette)

# --- Style Preview ---
st.sidebar.write("**Preview Style & Color (Data)**")

if {'RM', 'MEDV'}.issubset(df.columns):
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(x='RM', y='MEDV', data=df.sample(min(50, len(df))), ax=ax_scatter)
    ax_scatter.set_title("Scatter: RM vs MEDV")
    st.sidebar.pyplot(fig_scatter)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) >= 2:
    fig_heat, ax_heat = plt.subplots()
    sns.heatmap(df[num_cols].corr(), cmap='coolwarm', ax=ax_heat)
    ax_heat.set_title("Correlation Heatmap")
    st.sidebar.pyplot(fig_heat)

if 'LSTAT' in df.columns:
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df['LSTAT'], kde=True, ax=ax_hist)
    ax_hist.set_title("Histogram: LSTAT")
    st.sidebar.pyplot(fig_hist)

# --- Utility function for HTML table ---
def format_2dec(df):
    return df.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

def html_table(df, 
               header_gradient="linear-gradient(90deg, #00695C, #26A69A)", 
               text_color="white", 
               height_px=300, 
               highlight=True,
               highlight_residual_row=False):
    styled_df = df.copy()

    if highlight_residual_row and "Residual" in df.columns:
        min_idx = df["Residual"].abs().idxmin()
        max_idx = df["Residual"].abs().idxmax()

        # Format numbers
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                styled_df[col] = df[col].apply(lambda x: f"{x:.2f}")

        # Add HTML style for entire row
        def style_row(row_idx, color):
            for col in styled_df.columns:
                styled_df.at[row_idx, col] = f'<span style="background-color:{color}; display:block; width:100%">{styled_df.at[row_idx, col]}</span>'

        style_row(min_idx, "#90ee90")  # green for best
        style_row(max_idx, "#ff9999")  # red for worst

    elif highlight:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Assumption: higher is better for common metrics
                if col.lower() in ["r2", "accuracy", "precision", "recall", "f1", "auc"]:
                    best_val = df[col].max()
                    worst_val = df[col].min()
                else:
                    best_val = df[col].min()
                    worst_val = df[col].max()

                styled_df[col] = df[col].apply(
                    lambda x: f'<span style="background-color:#C8E6C9">{x:.2f}</span>' if x == best_val else
                              f'<span style="background-color:#FFCDD2">{x:.2f}</span>' if x == worst_val else
                              f'{x:.2f}'
                )

    df_html = styled_df.to_html(escape=False, index=True, border=0)
    css = f"""
    <style>
        .table-container {{
            height: {height_px}px;
            overflow-y: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            border: 1px solid #ddd;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 6px;
        }}
        thead th {{
            position: sticky;
            top: 0;
            background: {header_gradient};
            color: {text_color};
            font-weight: bold;
            text-align: center;
            padding: 8px;
            z-index: 2;
        }}
        tbody td {{
            text-align: center;
            padding: 6px;
            border-top: 1px solid #ddd;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tbody tr:nth-child(odd) {{ background-color: white; }}
        tbody tr:hover {{ background-color: #e0f2f1; color: #004d40; }}
    </style>
    """
    return css + f'<div class="table-container">{df_html}</div>'

# --- DDA Page ---
if page == "DDA":
    st.subheader("📊 Descriptive Data Analysis")
    desc_df = format_2dec(df.describe())
    st.write("**Descriptive Statistics:**")
    components.html(html_table(desc_df, height_px=305), height=305)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values:**")
        missing_df = format_2dec(df.isnull().sum().to_frame(name="Missing Count"))
        components.html(html_table(missing_df, header_gradient="linear-gradient(90deg, #FF9800, #FFC107)", height_px=300), height=340)
    with col2:
        st.write("**Correlation with MEDV:**")
        corr = df.corr(numeric_only=True)
        if "MEDV" in corr.columns:
            corr_medv_df = format_2dec(corr['MEDV'].sort_values(ascending=False).to_frame(name="Correlation"))
            components.html(html_table(corr_medv_df, header_gradient="linear-gradient(90deg, #9C27B0, #E1BEE7)", height_px=300), height=340)

# --- Visualization Page ---
elif page == "Visualization":
    st.subheader("📊 Visualization")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10}, cbar_kws={"shrink": 0.85})
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.write("**Scatter Plot**")
        col_x = st.selectbox(
            "X-axis", 
            df.columns, 
            index=df.columns.get_loc("RM") if "RM" in df.columns else 0, 
            key="scatter_x"
        )
        col_y = st.selectbox(
            "Y-axis", 
            df.columns, 
            index=df.columns.get_loc("MEDV") if "MEDV" in df.columns else 0, 
            key="scatter_y"
        )
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=col_x, y=col_y, data=df, ax=ax2, s=60, alpha=0.7, edgecolor="w")
        ax2.set_xlabel(col_x, fontsize=9)
        ax2.set_ylabel(col_y, fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)
    col3, col4 = st.columns([1, 1])
    with col3:
        st.write("**Histogram**")
        col_hist = st.selectbox(
            "Select column for histogram", 
            df.columns, 
            index=df.columns.get_loc("RM") if "RM" in df.columns else 0, 
            key="hist_col"
        )
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.histplot(df[col_hist], kde=True, ax=ax3, color="skyblue", edgecolor="black")
        ax3.set_xlabel(col_hist, fontsize=9)
        ax3.set_ylabel("Frequency", fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=True)
    with col4:
        st.write("**Box Plot**")
        col_box = st.selectbox(
            "Select column for box plot", 
            df.columns, 
            index=df.columns.get_loc("RM") if "RM" in df.columns else 0, 
            key="box_col"
        )
        fig4, ax4 = plt.subplots(figsize=(8,6))
        sns.boxplot(y=df[col_box], ax=ax4, color="lightgreen")
        ax4.set_ylabel(col_box, fontsize=9)
        ax4.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig4, clear_figure=True)

# --- Modeling Page ---
elif page == "Modeling":
    st.subheader("🤖 Modeling")
    if "MEDV" in df.columns:
        X = df.drop('MEDV', axis=1)
        y = df['MEDV']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }
        results = []
        fitted_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            fitted_models[name] = model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append([name, mse, rmse, mae, r2])
        results_df = pd.DataFrame(results, columns=['Model','MSE','RMSE','MAE','R2']).round(2)
        st.markdown("**📊 Model Evaluation Results:**", unsafe_allow_html=True)
        components.html(html_table(results_df, height_px=210), height=210)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Model Evaluation Results", csv, "model_evaluation.csv", "text/csv")
        st.write("### Feature Importance for All Models")
        cols = st.columns(3, gap="small")
        for idx, (name, model) in enumerate(fitted_models.items()):
            if hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                continue
            feat_importances = pd.Series(importances, index=X.columns)
            fig_imp, ax_imp = plt.subplots(figsize=(4,3))
            feat_importances.sort_values().plot(kind='barh', ax=ax_imp)
            ax_imp.set_title(f"Feature Importance - {name}")
            ax_imp.tick_params(labelsize=8)
            cols[idx % 3].pyplot(fig_imp)
    else:
        st.warning("'MEDV' column not found in dataset.")

# --- Prediction Page ---
elif page == "Prediction":
    st.subheader("🔮 Prediction")
    if "MEDV" in df.columns:
        model_choice = st.selectbox("Select Model for Prediction", 
                                    ["LinearRegression", "Ridge", "Lasso", "RandomForest", "GradientBoosting"])
        cols = st.columns(5)
        input_data = []
        for i, col_name in enumerate(df.drop('MEDV', axis=1).columns):
            col_index = i % 5
            with cols[col_index]:
                val = st.number_input(f"{col_name}",
                                      float(df[col_name].min()),
                                      float(df[col_name].max()),
                                      float(df[col_name].mean()))
                input_data.append(val)
        if st.button("Predict"):
            X = df.drop('MEDV', axis=1)
            y = df['MEDV']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.1),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42)
            }
            model = models[model_choice]
            model.fit(X_train, y_train)
            input_scaled = scaler.transform([input_data])
            pred = model.predict(input_scaled)[0]
            st.success(f"**Predicted MEDV ({model_choice}): {pred:.2f}**")
            y_pred_test = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            st.write("### 📊 Evaluation Metrics")
            st.write(f"- **MSE**: {mse:.4f}")
            st.write(f"- **RMSE**: {rmse:.4f}")
            st.write(f"- **MAE**: {mae:.4f}")
            st.write(f"- **R²**: {r2:.4f}")
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                fig_pred, ax_pred = plt.subplots()
                ax_pred.scatter(y_test, y_pred_test, alpha=0.7)
                ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax_pred.set_xlabel("Actual MEDV")
                ax_pred.set_ylabel("Predicted MEDV")
                ax_pred.set_title(f"Prediction vs Actual ({model_choice})")
                st.pyplot(fig_pred)
            with col_g2:
                residuals = y_test - y_pred_test
                fig_res, ax_res = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax_res)
                ax_res.set_title(f"Error (Residual) Distribution - {model_choice}")
                ax_res.set_xlabel("Residual")
                st.pyplot(fig_res)
            with col_g3:
                fig_res_scatter, ax_res_scatter = plt.subplots()
                ax_res_scatter.scatter(y_pred_test, residuals, alpha=0.7)
                ax_res_scatter.axhline(y=0, color='r', linestyle='--')
                ax_res_scatter.set_xlabel("Predicted MEDV")
                ax_res_scatter.set_ylabel("Residual")
                ax_res_scatter.set_title(f"Residual vs Predicted - {model_choice}")
                st.pyplot(fig_res_scatter)
            pred_df = pd.DataFrame({"Actual_MEDV": y_test, "Predicted_MEDV": y_pred_test, "Residual": residuals})
            csv_pred = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prediction Results", csv_pred, "predictions.csv", "text/csv")
            
            st.write("### 📋 Prediction Table with Best & Worst Highlight")
            components.html(html_table(pred_df, height_px=305, highlight=False, highlight_residual_row=True), height=350)

    else:
        st.warning("'MEDV' column not found in dataset.")
