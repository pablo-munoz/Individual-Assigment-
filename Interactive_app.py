import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

# Title and description
st.title("Machine Learning Model Trainer")
st.markdown("""
This Streamlit app allows you to train a machine learning model (regression or classification) on a chosen dataset.
You can select a built-in sample dataset or upload your own CSV, then choose features, target, and model parameters.
""")

# Section: Dataset Selection
st.header("1. Dataset Selection")
st.write("Select a built-in Seaborn sample dataset or upload a CSV file:")

# Data source selection: sample dataset or upload
data_source = st.radio("Data source:", ("Sample dataset", "Upload CSV"), index=0)

df = None
current_data_id = None

# Sample dataset selection
if data_source == "Sample dataset":
    # List of sample datasets (using seaborn's built-in examples)
    sample_datasets = ["iris", "tips", "titanic", "penguins", "diamonds"]
    dataset_name = st.selectbox("Choose a dataset:", sample_datasets, index=0)
    if dataset_name:
        # Load dataset (with caching to avoid reloading)
        @st.cache_data
        def load_sample_data(name):
            return sns.load_dataset(name)
        df = load_sample_data(dataset_name)
        current_data_id = f"sample-{dataset_name}"

# File upload selection
else:
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    if file is not None:
        @st.cache_data
        def load_csv_data(uploaded_file):
            return pd.read_csv(uploaded_file)
        df = load_csv_data(file)
        current_data_id = f"uploaded-{file.name}"

# Session state management to reset selections if dataset changes
for key, default in {
    "num_features": [],
    "cat_features": [],
    "target_col": "Select target...",
    "last_data_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if current_data_id and st.session_state['last_data_id'] != current_data_id:
    # New dataset loaded: reset feature and target selections
    st.session_state['num_features'] = []
    st.session_state['cat_features'] = []
    st.session_state['target_col'] = "Select target..."
    st.session_state['last_data_id'] = current_data_id

# If a dataset is loaded, continue to feature selection and model training
if df is not None:
    st.write(f"**Dataset loaded:** {df.shape[0]} rows, {df.shape[1]} columns.")
  
    st.dataframe(df.head())

    numeric_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    #Feature selection and Model configuration form
    with st.form("settings_form"):
        st.subheader("2. Feature and Target Selection")
        col1, col2 = st.columns(2)
        with col1:
            st.multiselect("Select numeric features (independent variables):", options=numeric_cols, default=st.session_state['num_features'], key='num_features')
        with col2:
            st.multiselect("Select categorical features (independent variables):", options=categorical_cols, default=st.session_state['cat_features'], key='cat_features')
        target_options = ["Select target..."] + list(df.columns)
        st.selectbox("Select target variable (dependent variable):", options=target_options, index=0, key='target_col')

        st.subheader("3. Model Configuration")
        # Model type selection
        model_choice = st.selectbox("Choose Model Type:", ["Linear Regression", "Random Forest (Classification)"])
        # Common parameter: test set size
        test_size = st.slider("Test set size (fraction of dataset):", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        # Model-specific parameters (shown only for Random Forest)
        n_estimators = None
        max_depth = None
        if model_choice == "Random Forest (Classification)":
            colA, colB = st.columns(2)
            with colA:
                n_estimators = st.slider("Number of trees (estimators):", min_value=50, max_value=300, value=100, step=50)
            with colB:
                max_depth = st.slider("Max depth of trees:", min_value=1, max_value=20, value=5, step=1)

        # Submit button to train the model
        submitted = st.form_submit_button("Fit Model")

    # If the user presses the "Fit Model" button
    if submitted:
        # Retrieve the selected features and target from session state
        selected_num = st.session_state['num_features']
        selected_cat = st.session_state['cat_features']
        target_col = st.session_state['target_col']
        # Validate that a target and features have been selected
        if target_col == "Select target..." or target_col is None:
            st.error("Please select a target variable.")
        elif len(selected_num) + len(selected_cat) == 0:
            st.error("Please select at least one feature for the model.")
        else:
            # Remove target from feature list if included (to avoid using target as a feature)
            features = [col for col in selected_num + selected_cat if col != target_col]
            # Prepare data for modeling: drop rows with missing values in any selected feature or target
            df_model = df[features + [target_col]].dropna()
            if df_model.empty:
                st.error("No data available after dropping missing values. Please adjust feature selection or handle missing data.")
            else:
                X = df_model[features]
                y = df_model[target_col]

                # Check if target is numeric when using regression
                if model_choice == "Linear Regression":
                    if not pd.api.types.is_numeric_dtype(y):
                        st.error("The selected target is not numeric. Please choose a numeric target for regression.")
                    else:
                        model_type = 'regression'
                else:
                    model_type = 'classification'

                # Continue only if the target type is appropriate for the chosen model
                if (model_choice == "Linear Regression" and pd.api.types.is_numeric_dtype(y)) or model_choice == "Random Forest (Classification)":
                    # Encode categorical features using one-hot encoding
                    X_encoded = pd.get_dummies(X, drop_first=True)
                    feature_names = X_encoded.columns.tolist()
                    # Encode target for classification (label encoding) if needed
                    if model_type == 'classification':
                        label_enc = LabelEncoder()
                        y_encoded = label_enc.fit_transform(y)
                        class_names = list(label_enc.classes_)
                    else:
                        y_encoded = y.values  # use numeric target values as is
                        class_names = None

                    # Split the data into training and testing sets
                    stratify_param = y_encoded if model_type == 'classification' else None
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=test_size, 
                                                                        stratify=stratify_param, random_state=0)

                    # Train the model (cache this to avoid unnecessary re-computation on the same data and parameters)
                    @st.cache_resource
                    def train_model(X_train, y_train, model_choice, n_estimators=None, max_depth=None):
                        if model_choice == "Linear Regression":
                            model = LinearRegression()
                        else:
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
                        model.fit(X_train, y_train)
                        return model

                    model = train_model(X_train, y_train, model_choice, n_estimators, max_depth)

                    # Make predictions on the test set
                    y_pred = model.predict(X_test)

                    # Evaluate and display results for classification vs regression
                    if model_type == 'classification':
                        # Compute and display accuracy
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"**Accuracy:** {accuracy*100:.2f}%")

                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                    xticklabels=(class_names if class_names else np.unique(y_test)),
                                    yticklabels=(class_names if class_names else np.unique(y_test)),
                                    ax=ax_cm)
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        ax_cm.set_title("Confusion Matrix")
                        plt.xticks(rotation=45)
                        st.pyplot(fig_cm)

                        # ROC curve (for binary classification only)
                        if len(np.unique(y_test)) == 2:
                            y_score = None
                            if hasattr(model, "predict_proba"):
                                # Probability of the positive class (assuming classes_[1] is positive)
                                y_score = model.predict_proba(X_test)[:, 1]
                            if y_score is not None:
                                fpr, tpr, _ = roc_curve(y_test, y_score)
                                roc_auc = auc(fpr, tpr)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel("False Positive Rate")
                                ax_roc.set_ylabel("True Positive Rate")
                                ax_roc.set_title("ROC Curve")
                                ax_roc.legend(loc="lower right")
                                st.pyplot(fig_roc)
                                st.write(f"**AUC:** {roc_auc:.3f}")

                        # Feature importance for classification (from Random Forest)
                        importances = model.feature_importances_
                        imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
                        if imp_series.size > 20:
                            imp_series = imp_series.iloc[-20:]  # show top 20 features
                            st.caption("Showing top 20 features by importance.")
                        fig_imp, ax_imp = plt.subplots()
                        ax_imp.barh(imp_series.index, imp_series.values)
                        ax_imp.set_title("Feature Importance")
                        ax_imp.set_xlabel("Importance")
                        st.pyplot(fig_imp)

                    else:
                        # Regression metrics: R sqaured and RMSE
                        r2_val = r2_score(y_test, y_pred)
                        rmse_val = mean_squared_error(y_test, y_pred, squared=False)
                        st.write(f"**R² (Coefficient of Determination):** {r2_val:.3f}")
                        st.write(f"**RMSE (Root Mean Squared Error):** {rmse_val:.3f}")

                        # Residual distribution plot
                        residuals = y_test - y_pred
                        fig_res, ax_res = plt.subplots()
                        sns.histplot(residuals, kde=True, ax=ax_res, color='teal')
                        ax_res.set_title("Residual Distribution")
                        ax_res.set_xlabel("Residual (Actual - Predicted)")
                        st.pyplot(fig_res)

                        # Feature importance for regression (based on absolute coefficients for Linear Regression)
                        if model_choice == "Linear Regression":
                            coefs = model.coef_
                            importance_vals = np.abs(coefs)
                            imp_series = pd.Series(importance_vals, index=feature_names).sort_values(ascending=True)
                            if imp_series.size > 20:
                                imp_series = imp_series.iloc[-20:]
                                st.caption("Showing top 20 features by absolute coefficient.")
                            fig_imp, ax_imp = plt.subplots()
                            ax_imp.barh(imp_series.index, imp_series.values)
                            ax_imp.set_title("Feature Importance (based on coefficients)")
                            ax_imp.set_xlabel("Absolute Coefficient")
                            st.pyplot(fig_imp)

                    # Allow the user to download the trained model as a pickle file
                    model_bytes = pickle.dumps(model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_bytes,
                        file_name="trained_model.pkl",
                        mime="application/octet-stream"
                    )
else:
    # If no dataset is loaded yet, prompt the user
    st.info("Awaiting dataset selection to start model training.")
