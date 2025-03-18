import streamlit as st
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set page config FIRST


# Load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv("train.csv")
    return train_df

train_df = load_data()

# Data preprocessing
def preprocess_data(df):
    # Convert transaction time to datetime
    # df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    # df['hour'] = df['trans_date_trans_time'].dt.hour
    # df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    # Drop irrelevant columns
    df = df.drop(['trans_date_trans_time', 'first', 'last', 'street', 'city', 'zip','job','dob', 'trans_num', 'unix_time'], axis=1)

    # One-hot encode categorical columns
    categorical_cols = ['category', 'gender', 'merchant', 'state']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

train_df = preprocess_data(train_df)

# Select features and target
X = train_df.drop('is_fraud', axis=1)
y = train_df['is_fraud']

# Check for non-numeric columns and drop them (if any)
non_numeric_columns = X.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    X = X.drop(non_numeric_columns, axis=1)

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the dataset into training and validation sets
train_X, val_X, train_Y, val_Y = train_test_split(X_res, y_res, test_size=0.25, random_state=42, stratify=y_res)

# Setting up the hyperparameter grid for Random Forest tuning
param_grid = {
    'n_estimators': [100, 150, 200],  # Number of trees in the forest
    'max_depth': [5, 10, 15],         # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum samples required at each leaf node
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Apply Grid Search with Cross Validation to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_X, train_Y)

# Retrieve the best model from grid search
best_rf = grid_search.best_estimator_

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stSidebar {
        background-color: #2e2e2e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: #2e2e2e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card h3 {
        margin-top: 0;
        color: #4CAF50;
    }
    .explanation {
        font-size: 14px;
        color: #cccccc;
        margin-bottom: 20px;
    }
    .highlight {
        color: #4CAF50;
        font-weight: bold;
    }
    /* Custom CSS for sidebar input labels */
    .stSidebar label {
        color: #4CAF50 !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown(
    """
    <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; color: white;">
        <h1 style="margin: 0;">Credit Card Fraud Detection</h1>
        <p style="margin: 0;">Detect fraudulent transactions using advanced machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input
st.sidebar.header("User Input")
st.sidebar.write("Enter transaction details:")

# Simulate raw transaction data input
amt = st.sidebar.number_input("Transaction Amount", min_value=0.0)
category = st.sidebar.selectbox("Category", ['grocery_pos', 'entertainment', 'gas_transport', 'misc_pos', 'shopping_net'])
hour = st.sidebar.slider("Hour of Day", 0, 23)
day_of_week = st.sidebar.slider("Day of Week", 0, 6)
lat = st.sidebar.number_input("Latitude")
long = st.sidebar.number_input("Longitude")
city_pop = st.sidebar.number_input("City Population", min_value=0)

# Add an "Enter" button to trigger prediction
if st.sidebar.button("Predict Fraud"):
    # Create raw data DataFrame
    raw_data = {
        'amt': [amt],
        'category_' + category: [1],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'lat': [lat],
        'long': [long],
        'city_pop': [city_pop]
    }

    # Ensure all columns are present
    raw_df = pd.DataFrame(raw_data)
    raw_df = raw_df.reindex(columns=X.columns, fill_value=0)

    # Predict the outcome for the new test dataset
    prediction = best_rf.predict(raw_df)
    prediction_proba = best_rf.predict_proba(raw_df)[0][1]

    # Display prediction and confidence score in a card
    st.markdown(
        f"""
        <div class="card">
            <h3>Prediction: {'Fraudulent Transaction ⚠️' if prediction[0] == 1 else 'Legitimate Transaction ✅'}</h3>
            <p>Confidence Score: {prediction_proba:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Explanation for Confidence Score
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">Confidence Score</span> represents the model's certainty in its prediction. 
            A score closer to <span class="highlight">1</span> indicates high confidence, while a score closer to <span class="highlight">0</span> indicates low confidence.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature Importance Plot
    st.markdown("<h2 style='color: white;'>Feature Importance</h2>", unsafe_allow_html=True)
    feature_importances = best_rf.feature_importances_
    features = X.columns
    fig_feature_importance = px.bar(x=features, y=feature_importances, labels={'x': 'Features', 'y': 'Importance'}, title="Feature Importance")
    st.plotly_chart(fig_feature_importance)

    # Confusion Matrix
    st.markdown("<h2 style='color: white;'>Confusion Matrix</h2>", unsafe_allow_html=True)
    predictions_val_rf = best_rf.predict(val_X)
    conf_matrix = confusion_matrix(val_Y, predictions_val_rf)
    fig_conf_matrix = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual", color="Count"), 
                                x=["Not Fraud", "Fraud"], y=["Not Fraud", "Fraud"], 
                                title="Confusion Matrix (Validation Data)")
    st.plotly_chart(fig_conf_matrix)

    # ROC Curve
    st.markdown("<h2 style='color: white;'>ROC Curve</h2>", unsafe_allow_html=True)
    fpr, tpr, thresholds = roc_curve(val_Y, best_rf.predict_proba(val_X)[:,1])
    roc_auc = auc(fpr, tpr)
    fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
    st.plotly_chart(fig_roc)

    # Model Accuracy
    st.markdown("<h2 style='color: white;'>Model Accuracy</h2>", unsafe_allow_html=True)
    accuracy_train_rf = metrics.accuracy_score(train_Y, best_rf.predict(train_X))
    accuracy_val_rf = metrics.accuracy_score(val_Y, predictions_val_rf)

    # Display Model Accuracy in a card
    st.markdown(
        f"""
        <div class="card">
            <h3>Model Accuracy</h3>
            <p>Training Data: {accuracy_train_rf:.2f}</p>
            <p>Validation Data: {accuracy_val_rf:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
