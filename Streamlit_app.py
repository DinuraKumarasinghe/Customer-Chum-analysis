from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import numpy as np

# Set Streamlit layout to wide
st.set_page_config(layout="wide", page_title="Customer Churn Prediction")


# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please ensure it's in the same directory.")
        return None


# Load the MinMaxScaler
@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
        return None


# Load resources
model = load_model()
scaler = load_scaler()

# Debug: Check scaler feature names if available
debug_mode = st.sidebar.checkbox("Show debug info", value=False)
if debug_mode and scaler is not None:
    if hasattr(scaler, 'feature_names_in_'):
        st.sidebar.write("Scaler was fitted with features:", scaler.feature_names_in_)
    else:
        st.sidebar.write("Scaler doesn't have feature names attribute")

# Get the ACTUAL feature names from the scaler or model
if scaler is not None and hasattr(scaler, 'feature_names_in_'):
    # Use the exact features from the scaler
    feature_names = list(scaler.feature_names_in_)
elif model is not None and hasattr(model, 'feature_names_in_'):
    # Fall back to model features
    feature_names = list(model.feature_names_in_)
else:
    # Default to our expected features (you may need to adjust this based on your actual training)
    feature_names = ["Age", "EstimatedSalary", "CreditScore", "Balance", "NumOfProducts", "Tenure"]

if debug_mode:
    st.sidebar.write("Using features:", feature_names)

# Feature importance data from your notebook output
feature_importance_data = {
    "Feature": ["Age", "EstimatedSalary", "CreditScore", "Balance", "NumOfProducts", "Tenure"],
    "Importance": [0.237533, 0.146100, 0.143051, 0.142969, 0.127711, 0.081423]
}
feature_importance_df = pd.DataFrame(feature_importance_data)

# Sidebar setup
st.sidebar.header("Customer Input Parameters")

# Try to display sidebar image
try:
    st.sidebar.image("Pic 11.png", use_container_width=True)
except:
    pass

# Customer Demographics Section
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35)

# Financial Information Section
st.sidebar.subheader("💰 Financial Information")
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0, max_value=300000, value=60000,
                                           step=1000)
credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=650)
balance = st.sidebar.number_input("Balance ($)", min_value=0, max_value=300000, value=80000, step=1000)

# Account Information Section
st.sidebar.subheader("🏦 Account Information")
num_products = st.sidebar.slider("Number of Products", min_value=1, max_value=4, value=2)
tenure = st.sidebar.slider("Tenure (years)", min_value=0, max_value=10, value=2)


# Create input data function - USING EXACT SAME ORDER AS ORIGINAL TRAINING
def prepare_input_data():
    # Create dictionary with all possible features
    input_dict = {}

    # Map our input variables to the expected feature names
    feature_mapping = {
        "Age": age,
        "EstimatedSalary": estimated_salary,
        "CreditScore": credit_score,
        "Balance": balance,
        "NumOfProducts": num_products,
        "Tenure": tenure
    }

    # Only include features that were used in training, in the correct order
    input_data = []
    for feature in feature_names:
        if feature in feature_mapping:
            input_data.append(feature_mapping[feature])
        else:
            st.error(f"Missing feature in mapping: {feature}")
            input_data.append(0)  # Default value for missing features

    # Convert to DataFrame with correct feature names and order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    return input_df


# Create input data
input_data = prepare_input_data()

if debug_mode:
    st.sidebar.write("Prepared input data columns:", list(input_data.columns))
    st.sidebar.write("Input data values:", input_data.values.tolist())
    st.sidebar.write("Input data shape:", input_data.shape)

# Apply scaling to the features
input_data_scaled = input_data.copy()
if scaler is not None:
    try:
        # Scale the data - ensure same feature order
        input_data_scaled[feature_names] = scaler.transform(input_data[feature_names])
        if debug_mode:
            st.sidebar.success("Scaling successful!")
            st.sidebar.write("Scaled data:", input_data_scaled.values.tolist())
    except Exception as e:
        st.error(f"Error scaling data: {e}")
        if debug_mode:
            st.sidebar.error(f"Scaling error details: {e}")
            st.sidebar.write("Scaler features:", getattr(scaler, 'feature_names_in_', 'Unknown'))
            st.sidebar.write("Input features:", list(input_data.columns))

# Main App Interface
try:
    st.image("Pic 12.png", use_container_width=True)
except:
    st.title("🏦 Customer Churn Prediction Dashboard")

st.markdown("---")

# Page Layout
left_col, right_col = st.columns([1, 1])

# Left Column: Feature Importance
with left_col:
    st.header("📈 Feature Importance Analysis")

    # Plot the feature importance bar chart
    fig = px.bar(
        feature_importance_df.sort_values(by="Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Model Feature Importance (6 Key Predictors)",
        labels={"Importance": "Importance Score", "Feature": "Features"},
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_range=[0, 0.3]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature descriptions
    with st.expander("💡 Feature Descriptions"):
        st.write("""
        - **Age**: Customer age (Most important feature)
        - **EstimatedSalary**: Annual salary estimate
        - **CreditScore**: Customer credit score
        - **Balance**: Current account balance
        - **NumOfProducts**: Number of bank products owned
        - **Tenure**: Years as bank customer
        """)

# Right Column: Prediction Interface
with right_col:
    st.header("🔮 Churn Prediction")

    # Display current input summary
    with st.expander("📋 Current Input Summary", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Age", f"{age} years", delta="23.75% impact")
            st.metric("Estimated Salary", f"${estimated_salary:,.0f}", delta="14.61% impact")
            st.metric("Credit Score", credit_score, delta="14.31% impact")

        with col2:
            st.metric("Balance", f"${balance:,.0f}", delta="14.30% impact")
            st.metric("Number of Products", num_products, delta="12.77% impact")
            st.metric("Tenure", f"{tenure} years", delta="8.14% impact")

    # Prediction button
    if st.button("🎯 Predict Churn Probability", type="primary", use_container_width=True):
        if model is not None and scaler is not None:
            try:
                # Debug info
                if debug_mode:
                    st.write("Input data for prediction:")
                    st.write(input_data_scaled)

                # Get the predicted probabilities and label
                probabilities = model.predict_proba(input_data_scaled)[0]
                prediction = model.predict(input_data_scaled)[0]

                # Map prediction to label
                prediction_label = "🚨 HIGH RISK OF CHURN" if prediction == 1 else "✅ CUSTOMER RETAINED"
                churn_probability = probabilities[1]

                # Display results with visual indicators
                st.markdown("---")
                st.subheader("Prediction Result")

                # Create a visual gauge for churn probability
                col1, col2 = st.columns([1, 3])

                with col1:
                    # Risk indicator
                    if churn_probability > 0.7:
                        st.error("**CRITICAL**")
                        risk_color = "red"
                    elif churn_probability > 0.4:
                        st.warning("**MEDIUM**")
                        risk_color = "orange"
                    else:
                        st.success("**LOW**")
                        risk_color = "green"

                    st.metric("Churn Probability", f"{churn_probability:.1%}")

                with col2:
                    # Progress bar with color
                    st.progress(float(churn_probability), text=f"Risk Level: {prediction_label}")

                # Detailed probabilities
                with st.expander("📊 Detailed Analysis", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Probability to Churn", f"{churn_probability:.1%}")
                    with col2:
                        st.metric("Probability to Stay", f"{probabilities[0]:.1%}")

                    # Recommendations based on risk level
                    st.subheader("🎯 Recommended Actions")
                    if churn_probability > 0.7:
                        st.error("""
                        **Immediate Retention Actions Needed:**
                        - Proactive customer service call
                        - Personalized retention offers
                        - Account review with relationship manager
                        """)
                    elif churn_probability > 0.4:
                        st.warning("""
                        **Monitor and Engage:**
                        - Regular check-ins
                        - Product recommendation campaigns
                        - Customer satisfaction survey
                        """)
                    else:
                        st.success("""
                        **Maintain and Grow:**
                        - Continue excellent service
                        - Cross-sell additional products
                        - Loyalty program enrollment
                        """)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                if debug_mode:
                    st.write("Model features expected:", getattr(model, 'feature_names_in_', 'Unknown'))
                    st.write("Input features provided:", list(input_data_scaled.columns))
        else:
            st.error("Model or scaler not loaded properly. Cannot make predictions.")

# Model Information
with st.expander("ℹ️ Model Information", expanded=False):
    st.write(f"""
    **Model Details:**
    - **Algorithm**: Random Forest Classifier
    - **Features**: {len(feature_names)} key customer attributes
    - **Feature Order**: {', '.join(feature_names)}
    - **Training**: Historical customer churn data
    - **Top Feature**: Age (23.75% impact)
    """)

# Footer
st.markdown("---")
st.markdown("**Customer Churn Prediction Dashboard** | Built with Streamlit")