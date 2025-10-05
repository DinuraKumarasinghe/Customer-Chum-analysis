# Customer Churn Analysis 

A comprehensive machine learning solution for predicting customer churn in the banking sector. This project includes both a Jupyter Notebook for analysis and model development, and an interactive Streamlit web application for real-time predictions.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📊 Project Overview

This project helps banks identify customers who are likely to churn (leave the bank) by analyzing key customer attributes. The solution provides:

- **Data Analysis**: Comprehensive EDA of customer data
- **Machine Learning**: Multiple models trained and evaluated
- **Web Dashboard**: Interactive interface for real-time predictions
- **Feature Importance**: Insights into what drives customer churn

## 🎯 Live Demo
<img width="1919" height="938" alt="Image" src="https://github.com/user-attachments/assets/84ba7c34-f874-4d97-8a28-9dd88db28c5b" />
<img width="1919" height="938" alt="Image" src="https://github.com/user-attachments/assets/93f861b4-513f-45c3-b247-1e2bb42c50f3" />
*Main Dashboard Interface*


<img width="1806" height="843" alt="Image" src="https://github.com/user-attachments/assets/618e14cc-8dd9-4f65-896d-91e8532014d5" />
*Real-time Prediction Results*


## 🚀 Features

### 📈 Analysis Notebook (`customer chum analisi.ipynb`)
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Feature selection using Chi-squared and Random Forest
- Handling class imbalance with SMOTE
- Multiple model training and comparison:
  - Random Forest Classifier
  - Logistic Regression
  - XGBoost Classifier
- Model evaluation and performance metrics

### 🌐 Streamlit Application (`Streamlit_app.py`)
- Interactive web dashboard with real-time predictions
- Feature importance visualization with interactive charts
- Customer input parameters via sidebar
- Risk assessment with color-coded indicators
- Actionable recommendations based on churn probability
- Responsive design with visual metrics

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📋 Requirements

Create a `requirements.txt` file with the following:

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.15.0
matplotlib==3.7.1
seaborn==0.12.2
imbalanced-learn==0.10.1
xgboost==1.7.5
missingno==0.5.2
jupyter==1.0.0
```

## 🎯 Usage

### Running the Analysis
1. Open the Jupyter notebook:
```bash
jupyter notebook "customer chum analisi.ipynb"
```

2. Execute cells sequentially to:
   - Load and explore the data
   - Perform feature engineering
   - Train machine learning models
   - Generate predictions

### Running the Web Application
1. Ensure you have the trained model files:
   - `best_model.pkl` (or `best_churn_model.pkl`)
   - `scaler.pkl`

2. Launch the Streamlit app:
```bash
streamlit run Streamlit_app.py
```

3. Open your browser to the provided URL (typically `http://localhost:8501`)

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Streamlit_app.py              # Main Streamlit application
├── customer chum analisi.ipynb   # Jupyter notebook analysis
├── best_model.pkl               # Trained model (generated)
├── scaler.pkl                   # Feature scaler (generated)
├── Churn_Modelling.csv          # Dataset
├── requirements.txt             # Python dependencies
├── Pic 11.png                   # Sidebar image
├── Pic 12.png                   # Header image
├── Screenshot 2025-10-04 231803.png  # Dashboard screenshot
├── Screenshot 2025-10-04 231827.png  # Prediction results screenshot
└── README.md                    # This file
```

## 🔧 Model Details

- **Algorithm**: Random Forest Classifier
- **Key Features** (by importance):
  - **Age** (23.75% impact) - Most significant predictor
  - **Estimated Salary** (14.61% impact)
  - **Credit Score** (14.31% impact)
  - **Balance** (14.30% impact)
  - **Number of Products** (12.77% impact)
  - **Tenure** (8.14% impact)

- **Performance Metrics**:
  - Accuracy: ~86%
  - ROC AUC: ~85%

## 📊 Dashboard Features

### Left Panel - Feature Importance
- Interactive horizontal bar chart showing feature impacts
- Color-coded importance scores
- Feature descriptions with impact percentages

### Right Panel - Prediction Interface
- **Current Input Summary**: Real-time display of customer parameters
- **Visual Metrics**: Impact indicators for each feature
- **Prediction Button**: Triggers churn probability calculation
- **Risk Assessment**: Color-coded results (Green/Yellow/Red)

### Prediction Results Include:
- **Churn Probability Percentage**
- **Risk Level Indicator**
- **Detailed Probability Breakdown**
- **Actionable Recommendations** based on risk level

## 🎮 How to Use the Dashboard

1. **Adjust Parameters**: Use the sidebar sliders and inputs to set customer attributes:
   - Age (18-100 years)
   - Estimated Salary ($0-$300,000)
   - Credit Score (300-850)
   - Balance ($0-$300,000)
   - Number of Products (1-4)
   - Tenure (0-10 years)

2. **View Feature Importance**: Understand which factors most influence churn predictions

3. **Get Prediction**: Click "🎯 Predict Churn Probability" to see real-time results

4. **Interpret Results**:
   - **🟢 LOW RISK** (< 40%): Customer likely to stay
   - **🟡 MEDIUM RISK** (40-70%): Monitor closely
   - **🔴 HIGH RISK** (> 70%): Immediate action needed

## 📈 Risk-Based Recommendations

### Low Risk (< 40%)
```
✅ Maintain and Grow:
• Continue excellent service
• Cross-sell additional products  
• Loyalty program enrollment
```

### Medium Risk (40-70%)
```
⚠️ Monitor and Engage:
• Regular check-ins
• Product recommendation campaigns
• Customer satisfaction survey
```

### High Risk (> 70%)
```
🚨 Immediate Retention Actions:
• Proactive customer service call
• Personalized retention offers  
• Account review with relationship manager
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Synthetic customer churn data for banking sector
- **Libraries**: Streamlit, Scikit-learn, Plotly, Pandas, NumPy
- **Icons**: Streamlit community and open-source contributors
- **Inspiration**: Real-world banking customer retention challenges

---

**Note**: This is a demonstration project for educational purposes. Always validate models with real business data and domain expertise before production deployment. The model performance may vary with different datasets and business contexts.

---

<div align="center">

**Built with ❤️ using Streamlit and Scikit-learn**

*Last updated: October 2024*

</div>
