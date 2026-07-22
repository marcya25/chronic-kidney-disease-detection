# 🩺 NephroScan – AI-Powered Chronic Kidney Disease Detection

An intelligent machine learning application that assists in the early detection of **Chronic Kidney Disease (CKD)** using patient biomarkers and clinical indicators.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![License](https://img.shields.io/badge/License-Open_Source-success?style=for-the-badge)

---

## 📖 Overview

**NephroScan** is a machine learning-powered clinical decision support system designed to identify patients who may be at risk of Chronic Kidney Disease (CKD).

The application combines a trained **Logistic Regression** model with a modern **Streamlit** web interface to deliver fast and user-friendly kidney disease predictions using common clinical biomarkers.

> **Model Accuracy:** **99%**

---

## ✨ Features

- 🤖 AI-powered CKD prediction
- 🧠 Logistic Regression classification model
- 🌐 Interactive Streamlit web application
- ⚡ Real-time prediction results
- 📊 Biomarker-based risk assessment
- 🏥 Clinical decision support interface
- 📈 Data preprocessing and feature scaling
- 💾 Saved trained model using Joblib

---

# 🏗️ System Architecture

```
Patient Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Scaling
(StandardScaler)
      │
      ▼
Logistic Regression Model
      │
      ▼
CKD Prediction
      │
      ▼
Risk Assessment Display
(Streamlit)
```

---

# 📊 Model Performance

| Metric | Value |
|---------|------:|
| Algorithm | Logistic Regression |
| Accuracy | **99.00%** |
| Train/Test Split | 80% / 20% |
| Framework | Scikit-learn |

---

# 🧬 Input Features

The prediction model uses the following patient information.

### Laboratory Parameters

- Serum Creatinine
- Albumin
- Hemoglobin
- Blood Pressure
- Age

### Clinical Indicators

- Diabetes Mellitus
- Hypertension
- Appetite
- Anemia

---

# 📁 Project Structure

```text
chronic-kidney-disease-detection/
│
├── myapp.py
├── PYTHON ,PROJECT.py
├── kidney_disease.csv
├── ckd_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

# 📄 File Description

| File | Description |
|------|-------------|
| `myapp.py` | Streamlit application |
| `PYTHON ,PROJECT.py` | Data preprocessing and model training |
| `kidney_disease.csv` | Dataset |
| `ckd_model.pkl` | Trained machine learning model |
| `scaler.pkl` | StandardScaler object |
| `requirements.txt` | Project dependencies |

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/yourusername/chronic-kidney-disease-detection.git

cd chronic-kidney-disease-detection
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run myapp.py
```

The application will launch at

```
http://localhost:8501
```

---

# 🧠 Machine Learning Pipeline

The project follows the complete machine learning workflow:

- Data Loading
- Exploratory Data Analysis
- Missing Value Handling
- Feature Engineering
- Encoding Categorical Variables
- Feature Scaling
- Model Training
- Model Evaluation
- Model Serialization
- Streamlit Deployment

---

# 📈 Training Process

The notebook/script performs:

### Exploratory Data Analysis

- Dataset overview
- Missing value analysis
- Statistical summary
- Correlation analysis
- Visualizations

### Data Cleaning

- Remove missing values
- Convert categorical values into numeric form
- Normalize numeric features

### Model Training

- Logistic Regression
- 80/20 Train-Test Split
- StandardScaler
- Accuracy Evaluation
- Confusion Matrix
- Classification Report

---

# 💻 Usage

1. Launch the application.
2. Enter patient biomarker values.
3. Select clinical indicators.
4. Click **Run Diagnostic Analysis**.
5. View the predicted CKD risk instantly.

---

# 📌 Prediction Results

## 🟥 High Risk

```
Elevated CKD Risk Detected
```

Recommendation:

- Immediate clinical review
- Referral to a nephrologist
- Further diagnostic testing

---

## 🟩 Low Risk

```
Low Risk — No CKD Detected
```

Recommendation:

- Routine monitoring
- Healthy lifestyle maintenance

---

# 📊 Visualizations

The training notebook includes several visual analyses:

- Box Plots
- Correlation Heatmap
- Scatter Plots
- Line Charts
- Pie Charts
- Bar Charts

---

# 💻 Example Code

```python
import joblib
import numpy as np

model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

patient = np.array([[
    1.8,
    4,
    9,
    150,
    1,
    1,
    60,
    0,
    1
]])

patient = scaler.transform(patient)

prediction = model.predict(patient)

if prediction[0] == 1:
    print("High Risk")
else:
    print("Low Risk")
```

---

# 📦 Dependencies

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Joblib

Install all packages using:

```bash
pip install -r requirements.txt
```

---

# 📚 Dataset

The dataset contains patient clinical records used for CKD prediction.

**Features**

- Clinical indicators
- Laboratory biomarkers
- Demographic information

**Target**

- CKD
- Non-CKD

---

# 🚀 Deployment

The application can be deployed using:

- Streamlit Community Cloud
- Docker
- Heroku
- Microsoft Azure
- Google Cloud Platform
- AWS

---

# ⚠️ Disclaimer

This application is intended for **educational and research purposes only**.

NephroScan is **not a substitute for professional medical advice, diagnosis, or treatment**. Clinical decisions should always be made by qualified healthcare professionals.

---

# 🤝 Contributing

Contributions are welcome.

You can help by:

- Reporting bugs
- Suggesting improvements
- Opening issues
- Submitting pull requests

---

# 👨‍💻 Author

**Marcy**

GitHub: **@marcya25**

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps support future improvements and makes the project easier for others to discover.
