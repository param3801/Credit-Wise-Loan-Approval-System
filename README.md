# 💳 CreditWise – Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-83%25-success)

---

## 📖 Project Overview

**CreditWise** is an end-to-end Machine Learning project designed to predict whether a loan application will be approved or not.

This project implements a complete ML pipeline including:

- Data preprocessing  
- Missing value handling  
- Exploratory Data Analysis (EDA)  
- Feature encoding  
- Feature scaling  
- Model training  
- Model evaluation  

Three classification algorithms were implemented and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  

The goal is to identify the best-performing model based on accuracy, precision, and recall.

---

## 🎯 Problem Statement

Financial institutions must assess loan applications carefully to minimize risk.

This project builds a binary classification model that predicts:

- **1 → Loan Approved**
- **0 → Loan Not Approved**

The model helps understand key factors such as:

- Credit Score  
- Applicant Income  
- Employment Status  
- DTI Ratio  
- Property Area  
- Education Level  

---

## 📂 Dataset Information

The dataset contains applicant details such as:

- Applicant_ID  
- Gender  
- Age  
- Education_Level  
- Employment_Status  
- Marital_Status  
- Applicant_Income  
- Coapplicant_Income  
- Credit_Score  
- DTI_Ratio  
- Loan_Purpose  
- Property_Area  
- Employer_Category  
- Loan_Approved (Target Variable)  

The dataset initially contained missing values which were handled during preprocessing.

---

## ⚙️ Technologies & Libraries Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🛠️ Project Workflow

### 1️⃣ Data Loading
- Loaded dataset using Pandas  
- Created a deep copy for safety  

### 2️⃣ Data Preprocessing
- Handled missing numerical values using **Mean Imputation**  
- Handled categorical missing values using **Most Frequent Imputation**  
- Removed unnecessary column: `Applicant_ID`  

### 3️⃣ Exploratory Data Analysis (EDA)
- Distribution of Loan Approval (Imbalanced dataset ~70:30)  
- Gender distribution  
- Education level distribution  
- Employment status analysis  
- Income distribution  
- Credit score vs Loan approval  
- Outlier detection using Boxplots  
- Correlation Heatmap  

### 4️⃣ Feature Engineering
- Label Encoding:
  - Education_Level  
  - Loan_Approved  

- One-Hot Encoding:
  - Employment_Status  
  - Marital_Status  
  - Loan_Purpose  
  - Property_Area  
  - Gender  
  - Employer_Category  

### 5️⃣ Feature Scaling
- Applied **StandardScaler** to normalize numerical features  

### 6️⃣ Train-Test Split
- 80% Training Data  
- 20% Testing Data  
- Random State = 42  

---

## 🤖 Models Implemented

### 🔹 Logistic Regression
- Linear classification model  
- Evaluated using Accuracy, Precision, Recall  

### 🔹 K-Nearest Neighbors (KNN)
- n_neighbors = 5  
- Distance-based classifier  

### 🔹 Naive Bayes (GaussianNB)
- Probabilistic classifier  
- Best performing model  

---

## 📊 Model Performance & Success Rate

| Model | Accuracy (Success Rate) |
|--------|--------------------------|
| Logistic Regression | ~78% |
| KNN | ~75% |
| 🥇 Naive Bayes | **~83%** |

---

## 🏆 Final Result

After evaluating all three models:

✅ **Naive Bayes performed the best**  
✅ Achieved a success rate (accuracy) of approximately **83%**  
✅ Maintained a strong balance between precision and recall  

This means the model correctly predicts loan approval status for **83 out of 100 applications**.

---

## 📈 Key Insights

- Credit Score above 650 significantly increases approval chances  
- Higher income applicants are more likely to be approved  
- Dataset shows class imbalance (~70:30)  
- DTI Ratio and Credit Score strongly impact loan decisions  

---




