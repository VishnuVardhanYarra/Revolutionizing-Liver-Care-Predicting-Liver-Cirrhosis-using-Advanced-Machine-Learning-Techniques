**Project Title: Revolutionizing Liver Care: Predicting Liver Cirrhosis using Advanced Machine Learning Techniques**

---

# 1. INTRODUCTION

## 1.1 Project Overview

This project aims to design a non-invasive, intelligent, and interpretable machine learning system to detect early signs of liver cirrhosis. By analyzing clinical and biochemical features, it supports doctors in making timely decisions.

## 1.2 Purpose

To enable healthcare professionals to perform early risk prediction of liver cirrhosis using readily available clinical data and explainable AI techniques.

---

# 2. IDEATION PHASE

## 2.1 Problem Statement

Delayed detection of liver cirrhosis due to expensive, invasive diagnostics prevents timely treatment. Early-stage symptoms are often missed, leading to complications.

## 2.2 Empathy Map Canvas

Empathy maps were developed for four contributors (VishnuVardhan, Sunny, Ujwala, SurendraReddy) to align technical ideas with user experience and real-world expectations.

## 2.3 Brainstorming

Each team member proposed approaches using ML models, web interfaces, explainability tools (like SHAP), and deployment strategies. Ideas were prioritized for feasibility and impact.

---

# 3. REQUIREMENT ANALYSIS

## 3.1 Customer Journey Map

Mapped from awareness to post-implementation, addressing pain points such as lack of tools, decision uncertainty, and time constraints.

## 3.2 Solution Requirement

* Functional: Registration, prediction, explainability, feedback, admin dashboard.
* Non-Functional: Usability, performance, scalability, security, availability.

## 3.3 Data Flow Diagram

Includes: Patient -> Web App -> ML Model -> SHAP Explanation -> Doctor

## 3.4 Technology Stack

Frontend: HTML, CSS, JS / Streamlit
Backend: Python, Flask / FastAPI
ML: scikit-learn, XGBoost, SHAP
Database: SQLite / Firebase
Deployment: IBM Cloud / Docker

---

# 4. PROJECT DESIGN

## 4.1 Problem-Solution Fit

Validated the gap between existing clinical practice and the need for interpretable, data-driven decisions. Mapped customer constraints, triggers, and behaviors.

## 4.2 Proposed Solution

A lightweight, browser-based platform that takes input parameters (age, ALT, AST, etc.) and provides real-time predictions with SHAP visualizations.

## 4.3 Solution Architecture

Multi-tiered: UI -> API -> ML Engine -> Explainability Module -> Result Viewer
Deployed via IBM Cloud, optionally containerized.

---

# 5. PROJECT PLANNING & SCHEDULING

## 5.1 Project Planning

Work was divided into sprints:

* Sprint 1: Data collection + UI setup
* Sprint 2: Model training + SHAP integration
* Sprint 3: Web deployment + validation

---

# 6. FUNCTIONAL AND PERFORMANCE TESTING

## 6.1 Performance Testing

### Classification Model Performance

| Metric           | Value                                                               |
| ---------------- | ------------------------------------------------------------------- |
| Accuracy         | 0.87                                                                |
| Precision        | 0.89                                                                |
| Recall           | 0.89                                                                |
| F1 Score         | 0.89                                                                |
| Confusion Matrix | ![Confusion Matrix](sandbox:/mnt/data/confusion_matrix_example.png) |

### Regression Model Performance

| Metric                        | Value                                                           |
| ----------------------------- | --------------------------------------------------------------- |
| MAE (Mean Absolute Error)     | 0.229                                                           |
| MSE (Mean Squared Error)      | 0.063                                                           |
| RMSE (Root Mean Square Error) | 0.251                                                           |
| R² Score                      | 0.985                                                           |
| Regression Plot               | ![Regression Plot](sandbox:/mnt/data/liver_regression_plot.png) |

---

# 7. RESULTS

## 7.1 Output Screenshots

Screenshots are available in the accompanying files.

---

# 8. ADVANTAGES & DISADVANTAGES

**Advantages:**

* Early detection
* Non-invasive approach
* Explainable predictions

**Disadvantages:**

* Dependent on data quality
* Needs validation across diverse populations

---

# 9. CONCLUSION

The system bridges the diagnostic gap using AI and clinical data. Doctors gain a second opinion without expensive tools, improving care in low-resource settings.

---

# 10. FUTURE SCOPE

* Mobile app version
* Multi-disease extension
* Integration with hospital EMR systems

---

# 11. APPENDIX

All relevant project materials including source code, dataset, screenshots, and demo video are available in the GitHub repository:

**GitHub Link:** [https://github.com/VishnuVardhanYarra/Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis-using-Advanced-Machine-Learning-Techniques.git](https://github.com/VishnuVardhanYarra/Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis-using-Advanced-Machine-Learning-Techniques.git)
