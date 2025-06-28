
# Milestone 3: Exploratory Data Analysis (EDA)

## Objective

The main goal of this milestone is to explore and understand the structure, distribution, and relationships within the liver patient dataset. EDA helps us discover trends, spot anomalies, and guide feature selection before training machine learning models.

---

## Steps Performed

### 1. Dataset Overview
- Loaded the cleaned CSV (`liver_data.csv`) from the previous milestone.
- Checked the shape of the dataset, column names, and data types.

### 2. Summary Statistics
- Generated descriptive statistics for numerical features using `.describe()`.
- Analyzed distributions, mean, median, and standard deviation.

### 3. Missing Value Analysis
- Verified there are no missing values after cleaning (from Milestone 2).
- Confirmed that all features are usable for modeling.

### 4. Target Variable Distribution
- Visualized the class distribution of `Liver_Cirrhosis` (0 vs 1).
- Ensured no extreme imbalance; slight skew towards positive cases.

### 5. Univariate Analysis
- Created histograms for numerical columns to check distributions.
- Boxplots were used to identify outliers in features like bilirubin, ALT, and AST.

### 6. Bivariate Analysis
- Created correlation heatmap to identify relationships between features.
- Used scatter plots and pair plots for highly correlated pairs.
- Observed positive correlation between certain liver enzymes and cirrhosis.

### 7. Feature Insights
- Features like `Direct Bilirubin`, `ALT`, `AST`, and `Albumin` showed strong trends in affected patients.
- Gender and age had modest impact.

---

## Conclusion

This EDA helped narrow down the most informative features and revealed that enzyme-related features, albumin levels, and bilirubin measures are likely to play a strong role in predicting liver cirrhosis.

