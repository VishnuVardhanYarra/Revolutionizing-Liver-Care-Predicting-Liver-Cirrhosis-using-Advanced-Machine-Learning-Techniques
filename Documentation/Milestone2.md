
# Milestone 2: Data Collection and Preparation

## Objective

The goal of this milestone is to gather and prepare a dataset that can be used to build a machine learning model for predicting liver cirrhosis. The dataset includes a range of medical and demographic details of patients.

---

## Step 1: Collecting the Dataset

We used a real-world Excel dataset that includes important liver-related metrics such as bilirubin levels, enzyme activity, hepatitis status, and patient history. The data also includes demographic information like gender, age, and location.

This dataset is useful for detecting early signs of liver cirrhosis because it includes medically relevant features such as:

- Liver function tests
- Platelet and hemoglobin levels
- Presence of hepatitis or alcohol history
- A clear target column indicating whether the patient is diagnosed with cirrhosis or not

---

## Step 2: Preparing the Dataset

### Cleaning
- Removed unwanted columns like serial numbers and overly descriptive labels that don't help the model.
- Checked for and removed rows with missing data.
- Simplified column names and standardized formatting for easier handling.

### Label Encoding
- Converted categorical features like gender (`male`, `female`) into numeric format.
- Mapped the target column `Predicted Value` from `YES`/`NO` to 1/0.

### Feature Scaling
- Applied `StandardScaler` to bring all numerical features to the same scale. This helps in improving model performance for many algorithms.

### Splitting the Dataset
- Split the data into training and testing sets using an 80-20 ratio. This is standard practice to test model generalization.

---

## Output Files
- `liver_data.csv` – Cleaned and ready-to-use dataset.
- `data_preparation.ipynb` – Jupyter Notebook that contains all preprocessing steps and code.

This milestone sets the stage for building a predictive model in the next phase.

