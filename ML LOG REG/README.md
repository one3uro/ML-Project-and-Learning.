# Student Performance Prediction

A Python script that uses **Logistic Regression** to predict whether a student will pass or fail an exam based on their daily habits and academic history.

## Features Used
* **Hours Studied**
* **Hours Slept**
* **Attendance Percentage**
* **Previous Test Score**

## How It Works
1. **Data Processing:** Loads student data from a CSV file and creates a feature matrix using NumPy.
2. **Feature Engineering:** Applies `PolynomialFeatures` (degree 2) to capture non-linear relationships between variables.
3. **Scaling:** Uses `StandardScaler` to normalize the data for better model convergence.
4. **Model Training:** Trains a `LogisticRegression` classifier (using the LBFGS solver) with a 60/40 train/test split.
5. **Evaluation:** Outputs the final accuracy percentage and compares predictions against actual test values alongside a breakdown of total passes vs. fails.
6. **Inference:** Tests model prediction capabilities using a custom entry of unseen student data.

## Prerequisites

Make sure you have the required libraries installed before running the script:

```bash
pip install numpy pandas scikit-learn
```

## Setup & Running

1. Update the dataset file path inside the script to match your local setup:
   ```python
   df = pd.read_csv("YOUR_PATH_TO/student_performance_logistic.csv")
   ```
2. Run the script:
   ```bash
   python LogReg.py
   ```

