import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import os

os.system('cls')


df = pd.read_csv("D:\\VSCode\\ML Python\\Logistic regression\\student_performance_logistic.csv")



#Features
Hours_Studied = df['hours_studied'].to_numpy()
Hours_Slept = df['hours_slept'].to_numpy()
Attendance_Percent = df['attendance_percent'].to_numpy()
Previous_Test_Score = df['previous_score'].to_numpy()

Passed_Or_Fail = df['passed_exam'].to_numpy()

Stacked_Matrix = np.column_stack((Hours_Studied, Hours_Slept, Attendance_Percent, Previous_Test_Score))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    Stacked_Matrix,
    Passed_Or_Fail,
    random_state=42,
    test_size=0.4,
    stratify=Passed_Or_Fail
)

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

Scaled_Values = StandardScaler()

Scaled_XTrain = Scaled_Values.fit_transform(X_train_poly)
Scaled_Xtest = Scaled_Values.transform(X_test_poly)

model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
model.fit(Scaled_XTrain, y_train)

prediction = model.predict(Scaled_Xtest)
Accuracy = accuracy_score(y_test, prediction)

# Unseen data
Unseen_Data = np.array([[7.0, 7.5, 85.0, 78.0]])

Unseen_Poly_Data = poly.transform(Unseen_Data)
Unseen_Scaled_Data = Scaled_Values.transform(Unseen_Poly_Data)

Unseen_Prediction = model.predict(Unseen_Scaled_Data)

print(f"Model Prediction Based On Data Used to Train Where Pass is [1] and Fail is [0] = {prediction} \n")

print(f"Actual Values Where Pass [1] and Fail [0] = {y_test}\n")

print(f"Model Prediction For Unseen Data Where Pass is [1] and Fail is [0] = {Unseen_Prediction}")

print(f"\nModel Accuracy: {100*Accuracy:.2f}% \n")

summed_model_pred_output = np.bincount(prediction)
summed_actual_model_output = np.bincount(y_test)
summed_unseen_data_output = np.bincount(Unseen_Prediction)

print(f"Total Passes For Model Prediction Based On Data Used to Train: {summed_model_pred_output[1]}")
print(f"Total Fails For Model Prediction Based On Data Used to Train: {summed_model_pred_output[0]}\n")

print(f"Total Passes For Actual Values: {summed_actual_model_output[1]}")
print(f"Total Fails For Actual Values: {summed_actual_model_output[0]}\n")

print(f"Total Passes For Unseen Values: {summed_unseen_data_output[1]}")
print(f"Total Fails For Unseen Values: {summed_unseen_data_output[0]}")

