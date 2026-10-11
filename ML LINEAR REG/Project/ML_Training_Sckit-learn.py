from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

#load in CSV to be read.
df = pd.read_csv("student_performance.csv")



#Features
#Convert data from the CSV into numpy arrays

#X1
Hours_Studied = df['hours_studied'].iloc[0:60].to_numpy()
#X2
Hours_Slept = df['hours_slept'].iloc[0:60].to_numpy()
#X3
Attendance_Percent = df['attendance_percent'].iloc[0:60].to_numpy()
#X4
Previous_Test_Score = df['previous_score'].iloc[0:60].to_numpy()
#y
Actual_Test_Score = df['test_score'].iloc[0:60].to_numpy()



#Convert the converted array data into Matrixes.
X_Matrix = np.column_stack((Hours_Studied, Hours_Slept, Attendance_Percent, Previous_Test_Score,))
Y_Matrix = Actual_Test_Score



#Load in Model.
model = LinearRegression(fit_intercept=True)
model.fit(X_Matrix,Y_Matrix)


#Inform when ML has completed its training.
print("*" *70)
print("Model Training Complete.")
print("*" *70)

#Output Bias and coefficients.
print("*" *70)
print(f"Algorithm Coefficients:{model.coef_}, Algorithm Bias: {model.intercept_}" )
print("*" *70)




#Feed unseen data and print prediction.
Unseen_Data = [[0.2, 2.5, 98.9, 84.5 ]]
NewPrediction = model.predict(Unseen_Data)
print(f"Ml ALgortihm Predicts: {NewPrediction} with given unseen data.")