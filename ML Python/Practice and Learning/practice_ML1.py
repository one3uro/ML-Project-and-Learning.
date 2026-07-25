from sklearn.linear_model import LinearRegression
import numpy as np



StudyTime = np.array([1, 2, 3, 4, 5]) #X
HoursSlept = np.array ([2, 5, 3, 8, 9]) #X 





X_InputMatrix = np.column_stack((StudyTime, HoursSlept))

y_TestScores = np.array([3, 5, 7, 9, 12]) #y

model = LinearRegression(fit_intercept=True)

model.fit(X_InputMatrix,y_TestScores)

print("Model Training complete.")
print(f"Coefficients:{model.coef_}, Bias: {model.intercept_}" )


NewData = [[6,4]]
Prediction = model.predict(NewData)
print(f"Prediction. Predicted Score :{Prediction}")
