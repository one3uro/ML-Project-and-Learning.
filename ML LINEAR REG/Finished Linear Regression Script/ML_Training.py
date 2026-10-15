# This Python ML algorithm was written in mainly scratch python only slight 3rd prty libraries were used such as pandas and numpy.


import numpy as np
import pandas as pd

df = pd.read_csv("student_performance.csv")



#Features
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


#Capture mean/std BEFORE scaling — needed later to scale new/unseen data the same way
hs_mean, hs_std = Hours_Studied.mean(), Hours_Studied.std()
sl_mean, sl_std = Hours_Slept.mean(), Hours_Slept.std()
at_mean, at_std = Attendance_Percent.mean(), Attendance_Percent.std()
pr_mean, pr_std = Previous_Test_Score.mean(), Previous_Test_Score.std()


#ML Algo explosion fool proof (feature scaling)
Hours_Studied = (Hours_Studied - hs_mean) / hs_std
Hours_Slept = (Hours_Slept - sl_mean) / sl_std
Attendance_Percent = (Attendance_Percent - at_mean) / at_std
Previous_Test_Score = (Previous_Test_Score - pr_mean) / pr_std

#Weights and Bias
W1 = 0.0
W2 = 0.0
W3 = 0.0
W4 = 0.0
B = 0.0


#Learning Loops
LearningRate = 0.01
epochs = 1000

for epoch in range(epochs):
    
    #Prediction
    
    prediction = (W1 * Hours_Studied) +  (W2 * Hours_Slept) + (W3 * Attendance_Percent) + (W4 * Previous_Test_Score) + B
    
    #Measure Error
    error = prediction - Actual_Test_Score
    
    
    #Compute Gradients
    Gradient_1 = np.mean( 2 * error * Hours_Studied)
    Gradient_2 = np.mean( 2 * error * Hours_Slept)
    Gradient_3 = np.mean( 2 * error * Attendance_Percent)
    Gradient_4 = np.mean( 2 * error * Previous_Test_Score)
    BiasGradient = np.mean(2 * error)
    
    #Updated Gradient Parameters
    W1 = W1 - (LearningRate * Gradient_1)
    W2 = W2 - (LearningRate * Gradient_2)
    W3 = W3 - (LearningRate * Gradient_3)
    W4 = W4 - (LearningRate * Gradient_4)
    B = B - (LearningRate * BiasGradient)
    
    #Print Iteration update to watch for errors and irregular/erronaeus data
    if epoch % 10 == 0:
        loss = np.mean(error ** 2)
        print(f"ML Algorithm on iteration {epoch}. Loss = {loss}")



#Print Final Training parameters.
print(f"Final Parameters Gradient 1: {W1:.4f}. \n Gradient_2: {W2:.4f}.\n Gradient_3: {W3:.4f}. \n Gradient_4: {W4:.4f}. \n  Bias Gradient: {B:.4f}. ")


#Output Final ML Algo prediction.
FinalPrediction =  (W1 * Hours_Studied) +  (W2 * Hours_Slept) + (W3 * Attendance_Percent) + (W4 * Previous_Test_Score) + B
print(f"final prediction: {FinalPrediction}")
print(f"Actual Test Scores: {Actual_Test_Score}")

#Feed the ML algo unseen data in order to test if it has learned properly.
UnseenHours_Studied = 0.2
UnseenHours_Slept = 2.5
UnseenAttendance_Percent = 98.9
UnseenPrevious_Test_Score = 84.5

#Scale the unseen data
scaled_hours_studied = (UnseenHours_Studied - hs_mean) / hs_std
scaled_hours_slept = (UnseenHours_Slept - sl_mean) / sl_std
scaled_attendance = (UnseenAttendance_Percent - at_mean) / at_std
scaled_previous_score = (UnseenPrevious_Test_Score - pr_mean) / pr_std

NewPrediction = (W1 * scaled_hours_studied) + (W2 * scaled_hours_slept) + (W3 * scaled_attendance) + (W4 * scaled_previous_score) + B

print(f"Using unseen data, ML Algorithm has predicted: {NewPrediction:.4f}")