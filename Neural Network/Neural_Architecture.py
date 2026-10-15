"""
This Python ML algorithm was written in mainly scratch python
only slight 3rd prty libraries were used such as pandas and numpy.
"""
import numpy as np
import pandas as pd


df = pd.read_csv("/media/zee/ExternalDrive/Python Projects/ML-Project-and-Learning./ML LINEAR REG/Finished Linear Regression Script/student_performance.csv")


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

stacked_features = np.column_stack(
    (Hours_Studied, Hours_Slept, Attendance_Percent, Previous_Test_Score)
    )


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
LEARNING_RATE = 0.01
EPOCHS = 1000
# Random weights for hidden layer (4 inputs, 3 neurons) 

HIDDEN_W = np.random.rand(4, 3) * 0.01
HIDDEN_B = np.zeros(3)

W_OUT = np.random.rand(3, 1) * 0.01
B_OUT = 0.0

hidden_raw = np.dot(stacked_features, HIDDEN_W) + HIDDEN_B



#Add an activation function in which we can squeeze,
#the output of the ML algorithm into a range of 0 to 1.
#This is is useful for preventing the model from predicting negative values
def relu(z):
    """
    Activation function that returns the,
    input value if it is greater than 0, otherwise it returns 0.
    """
    if z < 0:
        return np.maximum(0, z)
    else:
        return 

hidden_out = relu(hidden_raw)


fin_pred = np.dot(hidden_out, W_OUT) + B_OUT

for epoch in range(EPOCHS):
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
    W1 = W1 - (LEARNING_RATE * Gradient_1)
    W2 = W2 - (LEARNING_RATE * Gradient_2)
    W3 = W3 - (LEARNING_RATE * Gradient_3)
    W4 = W4 - (LEARNING_RATE * Gradient_4)
    B = B - (LEARNING_RATE * BiasGradient)  
    #Print Iteration update to watch for errors and irregular/erronaeus data
    if epoch % 10 == 0:
        loss = np.mean(error ** 2)
        print(f"ML Algorithm on iteration {epoch}. Loss = {loss}")



#Print Final Training parameters.
print(f"Final Parameters Gradient 1: {W1:.4f}. \n Gradient_2: {W2:.4f}.\n Gradient_3: {W3:.4f}. \n Gradient_4: {W4:.4f}. \n  Bias Gradient: {B:.4f}. ")


#Output Final ML Algo prediction.
FINAL_PREDICTION =  (W1 * Hours_Studied) +  (W2 * Hours_Slept) + (W3 * Attendance_Percent) + (W4 * Previous_Test_Score) + B
print(f"final prediction: {FINAL_PREDICTION}")
print(f"Actual Test Scores: {Actual_Test_Score}")

#Feed the ML algo unseen data in order to test if it has learned properly.
UNSEEN_HOURS_STUDIED = 0.2
UNSEEN_HOURS_SLEPT = 2.5
UNSEEN_ATTENDANCE_PERCENT = 98.9
UNSEEN_PREVIOUS_TEST_SCORE = 84.5

#Scale the unseen data
scaled_hours_studied = (UNSEEN_HOURS_STUDIED - hs_mean) / hs_std
scaled_hours_slept = (UNSEEN_HOURS_SLEPT - sl_mean) / sl_std
scaled_attendance = (UNSEEN_ATTENDANCE_PERCENT - at_mean) / at_std
scaled_previous_score = (UNSEEN_PREVIOUS_TEST_SCORE - pr_mean) / pr_std

NEW_PREDICTION = (W1 * scaled_hours_studied) + (W2 * scaled_hours_slept) + (W3 * scaled_attendance) + (W4 * scaled_previous_score) + B

print(f"Using unseen data, ML Algorithm has predicted: {NEW_PREDICTION:.4f}")