import binascii
from modulefinder import test

import numpy as np

#features
StudyTime = np.array([1, 2, 3, 4, 5]) #X
HoursSlept = np.array ([2, 5, 3, 8, 9]) #X 
TestScores = np.array([3, 5, 7, 9, 12]) #y

#Parameter Weights
StudyWeight_W1 = 0.0
SleepWeight_W2 = 0.0
bias = 0.0



LearningRate = 0.01

for epoch in range(1000):
    prediction = (StudyWeight_W1 * StudyTime) + (SleepWeight_W2 * HoursSlept) + bias
    error = prediction - TestScores
    
    Gradient1 = np.mean (2 * error * HoursSlept)
    Gradient2 = np.mean (2 * error * StudyTime)
    BiasGradient = np.mean (2 * error)
    
    if epoch % 10 == 0:
        loss = np.mean (error ** 2)
        print(f"Iteration update. On iteration No :{epoch}. Current ML Loss: {loss:.4f} WeightNo1: {StudyWeight_W1:.4f} WeightNo2: {SleepWeight_W2:.4f} BiasWeight: {bias:4f}")
    
    StudyWeight_W1 = StudyWeight_W1 - LearningRate * Gradient2
    SleepWeight_W2 = SleepWeight_W2 - LearningRate * Gradient1
    bias = bias - LearningRate * BiasGradient

print(f"Final Model :  WeightNo1: {StudyWeight_W1:.4f} WeightNo2: {SleepWeight_W2:.4f}  BiasWeight: {bias:4f} ")

FinalPrediction = (StudyWeight_W1 * StudyTime) + (SleepWeight_W2 * HoursSlept) + bias
print(f"Final Prediction = {FinalPrediction}")
print(f"Actual Values = {TestScores}")

new_hours_studied = 6
new_hours_slept = 4
new_prediction = (StudyWeight_W1 * new_hours_studied) + (SleepWeight_W2 * new_hours_slept) + bias
print(f" New prediction: {new_prediction}")
