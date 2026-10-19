# Credit Card Fraud Detection Using a Neural Network

A Python machine learning project that uses a **Multi-Layer Perceptron (MLP) Neural Network** to identify potentially fraudulent credit card transactions based on transaction data.

The project explores data preprocessing, feature scaling, neural network training, and model evaluation while developing an understanding of how machine learning can be applied to fraud detection.

## Features Used

- **Time:** Time elapsed between transactions and the first transaction in the dataset.
- **V1–V28:** Anonymised numerical features provided by the dataset.
- **Amount:** The transaction amount.
- **Class:** The target variable used to identify legitimate and fraudulent transactions (`0` = legitimate, `1` = fraudulent).

The `Class` column is used as the target and is excluded from the model's input features.

## How It Works

1. **Data Loading:** Loads `creditcard.csv` using Pandas and displays the dataset dimensions and first five rows.
2. **Feature Separation:** Separates the input features (`X`) from the target variable (`y`).
3. **Train/Test Split:** Uses `train_test_split` with an 80/20 split, `random_state=42`, and `stratify=y` to preserve the class distribution.
4. **Feature Scaling:** Uses `StandardScaler` to standardise the input features. The scaler is fitted on the training data and then applied to the testing data.
5. **Model Creation:** Builds an `MLPClassifier` neural network with one hidden layer containing 10 neurons and the ReLU activation function.
6. **Model Training:** Trains the neural network using the training data, with a maximum of 1,000 iterations.
7. **Prediction:** Uses the trained model to predict whether transactions in the testing set are legitimate or fraudulent.
8. **Evaluation:** Calculates accuracy and prints a classification report containing precision, recall, F1-score, and support.
9. **Fraud Analysis:** Uses a confusion matrix to examine correctly identified transactions, missed fraud cases, and legitimate transactions incorrectly flagged as fraudulent.

## Initial Model Results

The first neural network experiment produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 99.94% |
| Fraud Precision | 89% |
| Fraud Recall | 77% |
| Fraud F1-Score | 0.82 |
| Training Iterations | 22 |

These results are from the initial model run and may vary if the dataset, preprocessing, or model configuration changes.

**Important:** Accuracy alone is not a reliable measure of fraud detection performance because the dataset contains far more legitimate transactions than fraudulent ones.

- **Precision:** Measures how often transactions predicted as fraudulent are actually fraudulent.
- **Recall:** Measures how many actual fraudulent transactions the model successfully detects.
- **F1-Score:** Combines precision and recall into a single metric.

The initial recall of 77% indicates that the model detects most, but not all, fraudulent transactions in the test set.

## What I Have Learned

- **Data Preprocessing:** Learned how to load, inspect, and prepare a dataset for machine learning.
- **Feature Engineering and Selection:** Learned how to separate input features from the target variable and why the target must not be included in the model's inputs.
- **Train/Test Splitting:** Understood why training and testing data must be separated to evaluate performance on unseen examples.
- **Feature Scaling:** Learned how standardisation helps prepare numerical features for neural network training and why the scaler should be fitted only on training data.
- **Neural Networks:** Gained practical experience using `MLPClassifier`, configuring hidden layers and activation functions, and training a neural network.
- **Model Evaluation:** Learned how accuracy, precision, recall, and F1-score measure different aspects of classification performance.
- **Class Imbalance:** Understood why a model can achieve high accuracy while still missing fraudulent transactions.
- **Model Improvement:** Learned that an initial model should be treated as a baseline and that further experimentation is needed to evaluate and improve performance.

## Prerequisites

Make sure Python is installed and the required libraries are available:

```bash
pip install pandas scikit-learn numpy
```

## Setup and Running

1. Clone or download the repository.
2. Ensure that `creditcard.csv` is available in the project directory.
3. Update the dataset path in the notebook if necessary:

   ```python
   file_path = r"Neural Network Google Colab/creditcard.csv"
   ```

4. Open `Neural_network.ipynb` in Visual Studio Code with the Jupyter extension installed, or open it in a compatible Jupyter environment.
5. Select a Python kernel and run the notebook cells in order.

The dataset path must match your local folder structure. If the notebook is already inside the `Neural Network Google Colab` folder, you may be able to use:

```python
file_path = "creditcard.csv"
```

## Project Structure

```text
Neural Network Google Colab/
    ├── creditcard.csv
    ├── Neural_network.ipynb
    └── README.md
```

The structure above reflects the project folders shown in the development environment. Your actual repository may contain additional files.

## Future Improvements

- Compare the neural network against the logistic regression model developed in the other project folder.
- Print the exact number of fraudulent transactions detected and missed.
- Analyse the confusion matrix to understand false positives and false negatives.
- Investigate techniques for handling class imbalance.
- Experiment with different hidden-layer sizes and model parameters.
- Compare models using precision, recall, F1-score, and accuracy.
- Document experiment results and explain which changes improve performance.

## Project Status

**In Progress:** The initial neural network has been implemented, trained, and evaluated. Future work will focus on analysing individual predictions, comparing algorithms, and investigating improvements to fraud detection performance.

## Disclaimer

This project is for educational purposes and machine learning experimentation. The model has not been validated for real-world financial use and should not be used as the sole basis for financial decisions or transaction approvals.

## Dataset Acknowledgement

The project uses a credit card fraud detection dataset. The original dataset source is from Kaggle.
