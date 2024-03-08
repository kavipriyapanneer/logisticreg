# logisticreg
# Advertising Dataset Analysis

This repository contains Python code for analyzing the "advertising.csv" dataset, which comprises information related to online advertising. The code utilizes various libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, and machine learning models from `scikit-learn` for analysis and prediction.

## Overview

The provided Python script performs exploratory data analysis (EDA) and builds a logistic regression model to predict whether a user clicked on an advertisement based on various features from the dataset.

## Steps

1. **Data Loading and Exploration**:
   - Loads the dataset using `pandas`.
   - Displays the first few rows of the dataset using `df.head(10)`.
   - Provides information about the dataset using `df.info()`.
   - Generates descriptive statistics using `df.describe()`.
   - Displays the column names using `df.columns`.

2. **Data Visualization**:
   - Plots a histogram for the 'Age' column using `plt.hist()`.
   - Creates a joint plot for 'Area Income' versus 'Age' using `sns.jointplot()`.
   - Generates a joint plot for 'Daily Time Spent on Site' versus 'Daily Internet Usage' using `sns.jointplot()`.

3. **Data Preprocessing**:
   - Drops unnecessary columns ('Timestamp', 'Country', 'City', 'Ad Topic Line') from the dataset using `df.drop()`.
   - Visualizes missing values using a heatmap using `sns.heatmap()`.

4. **Model Building**:
   - Splits the dataset into features (X) and target variable (y) using `train_test_split` from `sklearn.model_selection`.
   - Instantiates a logistic regression model using `LogisticRegression()` from `sklearn.linear_model`.
   - Fits the model to the training data using `l.fit(X_train, y_train)`.

5. **Model Evaluation**:
   - Prints a classification report using `classification_report()` from `sklearn.metrics`.
   - Displays a confusion matrix using `confusion_matrix()` from `sklearn.metrics`.

## Prerequisites

Ensure that you have the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Ensure that you have the "advertising.csv" file available at the specified location.
2. Execute the Python script.
3. Analyze the generated visualizations and model evaluation metrics to understand the dataset and model performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact the repository owner:
- Name: kavipriya
- Email:kavipriyapanneerselvam22@gmail.com
