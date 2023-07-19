# Movie Profitability Predictor
This project aims to predict the profitability of movies using two different approaches:

A classification approach that predicts whether a movie will be profitable or not.
A regression approach that predicts the profitability of a movie in percentage terms.

## Requirements
Python 3
Libraries: pandas, numpy, scikit-learn. The specific versions of these libraries that are required can be found in the requirements.txt file.

## Dataset
The dataset used in this project is located in the ./data folder.

## Code
The source code for this project can be found in the ./src folder. It consists of two main Python scripts:

movie_profitability_classification.py: This script uses a Random Forest Classifier to predict whether a movie will be profitable or not, based on the defined categories. It uses features such as genre popularity, theme popularity, emotion rarity, reference, and budget to make these predictions.

movie_profitability_regression.py: This script uses Linear Regression to predict a movie's profitability in percentage terms. It uses the same features as the classification script.

Both scripts implement a flexible and reusable machine learning pipeline. This pipeline includes data loading, preprocessing, model training, and prediction.

Predictions made by the scripts are saved in the ./predictions/classification and ./predictions/regression folders respectively.

## Classification Profitability Definitions
In the context of the classification task, the level of profitability is defined as follows:

If the profitability is greater than 50%, it's considered a success.
If the profitability is between 0% and 50%, it's considered a failure.
If the profitability is less than 0%, it's considered a total failure.
These categories are encoded as follows: 0 for total failure, 1 for failure, and 2 for success.