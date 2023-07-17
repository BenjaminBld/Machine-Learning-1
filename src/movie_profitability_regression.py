import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from typing import List, Optional
from datetime import datetime
import logging
import os


class RegressionModel:
    """
    A class used to represent a Regression Model for movie profitability prediction.

    ...

    Attributes
    ----------
    model : RegressorMixin
        a scikit-learn regressor model
    features : list
        a list of feature names used in the model
    target : str
        the name of the target variable
    title : str
        the name of the title column in the dataset
    df : DataFrame
        the DataFrame containing the data
    X_test : DataFrame
        the test set features
    y_test : Series
        the test set target
    titles_test : Series
        the titles of the test set

    Methods
    -------
    load_data(filename: str):
        Loads data from a CSV file.
    preprocess():
        Preprocesses the data.
    train_and_evaluate(test_size: float):
        Trains the model and calculates the mean cross-validation MSE.
    make_predictions():
        Makes predictions on the test set and saves them to a file.
    """

    def __init__(
        self, 
        model: BaseEstimator, 
        features: List[str], 
        target: str, 
        title: str, 
        feature_to_encode: Optional[str] = None
    ):
        """
        Parameters
        ----------
        model : RegressorMixin
            a scikit-learn regressor model
        features : list
            a list of feature names used in the model
        target : str
            the name of the target variable
        title : str
            the name of the title column in the dataset
        """
        # Initialize model, features, target, and title attributes
        self.model = model
        self.features = features
        self.target = target
        self.title = title
        self.feature_to_encode = feature_to_encode
        # Initialize empty DataFrame and Series attributes
        self.df = None
        self.X_test = None
        self.y_test = None
        self.titles_test = None

    def load_data(self, filename: str) -> None:
        """Loads data from a CSV file."""
        if os.path.exists(filename):
            self.df = pd.read_csv(filename)
            logging.info(f"Data loaded from {filename}")
        else:
            raise FileNotFoundError(f"{filename} does not exist.")

    def preprocess(self) -> None:
        """Preprocesses the data."""
        if all(item in self.df.columns for item in [self.title] + self.features + [self.target]):
            # One-hot encode the specified feature if it exists
            if self.feature_to_encode in self.df.columns:
                self.one_hot_encode(self.feature_to_encode)
            self.df = self.df[[self.title] + self.features + [self.target]]
            scaler = StandardScaler()
            self.df[self.features] = scaler.fit_transform(self.df[self.features])
            print(self.df.columns)  # print the columns after preprocessing

            logging.info("Data preprocessing completed.")
        else:
            raise ValueError("Some of the specified features or target are not present in the DataFrame.")

    
    def one_hot_encode(self, column: str) -> None:
        """One-hot encodes a specified column."""
        # Create the encoder
        encoder = OneHotEncoder(sparse=False, drop='first')

        # Fit the encoder and transform the data
        transformed_data = encoder.fit_transform(self.df[[column]])

        # Get feature names
        encoded_features = encoder.get_feature_names_out([column])

        # Convert the transformed data into a dataframe
        encoded_df = pd.DataFrame(transformed_data, columns=encoded_features)

        # Drop the original column from the dataframe
        self.df = self.df.drop(column, axis=1)

        # Concatenate the original dataframe with the encoded dataframe
        self.df = pd.concat([self.df, encoded_df], axis=1)

        # Update the features list to include the new one-hot encoded features and exclude the original categorical feature
        self.features = [feature for feature in self.features if feature != column] + list(encoded_features)

    def train_and_evaluate(self, seed: int, test_size: float = 0.2) -> dict:
        """
        Trains the model and calculates various cross-validation metrics.

        Parameters
        ----------
        seed : int
            The random seed for splitting the data.
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2)

        Returns
        -------
        metrics : dict
            The metrics calculated for the model, including mean squared error (MSE),
            root mean squared error (RMSE), mean absolute error (MAE), and R-squared (R2).
        """

        # Split the DataFrame into feature and target variables
        X = self.df[self.features]
        y = self.df[self.target]
        titles = self.df[self.title]
        # Split the data into training and testing sets
        (
            X_train,
            self.X_test,
            y_train,
            self.y_test,
            _,
            self.titles_test,
        ) = train_test_split(X, y, titles, test_size=test_size, random_state=seed)
        # Train the model
        self.model.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Evaluate the model using cross-validation and calculate mean squared error
        scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        )
        mse_scores = -scores
        rmse_scores = np.sqrt(mse_scores)
        mae_scores = -cross_val_score(self.model, X_train, y_train, scoring="neg_mean_absolute_error", cv=KFold(n_splits=5, shuffle=True, random_state=seed))
        r2_scores = cross_val_score(self.model, X_train, y_train, scoring="r2", cv=KFold(n_splits=5, shuffle=True, random_state=seed))

        return {'mse': mse_scores.mean(), 'rmse': rmse_scores.mean(), 'mae': mae_scores.mean(), 'r2': r2_scores.mean()}

    def make_predictions(self, metrics: dict) -> None:
        """
        Makes predictions on the test set and saves them to a file.

        Parameters
        ----------
        metrics : dict
            The metrics calculated for the model.

        Notes
        -----
        The predictions are saved to a CSV file in the '../predictions/regression' directory. 
        The metrics are saved to a text file in the same directory.
        """

        y_pred = self.model.predict(self.X_test)
        predictions = pd.DataFrame({self.title: self.titles_test, "Predicted " + self.target: y_pred})
        file_name = f'../predictions/regression/predictions_{self.model.__class__.__name__}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        predictions.to_csv(file_name + ".csv", index=False)
        with open(file_name + "_metrics.txt", "w") as f:
            f.write(f"Metrics: {metrics}")
        logging.info("Predictions saved to file.")


def main():
    """
    The main function that orchestrates the movie profitability prediction process.
    It loads the data, preprocesses it, trains a linear regression model, evaluates it, and makes predictions.
    """
    logging.basicConfig(filename='regression_model.log', level=logging.INFO)

    # Define seed for random processes
    SEED = 42

    # File path
    filename = "../data/movies_data.csv"

    # Name of the title column
    title = "Titre"

    # Name of the target variable
    target = "Rentabilité (%)"

    # List of feature names
    features = [
        "Popularité genre",
        "Popularité thème",
        "Rareté émotion",
        "Référence",
        "Budget (M$)",
    ]
    from sklearn.linear_model import LinearRegression
    # Create a Linear Regression model
    model = LinearRegression()

    # Create a RegressionModel object
    regressor = RegressionModel(model, features, target, title, feature_to_encode='Genre')

    # Load the data
    regressor.load_data(filename)

    # Preprocess the data
    regressor.preprocess()

    # Train the model and calculate the mean MSE
    metrics = regressor.train_and_evaluate(seed=SEED, test_size=0.1)

    logging.info(f"Metrics: {metrics}")
    print(f"Metrics: {metrics}")

    # Make predictions and save them to a file
    regressor.make_predictions(metrics=metrics)


if __name__ == "__main__":
    main()
