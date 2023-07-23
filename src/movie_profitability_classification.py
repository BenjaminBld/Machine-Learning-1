# Import necessary libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Any
from datetime import datetime


class ClassificationModel:
    def __init__(
        self,
        model: RandomForestClassifier,
        numerical_features: List[str],
        categorical_features: List[str],
        target: str,
        title: str,
    ):
        """
        Initialize the classification model.

        :param model: The machine learning model to use for classification.
        :param numerical_features: A list of the names of numerical features.
        :param categorical_features: A list of the names of categorical features.
        :param target: The name of the target variable.
        :param title: The name of the title variable.
        """
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.features = numerical_features + categorical_features
        self.target = target
        self.title = title
        self.df = None
        self.X_test = None
        self.y_test = None
        self.titles_test = None

    def load_data(self, filename: str) -> None:
        """
        Load data from a CSV file.

        :param filename: The name of the file to load data from.
        """
        if os.path.exists(filename):
            self.df = pd.read_csv(filename)
        else:
            raise FileNotFoundError(f"{filename} does not exist.")

    def preprocess(self, method: str) -> None:
        """
        Preprocesses the data by handling missing values and encoding categorical features.

        :param method: The method to use for encoding categorical features. Must be 'frequency' or 'onehot'.
        """
        if method not in ["frequency", "onehot"]:
            raise ValueError("Method must be 'frequency' or 'onehot'.")

        # Fill missing values in categorical features
        self.df[self.categorical_features] = self.df[self.categorical_features].fillna(
            "Unknown"
        )

        # Apply frequency or one-hot encoding to categorical features
        if method == "frequency":
            for feature in self.categorical_features:
                encoding = self.df.groupby(feature).size()
                encoding = encoding / len(self.df)
                self.df[feature] = self.df[feature].map(encoding)
        elif method == "onehot":
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            onehot_features = encoder.fit_transform(self.df[self.categorical_features])
            onehot_feature_names = encoder.get_feature_names_out(
                self.categorical_features
            )
            onehot_features_df = pd.DataFrame(
                onehot_features, columns=onehot_feature_names, index=self.df.index
            )
            self.df.drop(self.categorical_features, axis=1, inplace=True)
            self.df = pd.concat([self.df, onehot_features_df], axis=1)
        non_feature_columns = [
            self.title,
            self.target,
            "Rentabilité (%)",
            "Box Office (M$)",
        ]
        self.features = list(set(self.df.columns) - set(non_feature_columns))

    def train(self, seed: int, test_size: float = 0.2) -> None:
        """
        Train the model on the data.

        :param seed: The random seed to use for splitting the data.
        :param test_size: The proportion of the data to use as the test set.
        """
        X = self.df[self.features]
        y = self.df[self.target]
        titles = self.df[self.title]
        (
            X_train,
            self.X_test,
            y_train,
            self.y_test,
            _,
            self.titles_test,
        ) = train_test_split(X, y, titles, test_size=test_size, random_state=seed)
        self.model.fit(X_train, y_train)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set and compute feature importances.

        :return: A dictionary with precision, recall, F-score, per class metrics, and feature importances.
        """
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Generate classification report and log it
        report = classification_report(
            self.y_test, y_pred, labels=[0, 1, 2], output_dict=True
        )
        logging.info(f"Classification report: {report}")
        print(f"Classification report: {report}")

        # Compute micro-averaged precision, recall, and F-score
        precision = precision_score(self.y_test, y_pred, average="micro")
        recall = recall_score(self.y_test, y_pred, average="micro")
        fscore = f1_score(self.y_test, y_pred, average="micro")

        metrics = {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "per_class": {
                str(i): report[str(i)] for i in [0, 1, 2]
            },  # Per-class metrics
        }

        # Compute feature importances
        feature_importances = self.model.feature_importances_
        top_features = sorted(
            list(zip(self.features, feature_importances)),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        metrics["top_features"] = dict(top_features)

        # Log the feature importances
        logging.info(f"Top 10 feature importances: {metrics['top_features']}")
        print(
            f"Top 10 feature importances: {metrics['top_features']}"
        )  # Print the top 10 feature importances

        return metrics

    def make_predictions(self, metrics: Dict[str, float]) -> None:
        """
        Make predictions on the test set and save them to a file, along with the evaluation metrics.

        :param metrics: The evaluation metrics to save.
        """
        y_pred = self.model.predict(self.X_test)
        predictions = pd.DataFrame(
            {self.title: self.titles_test, "y_pred": y_pred, "y_true": self.y_test}
        )
        file_name = f'../predictions/classification/predictions_{self.model.__class__.__name__}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        os.makedirs(
            os.path.dirname(file_name), exist_ok=True
        )  # Ensure the directory exists
        predictions.to_csv(file_name + ".csv", index=False)
        with open(file_name + "_metrics.txt", "w") as f:
            f.write(f"Metrics: {metrics}")
        logging.info("Predictions saved to file.")


def main():
    logging.basicConfig(filename="classification_model.log", level=logging.INFO)
    SEED = 42
    filename = "../data/movies_data.csv"
    title = "Titre"
    target = "Succès"
    numerical_features = [
        "Popularité genre",
        "Popularité thème",
        "Rareté émotion",
        "Référence",
        "Budget (M$)",
    ]
    categorical_features = [
        "Réalisateur",
        "Scénariste",
        "Compositeur",
        "Directeur photo",
        "Directeur montage",
        "Acteur 1",
        "Acteur 2",
        "Acteur 3",
        "Genre",
    ]
    model = RandomForestClassifier(
        class_weight="balanced",
        verbose=1,
        n_estimators=500,
        random_state=46,
        max_depth=6,
    )
    classifier = ClassificationModel(
        model, numerical_features, categorical_features, target, title
    )
    classifier.load_data(filename)
    classifier.preprocess(method="onehot")
    classifier.train(seed=SEED, test_size=0.1)
    metrics = classifier.evaluate()
    logging.info(f"Metrics: {metrics}")
    print(metrics)
    classifier.make_predictions(metrics)


if __name__ == "__main__":
    main()
