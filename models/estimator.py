""""Docstring placeholder"""
import random
import logging
from xgboost import XGBRegressor


class Estimator():
    """Represent an estimator and let us operate on it.
    Currently only works for eXtreme Gradient Boosting.
    """

    def __init__(self, param_choices=None):
        """Initialize our estimator.
        Args:
            param_choices (dict): Parameters for the estimator.
        """
        self.accuracy = 0.
        self.param_choices = param_choices
        self.parameters = {}
        self.model = {}

    def create_random(self):
        """Create a random estimator."""
        for key in self.param_choices:
            self.parameters[key] = random.choice(self.param_choices[key])

        self.model = self.get_model()
        logging.info('New random estimator created: %s', self.parameters)

    def create_set(self, estimator_parameters):
        """Set estimator properties.
        Args:
            estimator_parameters (dict): The estimator parameters
        """
        self.parameters = estimator_parameters

        self.model = self.get_model()
        logging.info('New estimator with defined parameters created: %s',
                     self.parameters)

    def fit(self, x_train, y_train):
        """Train the estimator.
        Args:
            x_train: Input labels.
            y_train: Output labels.
        """
        self.model.fit(x_train, y_train)

    def score(self, x_test, y_test):
        """Store the accuracy.
        Args:
            x_test: Input labels.
            y_test: Output labels.
        """
        self.accuracy = self.model.score(x_test, y_test)

    def print_estimator(self):
        """"Print out the estimator."""
        print("--- ESTIMATOR, accuracy: %.2f%%, params: ---" % (self.accuracy * 100))

        for key, value in self.parameters.items():
            print(key, ":", value)

        print("\n")

    def get_model(self):
        """"Return the model."""
        return XGBRegressor(max_depth=self.parameters['max_depth'],
                            min_child_weight=self.parameters['min_child_weight'],
                            gamma=self.parameters['gamma'],
                            colsample_bytree=self.parameters['colsample_bytree'])
