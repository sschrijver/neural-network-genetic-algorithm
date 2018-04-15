import random
import logging
from xgboost import XGBRegressor


class GeneticModel():
    def __init__(self, param_choices=None):
        self.accuracy = 0.
        self.param_choices = param_choices
        self.genetic_model_settings = {}

    def create_random(self):
        print('--- NEW MODEL RANDOM ---')
        for key in self.param_choices:
            self.genetic_model_settings[key] = random.choice(self.param_choices[key])

        self.model = self.get_model()
        print(self.print_genetic_model())
        print('--- END NEW MODEL RANDOM ---')

    def create_set(self, genetic_model_settings):
        print('--- NEW MODEL DEFINED PARAMS ---')
        self.genetic_model_settings = genetic_model_settings

        self.model = self.get_model()
        print(self.print_genetic_model())
        print('--- END NEW MODEL DEFINED PARAMS ---')

    # First train, then set accuracy
    def fit(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        self.accuracy = self.model.score(X, y)

    def print_genetic_model(self):
        print(self.genetic_model_settings)
        print("GeneticModel accuracy: %.2f%%" % (self.accuracy * 100))

    def get_model(self):
        return XGBRegressor(max_depth=self.genetic_model_settings['max_depth'],
                            min_child_weight=self.genetic_model_settings['min_child_weight'],
                            gamma=self.genetic_model_settings['gamma'],
                            colsample_bytree=self.genetic_model_settings['colsample_bytree'])