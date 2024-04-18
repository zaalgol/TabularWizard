# https://chat.openai.com/c/60d698a6-9405-4db9-8831-6b2491bb9111
import numpy as np
import pandas as pd
from scipy.stats import mode
from tabularwizard.src.regression.base_regressor_model import BaseRegressorModel
from src.classification.model.logistic_regression import LRegression
from src.regression.evaluate import Evaluate
from src.regression.model.lightgbm_regerssor import LightGBMRegressor
from src.regression.model.mlrpregressor import Mlrpegressor
from src.regression.model.xgboost_regressor import XgboostRegressor
from sklearn.ensemble import VotingRegressor
from itertools import islice

class Ensemble(BaseRegressorModel):
    def __init__(self, train_df, prediction_column, split_column=None, test_size=0.3, scoring='RMSE'):
        self.regressors = {}
        super().__init__(train_df=train_df, prediction_column=prediction_column, scoring=scoring, split_column=split_column, test_size=test_size)
        self.already_splited_data = {'X_train': self.X_train, 'X_test': self.X_test, 'y_train': self.y_train, 'y_test':self.y_test}
        self.evaluate = Evaluate()

    def create_models(self, df):
        self.regressors['lgbm_regressor'] = {'model':LightGBMRegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        self.regressors['mlr_regressor'] = {'model':Mlrpegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        self.regressors['xgb_regressor'] = {'model':XgboostRegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        
    def tune_hyper_parameters(self):
        for regressor_value in self.regressors.values():
            regressor_value['model'].tune_hyper_parameters(scoring=self.scoring)

    def train_all_models(self):
        for regressor_value in self.regressors.values():
            regressor_value['trained_model'] = regressor_value['model'].train()

    def evaluate_all_models(self):
        for regressor_value in self.regressors.values():
            regressor_value['evaluations'] = self.evaluate.evaluate_train_and_test(regressor_value['trained_model'], regressor_value['model'])
        self.regressors= dict(sorted(self.regressors.items(), key=lambda item: item[1]['evaluations']['test_metrics'][self.scoring], reverse=True))

    def create_voting_regressor(self, top_n_best_models=3):
        model_list = [(name, info['model'].estimator) for name, info in islice(self.regressors.items(), top_n_best_models)]
        self.voting_regressor = VotingRegressor(estimators=model_list, voting='soft')

    def train_voting_regressor(self):
        self.trained_voting_regressor = self.voting_regressor.fit(self.X_train, self.y_train)

    def evaluate_voting_regressor(self):
        self.voting_regressor_evaluations = self.evaluate.evaluate_train_and_test(self.trained_voting_regressor, self)

    # def hard_predict(self, trained_models):
    #     predictions = [model.predict(X_test) for model, X_test in trained_models]

    #     # Use mode to find the most common class label
    #     predictions = np.array(predictions)
    #     return mode(predictions, axis=0)[0].flatten()
    
    # def soft_predict(self, trained_models):
    #     if all(hasattr(model, 'predict_proba') for model, _ in trained_models):
    #         # Predict probabilities
    #         probabilities = [model.predict_proba(X_test) for model, X_test  in trained_models]

    #         # Average the probabilities
    #         avg_probabilities = np.mean(probabilities, axis=0)

    #         # Predict the class with the highest average probability
    #         soft_vote_predictions = np.argmax(avg_probabilities, axis=1)
    #     else:
    #         print("Not all models support predict_proba method necessary for soft voting.")



train_path = "tabularwizard/datasets/phone-price-classification/train.csv"
train_data = pd.read_csv(train_path)
ensemble = Ensemble(train_df=train_data, prediction_column='price_range')
ensemble.create_models(train_data)
ensemble.train_all_models()
ensemble.evaluate_all_models()

ensemble.create_voting_regressor()
ensemble.train_voting_regressor()
ensemble.evaluate_voting_regressor()
t=0




        


    