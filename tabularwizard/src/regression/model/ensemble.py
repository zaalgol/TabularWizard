# https://chat.openai.com/c/60d698a6-9405-4db9-8831-6b2491bb9111
import numpy as np
import pandas as pd
from scipy.stats import mode
from src.data_preprocessing import DataPreprocessing
from tabularwizard.src.regression.model.base_regressor_model import BaseRegressorModel
from src.regression.model.random_forest_regressor import RandomForestRegressorModel
from src.regression.model.svr_regressor import SVRRegressorModel
from src.regression.model.catboot_regressor import CatboostRegressor
from src.regression.evaluate import Evaluate
from src.regression.model.lightgbm_regerssor import LightGBMRegressor
from src.regression.model.mlrpregressor import MLPNetRegressor
from src.regression.model.xgboost_regressor import XgboostRegressor
from sklearn.ensemble import VotingRegressor
from itertools import islice

class Ensemble(BaseRegressorModel):
    def __init__(self, train_df, prediction_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                  create_transformations=False, apply_transformations=False, test_size=0.3, scoring='RMSE'):
        self.regressors = {}
        super().__init__(train_df=train_df, prediction_column=prediction_column, scoring=scoring, split_column=split_column, test_size=test_size,
                    create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                    create_transformations=create_transformations, apply_transformations=apply_transformations)
        self.already_splitted_data = {'X_train': self.X_train, 'X_test': self.X_test, 'y_train': self.y_train, 'y_test':self.y_test}
        self.evaluate = Evaluate()

    def create_models(self, df):
        self.regressors['lgbm_regressor'] = {'model':LightGBMRegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splitted_data=self.already_splitted_data)}
        self.regressors['mlr_regressor'] = {'model':MLPNetRegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splitted_data=self.already_splitted_data)}
        self.regressors['xgb_regressor'] = {'model':XgboostRegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splitted_data=self.already_splitted_data)}
        self.regressors['rf_regressor'] = {'model':RandomForestRegressorModel(train_df = df.copy(), prediction_column = self.prediction_column, already_splitted_data=self.already_splitted_data)}
        self.regressors['svr_regressor'] = {'model':SVRRegressorModel(train_df = df.copy(), prediction_column = self.prediction_column, already_splitted_data=self.already_splitted_data)}
        self.regressors['cat_regressor'] = {'model':CatboostRegressor(train_df = df.copy(), prediction_column = self.prediction_column, already_splitted_data=self.already_splitted_data)}
    def tune_hyper_parameters(self):
        for regressor_value in self.regressors.values():
            regressor_value['model'].tune_hyper_parameters(scoring=self.scoring)

    def train_all_models(self):
        for regressor_value in self.regressors.values():
            regressor_value['trained_model'] = regressor_value['model'].train()

    def evaluate_all_models(self):
        for regressor_value in self.regressors.values():
            regressor_value['evaluations'] = self.evaluate.evaluate_train_and_test(regressor_value['trained_model'], regressor_value['model'])
        self.regressors= dict(sorted(self.regressors.items(), key=lambda item:
            item[1]['evaluations']['test_metrics'][self.scoring], reverse=self.scoring=='R2')) # for R2 metrics, high is better, so reverse sroting.

    def create_voting_regressor(self, top_n_best_models=3):
        model_list = [(name, info['model'].estimator) for name, info in islice(self.regressors.items(), top_n_best_models)]
        self.voting_regressor = VotingRegressor(estimators=model_list)

    def train_voting_regressor(self):
        self.trained_voting_regressor = self.voting_regressor.fit(self.X_train, self.y_train)

    def evaluate_voting_regressor(self):
        self.voting_regressor_evaluations = self.evaluate.evaluate_train_and_test(self.trained_voting_regressor, self)


target_column = 'SalePrice'
train_path = "tabularwizard/datasets/house_prices_train.csv"
# # train_path = "tabularwizard/datasets/phone-price-classification/train.csv"
# train_path = "tabularwizard/datasets/titanic.csv"
# train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
train_data = pd.read_csv(train_path)
train_data_capy = train_data.copy()

data_preprocessing = DataPreprocessing()
train_data = data_preprocessing.sanitize_dataframe(train_data)
train_data = data_preprocessing.fill_missing_numeric_cells(train_data)
train_data = data_preprocessing.exclude_columns(train_data, [target_column])
train_data[target_column] = train_data_capy[target_column]
ensemble = Ensemble(train_df=train_data, prediction_column=target_column,
                        create_encoding_rules=True, apply_encoding_rules=True,
                        create_transformations=True, apply_transformations=True)
ensemble.create_models(train_data)
ensemble.train_all_models()
ensemble.evaluate_all_models()
for name, value in ensemble.regressors.items():
    print("<" * 20 +  f" Name {name}, train: {value['evaluations']['train_metrics'][ensemble.scoring]} test: {value['evaluations']['test_metrics'][ensemble.scoring]}")

ensemble.create_voting_regressor()
ensemble.train_voting_regressor()
ensemble.evaluate_voting_regressor()

for name, value in ensemble.regressors.items():
    print("<" * 20 +  f" Name {name}, train: {value['evaluations']['train_metrics'][ensemble.scoring]} test: {value['evaluations']['test_metrics'][ensemble.scoring]}")
print(ensemble.evaluate.format_train_and_test_evaluation(ensemble.voting_regressor_evaluations))

