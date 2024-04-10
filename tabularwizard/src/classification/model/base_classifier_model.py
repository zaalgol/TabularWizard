from abc import abstractmethod
import os
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from tabularwizard.src.base_model import BaseModel
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE


class BaseClassfierModel(BaseModel):
        def __init__(self, train_df, prediction_column, split_column=None, test_size=None):
            super().__init__(train_df, prediction_column, split_column, test_size)

            self.unique_classes = train_df[prediction_column].nunique()
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        def tune_hyper_parameters(self, params=None, scoring=None, kfold=5, n_iter=500):
            if params is None:
                params = self.default_params
            Kfold = KFold(n_splits=kfold)  
            
            self.search = BayesSearchCV(estimator=self.estimator,
                                        search_spaces=params,
                                        scoring=scoring,
                                        n_iter=n_iter,
                                        n_jobs=1, 
                                        n_points=3,
                                        cv=Kfold,
                                        verbose=0,
                                        random_state=0)
            
        def train(self):
            if self.search: # with hyperparameter tuining
                result = self.search.fit(self.X_train, self.y_train, callback=self.callbacks)
                print("Best Cross-Validation parameters:", self.search.best_params_)
                print("Best Cross-Validation score:", self.search.best_score_)
            else:
                result = self.estimator.fit(self.X_train, self.y_train)
                print("Best accuracy:", self.estimator.best_score_)
            return result
        
        def save_feature_importances(self, model_folder='', filename='feature_importances.png'):
            # Default implementation, to be overridden in derived classes
            feature_importances = self.search.best_estimator_.feature_importances_
            feature_names = self.X_train.columns
            plt.figure(figsize=(12, 6))
            plt.barh(feature_names, feature_importances)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.savefig(os.path.join(model_folder, filename))
            plt.close()

