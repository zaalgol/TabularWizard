from abc import abstractmethod
import os
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from src.base_model import BaseModel
import matplotlib.pyplot as plt


class BaseClassfierModel(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.default_params = {}
            
        @property
        @abstractmethod
        def default_params(self):
            return {}
        
        def tune_hyper_parameters(self, params=None, scoring='r2', kfold=10, n_iter=50):
            if params is None:
                params = self.default_params
            Kfold = KFold(n_splits=kfold)  
            
            self.search = BayesSearchCV(estimator=self.estimator,
                                        search_spaces=params,
                                        scoring=scoring,
                                        n_iter=n_iter,
                                        n_jobs=1, 
                                        cv=Kfold,
                                        verbose=0)
            
        def train(self):
            result = self.search.fit(self.X_train, self.y_train)
            print("Best parameters:", self.search.best_params_)
            print("Best accuracy:", self.search.best_score_)

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

