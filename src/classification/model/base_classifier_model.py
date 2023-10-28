from abc import abstractmethod
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from src.base_model import BaseModel


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

