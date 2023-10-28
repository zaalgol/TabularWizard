from abc import abstractmethod
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from src.base_model import BaseModel


class BaseRegressorrModel(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.default_params = self.estimator = LGBMClassifier()
            
        @property
        @abstractmethod
        def default_params(self):
            return {}
        
        def tune_hyper_parameters(self, params=None, *args, **kwargs):
            if params is None:
                params = self.default_params
            # Kfold = KFold(n_splits=kfold)  
            
            self.search = BayesSearchCV(estimator=self.estimator,
                                        search_spaces=params,
                                        *args, **kwargs)
            
        def train(self):
            result = self.search.fit(self.X_train, self.y_train)
            print("Best parameters:", self.search.best_params_)
            print("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
            return result

