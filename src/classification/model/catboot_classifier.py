import os
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from src.classification.model.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [100, 500, 1000]

}

class CatboostClassifier(BaseClassfierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = CatBoostClassifier(task_type = 'GPU', devices='0')

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def save_feature_importances(self, model_folder='', filename='catboost_feature_importances.png'):
        feature_importances = self.search.best_estimator_.get_feature_importance()
        feature_names = self.X_train.columns
        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.savefig(os.path.join(model_folder, filename),)
        plt.close()