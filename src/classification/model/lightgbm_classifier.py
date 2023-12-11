import os
from lightgbm import LGBMClassifier, plot_tree
from src.classification.model.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt

DEFAULT_PARAMS = {
            'class_weight': [None, 'balanced'],
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(30, 150)),
            'learning_rate': [0.01,0.05, 0.1 ,0.5],
            'subsample_for_bin': [20000,50000,100000,120000,150000],
            'min_child_samples': [20,50,100,200,500],
            'colsample_bytree': [0.6,0.8,1],
            "max_depth": (5,100, 'uniform'),
            'lambda_l1': (0.7, 5, 'log-uniform'),
            'lambda_l2': (0.7, 5, 'log-uniform')
        }

class LightgbmClassifier(BaseClassfierModel):
    def __init__(self, train_df, prediction_column, *args, split_column=None, test_size=0.3, **kwargs):
        super().__init__(train_df, prediction_column, split_column=split_column, test_size=test_size)
        self.estimator = LGBMClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    

    def save_tree_diagram(self, tree_index=0, model_folder='', filename='tree_diagram.png'):
        plot_tree(self.search.best_estimator_, tree_index=tree_index, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
        plt.savefig(os.path.join(model_folder, filename))
        plt.close()

