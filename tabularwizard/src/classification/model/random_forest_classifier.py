from sklearn.ensemble import RandomForestClassifier
from tabularwizard.src.classification.model.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

DEFAULT_PARAMS = {
    'n_estimators': list(range(50, 300, 50)),
    'max_depth': list(range(3, 10)) + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_features': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
}

class RandomForestClassifierCustom(BaseClassfierModel):
    def __init__(self, train_df, prediction_column, *args, split_column=None, test_size=0.3, **kwargs):
        super().__init__(train_df, prediction_column, split_column=split_column, test_size=test_size)
        self.estimator = RandomForestClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS

    def train(self):
        result = self.search.fit(self.X_train, self.y_train.values.ravel())  # using values.ravel() to get a 1-D array
        print("Best parameters:", self.search.best_params_)
        print("Best accuracy:", self.search.best_score_)

        return result
    
    def save_tree_diagram(self, tree_index=0, model_folder='', filename='random_forest_tree_diagram.png', dpi=300):
        plt.figure(figsize=(20, 10))
        plot_tree(self.estimator.estimators_[tree_index], filled=True, feature_names=self.X_train.columns, rounded=True, class_names=True)
        plt.savefig(os.path.join(model_folder, filename), format='png', dpi=dpi)
        plt.close()
