import os
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
from src.classification.model.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': list(range(50, 300, 50)),
    'max_depth': list(range(3, 10)),
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': list(range(1, 10)),
    'gamma': [i/10.0 for i in range(0, 5)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
    'scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000]
}

class XgboostClassifier(BaseClassfierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = XGBClassifier(use_label_encoder=False, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def save_tree_diagram(self, tree_index=0, model_folder='', filename='tree_diagram.png'):
        plot_tree(self.search.best_estimator_, num_trees=tree_index, rankdir='LR')
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        plt.savefig(os.path.join(model_folder, filename), bbox_inches='tight')
        plt.close()

    # def plot (self, result):
    #     from xgboost import plot_importance
    #     import matplotlib.pyplot as plt
    #     plt.style.use ('fivethirtyeight')
    #     plt.rcParams.update ({'font.size': 16})

    #     fig, ax = plt.subplots (figsize=(12, 6))
    #     plot_importance (result.best_estimator_, max_num_features=8, ax=ax)
    #     plt.show()

    #     # xgb.plot_tree (result.best_estimator_, num_trees=2)
    #     # fig = matplotlib.pyplot.gcf ()
    #     # fig.set_size_inches (150, 100)
    #     # fig.savefig ('tree.png')

    #     # plot_tree (result.best_estimator_)

    # def predict_test (self):
    #     self.test_predictions = self.search.predict (self.X_test)

    # def predict_train (self):
    #     self.train_predictions = self.search.predict (self.X_train)

    # def predict_true (self):
    #     self.true_predictions = self.search.predict (self.X_true)

    # def evaluate_predictions (self):
    #     cm = confusion_matrix (self.y_test, self.test_predictions)
    #     print (classification_report (self.y_test, self.test_predictions))
    #     print (f"confusion_matrix of test is {cm}")

    #     cm = confusion_matrix (self.y_train, self.train_predictions)
    #     print (classification_report (self.y_train, self.train_predictions))
    #     print (f"confusion_matrix of train is {cm}")
