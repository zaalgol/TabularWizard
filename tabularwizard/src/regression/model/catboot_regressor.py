from skopt.space import Real, Integer

from catboost import CatBoostRegressor
from tabularwizard.src.regression.model.base_regressor_model import BaseRegressorModel


DEFAULT_PARAMS = {
    'iterations': Integer(10, 2000),
    'depth': Integer(1, 12),
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'random_strength': Real(1e-9, 10, 'log-uniform'),  # randomness for scoring splits
    'bagging_temperature': Real(0.0, 1.0),  # settings of the Bayesian bootstrap
    'l2_leaf_reg': Integer(2, 100),  # L2 regularization
}


class CatboostRegressor(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None, test_size=0.3, 
                 create_encoding_rules=False, apply_encoding_rules=False, already_splitted_data=None, verbose=False, *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         already_splitted_data=already_splitted_data)
        self.estimator = CatBoostRegressor(task_type='GPU', devices='0', verbose=verbose, *args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
