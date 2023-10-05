# import matplotlib.pyplot as plt
# from xgboost import plot_tree, plot_importance


# def plot_model(model):
#     plot_tree(model)
#     plt.show()

# def plot_feature_importances(model):
#     plot_importance(model)
#     plt.show()
import lightgbm as lgb
import matplotlib.pyplot as plt

def plot_model(model, tree_index=0, figsize=(20, 15), **kwargs):
    # Check if the model is a fitted BayesSearchCV object
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_

    # Check if the model attribute is an LGBMModel instance
    if isinstance(model, lgb.LGBMModel):
        lgb.plot_tree(model.booster_, tree_index=tree_index, figsize=figsize, **kwargs)
    else:
        raise ValueError("Can't plot model of type {}".format(type(model)))
    plt.show()

def plot_feature_importances(model, importance_type='split', figsize=(10, 5), **kwargs):
    # Check if the model is a fitted BayesSearchCV object
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_

    # Check if the model attribute is an LGBMModel instance
    if isinstance(model, lgb.LGBMModel):
        lgb.plot_importance(model.booster_, importance_type=importance_type, figsize=figsize, **kwargs)
    else:
        raise ValueError("Can't plot model of type {}".format(type(model)))
    plt.show()
