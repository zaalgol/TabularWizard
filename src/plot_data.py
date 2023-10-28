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
import pandas as pd
import seaborn as sns

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

def plot_corelation_between_column_and_other_columns(df, target_column):
    corr_arr = []
    name_arr = []

    for col in range(len(df.columns)):
        if df.columns[col] == target_column:
            continue

        name_arr.append(df.columns[col])

        corr = df.corr()[target_column][col]
        if corr < 0: corr *= -1
        corr_arr.append(corr)

    pd.Series(corr_arr, name_arr).sort_values().plot(kind="barh")


def plot_corelation_between_all_columns(df):
    plt.figure(figsize=(15,12))
    r = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")

def plot_boxen_correlation_between_x_y(df, x, y):
    r = sns.boxenplot(data=df, y=y, x=x)

def plot_point_correlation_between_x_y(df, x, y):
    r = sns.pointplot(data=df, y=y, x=x)

def plot_box_correlation_between_x_y(df, x, y):
    r = sns.boxplot(data=df, y=y, x=x)