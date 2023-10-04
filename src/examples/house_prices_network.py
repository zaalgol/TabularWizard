import pickle
import pandas as pd
from src.regression.evaluate import Evaluate
from src.regression.model.lightgbm_regerssor import LightGBMRegressor
from src.data_preprocessing import DataPreprocessing
import matplotlib.pyplot as plt
from datetime import datetime

from src.regression.model.mlrpregressor import automatic_nn


SAVED_MODEL_PATH = 'results\\trained_models\\house_prices_finalized_mlrpregressor_model.sav'
train_data_path = 'datasets\house_prices_train.csv'

def train_model():
    df = pd.read_csv(train_data_path)
    print(df)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.exclude_columns(df, ['Id'])
    model = automatic_nn(df, 'SalePrice')

    # evaluate = Evaluate()

    # y_train_predict = evaluate.predict(model, xgboost_classifier.X_train)
    # print("Train evaluation:")
    # evaluate.evaluate_predictions(xgboost_classifier.y_train, y_train_predict)

    # y_test_predict = evaluate.predict(model, xgboost_classifier.X_test)
    # print("Test evaluation:")
    # evaluate.evaluate_predictions(xgboost_classifier.y_test, y_test_predict)
    
    pickle.dump(model, open(SAVED_MODEL_PATH, 'wb'))

# def use_traned_model():
#     df = pd.read_csv(train_data_path)
#     data_preprocessing = DataPreprocessing()
#     df = perprocess_data(df)
#     X_data = data_preprocessing.exclude_columns(df, ['SalePrice'])
#     loaded_model = pickle.load(open(SAVED_MODEL_PATH, 'rb'))
#     print(f'hyper params are: {loaded_model.best_params_}' )
#     evaluate = Evaluate()
#     y_predict = evaluate.predict(loaded_model, X_data)
#     evaluate.evaluate_predictions(df['SalePrice'], y_predict)


def perprocess_data(df):
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.exclude_columns(df, ['Id'])
    # df = data_preprocessing.map_order_column(df, 'Utilities', {'AllPub': 4, 'NoSewr': 3,'NoSeWa': 2, 'ELO': 1})
    # df = data_preprocessing.map_order_column(df, 'ExterQual', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1})
    # df = data_preprocessing.map_order_column(df, 'ExterCond', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1})
    # df = data_preprocessing.map_order_column(df, 'BsmtQual', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'BsmtCond', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'BsmtExposure', {'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'BsmtFinType1', {'GLQ': 6,'ALQ': 5,'BLQ': 4, 'Rec': 3,'LwQ': 2, 'Unf': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'BsmtFinType2', {'GLQ': 6,'ALQ': 5,'BLQ': 4, 'Rec': 3,'LwQ': 2, 'Unf': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'HeatingQC', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1})
    # df = data_preprocessing.map_order_column(df, 'KitchenQual', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1})
    # df = data_preprocessing.map_order_column(df, 'FireplaceQu', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'GarageQual', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'GarageCond', {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1, 'NA':0})
    # df = data_preprocessing.map_order_column(df, 'PoolQC', {'Ex': 4,'Gd': 3, 'TA': 2,'Fa': 1, 'NA':0})
    # df = data_preprocessing.one_hot_encode_all_categorical_columns(df)
    cat_features  =  data_preprocessing.get_all_categorical_columns_names(df)
    for feature in cat_features:
        df[feature] = df[feature].astype('category')
    return df
    # quantitative = [f for f in X_data.columns if X_data.dtypes[f] != 'object']
    # qualitative = [f for f in df.columns if df.dtypes[f] == 'object']
    # df = data_preprocessing.one_hot_encode_columns(df, qualitative)


if __name__ == '__main__':

    start_time = datetime.now().strftime("%H:%M:%S")
    train_model()
    # use_traned_model()
    end_time = datetime.now().strftime("%H:%M:%S")
    print("start time =", start_time)
    print("end time =", end_time)
    # print(f"total: {end_time - start_time}")
    # result = perprocess_data(df)
    # t =0

    # missing = df.isnull().sum()
    # missing = missing[missing > 0]
    # missing.sort_values(inplace=True)
    # missing.plot(kind='bar')
    # plt.show()

    # use_traned_model()

    # with gpu
    #start time = 21:35:40
    # end time = 21:45:26





