import pickle
import pandas as pd
from src.regression.evaluate import Evaluate
from src.regression.model.xgboost_rgressor import XgboostRegressor
from src.data_preprocessing import DataPreprocessing
import matplotlib.pyplot as plt


SAVED_MODEL_PATH = 'results\\trained_models\\house_prices_finalized_model.sav'
data_path = 'datasets\house_prices_train.csv'

def train_model():
    df = pd.read_csv(data_path)
    print(df)
    df = perprocess_data(df)
    xgboost_classifier = XgboostRegressor(train_df = df, prediction_column = 'SalePrice')
    xgboost_classifier.tune_hyper_parameters_with_bayesian()
    model = xgboost_classifier.train()

    evaluate = Evaluate()

    y_train_predict = evaluate.predict(model, xgboost_classifier.X_train)
    print("Train evaluation:")
    evaluate.evaluate_predictions(xgboost_classifier.y_train, y_train_predict)

    y_test_predict = evaluate.predict(model, xgboost_classifier.X_test)
    print("Test evaluation:")
    evaluate.evaluate_predictions(xgboost_classifier.y_test, y_test_predict)
    
    pickle.dump(model, open(SAVED_MODEL_PATH, 'wb'))

def use_traned_model():
    df = pd.read_csv(data_path)
    data_preprocessing = DataPreprocessing()
    df = perprocess_data(df)
    X_data = data_preprocessing.exclude_columns(df, ['SalePrice'])
    loaded_model = pickle.load(open(SAVED_MODEL_PATH, 'rb'))
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    evaluate.evaluate_predictions(df['SalePrice'], y_predict)


def perprocess_data(df):
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.exclude_columns(df, ['Id'])
    df = data_preprocessing.one_hot_encode_all_categorical_columns(df)
    return df
    # quantitative = [f for f in X_data.columns if X_data.dtypes[f] != 'object']
    # qualitative = [f for f in df.columns if df.dtypes[f] == 'object']
    # df = data_preprocessing.one_hot_encode_columns(df, qualitative)


if __name__ == '__main__':
    # data_path = 'datasets\house_prices_train.csv'
    # df = pd.read_csv(data_path)
    train_model()
    use_traned_model()
    # result = perprocess_data(df)
    # t =0

    # missing = df.isnull().sum()
    # missing = missing[missing > 0]
    # missing.sort_values(inplace=True)
    # missing.plot(kind='bar')
    # plt.show()

    # use_traned_model()





