

import pickle
import pandas as pd
from src.classification.evaluate import Evaluate
from src.classification.model.xgboost_classifier import XgboostClassifier
from src.data_preprocessing import DataPreprocessing

FILE_NAME = 'results\\trained_models\\finalized_model.sav'


def train_model():
    iris_path = 'datasets\IRIS.csv'
    df = pd.read_csv(iris_path)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.map_order_column(df, 'species', {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    print(df)
    xgboost_classifier = XgboostClassifier(train_df = df, prediction_column = 'species')
    xgboost_classifier.tune_hyper_parameters_with_bayesian()
    model = xgboost_classifier.train()
    
    pickle.dump(model, open(FILE_NAME, 'wb'))

def use_traned_model():
    iris_path = 'datasets\IRIS.csv'
    df = pd.read_csv(iris_path)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.map_order_column(df, 'species', {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    X_data = data_preprocessing.exclude_columns(df, 'species')
    loaded_model = pickle.load(open(FILE_NAME, 'rb'))
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    evaluate.evaluate_predictions(df['species'], y_predict)


if __name__ == '__main__':
    use_traned_model()





