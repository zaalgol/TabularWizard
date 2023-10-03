

import pickle
import pandas as pd
from src.classification.evaluate import Evaluate
from src.classification.model.xgboost_classifier import XgboostClassifier
from src.data_preprocessing import DataPreprocessing

SAVED_MODEL_PATH = 'results\\trained_models\\iris_finalized_model.sav'
iris_path = 'datasets\IRIS.csv'


def train_model():
    
    df = pd.read_csv(iris_path)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.map_order_column(df, 'species', {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    print(df)
    xgboost_classifier = XgboostClassifier(train_df = df, prediction_column = 'species')
    xgboost_classifier.tune_hyper_parameters_with_bayesian()
    model = xgboost_classifier.train()
    
    pickle.dump(model, open(SAVED_MODEL_PATH, 'wb'))

def use_traned_model():

    df = pd.read_csv(iris_path)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.map_order_column(df, 'species', {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    X_data = data_preprocessing.exclude_columns(df, ['species'])
    loaded_model = pickle.load(open(SAVED_MODEL_PATH, 'rb'))
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    evaluate.evaluate_predictions(df['species'], y_predict)


if __name__ == '__main__':
    train_model()
    use_traned_model()





