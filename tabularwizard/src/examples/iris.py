import os
from datetime import datetime
import pickle
import pandas as pd
from tabularwizard.src.classification.evaluate import Evaluate
from tabularwizard.src.classification.model.xgboost_classifier import XgboostClassifier
from tabularwizard.src.data_preprocessing import DataPreprocessing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# SAVED_MODEL_PATH = 'results\\trained_models\\iris_finalized_model.sav'
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'iris_regression', f"xgboost{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'iris_finalized_exboost_model.sav')
iris_path = 'datasets\IRIS.csv'


def train_model():
    
    df = pd.read_csv(iris_path)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.map_order_column(df, 'species', {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    print(df)
    xgboost_classifier = XgboostClassifier(train_df = df, prediction_column = 'species')
    xgboost_classifier.tune_hyper_parameters()
    model = xgboost_classifier.train()
    
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

def use_traned_model():

    df = pd.read_csv(iris_path)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.map_order_column(df, 'species', {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    X_data = data_preprocessing.exclude_columns(df, ['species'])
    loaded_model = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    evaluate.get_confution_matrix(df['species'], y_predict)


if __name__ == '__main__':
    train_model()
    use_traned_model()





