

from datetime import datetime
import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.classification.evaluate import Evaluate
from src.classification.model.xgboost_classifier import XgboostClassifier
from src.data_preprocessing import DataPreprocessing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"house_prices_xgboos_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'house_prices_finalized_xgboosr_classification_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'house_prices_finalized_xgboosr_classification_mode_eval')

dataset_Path = 'datasets\horse_survival_train.csv'


def train_model():
    
    df = pd.read_csv(dataset_Path)
    df = perprocess_data(df)

    xgboost_classifier = XgboostClassifier(train_df = df, prediction_column = 'outcome')
    xgboost_classifier.tune_hyper_parameters()
    model = xgboost_classifier.train()
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(model, xgboost_classifier)
    
    print(f"model evaluations: {evaluations}")
    with open(SAVED_MODEL_EVALUATION, 'w') as file:
        file.write(evaluations)


def use_traned_model():

    df = pd.read_csv(dataset_Path)
    data_preprocessing = DataPreprocessing()
    df = perprocess_data(df)
    X_data = data_preprocessing.exclude_columns(df, ['outcome'])
    loaded_model = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    print(evaluate.evaluate_classification(df['outcome'], y_predict))

def perprocess_data(df):
    data_preprocessing = DataPreprocessing()
    data_preprocessing.fill_missing_not_numeric_cells(df)
    data_preprocessing.fill_missing_numeric_cells(df)
    df['hospital_number'] = df['hospital_number'].astype('str')
    df['lesion_2'] = df['lesion_2'].apply(lambda x:1 if x>0 else 0)
    df["deviation_from_normal_temp"] = df["rectal_temp"].apply(lambda x: abs(x - 37.8))
    df = data_preprocessing.map_order_column(df, 'outcome', {'died':0, 'euthanized':1, 'lived':2})
    df = data_preprocessing.map_order_column(df, 'temp_of_extremities', {'cool':0, 'normal':1, 'warm':2})
    df = data_preprocessing.one_hot_encode_all_categorical_columns(df)

    return df

if __name__ == '__main__':
    train_model()
    # use_traned_model()





