

from datetime import datetime
import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.classification.evaluate import Evaluate
from src.classification.model.lightgbm_classifier import LightgbmClassifier
from src.data_preprocessing import DataPreprocessing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"house_prices_xgboos_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'house_prices_finalized_xgboosr_classification_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'house_prices_finalized_xgboosr_classification_mode_eval')

dataset_Path = 'datasets\horse_survival_train.csv'


def train_model():
    
    df = pd.read_csv(dataset_Path)
    data_preprocessing = DataPreprocessing()
    df = perprocess_data(df)

    lgbm_classifier = LightgbmClassifier(train_df = df, prediction_column = 'outcome', scorring='accuracy')
    lgbm_classifier.tune_hyper_parameters()
    model = lgbm_classifier.train()

    evaluate = Evaluate()
    print("Test eval:")
    y_predict = evaluate.predict(model, lgbm_classifier.X_test)
    evaluate.evaluate_predictions (lgbm_classifier.y_test, y_predict)
    print("Train eval:")
    y_predict = evaluate.predict(model, lgbm_classifier.X_train)
    evaluate.evaluate_predictions (lgbm_classifier.y_train, y_predict)
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

def use_traned_model():

    df = pd.read_csv(dataset_Path)
    data_preprocessing = DataPreprocessing()
    df = perprocess_data(df)
    X_data = data_preprocessing.exclude_columns(df, ['outcome'])
    loaded_model = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    evaluate.evaluate_predictions(df['outcome'], y_predict)

def perprocess_data(df):
    data_preprocessing = DataPreprocessing()
    data_preprocessing.fill_missing_not_numeric_cells(df)
    data_preprocessing.fill_missing_numeric_cells(df)
    df['hospital_number'] = df['hospital_number'].astype('str')
    df['lesion_2'] = df['lesion_2'].apply(lambda x:1 if x>0 else 0)
    df["deviation_from_normal_temp"] = df["rectal_temp"].apply(lambda x: abs(x - 37.8))
    df = data_preprocessing.map_order_column(df, 'outcome', {'died':0, 'euthanized':1, 'lived':2})
    df = data_preprocessing.map_order_column(df, 'temp_of_extremities', {'cool':0, 'normal':1, 'warm':2})
    # df = data_preprocessing.one_hot_encode_all_categorical_columns(df)
    cat_features  =  data_preprocessing.get_all_categorical_columns_names(df)
    for feature in cat_features:
        df[feature] = df[feature].astype('category')

    return df

if __name__ == '__main__':
    train_model()
    # use_traned_model()





