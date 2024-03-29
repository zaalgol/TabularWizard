

from datetime import datetime
import os
import pickle
import pandas as pd
from tabularwizard.src.classification.evaluate import Evaluate
from tabularwizard.src.classification.model.lightgbm_classifier import LightgbmClassifier
from tabularwizard.src.data_preprocessing import DataPreprocessing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"horse_survival_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)

SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'lightgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'lightgbm_model_eval')

dataset_Path = 'datasets\horse_survival_train.csv'

def train_model():
    
    df = pd.read_csv(dataset_Path)
    df = perprocess_data(df)

    lgbm_classifier = LightgbmClassifier(train_df = df, prediction_column = 'outcome')
    lgbm_classifier.tune_hyper_parameters(scoring='accuracy')
    model = lgbm_classifier.train()
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(model, lgbm_classifier)
    print(f"model evaluations: {evaluations}")
    with open(SAVED_MODEL_EVALUATION, 'w') as file:
        file.write(evaluations)

    lgbm_classifier.save_feature_importances(model_folder=SAVED_MODEL_FOLDER)
    lgbm_classifier.save_tree_diagram(tree_index=0, model_folder=SAVED_MODEL_FOLDER)

def use_traned_model():
    df = pd.read_csv(dataset_Path)
    data_preprocessing = DataPreprocessing()
    df = perprocess_data(df)
    X_data = data_preprocessing.exclude_columns(df, ['outcome'])
    SAVED_MODEL_FILE = "results\\trained_models\classification\horse_survival_lightgbm20240301_174113\lightgbm_model.sav"
    loaded_model = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    filename = "results\\trained_models\classification\horse_survival_lightgbm20240301_174113\gr"
    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    print(evaluate.evaluate_classification(df['outcome'], y_predict))



def perprocess_data(df):
    data_preprocessing = DataPreprocessing()
    data_preprocessing.fill_missing_not_numeric_cells(df)
    data_preprocessing.fill_missing_numeric_cells(df)
    # df['hospital_number'] = df['hospital_number'].astype('str')
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





