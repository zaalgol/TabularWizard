from datetime import datetime
import os
import pickle
import pandas as pd
from tabularwizard.src.classification.evaluate import Evaluate
from tabularwizard.src.classification.model.lightgbm_classifier import LightgbmClassifier
from tabularwizard.src.data_preprocessing import DataPreprocessing
from sklearn import preprocessing



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"titanic_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')
dataset_Path = "tabularwizard/datasets/titanic.csv"




def perprocess_data(df):
    data_preprocessing  = DataPreprocessing()
    df = df.set_index('PassengerId')
    df = df.drop('Ticket', axis=1)
    df['Name'] = df['Name'].str.extract('(\w+)\.')
    data_preprocessing.fill_missing_not_numeric_cells(df)
    data_preprocessing.fill_missing_numeric_cells(df)

    # df = transform_features(df)

    df = data_preprocessing.one_hot_encode_all_categorical_columns(df)
    df = data_preprocessing.sanitize_column_names(df)

    return df

def train_model():
    df = pd.read_csv(dataset_Path)
    df = perprocess_data(df)
    
    # results = []
    start_time = datetime.now().strftime("%H:%M:%S")
    lgbm_classifier = LightgbmClassifier(train_df = df, prediction_column = 'Survived')
    lgbm_classifier.tune_hyper_parameters(scoring='accuracy')
    model = lgbm_classifier.train()
    end_time = datetime.now().strftime("%H:%M:%S")
    print("start time =", start_time)
    print("end time =", end_time)
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(model, lgbm_classifier)
    
    print(f"model evaluations: {evaluations}")
    with open(SAVED_MODEL_EVALUATION, 'w') as file:
        file.write(evaluations)

    lgbm_classifier.save_feature_importances(model_folder=SAVED_MODEL_FOLDER)
    lgbm_classifier.save_tree_diagram(tree_index=0, model_folder=SAVED_MODEL_FOLDER)

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

if __name__ == '__main__':
    train_model()


