import pickle
import os
import pandas as pd
from src.plot_data import plot_feature_importances, plot_model
from src.regression.evaluate import Evaluate
from src.regression.model.lightgbm_regerssor import LightGBMRegressor
from src.data_preprocessing import DataPreprocessing
import matplotlib.pyplot as plt
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', f" lgbm_regression_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'house_prices_finalized_lightgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'house_prices_finalized_lightgbm_model_eval')
# SAVED_MODEL_PATH = 'results\\trained_models\\house_prices_finalized_lightgbm_model.sav'
# SAVED_MODEL_PATH = os.path.join('results', 'trained_models', 'house_prices_finalized_lightgbm_model.sav')
train_data_path = 'datasets\house_prices_train.csv'

def train_model():
    df = pd.read_csv(train_data_path)
    print(df)
    df = perprocess_data(df)
    lightgbm_classifier = LightGBMRegressor(train_df = df, prediction_column = 'SalePrice')
    lightgbm_classifier.tune_hyper_parameters_with_bayesian()
    model = lightgbm_classifier.train()

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_model(model, lightgbm_classifier.X_train, lightgbm_classifier.y_train,
                             lightgbm_classifier.X_test, lightgbm_classifier.y_test)
    with open(SAVED_MODEL_EVALUATION, 'w') as file:
        file.write(evaluations)
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

def use_traned_model():

    print(os.environ['PATH'])
    os.environ['PATH'] = r'C:\Program Files\Graphviz\bin' + ';' + os.environ['PATH']
    print(os.environ['PATH'])
    df = pd.read_csv(train_data_path)
    data_preprocessing = DataPreprocessing()
    df = perprocess_data(df)
    X_data = data_preprocessing.exclude_columns(df, ['SalePrice'])

    loaded_model = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    print(f'hyper params are: {loaded_model.best_params_}' )

    evaluate = Evaluate()
    y_predict = evaluate.predict(loaded_model, X_data)
    evaluate.evaluate_predictions(df['SalePrice'], y_predict)

    with open(SAVED_MODEL_EVALUATION, 'r') as file:
       print(file.read())

    plot_model(loaded_model)
    plot_feature_importances(loaded_model)
    t=0

    

    


def perprocess_data(df):
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.exclude_columns(df, ['Id'])
    cat_features  =  data_preprocessing.get_all_categorical_columns_names(df)
    for feature in cat_features:
        df[feature] = df[feature].astype('category')
    return df


if __name__ == '__main__':

    start_time = datetime.now().strftime("%H:%M:%S")
    train_model()
    use_traned_model()
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





