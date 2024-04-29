# https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data

from datetime import datetime
import os
import pickle
import pandas as pd
from src.models.classification.evaluate import Evaluate
from src.models.classification.implementations.catboot_classifier import CatboostClassifier
from src.models.classification.implementations.knn_classifier import KnnClassifier
from src.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from src.models.classification.implementations.logistic_regression import LRegression
from src.data_preprocessing import DataPreprocessing
from src.plot_data import plot_boxen_correlation_between_x_y, plot_corelation_between_all_columns, plot_point_correlation_between_x_y
import matplotlib.pyplot as plt

from src.visualize import plot_all_correlation, plot_correlation_one_vs_others, plot_correlation_two_columns, show_distrebution_of_categatial_column_valuse, show_distribution_of_numeric_column_values, show_missing



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"phone_price_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')
train_path = "tabularwizard/datasets/phone-price-classification/train.csv"
# test_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/test.csv"

data_preprocessing  = DataPreprocessing()

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

def use_traned_model():
    pass


def train_model():
    train_data = pd.read_csv(train_path)

    # print(train_data.head())

    # dataPreprocessing.describe_datafranme(train_data)
    # show_distribution_of_numeric_column_values(train_data,'ram')
    # plot_correlation_two_columns(train_data,'price_range', 'ram')
    


    # train_data = dataPreprocessing.one_hot_encode_column(train_data, 'color')
    # train_data = train_data.set_index('id')
    # print(train_data.head())
    # show_missing(df = train_data)
    # plot_corelation_between_all_columns(train_data)
    # plt.show()


    results = {}
    
   
    start_time = datetime.now().strftime("%H:%M:%S")

    lRegression = LRegression(train_df = train_data, target_column = 'price_range')
    lRegression.tune_hyper_parameters()
    trained_lRegression = lRegression.train()

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(trained_lRegression, lRegression)
    evaluate.format_train_and_test_evaluation(evaluations)
    results['lRegression'] = evaluations

    knnClassifier = KnnClassifier(train_df = train_data, target_column = 'price_range')
    knnClassifier.tune_hyper_parameters()
    trained_knnClassifier = knnClassifier.train()
    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(trained_knnClassifier, knnClassifier)
    evaluate.format_train_and_test_evaluation(evaluations)
    results['knnClassifier'] = evaluations



    lgbm_classifier = LightgbmClassifier(train_df = train_data, target_column = 'price_range')
    lgbm_classifier.tune_hyper_parameters()
    trained_lgbm = lgbm_classifier.train()
    # pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(trained_lgbm, lgbm_classifier)
    evaluate.format_train_and_test_evaluation(evaluations)
    results['lightbm'] = evaluations
    end_time = datetime.now().strftime("%H:%M:%S")
    print("start time =", start_time)
    print("end time =", end_time)


    # Printing the results after the loop
    # print(f"*" * 100)
    # for idx, res in enumerate(results, 1):
    #     print(f"Iteration {idx}:")
    #     print(f"Best Params: {res['best_params']}")
    #     print(f"CV Score: {res['cv_score']}")
    #     print(f"Test Score: {res['test_score']}")
    #     print("-" * 50)




if __name__ == '__main__':
    train_model()
    use_traned_model()