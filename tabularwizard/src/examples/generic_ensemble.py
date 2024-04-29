import pandas as pd
from src.models.classification.implementations.ensemble import Ensemble


train_path = "tabularwizard/datasets/phone-price-classification/train.csv"
train_data = pd.read_csv(train_path)
ensemble = Ensemble(train_df=train_data, target_column='price_range', scoring='accuracy')
ensemble.create_models(train_data)
ensemble.train_all_models()
ensemble.sort_models_by_score()
