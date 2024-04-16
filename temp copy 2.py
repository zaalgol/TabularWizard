import pandas as pd
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
xgb = XGBClassifier(enable_categorical=True)
df = pd.read_csv(train_path)
df['type']=df['type'].astype("category").cat.codes
df['color']=df['color'].astype("category")
X_train, X_test, y_train, y_test = train_test_split(df, df['type'], test_size=0.2, random_state=42)
result = xgb.fit(X_train, y_train)




