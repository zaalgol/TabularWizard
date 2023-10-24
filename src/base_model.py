from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

class BaseModel:
    def __init__(self, train_df, prediction_column, split_column=None, test_size=0.3):
        self.search = None

        if split_column is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df,
                                                                                        train_df[prediction_column],
                                                                                        test_size=test_size, random_state=42)
        else:
            splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7)
            split = splitter.split(train_df, groups=train_df[split_column])
            train_inds, test_inds = next(split)

            train = train_df.iloc[train_inds]
            self.y_train = train[[prediction_column]].astype(float)
            test = train_df.iloc[test_inds]
            self.y_test = test[[prediction_column]].astype(float)

        self.X_train = self.X_train.drop([prediction_column], axis=1)
        self.X_test = self.X_test.drop([prediction_column], axis=1)