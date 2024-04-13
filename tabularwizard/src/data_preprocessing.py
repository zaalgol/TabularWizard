import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

class DataPreprocessing:
    def exclude_columns(self, df, columns_to_exclude):
        # Use intersection to find the columns that exist in both the dataframe and the columns_to_exclude list
        valid_columns_to_exclude = [col for col in columns_to_exclude if col in df.columns]
        # Drop only the valid columns
        return df.drop(columns=valid_columns_to_exclude).copy()
        
    def exclude_other_columns(self, df, columns):
        columns_to_keep = [col for col in columns if col in df.columns]
        return df[columns_to_keep]

    def sanitize_column_names(self, df):
        df.columns = [col.replace(',', '').replace(':', '').replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '') for col in df.columns]
        return df
    
    def convert_column_categircal_values_to_numerical_values(self, df, column):
        df_copy = df.copy()
        labels, _ = pd.factorize(df_copy[column])
        df_copy[column] = labels
        return df_copy

    # example of mapping_dict: {'high': 3, 'medium': 2, 'low': 1}
    def map_order_column(self, df, column_name, mapping_dict):
        df_copy = df.copy()
        df_copy[column_name] = df_copy[column_name].map(mapping_dict)
        return df_copy
    
    # example of mapping_dict: {'high': 3, 'medium': 2, 'low': 1}
    def reverse_map_order_column(self, df, column_name, mapping_dict):
        # Inverting the mapping_dict
        inverted_dict = {v: k for k, v in mapping_dict.items()}
        df_copy = df.copy()
        df_copy[column_name] = df_copy[column_name].map(inverted_dict)
        return df_copy
    
    def one_hot_encode_columns(self, dataframe, column_name_array):
        df = dataframe.copy()
        for column_name in column_name_array:
            df = self.one_hot_encode_column(df, column_name)
        return df
    
    
    def one_hot_encode_all_categorical_columns(self, df):
        return pd.get_dummies(df)

    def one_hot_encode_column(self, dataframe, column_name):
        # Make a copy of the original DataFrame to avoid modifying it in place
        encoded_df = dataframe.copy()
        
        # Use pd.get_dummies to perform one-hot encoding on the specified column
        one_hot_encoded = pd.get_dummies(encoded_df[column_name], prefix=column_name)
        
        # Concatenate the one-hot encoded columns to the original DataFrame and drop the original column
        encoded_df = pd.concat([encoded_df, one_hot_encoded], axis=1)
        encoded_df.drop(columns=[column_name], inplace=True)
        
        return encoded_df
    
    def get_all_categorical_columns_names(self, df):
        return [f for f in df.columns if df.dtypes[f] == 'object']
    
    def get_numeric_columns(self, df):
        return [f for f in df.columns if df.dtypes[f] != 'object']
    
    def fill_missing_numeric_cells(self, df, median_stratay=True):
        new_df = df.copy()
        if median_stratay:
            new_df.fillna(df.median(numeric_only=True).round(1), inplace=True)
        else:
            new_df.fillna(df.mean(numeric_only=True).round(1), inplace=True)
        return new_df
    
    def fill_missing_not_numeric_cells(self, df):
        new_df = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
    
        # Filling missing values in categorical columns with their respective modes.
        for column in categorical_columns:
            mode_value = new_df[column].mode()[0]  # Getting the mode value of the column
            new_df[column].fillna(mode_value, inplace=True)
    

        return new_df
    
    def get_missing_values_per_coloun(self, df):
        return df.isnull().sum()
    
    def scale_values_netween_0_to_1(self, df, columns):
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=columns)
        return df
    
    def describe_datafranme(self, df):
        print(df.describe().T)
        return (df.describe().T)
    
    def oversampling_minority_class(self, df, column_name, minority_class, majority_class):
        new_df = df.copy()
        minority_rows = new_df[new_df[column_name]==minority_class]
        majority_rows = new_df[new_df[column_name]==majority_class]

        minority_upsampled = resample(minority_rows,
                          replace=True, # sample with replacement
                          n_samples=len(majority_rows), # match number in majority class
                          random_state=27) # reproducible results

        # combine majority and upsampled minority
        return pd.concat([majority_rows, minority_upsampled])
            
    def majority_minority_class(self, df, column_name, minority_class, majority_class):
        new_df = df.copy()
        minority_rows = new_df[new_df[column_name]==minority_class]
        majority_rows = new_df[new_df[column_name]==majority_class]

        majority_underampled = resample(majority_rows,
                          replace=False, # sample with replacement
                          n_samples=len(minority_rows), # match number in majority class
                          random_state=27) # reproducible results

        # combine majority and upsampled minority
        return pd.concat([majority_underampled, minority_rows])
    
    def oversampling_minority_classifier(self, classifier, minority_class, majority_class):
        sampling_df = self.oversampling_minority_class(pd.concat([classifier.X_train, classifier.y_train], axis=1),
                                                         classifier.y_train.name, minority_class, majority_class)
        print(sampling_df.Class.value_counts())
        classifier.y_train = sampling_df.Class
        classifier.X_train = sampling_df.drop('Class', axis=1)
        

    def majority_minority_classifier(self, classifier, minority_class, majority_class):
        sampling_df = self.majority_minority_class(pd.concat([classifier.X_train, classifier.y_train], axis=1),
                                                         classifier.y_train.name, minority_class, majority_class)
        classifier.y_train = sampling_df.Class
        classifier.X_train = sampling_df.drop('Class', axis=1)



