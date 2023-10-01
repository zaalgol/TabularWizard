import pandas as pd

class DataPreprocessing:
    def exclude_columns(self, df, columns_to_exclude):
        """
        Create a copy of a DataFrame with all columns except the specified ones.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - columns_to_exclude (list of str): List of column names to exclude.

        Returns:
        - pd.DataFrame: A copy of the input DataFrame with columns excluded.
        """
        return df.drop(columns=columns_to_exclude).copy()

    # example of mapping_dict: {'high': 3, 'medium': 2, 'low': 1}
    def map_order_column(self, df, column_name, mapping_dict):
        df_copy = df.copy()
        df_copy[column_name] = df_copy[column_name].map(mapping_dict)
        return df_copy

    def one_hot_encode_column(self, dataframe, column_name):
        """
        Perform one-hot encoding on a specific column in a Pandas DataFrame.
        
        Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.
        - column_name (str): The name of the column to be one-hot encoded.

        Returns:
        - pd.DataFrame: A copy of the input DataFrame with one-hot encoding for the specified column.
        """
        # Make a copy of the original DataFrame to avoid modifying it in place
        encoded_df = dataframe.copy()
        
        # Use pd.get_dummies to perform one-hot encoding on the specified column
        one_hot_encoded = pd.get_dummies(encoded_df[column_name], prefix=column_name)
        
        # Concatenate the one-hot encoded columns to the original DataFrame and drop the original column
        encoded_df = pd.concat([encoded_df, one_hot_encoded], axis=1)
        encoded_df.drop(columns=[column_name], inplace=True)
        
        return encoded_df