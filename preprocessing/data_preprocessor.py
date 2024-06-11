import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import List, Dict


class DataPreprocessor:
    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        """
        Initialize the DataPreprocessor class.

        Args:
            categorical_features (List[str]): List of categorical feature names.
            numerical_features (List[str]): List of numerical feature names.
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values.

        Args:
            data (pd.DataFrame): The input data to clean.
        
        Returns:
            pd.DataFrame: The cleaned data.
        """
        try:
            # Fill missing numerical values with the median
            for column in self.numerical_features:
                median_value = data[column].median()
                data[column].fillna(median_value, inplace=True)
                print(
                    f"Filled missing values in {column} with median: {median_value}")

            # Fill missing categorical values with the most frequent value
            for column in self.categorical_features:
                mode_value = data[column].mode()[0]
                data[column].fillna(mode_value, inplace=True)
                print(
                    f"Filled missing values in {column} with mode: {mode_value}")

            print("Data cleaned successfully.")
            return data
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return data

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by encoding categorical features and scaling numerical features.

        Args:
            data (pd.DataFrame): The input data to transform.
        
        Returns:
            pd.DataFrame: The transformed data.
        """
        try:
            # Create transformers for categorical and numerical features
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine transformers into a single preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )

            # Fit and transform the data
            transformed_data = preprocessor.fit_transform(data)

            # Get column names after transformation
            categorical_feature_names = preprocessor.named_transformers_[
                'cat']['onehot'].get_feature_names_out(self.categorical_features)
            all_feature_names = list(
                self.numerical_features) + list(categorical_feature_names)

            transformed_df = pd.DataFrame(
                transformed_data, columns=all_feature_names)

            print("Data transformed successfully.")
            return transformed_df
        except Exception as e:
            print(f"Error transforming data: {e}")
            return data

# Example usage:
# from preprocessing.data_preprocessor import DataPreprocessor

# Define categorical and numerical features
# categorical_features = ['category', 'region']
# numerical_features = ['views', 'likes', 'duration']

# Create a sample DataFrame
# data = pd.DataFrame({
#     'category': ['Music', 'Education', None, 'Music', 'Education'],
#     'region': ['US', 'EU', 'ASIA', None, 'EU'],
#     'views': [100, None, 300, 400, None],
#     'likes': [10, 20, None, 40, 50],
#     'duration': [2.5, None, 4.0, 5.0, 6.0]
# })

# Create the DataPreprocessor instance
# preprocessor = DataPreprocessor(categorical_features, numerical_features)

# Clean the data
# cleaned_data = preprocessor.clean_data(data)
# print(cleaned_data)

# Transform the data
# transformed_data = preprocessor.transform_data(cleaned_data)
# print(transformed_data)
