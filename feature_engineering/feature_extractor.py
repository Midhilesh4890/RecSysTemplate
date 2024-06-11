import pandas as pd
from typing import Dict, Any


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metadata features such as title length, category, and upload date.

        Args:
            data (pd.DataFrame): The input data containing video metadata.
        
        Returns:
            pd.DataFrame: The input data with added metadata features.
        """
        try:
            data['title_length'] = data['title'].apply(len)
            data['upload_day'] = pd.to_datetime(data['upload_date']).dt.day
            data['upload_month'] = pd.to_datetime(data['upload_date']).dt.month
            data['upload_year'] = pd.to_datetime(data['upload_date']).dt.year

            # Example: One-hot encode the category
            data = pd.get_dummies(data, columns=['category'], prefix='cat')

            print("Metadata features extracted successfully.")
            return data
        except KeyError as e:
            print(f"Error extracting metadata features: {e}")
            return data

    def extract_content_features(self, data: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Extract content-based features such as word count and average word length from video descriptions.

        Args:
            data (pd.DataFrame): The input data containing video content.
            text_column (str): The column name containing text data (e.g., descriptions).
        
        Returns:
            pd.DataFrame: The input data with added content-based features.
        """
        try:
            data['word_count'] = data[text_column].apply(
                lambda x: len(str(x).split()))
            data['avg_word_length'] = data[text_column].apply(
                lambda x: sum(len(word) for word in str(x).split()) /
                len(str(x).split()) if len(str(x).split()) > 0 else 0
            )

            print("Content features extracted successfully.")
            return data
        except KeyError as e:
            print(f"Error extracting content features: {e}")
            return data
