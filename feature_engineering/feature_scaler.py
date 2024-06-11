from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()

    def scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Scales features to have mean 0 and variance 1.

        Args:
            features (pd.DataFrame): The input DataFrame containing features to scale.
        
        Returns:
            pd.DataFrame: DataFrame with scaled features.
        """
        try:
            scaled_features = self.scaler.fit_transform(features)
            scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
            print("Features scaled successfully.")
            return scaled_df
        except Exception as e:
            print(f"Error scaling features: {e}")
            return features

    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes features to a range between 0 and 1.

        Args:
            features (pd.DataFrame): The input DataFrame containing features to normalize.
        
        Returns:
            pd.DataFrame: DataFrame with normalized features.
        """
        try:
            normalized_features = self.normalizer.fit_transform(features)
            normalized_df = pd.DataFrame(
                normalized_features, columns=features.columns)
            print("Features normalized successfully.")
            return normalized_df
        except Exception as e:
            print(f"Error normalizing features: {e}")
            return features

# Example usage:
# from feature_engineering.feature_scaler import FeatureScaler

# data = pd.DataFrame({
#     'feature1': [1, 2, 3, 4, 5],
#     'feature2': [10, 20, 30, 40, 50]
# })

# scaler = FeatureScaler()

# scaled_data = scaler.scale_features(data)
# print(scaled_data)

# norma
