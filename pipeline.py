from data.fetch_data import DataFetcher
from preprocessing.data_preprocessor import DataPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.feature_scaler import FeatureScaler
from models.recommender_model import RecommenderModel
from utils.config import Config
from monitoring import Monitoring


def main():
    # Load configuration
    config = Config("config.yaml")
    config.load_config()

    # Fetch data
    fetcher = DataFetcher(config.get_config("data_source"))
    raw_data = fetcher.fetch_from_db("SELECT * FROM video_data")

    # Preprocess data
    categorical_features = config.get_config("categorical_features")
    numerical_features = config.get_config("numerical_features")
    preprocessor = DataPreprocessor(categorical_features, numerical_features)
    cleaned_data = preprocessor.clean_data(raw_data)
    transformed_data = preprocessor.transform_data(cleaned_data)

    # Extract features
    extractor = FeatureExtractor()
    extracted_data = extractor.extract_metadata(transformed_data)
    extracted_data = extractor.extract_content_features(
        extracted_data, "description")

    # Scale features
    scaler = FeatureScaler()
    scaled_data = scaler.scale_features(extracted_data)

    # Train model
    model = RecommenderModel(config.get_config("model_params"))
    model.build_model()
    model.train_model(scaled_data, config.get_config("target_column"))

    # Evaluate model
    evaluation_metrics = model.evaluate_model(
        scaled_data, config.get_config("target_column"))
    print(f"Evaluation Metrics: {evaluation_metrics}")

    # Start monitoring
    monitoring = Monitoring(port=8000)
    monitoring.log_request()
    monitoring.log_response_time(0.2)  # Example time


if __name__ == "__main__":
    main()
