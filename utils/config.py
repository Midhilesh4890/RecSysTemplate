import json
import yaml
from typing import Any


class Config:
    def __init__(self, config_file: str):
        """
        Initialize the Config class.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.config_file = config_file
        self.config = {}

    def load_config(self):
        """
        Load configuration from the specified file.
        Supports JSON and YAML file formats.
        """
        try:
            if self.config_file.endswith('.json'):
                with open(self.config_file, 'r') as file:
                    self.config = json.load(file)
                print(
                    f"Configuration loaded from JSON file: {self.config_file}")

            elif self.config_file.endswith(('.yaml', '.yml')):
                with open(self.config_file, 'r') as file:
                    self.config = yaml.safe_load(file)
                print(
                    f"Configuration loaded from YAML file: {self.config_file}")

            else:
                print(f"Unsupported file format for: {self.config_file}")

        except Exception as e:
            print(f"Error loading configuration: {e}")

    def get_config(self, key: str) -> Any:
        """
        Get a specific configuration value by key.

        Args:
            key (str): The key of the configuration setting.
        
        Returns:
            Any: The value associated with the given key, or None if the key does not exist.
        """
        return self.config.get(key, None)

# Example usage:
# from utils.config import Config

# config = Config("config.yaml")
# config.load_config()
# db_host = config.get_config("database.host")
# print(f"Database Host: {db_host}")
