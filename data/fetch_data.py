import requests
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from typing import Any, Dict, List


class DataFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = db.create_engine(config["db_connection_string"])
        self.Session = sessionmaker(bind=self.engine)

    def fetch_from_db(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch data from a database using a given SQL query.
        
        Args:
            query (str): SQL query to execute.
        
        Returns:
            List[Dict[str, Any]]: List of records retrieved from the database.
        """
        try:
            session = self.Session()
            result = session.execute(query).fetchall()
            session.close()
            return [dict(row) for row in result]
        except Exception as e:
            print(f"Error fetching data from DB: {e}")
            return []

    def fetch_from_api(self, api_url: str, params: Dict[str, str] = None) -> Any:
        """
        Fetch data from an API endpoint.
        
        Args:
            api_url (str): URL of the API endpoint.
            params (Dict[str, str], optional): Query parameters to include in the request.
        
        Returns:
            Any: JSON response from the API.
        """
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Raises HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return None

# Example of how the DataFetcher class can be used in another script or module
# from data.fetch_data import DataFetcher

# config = {
#     "db_connection_string": "sqlite:///example.db"  # Replace with your actual connection string
# }

# fetcher = DataFetcher(config)

# # Fetch from database
# query = "SELECT * FROM videos WHERE views > 1000"
# db_data = fetcher.fetch_from_db(query)
# print(f"Fetched from DB: {db_data}")

# # Fetch from API
# api_url = "https://api.example.com/videos"
# api_params = {"category": "technology", "limit": "10"}
# api_data = fetcher.fetch_from_api(api_url, params=api_params)
# print(f"Fetched from API: {api_data}")
