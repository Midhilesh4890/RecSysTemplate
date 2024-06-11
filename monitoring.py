from prometheus_client import start_http_server, Gauge, Counter


class Monitoring:
    def __init__(self, port: int = 8000):
        """
        Initialize the Monitoring class and start the Prometheus HTTP server.

        Args:
            port (int): Port number for the Prometheus HTTP server.
        """
        self.request_count = Counter(
            'request_count', 'Total number of requests served')
        self.error_count = Counter(
            'error_count', 'Total number of errors occurred')
        self.response_time = Gauge(
            'response_time', 'Response time for requests in seconds')
        self.db_query_time = Gauge(
            'db_query_time', 'Database query time in seconds')

        start_http_server(port)
        print(f"Prometheus HTTP server started on port {port}")

    def log_request(self):
        """
        Increment the request count.
        """
        self.request_count.inc()
        print("Request count incremented.")

    def log_error(self):
        """
        Increment the error count.
        """
        self.error_count.inc()
        print("Error count incremented.")

    def log_response_time(self, time: float):
        """
        Set the response time gauge.

        Args:
            time (float): Response time in seconds.
        """
        self.response_time.set(time)
        print(f"Response time set to {time} seconds.")

    def log_db_query_time(self, time: float):
        """
        Set the database query time gauge.

        Args:
            time (float): Database query time in seconds.
        """
        self.db_query_time.set(time)
        print(f"Database query time set to {time} seconds.")

# Example usage:
# from utils.monitoring import Monitoring

# monitoring = Monitoring(port=8000)

# # Log a request
# monitoring.log_request()

# # Log an error
# monitoring.log_error()

# # Log response time
# monitoring.log_response_time(0.245)

# # Log database query time
# monitoring.log_db_query_time(0.123)
