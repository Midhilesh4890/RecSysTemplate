import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict


class ExploratoryDataAnalysis:
    def __init__(self):
        sns.set(style="whitegrid")

    def visualize_data(self, data: pd.DataFrame, features: Dict[str, str]):
        """
        Visualize data using various plots.

        Args:
            data (pd.DataFrame): The data to visualize.
            features (Dict[str, str]): Dictionary with keys as plot types and values as column names to visualize.
        """
        try:
            for plot_type, column in features.items():
                if plot_type == "histogram":
                    self._plot_histogram(data, column)
                elif plot_type == "boxplot":
                    self._plot_boxplot(data, column)
                elif plot_type == "scatter":
                    x_col, y_col = column.split(',')
                    self._plot_scatter(data, x_col.strip(), y_col.strip())
                else:
                    print(f"Unknown plot type: {plot_type}")
        except Exception as e:
            print(f"Error in visualization: {e}")

    def generate_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical summaries for the data.

        Args:
            data (pd.DataFrame): The data to analyze.
        
        Returns:
            pd.DataFrame: Summary statistics of the data.
        """
        try:
            statistics = data.describe()
            print("Statistical Summary:\n", statistics)
            return statistics
        except Exception as e:
            print(f"Error generating statistics: {e}")
            return pd.DataFrame()

    def _plot_histogram(self, data: pd.DataFrame, column: str):
        """
        Helper function to plot histogram.

        Args:
            data (pd.DataFrame): The data to plot.
            column (str): The column to plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def _plot_boxplot(self, data: pd.DataFrame, column: str):
        """
        Helper function to plot boxplot.

        Args:
            data (pd.DataFrame): The data to plot.
            column (str): The column to plot.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.grid(True)
        plt.show()

    def _plot_scatter(self, data: pd.DataFrame, x_col: str, y_col: str):
        """
        Helper function to plot scatter plot.

        Args:
            data (pd.DataFrame): The data to plot.
            x_col (str): The column for x-axis.
            y_col (str): The column for y-axis.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_col, y=y_col)
        plt.title(f'Scatter Plot of {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()
