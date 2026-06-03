from lib.DataCollector import DataCollector
from pykalman import KalmanFilter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class StochasticModels:
    def __init__(self, data_collector : DataCollector | None = None):
        self.data_collector = data_collector

    def apply_kalman_filter(self,column : str = 'close', df : pd.DataFrame | None = None) -> pd.DataFrame | None:
        """
        Applies the Kalman Filter to the specified column of the data.
 
        By default applies to the 'close' column of the data collected by the DataCollector instance, unless a DataFrame is specified.
        """
        if df is not None:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in the provided DataFrame.")
            kf = KalmanFilter(
                initial_state_mean=df[column].iloc[0],
                n_dim_obs=1,
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01,
                transition_matrices=[1],
                observation_matrices=[1]
            )
            state_means, _ = kf.filter(df[column].values)
            df[f'{column}_kalman'] = state_means.flatten()

            return df

        if self.data_collector is None:
            raise ValueError("No DataFrame provided and no DataCollector instance available.")
        
        try:
            for ticker, df_temp in self.data_collector.data.items():
                kf = KalmanFilter(
                    initial_state_mean=df_temp[column].iloc[0],
                    n_dim_obs=1,
                    initial_state_covariance=1,
                    observation_covariance=1,
                    transition_covariance=0.01,
                    transition_matrices=[1],
                    observation_matrices=[1]
                )
                state_means, _ = kf.filter(df_temp[column].values)
                df_temp[f'{column}_kalman'] = state_means.flatten()
                self.data_collector.data[ticker] = df_temp
        except Exception as e:
            print(f"An error occurred while applying the Kalman Filter on DataCollector object: {e}")
            return None