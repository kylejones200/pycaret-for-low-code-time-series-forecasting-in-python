"""Auto-split from legacy monolithic script."""

from arch import arch_model
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from pmdarima import auto_arima
from pycaret.time_series import TSForecastingExperiment
from pycaret.time_series import setup, compare_models, tune_model, predict_model, plot_model, pull, save_model, load_model
from pycaret.time_series import stack_models
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
import arviz as az
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

def create_sample_data() -> None:
    data = pd.Series([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name='Sales')

    data.index = pd.date_range(start='2010-01-01', periods=len(data), freq='M')

    df = data.to_frame()

    setup_data = setup(data=df, target='Sales', session_id=123, seasonal_period=12)

    X_train = pull(variable='X_train')

    print('Training data head:')

    print(X_train.head())

    best_model = compare_models()

    tuned_model = tune_model(best_model)

    future_forecast = predict_model(tuned_model, fh=12)

    print('\nForecast:')

    print(future_forecast)

    plot_model(tuned_model, plot='forecast')

    backtest_metrics = pull()

    print('\nBacktest metrics:')

    print(backtest_metrics)

    stacked_model = stack_models(top_n=3)

    save_model(tuned_model, 'time_series_model')

    loaded_model = load_model('time_series_model')

    df['Marketing_Spend'] = [50 + i % 10 for i in range(len(df))]

    setup_multivariate = setup(data=df, target='Sales', session_id=123)

    best_multivariate_model = compare_models()

    future_forecast_multivariate = predict_model(best_multivariate_model, fh=12)

    print('\nMultivariate forecast:')

    print(future_forecast_multivariate)

    plot_model(best_multivariate_model, plot='forecast')

