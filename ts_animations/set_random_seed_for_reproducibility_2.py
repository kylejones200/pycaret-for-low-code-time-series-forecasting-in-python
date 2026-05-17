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

def set_random_seed_for_reproducibility_2() -> None:
    np.random.seed(42)

    n = 1000

    omega = 0.1

    alpha = 0.8

    errors = np.random.normal(size=n)

    volatility = np.zeros(n)

    returns = np.zeros(n)

    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * errors[t - 1] ** 2)
        returns[t] = volatility[t] * np.random.normal()

    data = pd.DataFrame({'returns': returns, 'volatility': volatility})

    data.plot(subplots=True, figsize=(10, 6), title='Simulated Returns and Volatility')

    plt.savefig('returns_and_volatility.png')

    plt.show()

    arch_model_fit = arch_model(data['returns'], vol='ARCH', p=1).fit()

    print(arch_model_fit.summary())

    forecast = arch_model_fit.forecast(horizon=10)

    forecast_variance = forecast.variance.iloc[-1]

    plt.figure(figsize=(10, 6))

    plt.plot(forecast_variance, marker='o', label='Forecasted Variance')

    plt.title('Forecasted Volatility')

    plt.xlabel('Horizon')

    plt.ylabel('Variance')

    plt.legend()

    plt.grid()

    plt.savefig('forecasted_volatility.png')

    plt.show()

