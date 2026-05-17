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

class BayesianARIMAVisualizer:

    def __init__(self):
        np.random.seed(42)
        self.n = 100
        self.x = np.arange(self.n)
        self.y = 5 + 0.5 * self.x + np.random.normal(0, 2, self.n)
        self.fit_models()

    def fit_models(self):
        with pm.Model() as model:
            phi = pm.Normal('phi', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            init_dist = pm.Normal.dist(0, 10)
            y_obs = pm.GaussianRandomWalk('y_obs', sigma=sigma, init_dist=init_dist, shape=self.n)
            y_like = pm.Normal('y_like', mu=y_obs, sigma=sigma, observed=self.y)
            self.trace = pm.sample(1000, tune=1000, return_inferencedata=True, cores=1)
        self.arima_model = auto_arima(self.y, error_action='ignore', suppress_warnings=True, seasonal=False)
        self.arima_pred = self.arima_model.predict(n_periods=self.n)
        with model:
            posterior_pred = pm.sample_posterior_predictive(self.trace)
            self.posterior_samples = posterior_pred.posterior_predictive['y_like'].values
        self.posterior_mean = np.mean(self.posterior_samples, axis=(0, 1))
        self.credible_intervals = np.percentile(self.posterior_samples.reshape(-1, self.n), [2.5, 97.5], axis=0)
        self.param_traces = {'phi': self.trace.posterior['phi'].values.flatten(), 'sigma': self.trace.posterior['sigma'].values.flatten()}

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        sigma_range = (min(self.param_traces['sigma']), max(self.param_traces['sigma']))
        phi_range = (min(self.param_traces['phi']), max(self.param_traces['phi']))

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 30
            start_idx = frame % (self.n - window)
            end_idx = start_idx + window
            n_samples = int((frame + 1) * len(self.param_traces['sigma']) / 100)
            n_samples = min(n_samples, len(self.param_traces['sigma']))
            ax1.plot(self.x[start_idx:end_idx], self.y[start_idx:end_idx], 'b-', label='Actual', alpha=0.7)
            ax1.plot(self.x[start_idx:end_idx], self.posterior_mean[start_idx:end_idx], 'r-', label='Bayesian AR(1)')
            ax1.plot(self.x[start_idx:end_idx], self.arima_pred[start_idx:end_idx], 'g-', label='ARIMA')
            ax1.fill_between(self.x[start_idx:end_idx], self.credible_intervals[0, start_idx:end_idx], self.credible_intervals[1, start_idx:end_idx], color='r', alpha=0.2)
            ax1.set_title('Time Series Predictions')
            ax1.legend()
            ax1.grid(True)
            sigma_values = self.param_traces['sigma'][:n_samples]
            ax2.hist(sigma_values, bins=30, density=True, alpha=0.6, color='blue')
            ax2.axvline(np.mean(sigma_values), color='r', linestyle='--', label=f'Mean: {np.mean(sigma_values):.3f}')
            ax2.set_title(f'Posterior Distribution: σ (n={n_samples})')
            ax2.set_xlim(sigma_range)
            ax2.legend()
            ax2.grid(True)
            phi_values = self.param_traces['phi'][:n_samples]
            ax3.hist(phi_values, bins=30, density=True, alpha=0.6, color='green')
            ax3.axvline(np.mean(phi_values), color='r', linestyle='--', label=f'Mean: {np.mean(phi_values):.3f}')
            ax3.set_title(f'Posterior Distribution: φ (n={n_samples})')
            ax3.set_xlim(phi_range)
            ax3.legend()
            ax3.grid(True)
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=100, interval=200, repeat=True)
        return anim

