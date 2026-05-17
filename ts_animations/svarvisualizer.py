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

class SVARVisualizer:

    def __init__(self):
        np.random.seed(42)
        self.n = 200
        self.generate_data()
        self.fit_models()

    def generate_data(self):
        eps1 = np.random.normal(0, 1, self.n)
        eps2 = np.random.normal(0, 1, self.n)
        gdp = np.zeros(self.n)
        inf = np.zeros(self.n)
        for t in range(1, self.n):
            gdp[t] = 0.5 * gdp[t - 1] + 0.1 * inf[t - 1] + eps1[t]
            inf[t] = 0.2 * gdp[t - 1] + 0.7 * inf[t - 1] + eps2[t]
        self.data = pd.DataFrame({'GDP': gdp, 'Inflation': inf})

    def fit_models(self):
        self.var_model = VAR(self.data)
        self.var_results = self.var_model.fit(2)
        n_vars = len(self.data.columns)
        n_lags = 2
        coef_matrix = self.var_results.coefs
        self.coef = coef_matrix.reshape(n_lags, n_vars, n_vars).transpose(1, 2, 0)
        self.A = np.array([[1.0, 0.0], [-0.5, 1.0]])
        self.irf_periods = 20
        self.irf = self._calculate_irf()
        self.fevd = self._calculate_fevd()

    def _calculate_irf(self):
        n_vars = len(self.data.columns)
        irf = np.zeros((self.irf_periods, n_vars, n_vars))
        irf[0] = np.linalg.inv(self.A)
        for i in range(1, self.irf_periods):
            lag_effects = np.zeros((n_vars, n_vars))
            for lag in range(min(i, self.coef.shape[2])):
                if i - 1 - lag >= 0:
                    lag_effects += np.dot(self.coef[:, :, lag], irf[i - 1 - lag])
            irf[i] = lag_effects
        return irf

    def _calculate_fevd(self):
        n_vars = len(self.data.columns)
        fevd = np.zeros((self.irf_periods, n_vars, n_vars))
        cum_effects = np.cumsum(self.irf ** 2, axis=0)
        for t in range(self.irf_periods):
            total_var = np.sum(cum_effects[t], axis=1)
            for i in range(n_vars):
                fevd[t, i, :] = cum_effects[t, i, :] / total_var[i]
        return fevd

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 50
            start_idx = frame % (self.n - window)
            end_idx = start_idx + window
            for col in self.data.columns:
                ax1.plot(self.data.index[start_idx:end_idx], self.data[col][start_idx:end_idx], label=col)
            ax1.set_title('Time Series Data')
            ax1.legend()
            ax1.grid(True)
            response_var = frame % 2
            shock_var = frame // 2 % 2
            var_names = ['GDP', 'Inflation']
            irf_data = self.irf[:, response_var, shock_var]
            ax2.plot(range(len(irf_data)), irf_data, 'b-', marker='o')
            ax2.fill_between(range(len(irf_data)), irf_data - 0.2, irf_data + 0.2, alpha=0.3)
            ax2.set_title(f'IRF: Response of {var_names[response_var]} to {var_names[shock_var]} shock')
            ax2.set_xlabel('Periods')
            ax2.grid(True)
            current_fevd = self.fevd[frame % self.irf_periods]
            sns.heatmap(current_fevd, annot=True, fmt='.2f', xticklabels=var_names, yticklabels=var_names, ax=ax3, cmap='coolwarm', vmin=0, vmax=1)
            ax3.set_title(f'FEVD at horizon {frame % self.irf_periods + 1}')
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=80, interval=200, repeat=True)
        return anim

