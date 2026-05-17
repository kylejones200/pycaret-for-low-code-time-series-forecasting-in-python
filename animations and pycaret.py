from arch import arch_model
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from pmdarima import auto_arima
from pycaret.time_series import TSForecastingExperiment
from pycaret.time_series import setup, compare_models, tune_model, predict_model, plot_model, pull, save_model, load_model
from pycaret.time_series import stack_models
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
import arviz as az
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

class ARCHVisualizer:

    def __init__(self):
        np.random.seed(42)
        self.n = 1000
        self.omega = 0.1
        self.alpha = 0.8
        self.generate_data()
        self.fit_models()

    def generate_data(self):
        errors = np.random.normal(size=self.n)
        self.volatility = np.zeros(self.n)
        self.returns = np.zeros(self.n)
        for t in range(1, self.n):
            self.volatility[t] = np.sqrt(self.omega + self.alpha * errors[t - 1] ** 2)
            self.returns[t] = self.volatility[t] * np.random.normal()
        self.data = pd.DataFrame({'returns': self.returns, 'volatility': self.volatility})

    def fit_models(self):
        self.model = arch_model(self.data['returns'], vol='ARCH', p=1)
        self.result = self.model.fit(disp='off')
        self.forecasts = []
        window = 100
        for i in range(window, self.n):
            train = self.returns[i - window:i]
            model = arch_model(train, vol='ARCH', p=1)
            result = model.fit(disp='off')
            forecast = result.forecast(horizon=10)
            self.forecasts.append(forecast.variance.iloc[-1])
        self.forecasts = np.array(self.forecasts)
        self.forecast_ylim = (0, np.percentile(self.forecasts, 99))

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 100
            start_idx = frame % (self.n - window)
            end_idx = start_idx + window
            ax1.plot(self.data.index[start_idx:end_idx], self.returns[start_idx:end_idx], 'b-', label='Returns', alpha=0.7)
            ax1.plot(self.data.index[start_idx:end_idx], self.volatility[start_idx:end_idx], 'r-', label='True Volatility')
            ax1.fill_between(self.data.index[start_idx:end_idx], -self.volatility[start_idx:end_idx], self.volatility[start_idx:end_idx], color='r', alpha=0.2)
            ax1.set_title('Returns and Volatility')
            ax1.legend()
            ax1.grid(True)
            if start_idx >= window:
                forecast_idx = start_idx - window
                horizons = np.arange(1, 11)
                current_forecast = self.forecasts[forecast_idx]
                ax2.plot(horizons, current_forecast, 'g-', marker='o', label='Forecast')
                ax2.set_ylim(self.forecast_ylim)
                ax2.set_xlim(0.5, 10.5)
                ax2.set_title('10-step Volatility Forecast')
                ax2.set_xlabel('Forecast Horizon')
                ax2.set_ylabel('Predicted Variance')
                ax2.legend()
                ax2.grid(True)
                ax2.axhline(y=self.volatility[start_idx] ** 2, color='r', linestyle='--', label='Current Volatility')
            residuals = self.result.resid[start_idx:end_idx]
            ax3.hist(residuals, bins=30, density=True, alpha=0.7, range=(-4, 4))
            ax3.set_xlim(-4, 4)
            ax3.set_title('Standardized Residuals Distribution')
            ax3.grid(True)
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=200, interval=100, repeat=True)
        return anim

class AnimatedCrossValidation:

    def __init__(self, data_length=200):
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=data_length)
        self.data = np.cumsum(np.random.randn(data_length)) + 20

    def create_expanding_window_animation(self, n_splits=3):
        fig, ax = plt.subplots(figsize=(12, 6))
        total_points = len(self.data)
        split_size = total_points // (n_splits + 1)

        def animate(i):
            ax.clear()
            ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
            train_end = split_size * (i // 15 + 1)
            test_start = train_end
            test_end = train_end + split_size
            if test_end <= total_points:
                ax.plot(self.dates[:train_end], self.data[:train_end], 'b-', label='Training')
                ax.plot(self.dates[test_start:test_end], self.data[test_start:test_end], 'r--', label='Test')
                ax.axvspan(self.dates[0], self.dates[train_end - 1], alpha=0.1, color='blue')
                ax.axvspan(self.dates[test_start], self.dates[test_end - 1], alpha=0.2, color='red')
            ax.set_title('Expanding Window Cross-Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=15 * n_splits, interval=500, repeat=True)
        return anim

    def create_rolling_window_animation(self, window_size=40, step_size=20):
        fig, ax = plt.subplots(figsize=(12, 6))
        total_points = len(self.data)
        n_splits = (total_points - window_size) // step_size

        def animate(i):
            ax.clear()
            ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
            start = i // 15 * step_size
            train_end = start + window_size
            if train_end <= total_points:
                ax.plot(self.dates[start:train_end], self.data[start:train_end], 'b-', label='Training Window')
                ax.axvspan(self.dates[start], self.dates[train_end - 1], alpha=0.2, color='blue')
            ax.set_title('Rolling Window Cross-Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=15 * n_splits, interval=500, repeat=True)
        return anim

    def create_sliding_window_animation(self, window_size=40, horizon=10):
        fig, ax = plt.subplots(figsize=(12, 6))
        total_points = len(self.data)
        n_splits = (total_points - window_size - horizon) // horizon

        def animate(i):
            ax.clear()
            ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
            start = i // 15 * horizon
            train_end = start + window_size
            test_end = train_end + horizon
            if test_end <= total_points:
                ax.plot(self.dates[start:train_end], self.data[start:train_end], 'b-', label='Training')
                ax.plot(self.dates[train_end:test_end], self.data[train_end:test_end], 'r--', label='Forecast')
                ax.axvspan(self.dates[start], self.dates[train_end - 1], alpha=0.1, color='blue')
                ax.axvspan(self.dates[train_end], self.dates[test_end - 1], alpha=0.2, color='red')
            ax.set_title('Sliding Window Cross-Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=15 * n_splits, interval=500, repeat=True)
        return anim

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

class CausalInferenceVisualizer:

    def __init__(self):
        np.random.seed(42)
        self.t = np.linspace(0, 10, 500)
        self.x = np.sin(self.t) + np.random.normal(0, 0.1, len(self.t))
        self.y = np.sin(self.t + 0.5) + np.random.normal(0, 0.1, len(self.t))

    def create_animation(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Causal Analysis in Time Series', fontsize=16)

        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            window = 100
            start_idx = frame % (len(self.t) - window)
            end_idx = start_idx + window
            ax1.plot(self.t[start_idx:end_idx], self.x[start_idx:end_idx], label='Series X', color='blue')
            ax1.plot(self.t[start_idx:end_idx], self.y[start_idx:end_idx], label='Series Y', color='red')
            ax1.set_title('Time Series Data')
            ax1.legend()
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            lags = 20
            if start_idx > lags:
                x_lagged = self.x[start_idx - lags:end_idx - lags]
                y_current = self.y[start_idx:end_idx]
                ax2.scatter(x_lagged, y_current, alpha=0.5, color='purple')
                ax2.set_title('Granger Causality Analysis')
                ax2.set_xlabel('X (lagged)')
                ax2.set_ylabel('Y (current)')
            treatment_period = len(self.t) // 2
            ax3.plot(self.t[:treatment_period], self.y[:treatment_period], color='blue', label='Pre-treatment')
            ax3.plot(self.t[treatment_period:], self.y[treatment_period:], color='red', label='Post-treatment')
            ax3.axvline(x=self.t[treatment_period], color='black', linestyle='--', label='Treatment')
            ax3.set_title('Treatment Effect Analysis')
            ax3.legend()
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Outcome')
            impulse = np.zeros(50)
            impulse[0] = 1
            response = np.exp(-np.arange(50) * 0.1) * np.sin(np.arange(50) * 0.5)
            ax4.plot(range(50), response, color='green')
            ax4.set_title('Impulse Response Function')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Response')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        anim = FuncAnimation(fig, animate, frames=len(self.t) - 100, interval=50, repeat=True)
        return anim

class ConfidenceIntervalVisualizer:

    def __init__(self):
        self.population_mean = 100
        self.population_std = 15
        self.max_samples = 200
        np.random.seed(42)
        self.population = np.random.normal(self.population_mean, self.population_std, 10000)
        self.samples = []
        self.means = []
        self.intervals = []

    def create_animation(self):
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        ax_count = fig.add_subplot(gs[0, 0])
        ax_mean = fig.add_subplot(gs[0, 1])
        ax_main = fig.add_subplot(gs[1, :])

        def animate(frame):
            for ax in [ax_count, ax_mean, ax_main]:
                ax.clear()
            new_sample = np.random.choice(self.population, size=30)
            self.samples.append(new_sample)
            current_mean = np.mean(new_sample)
            self.means.append(current_mean)
            ci = stats.t.interval(alpha=0.95, df=len(new_sample) - 1, loc=current_mean, scale=stats.sem(new_sample))
            self.intervals.append(ci)
            ax_count.text(0.5, 0.5, f'Samples: {len(self.samples)}', ha='center', va='center', fontsize=20)
            ax_count.axis('off')
            ax_mean.text(0.5, 0.5, f'Current Mean:\n{current_mean:.2f}', ha='center', va='center', fontsize=20)
            ax_mean.axis('off')
            for i, (mean, interval) in enumerate(zip(self.means, self.intervals)):
                ax_main.plot([i, i], interval, 'b-', alpha=0.1)
            ax_main.plot(self.means, 'bo-', label='Sample Means', alpha=0.5)
            ax_main.axhline(y=self.population_mean, color='r', linestyle='--', label='True Mean')
            coverage = sum([ci[0] <= self.population_mean <= ci[1] for ci in self.intervals]) / len(self.intervals)
            stats_text = f'Coverage: {coverage:.1%}\nMean Width: {np.mean([i[1] - i[0] for i in self.intervals]):.2f}'
            ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax_main.set_title('Confidence Intervals Over Time')
            ax_main.set_xlabel('Sample Number')
            ax_main.set_ylabel('Value')
            ax_main.legend()
            y_min = min([ci[0] for ci in self.intervals])
            y_max = max([ci[1] for ci in self.intervals])
            margin = (y_max - y_min) * 0.1
            ax_main.set_ylim(y_min - margin, y_max + margin)
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=self.max_samples, interval=100, repeat=False)
        return anim

class CrossValidationVisualizer:

    def __init__(self, data_length=200):
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=data_length)
        self.data = np.cumsum(np.random.randn(data_length)) + 20

    def plot_expanding_window(self, n_splits=3):
        """Visualize expanding window cross-validation"""
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
        total_points = len(self.data)
        split_size = total_points // (n_splits + 1)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_splits))
        for i, color in enumerate(colors):
            train_end = split_size * (i + 1)
            test_start = train_end
            test_end = train_end + split_size
            if test_end > total_points:
                break
            ax.plot(self.dates[:train_end], self.data[:train_end], color=color, label=f'Train {i + 1}')
            ax.plot(self.dates[test_start:test_end], self.data[test_start:test_end], '--', color=color, label=f'Test {i + 1}')
            ax.axvspan(self.dates[0], self.dates[train_end - 1], alpha=0.1, color=color)
            ax.axvspan(self.dates[test_start], self.dates[test_end - 1], alpha=0.2, color=color)
        ax.set_title('Expanding Window Cross-Validation')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    def plot_rolling_window(self, window_size=40, step_size=20):
        """Visualize rolling window cross-validation"""
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
        total_points = len(self.data)
        n_splits = (total_points - window_size) // step_size
        colors = plt.cm.rainbow(np.linspace(0, 1, n_splits))
        for i, color in enumerate(colors):
            start = i * step_size
            train_end = start + window_size
            if train_end >= total_points:
                break
            ax.plot(self.dates[start:train_end], self.data[start:train_end], color=color, label=f'Train {i + 1}')
            ax.axvspan(self.dates[start], self.dates[train_end - 1], alpha=0.1, color=color)
        ax.set_title('Rolling Window Cross-Validation')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    def plot_sliding_window(self, window_size=40, horizon=10):
        """Visualize sliding window cross-validation"""
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
        total_points = len(self.data)
        n_splits = (total_points - window_size - horizon) // horizon
        colors = plt.cm.rainbow(np.linspace(0, 1, max(1, n_splits)))
        for i, color in enumerate(colors):
            start = i * horizon
            train_end = start + window_size
            test_end = train_end + horizon
            if test_end >= total_points:
                break
            ax.plot(self.dates[start:train_end], self.data[start:train_end], color=color, label=f'Train {i + 1}')
            ax.plot(self.dates[train_end:test_end], self.data[train_end:test_end], '--', color=color, label=f'Forecast {i + 1}')
            ax.axvspan(self.dates[start], self.dates[train_end - 1], alpha=0.1, color=color)
            ax.axvspan(self.dates[train_end], self.dates[test_end - 1], alpha=0.2, color=color)
        ax.set_title('Sliding Window Cross-Validation')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

class KalmanFilterVisualizer:

    def __init__(self):
        self.n_steps = 100
        self.t = np.linspace(0, 4 * np.pi, self.n_steps)
        self.true_position = 10 * np.sin(self.t)
        self.true_velocity = 10 * np.cos(self.t)
        self.measurements = self.true_position + np.random.normal(0, 1, self.n_steps)
        self.run_kalman_filter()

    def run_kalman_filter(self):
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])
        x = np.array([0.0, 0.0]).reshape(2, 1)
        P = np.eye(2)
        self.estimated_states = []
        self.estimation_covs = []
        for measurement in self.measurements:
            x = F @ x
            P = F @ P @ F.T + Q
            y = measurement - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K * y
            P = (np.eye(2) - K @ H) @ P
            self.estimated_states.append(x.flatten())
            self.estimation_covs.append(P.copy())
        self.estimated_states = np.array(self.estimated_states)
        self.estimation_covs = np.array(self.estimation_covs)

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 30
            start_idx = max(0, frame - window)
            end_idx = frame + 1
            ax1.plot(self.t[start_idx:end_idx], self.true_position[start_idx:end_idx], 'g-', label='True Position')
            ax1.plot(self.t[start_idx:end_idx], self.measurements[start_idx:end_idx], 'r.', label='Measurements')
            ax1.plot(self.t[start_idx:end_idx], self.estimated_states[start_idx:end_idx, 0], 'b-', label='Kalman Filter')
            ax1.set_title('Position Tracking')
            ax1.legend()
            ax2.plot(self.true_position[start_idx:end_idx], self.true_velocity[start_idx:end_idx], 'g-', label='True State')
            ax2.plot(self.estimated_states[start_idx:end_idx, 0], self.estimated_states[start_idx:end_idx, 1], 'b-', label='Estimated State')
            ax2.set_title('Phase Space')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Velocity')
            ax2.legend()
            if frame > 0:
                mean = self.estimated_states[frame]
                cov = self.estimation_covs[frame]
                theta = np.linspace(0, 2 * np.pi, 100)
                epsilon = 1e-06
                eigenvals, eigenvecs = np.linalg.eigh(cov + epsilon * np.eye(2))
                sqrt_eigenvals = np.sqrt(eigenvals)
                ellipse_x = sqrt_eigenvals[0] * np.cos(theta) * eigenvecs[0, 0] + sqrt_eigenvals[1] * np.sin(theta) * eigenvecs[0, 1]
                ellipse_y = sqrt_eigenvals[0] * np.cos(theta) * eigenvecs[1, 0] + sqrt_eigenvals[1] * np.sin(theta) * eigenvecs[1, 1]
                ax3.plot(mean[0] + ellipse_x, mean[1] + ellipse_y, 'r-', label='Uncertainty (1σ)')
                ax3.plot(mean[0], mean[1], 'bo', label='Current Estimate')
                ax3.set_title('State Uncertainty')
                ax3.set_xlabel('Position')
                ax3.set_ylabel('Velocity')
                ax3.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=len(self.t), interval=100, repeat=True)
        return anim

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

class TSClassificationVisualizer:

    def __init__(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_timesteps = 100
        t = np.linspace(0, 4 * np.pi, self.n_timesteps)
        self.class1 = np.array([np.sin(t) + np.random.normal(0, 0.2, self.n_timesteps) for _ in range(self.n_samples)])
        self.class2 = np.array([np.where(np.sin(t) > 0, 1, -1) + np.random.normal(0, 0.2, self.n_timesteps) for _ in range(self.n_samples)])
        self.prepare_data()

    def prepare_data(self):
        self.features1 = self.extract_features(self.class1)
        self.features2 = self.extract_features(self.class2)
        self.dtw_matrix = self.compute_dtw_matrix(self.class1[0], self.class2[0])
        self.cnn_features = self.simulate_cnn_features()

    def extract_features(self, data):
        return np.array([[np.mean(series), np.std(series), np.percentile(series, 75) - np.percentile(series, 25)] for series in data])

    def compute_dtw_matrix(self, s1, s2):
        n, m = (len(s1), len(s2))
        dtw_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                cost = (s1[i] - s2[j]) ** 2
                if i > 0 and j > 0:
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                elif i > 0:
                    dtw_matrix[i, j] = cost + dtw_matrix[i - 1, j]
                elif j > 0:
                    dtw_matrix[i, j] = cost + dtw_matrix[i, j - 1]
                else:
                    dtw_matrix[i, j] = cost
        return dtw_matrix

    def simulate_cnn_features(self):
        features = []
        for i in range(self.n_timesteps - 10):
            feature = np.mean(self.class1[0][i:i + 10])
            features.append(feature)
        return np.array(features)

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
            start_idx = frame % (self.n_timesteps - window)
            end_idx = start_idx + window
            ax1.plot(self.class1[0][start_idx:end_idx], label='Class 1', color='blue')
            ax1.plot(self.class2[0][start_idx:end_idx], label='Class 2', color='red')
            if frame % 3 == 0:
                current_window = self.class1[0][start_idx:end_idx]
                ax1.axhline(y=np.mean(current_window), color='green', linestyle='--', label='Mean')
                ax1.axhspan(np.percentile(current_window, 25), np.percentile(current_window, 75), color='yellow', alpha=0.3, label='IQR')
            ax1.set_title('Time Series Classification')
            ax1.legend()
            im = ax2.imshow(self.dtw_matrix[:end_idx, :end_idx], aspect='auto', cmap='viridis')
            ax2.set_title('DTW Alignment Matrix')
            plt.colorbar(im, ax=ax2)
            ax3.plot(self.cnn_features[:end_idx], label='CNN Features', color='purple')
            ax3.axvline(x=frame, color='red', linestyle='--')
            ax3.set_title('Deep Learning Features')
            ax3.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=self.n_timesteps, interval=100, repeat=True)
        return anim

class TimeSeriesFeatureVisualizer:

    def __init__(self):
        np.random.seed(42)
        n = 500
        self.dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        self.values = 100 + np.cumsum(np.random.normal(0, 1, n))
        self.df = pd.DataFrame({'date': self.dates, 'value': self.values})
        self.calculate_features()

    def calculate_features(self):
        self.df['rolling_mean_7'] = self.df['value'].rolling(window=7).mean()
        self.df['rolling_std_7'] = self.df['value'].rolling(window=7).std()
        self.df['lag_1'] = self.df['value'].shift(1)
        self.df['lag_7'] = self.df['value'].shift(7)
        t = np.arange(len(self.df))
        period = 30
        self.df['fourier_sin'] = np.sin(2 * np.pi * t / period)
        self.df['fourier_cos'] = np.cos(2 * np.pi * t / period)

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        self.window = 100

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            start_idx = max(0, frame - self.window)
            end_idx = frame + 1
            ax1.plot(self.df['date'][start_idx:end_idx], self.df['value'][start_idx:end_idx], label='Original', color='blue')
            ax1.plot(self.df['date'][start_idx:end_idx], self.df['rolling_mean_7'][start_idx:end_idx], label='7-day MA', color='red')
            ax1.fill_between(self.df['date'][start_idx:end_idx], self.df['rolling_mean_7'][start_idx:end_idx] - self.df['rolling_std_7'][start_idx:end_idx], self.df['rolling_mean_7'][start_idx:end_idx] + self.df['rolling_std_7'][start_idx:end_idx], alpha=0.2, color='red')
            ax1.set_title('Time Series with Rolling Statistics')
            ax1.legend()
            if frame > 7:
                ax2.scatter(self.df['lag_1'][start_idx:end_idx], self.df['value'][start_idx:end_idx], alpha=0.5, label='Lag 1')
                ax2.scatter(self.df['lag_7'][start_idx:end_idx], self.df['value'][start_idx:end_idx], alpha=0.5, label='Lag 7')
                ax2.set_title('Lagged Features')
                ax2.set_xlabel('Lagged Value')
                ax2.set_ylabel('Current Value')
                ax2.legend()
            ax3.plot(self.df['date'][start_idx:end_idx], self.df['fourier_sin'][start_idx:end_idx], label='Sine', color='green')
            ax3.plot(self.df['date'][start_idx:end_idx], self.df['fourier_cos'][start_idx:end_idx], label='Cosine', color='orange')
            ax3.set_title('Fourier Components')
            ax3.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=len(self.df) - self.window, interval=200, repeat=True)
        return anim

def animate(i):
    data = df.iloc[:i + 1]
    line.set_data(data['Time'], data['Value'])
    out_of_control = (data['Value'] > ucl) | (data['Value'] < lcl)
    ax.scatter(data['Time'][out_of_control], data['Value'][out_of_control], color='red')
    for j in range(i):
        if out_of_control.iloc[j]:
            ax.axvspan(data['Time'].iloc[j], data['Time'].iloc[j + 1], facecolor='red', alpha=0.1)
    return (line, ucl_line, lcl_line, mean_line)

def plot_cv_scheme(cv_type='expanding', n_splits=4):
    """Create schematic representation of CV schemes"""
    fig, ax = plt.subplots(figsize=(12, 4))
    total_length = 100
    height = 1
    y_positions = np.arange(n_splits) * height * 1.5
    if cv_type == 'expanding':
        initial_size = total_length // (n_splits + 1)
        for i in range(n_splits):
            train_end = initial_size + i * initial_size
            test_end = train_end + initial_size
            ax.add_patch(Rectangle((0, y_positions[i]), train_end, height, facecolor='blue', alpha=0.3))
            ax.add_patch(Rectangle((train_end, y_positions[i]), initial_size, height, facecolor='red', alpha=0.3))
    elif cv_type == 'rolling':
        window_size = total_length // 3
        step_size = window_size // 2
        for i in range(n_splits):
            start = i * step_size
            train_end = start + window_size
            test_end = train_end + step_size
            ax.add_patch(Rectangle((start, y_positions[i]), window_size, height, facecolor='blue', alpha=0.3))
            ax.add_patch(Rectangle((train_end, y_positions[i]), step_size, height, facecolor='red', alpha=0.3))
    elif cv_type == 'blocked':
        block_size = total_length // n_splits
        for i in range(n_splits):
            test_block = i
            for j in range(n_splits):
                if j != test_block:
                    ax.add_patch(Rectangle((j * block_size, y_positions[i]), block_size, height, facecolor='blue', alpha=0.3))
            ax.add_patch(Rectangle((test_block * block_size, y_positions[i]), block_size, height, facecolor='red', alpha=0.3))
    ax.set_xlim(-5, total_length + 5)
    ax.set_ylim(-height, max(y_positions) + height * 2)
    ax.set_title(f'{cv_type.capitalize()} Window Cross-Validation Scheme')
    ax.set_xlabel('Time')
    ax.set_yticks(y_positions + height / 2)
    ax.set_yticklabels([f'Split {i + 1}' for i in range(n_splits)])
    ax.add_patch(Rectangle((0, max(y_positions) + height), 20, height, facecolor='blue', alpha=0.3, label='Training'))
    ax.add_patch(Rectangle((25, max(y_positions) + height), 20, height, facecolor='red', alpha=0.3, label='Testing'))
    plt.legend()
    plt.tight_layout()
    return fig


def notebook_step_001() -> None:
    'Generated from Jupyter notebook: animations and pycaret\n\nMagics and shell lines are commented out. Run with a normal Python interpreter.'


def assuming_you_have_your_taxi_data_in_a_dataframe() -> None:
    dates = pd.date_range(start='2024-01-15', end='2024-01-29', freq='D')

    np.random.seed(42)

    observed = np.random.normal(500, 200, len(dates)) + np.sin(np.arange(len(dates)) / 12) * 300

    predicted = observed + np.random.normal(0, 50, len(dates))

    confidence = np.random.normal(50, 10, len(dates))

    df = pd.DataFrame({'date': dates, 'observed': observed, 'predicted': predicted, 'confidence': confidence})

    plt.figure(figsize=(12, 6))

    plt.plot(df['date'], df['observed'], color='red', label='Observed', linewidth=1)

    plt.plot(df['date'], df['predicted'], color='blue', label='Predicted', linewidth=1)

    plt.fill_between(df['date'], df['predicted'] - df['confidence'], df['predicted'] + df['confidence'], color='grey', alpha=0.2)

    plt.title('Taxicab Pickup Count in Times Square by Time')

    plt.xlabel('Date')

    plt.ylabel('Pickup Count')

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.show()


def visualizing_time_series_cross_validation_methods() -> None:
    cv_viz = CrossValidationVisualizer(data_length=200)

    expanding_window_fig = cv_viz.plot_expanding_window(n_splits=4)

    rolling_window_fig = cv_viz.plot_rolling_window(window_size=50, step_size=20)

    sliding_window_fig = cv_viz.plot_sliding_window(window_size=50, horizon=20)

    blocked_cv_fig = cv_viz.plot_blocked_cv(n_blocks=5)

    expanding_window_fig.savefig('expanding_window_cv.png', bbox_inches='tight', dpi=300)

    rolling_window_fig.savefig('rolling_window_cv.png', bbox_inches='tight', dpi=300)

    sliding_window_fig.savefig('sliding_window_cv.png', bbox_inches='tight', dpi=300)

    blocked_cv_fig.savefig('blocked_cv.png', bbox_inches='tight', dpi=300)

    plt.show()

    schemes = ['expanding', 'rolling', 'blocked']

    for scheme in schemes:
        fig = plot_cv_scheme(scheme)
        fig.savefig(f'{scheme}_scheme.png', bbox_inches='tight', dpi=300)
        plt.close()


def generate_sample_time_series_data() -> None:
    cv_viz = CrossValidationVisualizer(data_length=200)

    expanding_window_fig = cv_viz.plot_expanding_window(n_splits=3)

    rolling_window_fig = cv_viz.plot_rolling_window(window_size=40, step_size=30)

    sliding_window_fig = cv_viz.plot_sliding_window(window_size=40, horizon=15)

    blocked_cv_fig = cv_viz.plot_blocked_cv(n_blocks=4)

    expanding_window_fig.savefig('expanding_window_cv.png', bbox_inches='tight', dpi=300)

    rolling_window_fig.savefig('rolling_window_cv.png', bbox_inches='tight', dpi=300)

    sliding_window_fig.savefig('sliding_window_cv.png', bbox_inches='tight', dpi=300)

    blocked_cv_fig.savefig('blocked_cv.png', bbox_inches='tight', dpi=300)

    plt.show()


def plot_full_dataset() -> None:
    cv_viz = CrossValidationVisualizer(data_length=200)

    fig1 = cv_viz.plot_expanding_window(n_splits=3)

    fig2 = cv_viz.plot_rolling_window(window_size=40, step_size=20)

    fig3 = cv_viz.plot_sliding_window(window_size=40, horizon=10)

    fig1.savefig('expanding_window.png', bbox_inches='tight', dpi=300)

    fig2.savefig('rolling_window.png', bbox_inches='tight', dpi=300)

    fig3.savefig('sliding_window.png', bbox_inches='tight', dpi=300)

    plt.show()


def plot_full_dataset_2() -> None:
    cv_anim = AnimatedCrossValidation(data_length=200)

    expanding_anim = cv_anim.create_expanding_window_animation(n_splits=3)

    rolling_anim = cv_anim.create_rolling_window_animation(window_size=40, step_size=20)

    sliding_anim = cv_anim.create_sliding_window_animation(window_size=40, horizon=10)

    expanding_anim.save('expanding_window.gif', writer='pillow')

    rolling_anim.save('rolling_window.gif', writer='pillow')

    sliding_anim.save('sliding_window.gif', writer='pillow')

    plt.show()


def generate_two_related_time_series() -> None:
    visualizer = CausalInferenceVisualizer()

    anim = visualizer.create_animation()

    anim.save('causal_inference.gif', writer='pillow')

    plt.close()


def generate_true_trajectory() -> None:
    visualizer = KalmanFilterVisualizer()

    anim = visualizer.create_animation()

    anim.save('kalman_filter.gif', writer='pillow', fps=30)

    plt.close()


def set_random_seed_for_reproducibility() -> None:
    '\nworks\n'

    np.random.seed(42)

    time = pd.date_range(start='2023-01-01', periods=100, freq='D')

    values = np.random.normal(50, 2, 100)

    values[30:35] += 8

    values[70:75] -= 8

    df = pd.DataFrame({'Time': time, 'Value': values})

    mean = df['Value'].mean()

    std_dev = df['Value'].std()

    ucl = mean + 3 * std_dev

    lcl = mean - 3 * std_dev

    fig, ax = plt.subplots(figsize=(12, 6))

    line, = ax.plot([], [], label='Process Data', marker='o', linestyle='-')

    ucl_line = ax.axhline(ucl, color='red', linestyle='--', label='Upper Control Limit (UCL)')

    lcl_line = ax.axhline(lcl, color='red', linestyle='--', label='Lower Control Limit (LCL)')

    mean_line = ax.axhline(mean, color='blue', linestyle='--', label='Mean')

    ax.set_xlim(df['Time'].min(), df['Time'].max())

    ax.set_ylim(min(lcl, df['Value'].min()) - 1, max(ucl, df['Value'].max()) + 1)

    ax.set_title('Control Chart with Out-of-Control Areas')

    ax.set_xlabel('Time')

    ax.set_ylabel('Value')

    ax.legend()

    ax.grid(True)

    anim = FuncAnimation(fig, animate, frames=len(df), interval=100, blit=True, repeat=False)

    anim.save('control_chart_animation.gif', writer='pillow', fps=10)

    plt.close(fig)


def don_t_use_notebook_markdown() -> None:
    """
    # don't use  # notebook markdown
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import seaborn as sns

    class TimeSeriesFeatureVisualizer:
        def __init__(self):
            # Generate sample data
            np.random.seed(42)
            n = 500
            self.dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
            self.values = 100 + np.cumsum(np.random.normal(0, 1, n))
            self.df = pd.DataFrame({"date": self.dates, "value": self.values})

            # Calculate features
            self.calculate_features()

        def calculate_features(self):
            # Rolling statistics
            self.df['rolling_mean_7'] = self.df['value'].rolling(window=7).mean()
            self.df['rolling_std_7'] = self.df['value'].rolling(window=7).std()

            # Lagged features
            self.df['lag_1'] = self.df['value'].shift(1)
            self.df['lag_7'] = self.df['value'].shift(7)

            # Fourier features
            t = np.arange(len(self.df))
            period = 30  # 30-day seasonality
            self.df['fourier_sin'] = np.sin(2 * np.pi * t / period)
            self.df['fourier_cos'] = np.cos(2 * np.pi * t / period)

        def create_animation(self):
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2)

            # Create subplots
            ax1 = fig.add_subplot(gs[0, :])  # Original series + rolling stats
            ax2 = fig.add_subplot(gs[1, 0])  # Lagged features
            ax3 = fig.add_subplot(gs[1, 1])  # Fourier components
            ax4 = fig.add_subplot(gs[2, :])  # Feature importance

            # Define window size as class attribute
            self.window = 100

            def animate(frame):
                # Clear previous frame
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.clear()

                # Use window size from class attribute
                start_idx = max(0, frame - self.window)
                end_idx = frame + 1

                # Plot 1: Original series with rolling statistics
                ax1.plot(self.df['date'][start_idx:end_idx],
                        self.df['value'][start_idx:end_idx],
                        label='Original', color='blue')
                ax1.plot(self.df['date'][start_idx:end_idx],
                        self.df['rolling_mean_7'][start_idx:end_idx],
                        label='7-day MA', color='red')
                ax1.fill_between(self.df['date'][start_idx:end_idx],
                               self.df['rolling_mean_7'][start_idx:end_idx] -
                               self.df['rolling_std_7'][start_idx:end_idx],
                               self.df['rolling_mean_7'][start_idx:end_idx] +
                               self.df['rolling_std_7'][start_idx:end_idx],
                               alpha=0.2, color='red')
                ax1.set_title('Time Series with Rolling Statistics')
                ax1.legend()

                # Plot 2: Lagged features
                if frame > 7:
                    ax2.scatter(self.df['lag_1'][start_idx:end_idx],
                              self.df['value'][start_idx:end_idx],
                              alpha=0.5, label='Lag 1')
                    ax2.scatter(self.df['lag_7'][start_idx:end_idx],
                              self.df['value'][start_idx:end_idx],
                              alpha=0.5, label='Lag 7')
                    ax2.set_title('Lagged Features')
                    ax2.set_xlabel('Lagged Value')
                    ax2.set_ylabel('Current Value')
                    ax2.legend()

                # Plot 3: Fourier components
                ax3.plot(self.df['date'][start_idx:end_idx],
                        self.df['fourier_sin'][start_idx:end_idx],
                        label='Sine', color='green')
                ax3.plot(self.df['date'][start_idx:end_idx],
                        self.df['fourier_cos'][start_idx:end_idx],
                        label='Cosine', color='orange')
                ax3.set_title('Fourier Components')
                ax3.legend()

                # Plot 4: Feature importance (simulated) — disabled in animation

            anim = FuncAnimation(fig, animate,
                               frames=len(self.df)-self.window,
                               interval=50, repeat=True)
            return anim

    # Create and save animation
    visualizer = TimeSeriesFeatureVisualizer()
    anim = visualizer.create_animation()
    anim.save('pytimetk_features.gif', writer='pillow', fps=30)
    plt.close()


def generate_sample_data() -> None:
    visualizer = TimeSeriesFeatureVisualizer()

    anim = visualizer.create_animation()

    anim.save('pytimetk_features.gif', writer='pillow', fps=10)

    plt.close()


def generate_sample_data_2() -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.stats import norm

    class StatsVisualizer:
        def __init__(self):
            # Generate sample data
            np.random.seed(42)
            self.data = [3, 21, 98, 203, 17, 9]
            self.mean = np.mean(self.data)
            self.variance = np.var(self.data, ddof=1)
            self.std_dev = np.std(self.data, ddof=1)

            # Create normal distribution for comparison
            self.x = np.linspace(self.mean - 4*self.std_dev,
                               self.mean + 4*self.std_dev, 100)
            self.normal_dist = norm.pdf(self.x, self.mean, self.std_dev)

        def create_animation(self):
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)

            # Create subplots
            ax1 = fig.add_subplot(gs[0, :])  # Dot plot and deviations
            ax2 = fig.add_subplot(gs[1, 0])  # Squared deviations
            ax3 = fig.add_subplot(gs[1, 1])  # Normal distribution

            def animate(frame):
                # Clear axes
                for ax in [ax1, ax2, ax3]:
                    ax.clear()

                # Plot 1: Dot plot with deviations
                ax1.scatter(self.data, np.zeros_like(self.data),
                           color='blue', s=100, label='Data points')
                ax1.axvline(self.mean, color='red', linestyle='--',
                           label=f'Mean = {self.mean:.2f}')

                # Animate deviations
                current_point = frame % len(self.data)
                for i in range(current_point + 1):
                    deviation = self.data[i] - self.mean
                    ax1.arrow(self.mean, 0, deviation, 0,
                             color='green', alpha=0.5,
                             head_width=0.1, head_length=2)

                ax1.set_title('Data Points and Deviations from Mean')
                ax1.legend()

                # Plot 2: Squared deviations
                deviations = [(x - self.mean)**2 for x in self.data[:current_point+1]]
                ax2.bar(range(current_point+1), deviations,
                       color='purple', alpha=0.6)
                if deviations:
                    current_variance = np.mean(deviations)
                    ax2.axhline(current_variance, color='red', linestyle='--',
                               label=f'Current Variance = {current_variance:.2f}')
                ax2.set_title('Squared Deviations')
                ax2.set_xlabel('Data Point Index')
                ax2.set_ylabel('Squared Deviation')
                ax2.legend()

                # Plot 3: Normal distribution with standard deviations
                ax3.plot(self.x, self.normal_dist, 'b-', label='Normal Distribution')
                ax3.axvline(self.mean, color='red', linestyle='--', label='Mean')

                # Add standard deviation bands
                for i in range(1, 4):
                    ax3.axvline(self.mean + i*self.std_dev, color='green',
                              alpha=0.3, linestyle=':')
                    ax3.axvline(self.mean - i*self.std_dev, color='green',
                              alpha=0.3, linestyle=':')
                    ax3.fill_between(self.x,
                                   np.zeros_like(self.x),
                                   self.normal_dist,
                                   where=(self.x >= self.mean - i*self.std_dev) &
                                        (self.x <= self.mean + i*self.std_dev),
                                   alpha=0.1, color='green',
                                   label=f'{i}σ = {i*68.27:.1f}%')

                ax3.set_title('Normal Distribution with Standard Deviations')
                ax3.legend()

                plt.tight_layout()

            anim = FuncAnimation(fig, animate,
                               frames=len(self.data) * 2,  # Slow down animation
                               interval=1000,  # 1 second between frames
                               repeat=True)
            return anim

    # Create and save animation
    visualizer = StatsVisualizer()
    anim = visualizer.create_animation()
    anim.save('variance_std_dev.gif', writer='pillow', fps=2)
    plt.close()


def generate_larger_sample_data() -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.stats import norm

    class StatsVisualizer:
        def __init__(self):
            # Generate larger sample data
            np.random.seed(42)
            self.n_points = 200
            # Generate data from a mixture of normal distributions for interesting visualization
            dist1 = np.random.normal(100, 15, self.n_points // 2)
            dist2 = np.random.normal(130, 20, self.n_points // 2)
            self.data = np.concatenate([dist1, dist2])
            self.data.sort()  # Sort for better visualization

            self.mean = np.mean(self.data)
            self.variance = np.var(self.data, ddof=1)
            self.std_dev = np.std(self.data, ddof=1)

            # Create normal distribution for comparison
            self.x = np.linspace(self.mean - 4*self.std_dev,
                               self.mean + 4*self.std_dev, 200)
            self.normal_dist = norm.pdf(self.x, self.mean, self.std_dev)

        def create_animation(self):
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)

            # Create subplots
            ax1 = fig.add_subplot(gs[0, :])  # Histogram and normal distribution
            ax2 = fig.add_subplot(gs[1, 0])  # Cumulative mean
            ax3 = fig.add_subplot(gs[1, 1])  # Cumulative standard deviation

            def animate(frame):
                # Clear axes
                for ax in [ax1, ax2, ax3]:
                    ax.clear()

                # Calculate current slice of data
                current_idx = int((frame / 100) * len(self.data))
                current_data = self.data[:current_idx+1]

                if len(current_data) > 0:
                    current_mean = np.mean(current_data)
                    current_std = np.std(current_data, ddof=1) if len(current_data) > 1 else 0

                    # Plot 1: Histogram with normal distribution
                    ax1.hist(current_data, bins=30, density=True, alpha=0.6,
                            color='blue', label='Data Distribution')

                    # Update normal distribution based on current data
                    if len(current_data) > 1:
                        x_current = np.linspace(min(current_data), max(current_data), 100)
                        current_normal = norm.pdf(x_current, current_mean, current_std)
                        ax1.plot(x_current, current_normal, 'r-',
                                label='Normal Distribution')

                        # Add standard deviation bands
                        for i in range(1, 4):
                            ax1.axvline(current_mean + i*current_std, color='green',
                                      alpha=0.3, linestyle=':')
                            ax1.axvline(current_mean - i*current_std, color='green',
                                      alpha=0.3, linestyle=':')
                            ax1.fill_between(x_current,
                                           np.zeros_like(x_current),
                                           current_normal,
                                           where=(x_current >= current_mean - i*current_std) &
                                                (x_current <= current_mean + i*current_std),
                                           alpha=0.1, color='green',
                                           label=f'{i}σ = {i*68.27:.1f}%')

                    ax1.axvline(current_mean, color='red', linestyle='--',
                              label=f'Mean = {current_mean:.2f}')
                    ax1.set_title(f'Distribution of {len(current_data)} Data Points')
                    ax1.legend()

                    # Plot 2: Running mean
                    means = [np.mean(self.data[:i+1]) for i in range(len(current_data))]
                    ax2.plot(range(len(means)), means, 'b-', label='Running Mean')
                    ax2.axhline(self.mean, color='r', linestyle='--',
                              label=f'True Mean = {self.mean:.2f}')
                    ax2.set_title('Running Mean')
                    ax2.set_xlabel('Number of Points')
                    ax2.set_ylabel('Mean Value')
                    ax2.legend()

                    # Plot 3: Running standard deviation
                    stds = [np.std(self.data[:i+1], ddof=1) if i > 0 else 0
                           for i in range(len(current_data))]
                    ax3.plot(range(len(stds)), stds, 'g-', label='Running Std Dev')
                    ax3.axhline(self.std_dev, color='r', linestyle='--',
                              label=f'True Std Dev = {self.std_dev:.2f}')
                    ax3.set_title('Running Standard Deviation')
                    ax3.set_xlabel('Number of Points')
                    ax3.set_ylabel('Standard Deviation')
                    ax3.legend()

                plt.tight_layout()

            anim = FuncAnimation(fig, animate,
                               frames=100,  # 100 frames for smooth animation
                               interval=200,  # 100ms between frames
                               repeat=False)
            return anim

    # Create and save animation
    visualizer = StatsVisualizer()
    anim = visualizer.create_animation()
    anim.save('variance_std_dev.gif', writer='pillow', fps=5)
    plt.close()


def generate_true_trajectory_2() -> None:
    visualizer = KalmanFilterVisualizer()

    anim = visualizer.create_animation()

    anim.save('kalman_filter.gif', writer='pillow', fps=10)

    plt.close()


def generate_true_trajectory_3() -> None:
    visualizer = KalmanFilterVisualizer()

    anim = visualizer.create_animation()

    anim.save('kalman_filter.gif', writer='pillow', fps=10)

    plt.close()


def generate_synthetic_time_series_data_for_differen() -> None:
    visualizer = TSClassificationVisualizer()

    anim = visualizer.create_animation()

    anim.save('ts_classification.gif', writer='pillow', fps=10)

    plt.close()


def parameters() -> None:
    visualizer = ConfidenceIntervalVisualizer()

    anim = visualizer.create_animation()

    anim.save('confidence_intervals.gif', writer='pillow', fps=10)

    plt.close()


def time_domain_parameters() -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    class FourierVisualizer:
        def __init__(self):
            # Time domain parameters
            self.t = np.linspace(0, 10, 1000)
            self.dt = self.t[1] - self.t[0]

            # Generate composite signal
            self.f1, self.f2 = 1.0, 2.5  # frequencies
            self.signal = (np.sin(2*np.pi*self.f1*self.t) +
                          0.5*np.sin(2*np.pi*self.f2*self.t) +
                          0.2*np.random.randn(len(self.t)))

            # Compute FFT
            self.fft_vals = np.fft.fft(self.signal)
            self.freqs = np.fft.fftfreq(len(self.t), self.dt)

            # Store reconstructed signals
            self.reconstruct_signals()

        def reconstruct_signals(self):
            """Reconstruct signals with different numbers of frequencies"""
            self.reconstructions = []
            sorted_indices = np.argsort(np.abs(self.fft_vals))[::-1]

            for n_freqs in range(1, 51):
                fft_filtered = np.zeros_like(self.fft_vals, dtype=complex)
                fft_filtered[sorted_indices[:n_freqs]] = self.fft_vals[sorted_indices[:n_freqs]]
                fft_filtered[sorted_indices[-n_freqs + 1 :]] = self.fft_vals[
                    sorted_indices[-n_freqs + 1 :]
                ]
                reconstructed = np.fft.ifft(fft_filtered).real
                self.reconstructions.append(reconstructed)

        def create_animation(self):
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(3, 2)

            # Create subplots
            ax1 = fig.add_subplot(gs[0, :])    # Original signal
            ax2 = fig.add_subplot(gs[1, 0])    # Frequency domain
            ax3 = fig.add_subplot(gs[1, 1])    # Phase spectrum
            ax4 = fig.add_subplot(gs[2, :])    # Reconstruction

            def animate(frame):
                # Clear axes
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.clear()

                # Plot 1: Original Signal
                window = 200
                start_idx = frame % (len(self.t) - window)
                end_idx = start_idx + window

                ax1.plot(self.t[start_idx:end_idx],
                        self.signal[start_idx:end_idx],
                        'b-', label='Original Signal')
                ax1.set_title('Time Domain Signal')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Amplitude')
                ax1.legend()
                ax1.grid(True)

                # Plot 2: Frequency Spectrum
                positive_freq_mask = self.freqs >= 0
                ax2.plot(self.freqs[positive_freq_mask],
                        np.abs(self.fft_vals)[positive_freq_mask],
                        'r-', label='Magnitude Spectrum')
                ax2.set_title('Frequency Spectrum')
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Magnitude')
                ax2.grid(True)

                # Plot 3: Phase Spectrum
                ax3.plot(self.freqs[positive_freq_mask],
                        np.angle(self.fft_vals)[positive_freq_mask],
                        'g-', label='Phase Spectrum')
                ax3.set_title('Phase Spectrum')
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('Phase (radians)')
                ax3.grid(True)

                # Plot 4: Signal Reconstruction
                reconstruction_idx = min(frame // 2, len(self.reconstructions) - 1)
                ax4.plot(self.t[start_idx:end_idx],
                        self.signal[start_idx:end_idx],
                        'b-', alpha=0.5, label='Original')
                ax4.plot(self.t[start_idx:end_idx],
                        self.reconstructions[reconstruction_idx][start_idx:end_idx],
                        'r-', label=f'Reconstruction\n({reconstruction_idx+1} frequencies)')
                ax4.set_title('Signal Reconstruction')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Amplitude')
                ax4.legend()
                ax4.grid(True)

                plt.tight_layout()

            anim = FuncAnimation(fig, animate,
                               frames=200,  # Adjust for smoothness
                               interval=100,  # 100ms between frames
                               repeat=True)
            return anim

    # Create and save animation
    visualizer = FourierVisualizer()
    anim = visualizer.create_animation()
    anim.save('fourier_transform.gif', writer='pillow', fps=10)
    plt.close()


def notebook_step_019() -> None:
    data = pd.Series([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name='Sales')

    data.index = pd.date_range(start='2010-01-01', periods=len(data), freq='M')

    df = data.to_frame()

    s = setup(data=df, target='Sales', session_id=123, seasonal_period=12)

    best_model = compare_models()

    tuned_model = tune_model(best_model)

    future_forecast = predict_model(tuned_model, fh=12)

    print(future_forecast)

    plot_model(tuned_model, plot='forecast')

    backtest_metrics = pull()

    print(backtest_metrics)

    print(s.get_config('X_train').head())

    stacked_model = stack_models(top_n=3)

    save_model(tuned_model, 'time_series_model')

    loaded_model = load_model('time_series_model')

    df['Marketing_Spend'] = [50 + i % 10 for i in range(len(df))]

    s = setup(data=df, target='Sales', session_id=123)

    best_multivariate_model = compare_models()

    future_forecast_multivariate = predict_model(best_multivariate_model, fh=12)

    print(future_forecast_multivariate)

    plot_model(best_multivariate_model, plot='forecast')


def pip_install_pycaret_jupyter_only() -> None:
    # !pip install pycaret  # Jupyter-only
    # pip install pycaret  # Jupyter-only


    pass
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


def create_sample_data_2() -> None:
    data = pd.Series([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name='Sales')

    data.index = pd.date_range(start='2010-01-01', periods=len(data), freq='M')

    df = data.to_frame()

    setup_data = setup(data=df, target='Sales', session_id=123, seasonal_period=12)

    best_model = compare_models()

    tuned_model = tune_model(best_model)

    future_forecast = predict_model(tuned_model, fh=12)

    print('\nForecast:')

    print(future_forecast)

    plot_model(tuned_model, plot='forecast')

    metrics = pull()

    print('\nMetrics:')

    print(metrics)

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


def pip_show_pycaret_jupyter_only() -> None:
    # !pip show pycaret  # Jupyter-only
    # pip show pycaret  # Jupyter-only


    pass
def create_sample_data_3() -> None:
    data = pd.Series([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name='Sales')

    data.index = pd.date_range(start='2010-01-01', periods=len(data), freq='M')

    df = data.to_frame()

    exp = TSForecastingExperiment()

    exp.setup(data=df, target='Sales', session_id=123, seasonal_period=12)

    best_model = exp.compare_models()

    tuned_model = exp.tune_model(best_model)

    future_forecast = exp.predict_model(tuned_model, fh=12)

    print('\nForecast:')

    print(future_forecast)

    exp.plot_model(tuned_model, plot='forecast')

    metrics = exp.pull()

    print('\nMetrics:')

    print(metrics)

    exp.save_model(tuned_model, 'time_series_model')

    loaded_model = exp.load_model('time_series_model')

    df['Marketing_Spend'] = [50 + i % 10 for i in range(len(df))]

    exp_multi = TSForecastingExperiment()

    exp_multi.setup(data=df, target='Sales', session_id=123)

    best_multivariate_model = exp_multi.compare_models()

    future_forecast_multivariate = exp_multi.predict_model(best_multivariate_model, fh=12)

    print('\nMultivariate forecast:')

    print(future_forecast_multivariate)

    exp_multi.plot_model(best_multivariate_model, plot='forecast')


def generate_sample_data_3() -> None:
    visualizer = BayesianARIMAVisualizer()

    anim = visualizer.create_animation()

    anim.save('bayesian_arima_comparison.gif', writer='pillow', fps=5)

    plt.close()


def pip_install_pmdarima_jupyter_only() -> None:
    # !pip install pmdarima  # Jupyter-only
    # pip install pmdarima  # Jupyter-only


    pass
def simulate_data_with_volatility_clustering() -> None:
    visualizer = ARCHVisualizer()

    anim = visualizer.create_animation()

    anim.save('arch_visualization.gif', writer='pillow', fps=5)

    plt.close()


def pip_install_arch_jupyter_only() -> None:
    # !pip install arch  # Jupyter-only
    # pip install arch  # Jupyter-only


    pass
def generate_rolling_forecasts() -> None:
    visualizer = ARCHVisualizer()

    anim = visualizer.create_animation()

    anim.save('arch_visualization.gif', writer='pillow', fps=5)

    plt.close()


def generate_sample_data_4() -> None:
    visualizer = SVARVisualizer()

    anim = visualizer.create_animation()

    anim.save('svar_visualization.gif', writer='pillow', fps=5)

    plt.close()


def simulate_a_two_variable_system() -> None:
    visualizer = SVARVisualizer()

    anim = visualizer.create_animation()

    anim.save('svar_visualization.gif', writer='pillow', fps=5)

    plt.close()


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


def generate_sample_data_5() -> None:
    visualizer = BayesianARIMAVisualizer()

    anim = visualizer.create_animation()

    anim.save('bayesian_arima_comparison.gif', writer='pillow', fps=5)

    plt.close()


def main() -> None:
    notebook_step_001()
    assuming_you_have_your_taxi_data_in_a_dataframe()
    visualizing_time_series_cross_validation_methods()
    generate_sample_time_series_data()
    plot_full_dataset()
    plot_full_dataset_2()
    generate_two_related_time_series()
    generate_true_trajectory()
    set_random_seed_for_reproducibility()
    don_t_use_notebook_markdown()
    generate_sample_data()
    generate_sample_data_2()
    generate_larger_sample_data()
    generate_true_trajectory_2()
    generate_true_trajectory_3()
    generate_synthetic_time_series_data_for_differen()
    parameters()
    time_domain_parameters()
    notebook_step_019()
    pip_install_pycaret_jupyter_only()
    create_sample_data()
    create_sample_data_2()
    pip_show_pycaret_jupyter_only()
    create_sample_data_3()
    generate_sample_data_3()
    pip_install_pmdarima_jupyter_only()
    simulate_data_with_volatility_clustering()
    pip_install_arch_jupyter_only()
    generate_rolling_forecasts()
    generate_sample_data_4()
    simulate_a_two_variable_system()
    set_random_seed_for_reproducibility_2()
    generate_sample_data_5()

if __name__ == "__main__":
    main()
