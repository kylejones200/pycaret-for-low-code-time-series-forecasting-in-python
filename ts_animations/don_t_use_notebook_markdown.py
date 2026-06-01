"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


def don_t_use_notebook_markdown() -> None:
    """
    # don't use  # notebook markdown
    """

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
            self.df["rolling_mean_7"] = self.df["value"].rolling(window=7).mean()
            self.df["rolling_std_7"] = self.df["value"].rolling(window=7).std()
            # Lagged features
            self.df["lag_1"] = self.df["value"].shift(1)
            self.df["lag_7"] = self.df["value"].shift(7)
            # Fourier features
            t = np.arange(len(self.df))
            period = 30  # 30-day seasonality
            self.df["fourier_sin"] = np.sin(2 * np.pi * t / period)
            self.df["fourier_cos"] = np.cos(2 * np.pi * t / period)

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
                ax1.plot(
                    self.df["date"][start_idx:end_idx],
                    self.df["value"][start_idx:end_idx],
                    label="Original",
                    color="blue",
                )
                ax1.plot(
                    self.df["date"][start_idx:end_idx],
                    self.df["rolling_mean_7"][start_idx:end_idx],
                    label="7-day MA",
                    color="red",
                )
                ax1.fill_between(
                    self.df["date"][start_idx:end_idx],
                    self.df["rolling_mean_7"][start_idx:end_idx]
                    - self.df["rolling_std_7"][start_idx:end_idx],
                    self.df["rolling_mean_7"][start_idx:end_idx]
                    + self.df["rolling_std_7"][start_idx:end_idx],
                    alpha=0.2,
                    color="red",
                )
                ax1.set_title("Time Series with Rolling Statistics")
                ax1.legend()
                # Plot 2: Lagged features
                if frame > 7:
                    ax2.scatter(
                        self.df["lag_1"][start_idx:end_idx],
                        self.df["value"][start_idx:end_idx],
                        alpha=0.5,
                        label="Lag 1",
                    )
                    ax2.scatter(
                        self.df["lag_7"][start_idx:end_idx],
                        self.df["value"][start_idx:end_idx],
                        alpha=0.5,
                        label="Lag 7",
                    )
                    ax2.set_title("Lagged Features")
                    ax2.set_xlabel("Lagged Value")
                    ax2.set_ylabel("Current Value")
                    ax2.legend()

                # Plot 3: Fourier components
                ax3.plot(
                    self.df["date"][start_idx:end_idx],
                    self.df["fourier_sin"][start_idx:end_idx],
                    label="Sine",
                    color="green",
                )
                ax3.plot(
                    self.df["date"][start_idx:end_idx],
                    self.df["fourier_cos"][start_idx:end_idx],
                    label="Cosine",
                    color="orange",
                )
                ax3.set_title("Fourier Components")
                ax3.legend()
                # Plot 4: Feature importance (simulated) — disabled in animation

            anim = FuncAnimation(
                fig,
                animate,
                frames=len(self.df) - self.window,
                interval=50,
                repeat=True,
            )
            return anim

    # Create and save animation
    visualizer = TimeSeriesFeatureVisualizer()
    anim = visualizer.create_animation()
    anim.save("pytimetk_features.gif", writer="pillow", fps=30)
    plt.close()
