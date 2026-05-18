"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class CausalInferenceVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.t = np.linspace(0, 10, 500)
        self.x = np.sin(self.t) + np.random.normal(0, 0.1, len(self.t))
        self.y = np.sin(self.t + 0.5) + np.random.normal(0, 0.1, len(self.t))

    def create_animation(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Causal Analysis in Time Series", fontsize=16)

        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            window = 100
            start_idx = frame % (len(self.t) - window)
            end_idx = start_idx + window
            ax1.plot(
                self.t[start_idx:end_idx],
                self.x[start_idx:end_idx],
                label="Series X",
                color="blue",
            )
            ax1.plot(
                self.t[start_idx:end_idx],
                self.y[start_idx:end_idx],
                label="Series Y",
                color="red",
            )
            ax1.set_title("Time Series Data")
            ax1.legend()
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Value")
            lags = 20
            if start_idx > lags:
                x_lagged = self.x[start_idx - lags : end_idx - lags]
                y_current = self.y[start_idx:end_idx]
                ax2.scatter(x_lagged, y_current, alpha=0.5, color="purple")
                ax2.set_title("Granger Causality Analysis")
                ax2.set_xlabel("X (lagged)")
                ax2.set_ylabel("Y (current)")
            treatment_period = len(self.t) // 2
            ax3.plot(
                self.t[:treatment_period],
                self.y[:treatment_period],
                color="blue",
                label="Pre-treatment",
            )
            ax3.plot(
                self.t[treatment_period:],
                self.y[treatment_period:],
                color="red",
                label="Post-treatment",
            )
            ax3.axvline(
                x=self.t[treatment_period],
                color="black",
                linestyle="--",
                label="Treatment",
            )
            ax3.set_title("Treatment Effect Analysis")
            ax3.legend()
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Outcome")
            impulse = np.zeros(50)
            impulse[0] = 1
            response = np.exp(-np.arange(50) * 0.1) * np.sin(np.arange(50) * 0.5)
            ax4.plot(range(50), response, color="green")
            ax4.set_title("Impulse Response Function")
            ax4.set_xlabel("Time")
            ax4.set_ylabel("Response")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        anim = FuncAnimation(
            fig, animate, frames=len(self.t) - 100, interval=50, repeat=True
        )
        return anim
