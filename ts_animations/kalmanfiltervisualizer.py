"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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
            ax1.plot(
                self.t[start_idx:end_idx],
                self.true_position[start_idx:end_idx],
                "g-",
                label="True Position",
            )
            ax1.plot(
                self.t[start_idx:end_idx],
                self.measurements[start_idx:end_idx],
                "r.",
                label="Measurements",
            )
            ax1.plot(
                self.t[start_idx:end_idx],
                self.estimated_states[start_idx:end_idx, 0],
                "b-",
                label="Kalman Filter",
            )
            ax1.set_title("Position Tracking")
            ax1.legend()
            ax2.plot(
                self.true_position[start_idx:end_idx],
                self.true_velocity[start_idx:end_idx],
                "g-",
                label="True State",
            )
            ax2.plot(
                self.estimated_states[start_idx:end_idx, 0],
                self.estimated_states[start_idx:end_idx, 1],
                "b-",
                label="Estimated State",
            )
            ax2.set_title("Phase Space")
            ax2.set_xlabel("Position")
            ax2.set_ylabel("Velocity")
            ax2.legend()
            if frame > 0:
                mean = self.estimated_states[frame]
                cov = self.estimation_covs[frame]
                theta = np.linspace(0, 2 * np.pi, 100)
                epsilon = 1e-06
                eigenvals, eigenvecs = np.linalg.eigh(cov + epsilon * np.eye(2))
                sqrt_eigenvals = np.sqrt(eigenvals)
                ellipse_x = (
                    sqrt_eigenvals[0] * np.cos(theta) * eigenvecs[0, 0]
                    + sqrt_eigenvals[1] * np.sin(theta) * eigenvecs[0, 1]
                )
                ellipse_y = (
                    sqrt_eigenvals[0] * np.cos(theta) * eigenvecs[1, 0]
                    + sqrt_eigenvals[1] * np.sin(theta) * eigenvecs[1, 1]
                )
                ax3.plot(
                    mean[0] + ellipse_x,
                    mean[1] + ellipse_y,
                    "r-",
                    label="Uncertainty (1σ)",
                )
                ax3.plot(mean[0], mean[1], "bo", label="Current Estimate")
                ax3.set_title("State Uncertainty")
                ax3.set_xlabel("Position")
                ax3.set_ylabel("Velocity")
                ax3.legend()
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=len(self.t), interval=100, repeat=True
        )
        return anim
