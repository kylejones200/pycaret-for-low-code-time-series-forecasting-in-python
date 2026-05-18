"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def generate_larger_sample_data() -> None:
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
            self.x = np.linspace(
                self.mean - 4 * self.std_dev, self.mean + 4 * self.std_dev, 200
            )
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
                current_data = self.data[: current_idx + 1]
                if len(current_data) > 0:
                    current_mean = np.mean(current_data)
                    current_std = (
                        np.std(current_data, ddof=1) if len(current_data) > 1 else 0
                    )
                    # Plot 1: Histogram with normal distribution
                    ax1.hist(
                        current_data,
                        bins=30,
                        density=True,
                        alpha=0.6,
                        color="blue",
                        label="Data Distribution",
                    )
                    # Update normal distribution based on current data
                    if len(current_data) > 1:
                        x_current = np.linspace(
                            min(current_data), max(current_data), 100
                        )
                        current_normal = norm.pdf(x_current, current_mean, current_std)
                        ax1.plot(
                            x_current, current_normal, "r-", label="Normal Distribution"
                        )
                        # Add standard deviation bands
                        for i in range(1, 4):
                            ax1.axvline(
                                current_mean + i * current_std,
                                color="green",
                                alpha=0.3,
                                linestyle=":",
                            )
                            ax1.axvline(
                                current_mean - i * current_std,
                                color="green",
                                alpha=0.3,
                                linestyle=":",
                            )
                            ax1.fill_between(
                                x_current,
                                np.zeros_like(x_current),
                                current_normal,
                                where=(x_current >= current_mean - i * current_std)
                                & (x_current <= current_mean + i * current_std),
                                alpha=0.1,
                                color="green",
                                label=f"{i}σ = {i * 68.27:.1f}%",
                            )

                    ax1.axvline(
                        current_mean,
                        color="red",
                        linestyle="--",
                        label=f"Mean = {current_mean:.2f}",
                    )
                    ax1.set_title(f"Distribution of {len(current_data)} Data Points")
                    ax1.legend()
                    # Plot 2: Running mean
                    means = [
                        np.mean(self.data[: i + 1]) for i in range(len(current_data))
                    ]
                    ax2.plot(range(len(means)), means, "b-", label="Running Mean")
                    ax2.axhline(
                        self.mean,
                        color="r",
                        linestyle="--",
                        label=f"True Mean = {self.mean:.2f}",
                    )
                    ax2.set_title("Running Mean")
                    ax2.set_xlabel("Number of Points")
                    ax2.set_ylabel("Mean Value")
                    ax2.legend()
                    # Plot 3: Running standard deviation
                    stds = [
                        np.std(self.data[: i + 1], ddof=1) if i > 0 else 0
                        for i in range(len(current_data))
                    ]
                    ax3.plot(range(len(stds)), stds, "g-", label="Running Std Dev")
                    ax3.axhline(
                        self.std_dev,
                        color="r",
                        linestyle="--",
                        label=f"True Std Dev = {self.std_dev:.2f}",
                    )
                    ax3.set_title("Running Standard Deviation")
                    ax3.set_xlabel("Number of Points")
                    ax3.set_ylabel("Standard Deviation")
                    ax3.legend()

                plt.tight_layout()

            anim = FuncAnimation(
                fig,
                animate,
                frames=100,  # 100 frames for smooth animation
                interval=200,  # 100ms between frames
                repeat=False,
            )
            return anim

    # Create and save animation
    visualizer = StatsVisualizer()
    anim = visualizer.create_animation()
    anim.save("variance_std_dev.gif", writer="pillow", fps=5)
    plt.close()
