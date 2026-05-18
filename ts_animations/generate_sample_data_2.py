"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def generate_sample_data_2() -> None:
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
            self.x = np.linspace(
                self.mean - 4 * self.std_dev, self.mean + 4 * self.std_dev, 100
            )
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
                ax1.scatter(
                    self.data,
                    np.zeros_like(self.data),
                    color="blue",
                    s=100,
                    label="Data points",
                )
                ax1.axvline(
                    self.mean,
                    color="red",
                    linestyle="--",
                    label=f"Mean = {self.mean:.2f}",
                )
                # Animate deviations
                current_point = frame % len(self.data)
                for i in range(current_point + 1):
                    deviation = self.data[i] - self.mean
                    ax1.arrow(
                        self.mean,
                        0,
                        deviation,
                        0,
                        color="green",
                        alpha=0.5,
                        head_width=0.1,
                        head_length=2,
                    )

                ax1.set_title("Data Points and Deviations from Mean")
                ax1.legend()
                # Plot 2: Squared deviations
                deviations = [
                    (x - self.mean) ** 2 for x in self.data[: current_point + 1]
                ]
                ax2.bar(range(current_point + 1), deviations, color="purple", alpha=0.6)
                if deviations:
                    current_variance = np.mean(deviations)
                    ax2.axhline(
                        current_variance,
                        color="red",
                        linestyle="--",
                        label=f"Current Variance = {current_variance:.2f}",
                    )
                ax2.set_title("Squared Deviations")
                ax2.set_xlabel("Data Point Index")
                ax2.set_ylabel("Squared Deviation")
                ax2.legend()
                # Plot 3: Normal distribution with standard deviations
                ax3.plot(self.x, self.normal_dist, "b-", label="Normal Distribution")
                ax3.axvline(self.mean, color="red", linestyle="--", label="Mean")
                # Add standard deviation bands
                for i in range(1, 4):
                    ax3.axvline(
                        self.mean + i * self.std_dev,
                        color="green",
                        alpha=0.3,
                        linestyle=":",
                    )
                    ax3.axvline(
                        self.mean - i * self.std_dev,
                        color="green",
                        alpha=0.3,
                        linestyle=":",
                    )
                    ax3.fill_between(
                        self.x,
                        np.zeros_like(self.x),
                        self.normal_dist,
                        where=(self.x >= self.mean - i * self.std_dev)
                        & (self.x <= self.mean + i * self.std_dev),
                        alpha=0.1,
                        color="green",
                        label=f"{i}σ = {i * 68.27:.1f}%",
                    )

                ax3.set_title("Normal Distribution with Standard Deviations")
                ax3.legend()
                plt.tight_layout()

            anim = FuncAnimation(
                fig,
                animate,
                frames=len(self.data) * 2,  # Slow down animation
                interval=1000,  # 1 second between frames
                repeat=True,
            )
            return anim

    # Create and save animation
    visualizer = StatsVisualizer()
    anim = visualizer.create_animation()
    anim.save("variance_std_dev.gif", writer="pillow", fps=2)
    plt.close()
