"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


def set_random_seed_for_reproducibility() -> None:
    "\nworks\n"
    np.random.seed(42)
    time = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.normal(50, 2, 100)
    values[30:35] += 8
    values[70:75] -= 8
    df = pd.DataFrame({"Time": time, "Value": values})
    mean = df["Value"].mean()
    std_dev = df["Value"].std()
    ucl = mean + 3 * std_dev
    lcl = mean - 3 * std_dev
    fig, ax = plt.subplots(figsize=(12, 6))
    (line,) = ax.plot([], [], label="Process Data", marker="o", linestyle="-")
    ax.axhline(ucl, color="red", linestyle="--", label="Upper Control Limit (UCL)")
    ax.axhline(lcl, color="red", linestyle="--", label="Lower Control Limit (LCL)")
    ax.axhline(mean, color="blue", linestyle="--", label="Mean")
    ax.set_xlim(df["Time"].min(), df["Time"].max())
    ax.set_ylim(min(lcl, df["Value"].min()) - 1, max(ucl, df["Value"].max()) + 1)
    ax.set_title("Control Chart with Out-of-Control Areas")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    anim = FuncAnimation(
        fig, animate, frames=len(df), interval=100, blit=True, repeat=False
    )
    anim.save("control_chart_animation.gif", writer="pillow", fps=10)
    plt.close(fig)
