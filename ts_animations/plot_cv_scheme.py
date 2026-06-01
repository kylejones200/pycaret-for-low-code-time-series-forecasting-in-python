"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_cv_scheme(cv_type="expanding", n_splits=4):
    """Create schematic representation of CV schemes"""
    fig, ax = plt.subplots(figsize=(12, 4))
    total_length = 100
    height = 1
    y_positions = np.arange(n_splits) * height * 1.5
    if cv_type == "expanding":
        initial_size = total_length // (n_splits + 1)
        for i in range(n_splits):
            train_end = initial_size + i * initial_size
            train_end + initial_size
            ax.add_patch(
                Rectangle(
                    (0, y_positions[i]), train_end, height, facecolor="blue", alpha=0.3
                )
            )
            ax.add_patch(
                Rectangle(
                    (train_end, y_positions[i]),
                    initial_size,
                    height,
                    facecolor="red",
                    alpha=0.3,
                )
            )
    elif cv_type == "rolling":
        window_size = total_length // 3
        step_size = window_size // 2
        for i in range(n_splits):
            start = i * step_size
            train_end = start + window_size
            train_end + step_size
            ax.add_patch(
                Rectangle(
                    (start, y_positions[i]),
                    window_size,
                    height,
                    facecolor="blue",
                    alpha=0.3,
                )
            )
            ax.add_patch(
                Rectangle(
                    (train_end, y_positions[i]),
                    step_size,
                    height,
                    facecolor="red",
                    alpha=0.3,
                )
            )
    elif cv_type == "blocked":
        block_size = total_length // n_splits
        for i in range(n_splits):
            test_block = i
            for j in range(n_splits):
                if j != test_block:
                    ax.add_patch(
                        Rectangle(
                            (j * block_size, y_positions[i]),
                            block_size,
                            height,
                            facecolor="blue",
                            alpha=0.3,
                        )
                    )
            ax.add_patch(
                Rectangle(
                    (test_block * block_size, y_positions[i]),
                    block_size,
                    height,
                    facecolor="red",
                    alpha=0.3,
                )
            )
    ax.set_xlim(-5, total_length + 5)
    ax.set_ylim(-height, max(y_positions) + height * 2)
    ax.set_title(f"{cv_type.capitalize()} Window Cross-Validation Scheme")
    ax.set_xlabel("Time")
    ax.set_yticks(y_positions + height / 2)
    ax.set_yticklabels([f"Split {i + 1}" for i in range(n_splits)])
    ax.add_patch(
        Rectangle(
            (0, max(y_positions) + height),
            20,
            height,
            facecolor="blue",
            alpha=0.3,
            label="Training",
        )
    )
    ax.add_patch(
        Rectangle(
            (25, max(y_positions) + height),
            20,
            height,
            facecolor="red",
            alpha=0.3,
            label="Testing",
        )
    )
    plt.legend()
    plt.tight_layout()
    return fig
