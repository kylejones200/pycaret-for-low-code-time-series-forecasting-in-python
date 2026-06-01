"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def animate(i):
    data = df.iloc[: i + 1]
    line.set_data(data["Time"], data["Value"])
    out_of_control = (data["Value"] > ucl) | (data["Value"] < lcl)
    ax.scatter(data["Time"][out_of_control], data["Value"][out_of_control], color="red")
    for j in range(i):
        if out_of_control.iloc[j]:
            ax.axvspan(
                data["Time"].iloc[j],
                data["Time"].iloc[j + 1],
                facecolor="red",
                alpha=0.1,
            )
    return (line, ucl_line, lcl_line, mean_line)


def notebook_step_001() -> None:
    "Generated from Jupyter notebook: annimations_and_pycaret\n\nMagics and shell lines are commented out. Run with a normal Python interpreter."


def assuming_you_have_your_taxi_data_in_a_dataframe() -> None:
    dates = pd.date_range(start="2024-01-15", end="2024-01-29", freq="D")
    np.random.seed(42)
    observed = (
        np.random.normal(500, 200, len(dates))
        + np.sin(np.arange(len(dates)) / 12) * 300
    )
    predicted = observed + np.random.normal(0, 50, len(dates))
    confidence = np.random.normal(50, 10, len(dates))
    df = pd.DataFrame(
        {
            "date": dates,
            "observed": observed,
            "predicted": predicted,
            "confidence": confidence,
        }
    )
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["observed"], color="red", label="Observed", linewidth=1)
    plt.plot(df["date"], df["predicted"], color="blue", label="Predicted", linewidth=1)
    plt.fill_between(
        df["date"],
        df["predicted"] - df["confidence"],
        df["predicted"] + df["confidence"],
        color="grey",
        alpha=0.2,
    )
    plt.title("Taxicab Pickup Count in Times Square by Time")
    plt.xlabel("Date")
    plt.ylabel("Pickup Count")
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
    expanding_window_fig.savefig(
        "expanding_window_cv.png", bbox_inches="tight", dpi=300
    )
    rolling_window_fig.savefig("rolling_window_cv.png", bbox_inches="tight", dpi=300)
    sliding_window_fig.savefig("sliding_window_cv.png", bbox_inches="tight", dpi=300)
    blocked_cv_fig.savefig("blocked_cv.png", bbox_inches="tight", dpi=300)
    plt.show()
    schemes = ["expanding", "rolling", "blocked"]
    for scheme in schemes:
        fig = plot_cv_scheme(scheme)
        fig.savefig(f"{scheme}_scheme.png", bbox_inches="tight", dpi=300)
        plt.close()


def plot_full_dataset() -> None:
    cv_viz = CrossValidationVisualizer(data_length=200)
    fig1 = cv_viz.plot_expanding_window(n_splits=3)
    fig2 = cv_viz.plot_rolling_window(window_size=40, step_size=20)
    fig3 = cv_viz.plot_sliding_window(window_size=40, horizon=10)
    fig1.savefig("expanding_window.png", bbox_inches="tight", dpi=300)
    fig2.savefig("rolling_window.png", bbox_inches="tight", dpi=300)
    fig3.savefig("sliding_window.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_full_dataset_2() -> None:
    cv_anim = AnimatedCrossValidation(data_length=200)
    expanding_anim = cv_anim.create_expanding_window_animation(n_splits=3)
    rolling_anim = cv_anim.create_rolling_window_animation(window_size=40, step_size=20)
    sliding_anim = cv_anim.create_sliding_window_animation(window_size=40, horizon=10)
    expanding_anim.save("expanding_window.gif", writer="pillow")
    rolling_anim.save("rolling_window.gif", writer="pillow")
    sliding_anim.save("sliding_window.gif", writer="pillow")
    plt.show()


def generate_two_related_time_series() -> None:
    visualizer = CausalInferenceVisualizer()
    anim = visualizer.create_animation()
    anim.save("causal_inference.gif", writer="pillow")
    plt.close()


def generate_sample_data() -> None:
    visualizer = TimeSeriesFeatureVisualizer()
    anim = visualizer.create_animation()
    anim.save("pytimetk_features.gif", writer="pillow", fps=10)
    plt.close()


def generate_true_trajectory() -> None:
    visualizer = KalmanFilterVisualizer()
    anim = visualizer.create_animation()
    anim.save("kalman_filter.gif", writer="pillow", fps=10)
    plt.close()


def generate_synthetic_time_series_data_for_differen() -> None:
    visualizer = TSClassificationVisualizer()
    anim = visualizer.create_animation()
    anim.save("ts_classification.gif", writer="pillow", fps=10)
    plt.close()


def pip_install_pmdarima_jupyter_only() -> None:
    # !pip install pmdarima  # Jupyter-only
    # pip install pmdarima  # Jupyter-only
    pass


def simulate_data_with_volatility_clustering() -> None:
    visualizer = ARCHVisualizer()
    anim = visualizer.create_animation()
    anim.save("arch_visualization.gif", writer="pillow", fps=5)
    plt.close()


def generate_rolling_forecasts() -> None:
    visualizer = ARCHVisualizer()
    anim = visualizer.create_animation()
    anim.save("arch_visualization.gif", writer="pillow", fps=5)
    plt.close()


def simulate_a_two_variable_system() -> None:
    visualizer = SVARVisualizer()
    anim = visualizer.create_animation()
    anim.save("svar_visualization.gif", writer="pillow", fps=5)
    plt.close()


def generate_sample_data_3() -> None:
    visualizer = BayesianARIMAVisualizer()
    anim = visualizer.create_animation()
    anim.save("bayesian_arima_comparison.gif", writer="pillow", fps=5)
    plt.close()


def main() -> None:
    notebook_step_001()
    assuming_you_have_your_taxi_data_in_a_dataframe()
    visualizing_time_series_cross_validation_methods()
    plot_full_dataset()
    plot_full_dataset_2()
    generate_two_related_time_series()
    set_random_seed_for_reproducibility()
    don_t_use_notebook_markdown()
    generate_sample_data()
    generate_sample_data_2()
    generate_larger_sample_data()
    generate_true_trajectory()
    generate_synthetic_time_series_data_for_differen()
    time_domain_parameters()
    notebook_step_015()
    create_sample_data()
    create_sample_data_2()
    create_sample_data_3()
    pip_install_pmdarima_jupyter_only()
    simulate_data_with_volatility_clustering()
    generate_rolling_forecasts()
    simulate_a_two_variable_system()
    set_random_seed_for_reproducibility_2()
    generate_sample_data_3()
