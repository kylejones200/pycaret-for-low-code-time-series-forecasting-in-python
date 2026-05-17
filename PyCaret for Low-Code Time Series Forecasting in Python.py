"""Generated from Jupyter notebook: PyCaret for Time Series Forecasting in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import logging
import warnings

import pandas as pd
from pycaret.time_series import *
from pycaret.time_series import (
    TSForecastingExperiment,
    compare_models,
    load_model,
    plot_model,
    predict_model,
    pull,
    save_model,
    setup,
    stack_models,
    tune_model,
)


def create_a_sample_time_series_dataset() -> None:
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )

    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

    df = data.to_frame()

    df.head()


def notebook_step_004() -> None:
    warnings.filterwarnings("ignore")

    logging.disable(logging.CRITICAL)


def initialize_pycaret() -> None:
    s = setup(data=df, target="Sales", session_id=123, seasonal_period=12)


def create_a_sample_time_series_dataset_2() -> None:
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )

    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

    df = data.to_frame()

    exp = TSForecastingExperiment()

    exp.setup(data=df, target="Sales", fold=3, seasonal_period=12)

    best_model = exp.compare_models()

    future_forecasts = exp.predict_model(best_model, fh=12)

    print(future_forecasts)


def create_a_sample_time_series_dataset_3() -> None:
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )

    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

    df = data.to_frame()

    s = setup(data=df, target="Sales", train_size=0.8, fold=3, seasonal_period="M")

    best_model = compare_models()

    future_forecasts = predict_model(best_model, fh=12)

    print(future_forecasts)


def compare_models() -> None:
    best_model = compare_models()


def tune_the_best_model() -> None:
    tuned_model = tune_model(best_model)


def forecast_the_next_12_months() -> None:
    future_forecast = predict_model(tuned_model, fh=12)

    print(future_forecast)


def plot_forecasts() -> None:
    plot_model(tuned_model, plot="forecast")


def get_backtesting_metrics() -> None:
    backtest_metrics = pull()

    print(backtest_metrics)


def inspect_generated_features() -> None:
    print(s.get_config("X_train").head())


def stack_the_top_3_models() -> None:
    stacked_model = stack_models(top_n=3)


def save_the_model() -> None:
    save_model(tuned_model, "time_series_model")


def load_the_model() -> None:
    loaded_model = load_model("time_series_model")


def add_a_secondary_feature_e_g_marketing_spend() -> None:
    df["Marketing_Spend"] = [50 + i % 10 for i in range(len(df))]

    s = setup(data=df, target="Sales", session_id=123)


def main() -> None:
    create_a_sample_time_series_dataset()
    notebook_step_004()
    initialize_pycaret()
    create_a_sample_time_series_dataset_2()
    create_a_sample_time_series_dataset_3()
    compare_models()
    tune_the_best_model()
    forecast_the_next_12_months()
    plot_forecasts()
    get_backtesting_metrics()
    inspect_generated_features()
    stack_the_top_3_models()
    save_the_model()
    load_the_model()
    add_a_secondary_feature_e_g_marketing_spend()


if __name__ == "__main__":
    main()
