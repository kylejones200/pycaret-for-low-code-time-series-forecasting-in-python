"""Generated from Jupyter notebook: PyCaret for Time Series Forecasting in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import pandas as pd


def main():
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")
    df = data.to_frame()
    df.head()
    import logging

    logging.disable(logging.CRITICAL)
    from pycaret.time_series import setup

    s = setup(data=df, target="Sales", session_id=123, seasonal_period=12)
    import pandas as pd
    from pycaret.time_series import TSForecastingExperiment

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
    import pandas as pd
    from pycaret.time_series import *

    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")
    df = data.to_frame()
    s = setup(data=df, target="Sales", train_size=0.8, fold=3, seasonal_period="M")
    best_model = compare_models()
    future_forecasts = predict_model(best_model, fh=12)
    print(future_forecasts)
    from pycaret.time_series import compare_models

    best_model = compare_models()
    from pycaret.time_series import tune_model

    tuned_model = tune_model(best_model)
    from pycaret.time_series import predict_model

    future_forecast = predict_model(tuned_model, fh=12)
    print(future_forecast)
    from pycaret.time_series import plot_model

    plot_model(tuned_model, plot="forecast")
    from pycaret.time_series import pull

    backtest_metrics = pull()
    print(backtest_metrics)
    print(s.get_config("X_train").head())
    from pycaret.time_series import stack_models

    stacked_model = stack_models(top_n=3)
    from pycaret.time_series import save_model

    save_model(tuned_model, "time_series_model")
    from pycaret.time_series import load_model

    loaded_model = load_model("time_series_model")
    df["Marketing_Spend"] = [50 + i % 10 for i in range(len(df))]
    s = setup(data=df, target="Sales", session_id=123)


def main() -> None:
    main()


if __name__ == "__main__":
    main()
