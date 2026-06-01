"""Auto-split from legacy monolithic script."""

import pandas as pd
from pycaret.time_series import TSForecastingExperiment


def create_sample_data_3() -> None:
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")
    df = data.to_frame()
    exp = TSForecastingExperiment()
    exp.setup(data=df, target="Sales", session_id=123, seasonal_period=12)
    best_model = exp.compare_models()
    tuned_model = exp.tune_model(best_model)
    future_forecast = exp.predict_model(tuned_model, fh=12)
    print("\nForecast:")
    print(future_forecast)
    exp.plot_model(tuned_model, plot="forecast")
    metrics = exp.pull()
    print("\nMetrics:")
    print(metrics)
    exp.save_model(tuned_model, "time_series_model")
    exp.load_model("time_series_model")
    df["Marketing_Spend"] = [50 + i % 10 for i in range(len(df))]
    exp_multi = TSForecastingExperiment()
    exp_multi.setup(data=df, target="Sales", session_id=123)
    best_multivariate_model = exp_multi.compare_models()
    future_forecast_multivariate = exp_multi.predict_model(
        best_multivariate_model, fh=12
    )
    print("\nMultivariate forecast:")
    print(future_forecast_multivariate)
    exp_multi.plot_model(best_multivariate_model, plot="forecast")
