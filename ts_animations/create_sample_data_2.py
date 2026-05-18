"""Auto-split from legacy monolithic script."""

import pandas as pd
from pycaret.time_series import (
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


def create_sample_data_2() -> None:
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")
    df = data.to_frame()
    setup(data=df, target="Sales", session_id=123, seasonal_period=12)
    best_model = compare_models()
    tuned_model = tune_model(best_model)
    future_forecast = predict_model(tuned_model, fh=12)
    print("\nForecast:")
    print(future_forecast)
    plot_model(tuned_model, plot="forecast")
    metrics = pull()
    print("\nMetrics:")
    print(metrics)
    stack_models(top_n=3)
    save_model(tuned_model, "time_series_model")
    load_model("time_series_model")
    df["Marketing_Spend"] = [50 + i % 10 for i in range(len(df))]
    setup(data=df, target="Sales", session_id=123)
    best_multivariate_model = compare_models()
    future_forecast_multivariate = predict_model(best_multivariate_model, fh=12)
    print("\nMultivariate forecast:")
    print(future_forecast_multivariate)
    plot_model(best_multivariate_model, plot="forecast")
