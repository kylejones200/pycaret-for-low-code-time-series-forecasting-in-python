# Description: Short example for PyCaret for Low Code Time Series Forecasting in Python.

import logging

import pandas as pd


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Create a sample time series dataset
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")
    # Convert to DataFrame
    df = data.to_frame()
    logger.info(df.head())
    # Initialize time series experiment
    exp = TSForecastingExperiment()
    # Setup the environment
    exp.setup(data=df, target="Sales", session_id=123, seasonal_period=12)
    # Compare models
    best_model = exp.compare_models()
    # Tune the best model
    tuned_model = exp.tune_model(best_model)
    # Make predictions
    future_forecast = exp.predict_model(tuned_model, fh=12)
    logger.info("\nForecast:")
    logger.info(future_forecast)
    # Plot results
    exp.plot_model(tuned_model, plot="forecast")
    # Get metrics
    metrics = exp.pull()
    logger.info("\nMetrics:")
    logger.info(metrics)
    # Save and load model
    exp.save_model(tuned_model, "time_series_model")
    exp.load_model("time_series_model")
    # Multivariate analysis
    df["Marketing_Spend"] = [50 + (i % 10) for i in range(len(df))]
    # New experiment for multivariate analysis
    exp_multi = TSForecastingExperiment()
    exp_multi.setup(data=df, target="Sales", session_id=123)
    best_multivariate_model = exp_multi.compare_models()
    future_forecast_multivariate = exp_multi.predict_model(
        best_multivariate_model, fh=12
    )
    logger.info("\nMultivariate forecast:")
    logger.info(future_forecast_multivariate)
    exp_multi.plot_model(best_multivariate_model, plot="forecast")


if __name__ == "__main__":
    main()
