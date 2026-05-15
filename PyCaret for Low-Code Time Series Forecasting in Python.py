"""Generated from Jupyter notebook: PyCaret for Time Series Forecasting in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# ! pip install pycaret[time_series]  # Jupyter-only


# --- code cell ---

import pandas as pd


def main():
    # Create a sample time series dataset
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

    # Convert to DataFrame
    df = data.to_frame()

    df.head()


    # --- code cell ---

    # !pip install --pre pycaret  # Jupyter-only


    # --- code cell ---

    import warnings

    warnings.filterwarnings("ignore")
    import logging

    logging.disable(logging.CRITICAL)


    # --- code cell ---

    from pycaret.time_series import setup

    # Initialize PyCaret
    s = setup(
        data=df,
        target="Sales",
        session_id=123,  # For reproducibility
        seasonal_period=12,  # Monthly data (12 periods per year)
    )


    # --- code cell ---

    import pandas as pd
    from pycaret.time_series import TSForecastingExperiment

    # Create a sample time series dataset
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

    # Convert to DataFrame
    df = data.to_frame()

    # Initialize the PyCaret Time Series Forecasting Experiment
    exp = TSForecastingExperiment()

    # Set up the experiment
    exp.setup(data=df, target="Sales", fold=3, seasonal_period=12)

    # Compare different models
    best_model = exp.compare_models()

    # Generate future forecasts
    future_forecasts = exp.predict_model(best_model, fh=12)  # Forecast next 12 months

    # Print the forecasts
    print(future_forecasts)


    # --- code cell ---

    # !pip install joblib==0.16.0  # Or another version that's known to work with your PyCaret version  # Jupyter-only


    # --- code cell ---

    import pandas as pd
    from pycaret.time_series import *

    # Create a sample time series dataset
    data = pd.Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10, name="Sales"
    )
    data.index = pd.date_range(start="2010-01-01", periods=len(data), freq="M")

    # Convert to DataFrame
    df = data.to_frame()

    # Initialize the PyCaret setup
    s = setup(data=df, target="Sales", train_size=0.8, fold=3, seasonal_period="M")

    # Compare different models
    best_model = compare_models()

    # Generate future forecasts
    future_forecasts = predict_model(best_model, fh=12)  # Forecast next 12 months

    # Print the forecasts
    print(future_forecasts)


    # --- code cell ---

    from pycaret.time_series import compare_models

    # Compare models
    best_model = compare_models()


    # --- code cell ---

    # ! pip install pycaret[full]  # Jupyter-only


    # --- code cell ---

    from pycaret.time_series import tune_model

    # Tune the best model
    tuned_model = tune_model(best_model)


    # --- code cell ---

    from pycaret.time_series import predict_model

    # Forecast the next 12 months
    future_forecast = predict_model(tuned_model, fh=12)  # fh = forecast horizon
    print(future_forecast)


    # --- code cell ---

    from pycaret.time_series import plot_model

    # Plot forecasts
    plot_model(tuned_model, plot="forecast")


    # --- code cell ---

    from pycaret.time_series import pull

    # Get backtesting metrics
    backtest_metrics = pull()
    print(backtest_metrics)


    # --- code cell ---

    # Inspect generated features
    print(s.get_config("X_train").head())


    # --- code cell ---

    from pycaret.time_series import stack_models

    # Stack the top 3 models
    stacked_model = stack_models(top_n=3)


    # --- code cell ---

    from pycaret.time_series import save_model

    # Save the model
    save_model(tuned_model, "time_series_model")


    # --- code cell ---

    from pycaret.time_series import load_model

    # Load the model
    loaded_model = load_model("time_series_model")


    # --- code cell ---

    # Add a secondary feature (e.g., marketing spend)
    df["Marketing_Spend"] = [50 + (i % 10) for i in range(len(df))]

    # Reinitialize setup for multivariate forecasting
    s = setup(data=df, target="Sales", session_id=123)


if __name__ == "__main__":
    main()
