"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model


def set_random_seed_for_reproducibility_2() -> None:
    np.random.seed(42)
    n = 1000
    omega = 0.1
    alpha = 0.8
    errors = np.random.normal(size=n)
    volatility = np.zeros(n)
    returns = np.zeros(n)
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * errors[t - 1] ** 2)
        returns[t] = volatility[t] * np.random.normal()

    data = pd.DataFrame({"returns": returns, "volatility": volatility})
    data.plot(subplots=True, figsize=(10, 6), title="Simulated Returns and Volatility")
    plt.savefig("returns_and_volatility.png")
    plt.show()
    arch_model_fit = arch_model(data["returns"], vol="ARCH", p=1).fit()
    print(arch_model_fit.summary())
    forecast = arch_model_fit.forecast(horizon=10)
    forecast_variance = forecast.variance.iloc[-1]
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_variance, marker="o", label="Forecasted Variance")
    plt.title("Forecasted Volatility")
    plt.xlabel("Horizon")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid()
    plt.savefig("forecasted_volatility.png")
    plt.show()
