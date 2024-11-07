"""
This module contains functions for factor anlaysis

Reference
https://goldinlocks.github.io/ARCH_GARCH-Volatility-Forecasting/
https://www.kaggle.com/code/nholloway/volatility-clustering-and-garch
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
import numpy as np
import matplotlib.pyplot as plt
from numpy import log, polyfit, sqrt, std, subtract
from scipy.stats import probplot, moment
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.stattools as tsa
import pandas as pd
import statsmodels.api as sm


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x) / 5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f"Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}"
    axes[0][0].text(x=0.02, y=0.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f"Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}"
    axes[0][1].text(x=0.02, y=0.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel("Lag")
    axes[1][1].set_xlabel("Lag")
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)


def plot_returns_corr(df, s_big_market):
    df = df.copy()
    cols = []
    for i in df.columns:  # find coin columns
        if "_prices" in i:
            cols.append(i)
    df_rets = df[cols].copy()
    df_rets["spy"] = df["SPY_Close"]
    df_rets["big_market"] = s_big_market
    df_rets = df_rets.pct_change().dropna()

    return df_rets


def test_ljung_box(returns):
    ljung_res = acorr_ljungbox(returns, lags=40, boxpierce=True)
    sig = ljung_res[ljung_res["bp_pvalue"] < 0.05]
    print(sig)


def adf_test(series):
    p = tsa.adfuller(series)[1]
    if p < 0.01:
        print(f"The series is stationary, with p-value from ADF {p}")
    else:
        print(f"The series is NOT stationary, with p-value from ADF {p}")


def find_volitile_month(market_returns, conditional_volatility, state="high vol"):
    """
    Classify market_returns using GARCH conditional_volatility into 4 markets

    Use 75 percentile as high volatility threshold

    Use 23 percentile as low volatility threshold

    Return the day counts by month

    """

    def market_condition(row):
        if row["rolling_returns"] > 0 and row["conditional_volatility"] > threshold_high:
            return "Bull Market - High Volatility"
        elif row["rolling_returns"] > 0 and row["conditional_volatility"] < threshold_low:
            return "Bull Market - Low Volatility"
        elif row["rolling_returns"] < 0 and row["conditional_volatility"] > threshold_high:
            return "Bear Market - High Volatility"
        elif row["rolling_returns"] < 0 and row["conditional_volatility"] < threshold_low:
            return "Bear Market - Low Volatility"
        else:
            return "Neutral"

    threshold_high = conditional_volatility.quantile(0.75)
    threshold_low = conditional_volatility.quantile(0.25)
    df_mkt = pd.DataFrame({"market_returns": market_returns})
    df_mkt["rolling_returns"] = df_mkt["market_returns"].rolling(window=20).mean()
    df_mkt["conditional_volatility"] = conditional_volatility
    df_mkt["market_condition"] = df_mkt.apply(market_condition, axis=1)

    if state == "high vol":
        mask = (df_mkt["market_condition"] == "Bear Market - High Volatility") | (df_mkt["market_condition"] == "Bull Market - High Volatility")
    elif state == "low vol":
        mask = (df_mkt["market_condition"] == "Bear Market - Low Volatility") | (df_mkt["market_condition"] == "Low Market - High Volatility")

    s_index_high_vol = df_mkt[mask].index
    b = pd.DataFrame(s_index_high_vol)
    b = b.set_index(0)
    b["count"] = 1
    r = b.groupby(pd.Grouper(freq="ME"))["count"].sum()
    return df_mkt, r


def run_regression(df1, x_column, y_column, plot_chart=True):
    X_data = df1[x_column].copy()
    y = df1[y_column].copy()
    # Add a constant to the independent variable (for the intercept)
    X = sm.add_constant(X_data)
    model = sm.OLS(y, X).fit()

    if plot_chart:
        print(model.summary())
        plt.scatter(X_data, y, color="blue", label="Data Points")
        plt.plot(X_data, model.predict(X), color="red", label="Regression Line")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"Linear Regression between {x_column} returns and {y_column} returns")
        plt.legend()
        plt.show()
    return model


def regress_all_macro_factors(df_rets):
    cryptos = [i for i in df_rets.columns if "_prices" in i]
    macro_factors = ["SPY_Close", "RIOT_Close", "NVDA_Close", "^IRX_Close", "^TNX_Close", "^XAU_Close"]
    result = {}
    result_sig = {}
    for macro_f in macro_factors:
        r = {}
        r_sig = {}
        for coin in cryptos:
            m = run_regression(df_rets, x_column=macro_f, y_column=coin, plot_chart=False)
            sensitivity = m.params[macro_f]
            p_value = m.pvalues[macro_f]
            current_result = {f"sensitivity_{coin}": sensitivity, f"p_value_{coin}": p_value}
            r = r | current_result
            if p_value < 0.01:
                r_sig = r_sig | current_result
        result[macro_f] = r
        result_sig[macro_f] = r_sig
    return pd.DataFrame(result), pd.DataFrame(result_sig)


def identify_factor_contribution(df_rets):
    """
    run linear regression iterative using
    PC1
    PC1 and PC2
    PC1, PC2 and PC3
    And calculte the R-Square change for each asset
    """
    result = {}
    for asset in df_rets.columns:
        if "_prices" not in asset:
            continue
        m_full = run_regression(df_rets, ["PC1", "PC2", "PC3"], asset, plot_chart=False)
        r_square_full = m_full.rsquared

        m = run_regression(df_rets, ["PC2", "PC3"], asset, plot_chart=False)
        pc1 = r_square_full - m.rsquared

        m = run_regression(df_rets, ["PC1", "PC3"], asset, plot_chart=False)
        pc2 = r_square_full - m.rsquared

        m = run_regression(df_rets, ["PC1", "PC2"], asset, plot_chart=False)
        pc3 = r_square_full - m.rsquared
        result[asset] = {"R2_3 PCs": r_square_full, "PC1": pc1, "PC2": pc2, "PC3": pc3}
    return pd.DataFrame(result)


def restore_rets(df_factor_loadings, pc_df, df_rets, n=3):
    proj = np.dot(pc_df.iloc[:, :n], df_factor_loadings[:n])
    proj = pd.DataFrame(proj)
    proj.columns = df_factor_loadings.columns
    proj.index = df_rets.index
    stds = df_rets.std()
    means = df_rets.mean()
    restored = proj.mul(stds) + means
    return restored


def shock_from_pca(df_factor_loadings, df_rets, component, score):
    """
    Give a score of a component, what's the corresponding return?

    To reproduce the 3 factor estimated return of bitcoin on 2024-11-01, the bitcoin's result should be -0.003859
    
    shock_from_pca(df_factor_loadings0, df_rets0, component=1, score=-0.492307) +\
    shock_from_pca(df_factor_loadings0, df_rets0, component=2, score=-0.924682) +\
    shock_from_pca(df_factor_loadings0, df_rets0, component=3, score=0.313774)  + df_rets0.mean()    
    """
    df_rets = df_rets.copy()
    df_factor_loadings = df_factor_loadings.copy()
    loadings = df_factor_loadings.loc[component - 1]  # first component starts from index 0
    proj = score * loadings
    restored = proj * df_rets.std()
    return restored


def stress_calc(shocks, df_factor_loadings, df_rets):
    """
    shocks ={'PC1': 5, "PC2':7}

    stress_calc({'PC1': -0.492307, "PC2":-0.924682, 'PC3':0.313774}, df_factor_loadings0,df_rets0)

    result for bitcoin should be -0.003859
    """
    result = 0
    for pc, shock in shocks.items():
        component = int(pc[-1])
        result += shock_from_pca(df_factor_loadings, df_rets, component, shock)
    result = result + df_rets.mean()
    return result
