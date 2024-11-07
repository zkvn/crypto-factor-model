"""
This module has the following functions
1. construct_crypto_index(): construct crptyo market index that uses adjusted market cap
2. construct_big_market_index(): construct a big market index that combines SPX index and crypto index
3. construct PCA factors using the following
    ['bitcoin_prices', 'usd-coin_prices', 'tether_prices', 'solana_prices',
       'chainlink_prices', 'ripple_prices', 'ethereum_prices',
       'binancecoin_prices', 'dogecoin_prices', 'SQ_Close', '^XAU_Close',
       'RIOT_Close', 'NVDA_Close', 'SPY_Close', 'spy', 'big_market',
       'crypto_market', '^TNX_Close', '^VIX_Close', '^IRX_Close', '^TYX_Close',
       '^FVX_Close', 'median_spread']
"""

import pandas as pd
from arch import arch_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def construct_PCA(df_pca_data, n=5, standardize_return=True):

    if standardize_return:
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(df_pca_data)
    else:
        scaled_returns = df_pca_data

    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(scaled_returns)
    pc_df = pd.DataFrame(data=principal_components, columns=[f"PC{i}" for i in range(1, n + 1)])

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance_ratio = explained_variance_ratio.sum()

    # Print results
    print(f"\nExplained Variance Ratio:\n{explained_variance_ratio}")
    print(f"Total Explained Variance Ratio: {total_explained_variance_ratio:.4f}")

    # Plot explained variance ratio
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(cumulative_variance_ratio, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Cumulative Explained Variance Ratio by Principal Components")
    plt.show()

    # print factor loadings
    df_factor_loadings = pd.DataFrame(pca.components_)
    df_factor_loadings.columns = df_pca_data.columns

    # Reconstruct scaled returns
    # reconstructed_data = np.dot(pc_df, df_factor_loadings)
    # df_reconstructed_data=pd.DataFrame(reconstructed_data)
    # df_reconstructed_data.columns = df_pca_data.columns
    # df_reconstructed_data.index = df_pca_data.index
    # df_reconstructed_data=df_reconstructed_data
    proj = pca.inverse_transform(pc_df)
    loss = np.sum((scaled_returns - proj) ** 2, axis=1).mean()
    print(f"PCA Projection Loss {loss}")

    return pc_df, df_factor_loadings


def check_PCA_construction(df_pca_source_data, pc_df, df_factor_loadings):
    """
    Generate projection and calculate loss
    If PCA has full dimension, the loss should be zero
    """
    reconstructed_data = np.dot(pc_df, df_factor_loadings)
    df_reconstructed_data = pd.DataFrame(reconstructed_data)
    df_reconstructed_data.columns = df_pca_source_data.columns
    df_reconstructed_data.index = df_pca_source_data.index
    loss = np.sum((df_pca_source_data - df_reconstructed_data) ** 2, axis=1).mean()
    print(f"PCA Projection Loss {loss}")
    return df_reconstructed_data


def adjust_mkt_cap_weight(a_series):
    """
    Adjust this series of market cap weight as of 2024-11-01:

    bitcoin_market_caps        0.669255
    usd-coin_market_caps       0.016967
    tether_market_caps         0.058664
    solana_market_caps         0.038095
    chainlink_market_caps      0.003437
    ripple_market_caps         0.014199
    ethereum_market_caps       0.147335
    binancecoin_market_caps    0.040686
    dogecoin_market_caps       0.011361

    Into this  ->


    bitcoin_market_caps        0.555556
    ethereum_market_caps       0.183391
    tether_market_caps         0.073021
    binancecoin_market_caps    0.050643
    solana_market_caps         0.047418
    chainlink_market_caps      0.037037
    usd-coin_market_caps       0.021119
    ripple_market_caps         0.017674
    dogecoin_market_caps       0.014141

    Take average weight as 1/n, where n is number of tokens in the crypto portfollio
    Set minimum to be 1/5 of the average weight
    And maximum to be 5 times the average wegiht
    Scale the rest
    """
    a_series = a_series.copy()
    n = len(a_series)
    a_series = a_series.sort_values()
    max_weight = 5 * (1 / n)
    min_weight = (1 / n) / 3
    other_total_weight = 1 - max_weight - min_weight
    scaler = other_total_weight / sum(a_series.iloc[1:-1])
    a_series.iloc[0] = min_weight
    a_series.iloc[-1] = max_weight
    a_series.iloc[1:-1] = a_series.iloc[1:-1] * scaler
    return a_series


def construct_crypto_index(df):
    """
    Use adjusted market cap weight to construct a crypto index using
    bitcoin, ethereum, tether, binancecoin, solana, chainlink, usd-coin, ripple and dogecoin
    """
    df1 = df.copy()
    mkt_cap_cols = []
    for i in df.columns:
        if "_market_caps" in i:
            mkt_cap_cols.append(i)
    # construct a weight column that is based on market cap
    df_cap = df1[mkt_cap_cols]
    total_market_cap_series = df_cap.sum(axis=1)
    df_weights = {}
    for i in mkt_cap_cols:
        df_weights[i] = df_cap[i] / total_market_cap_series
    df_weights = pd.DataFrame(df_weights)
    latest = df1[mkt_cap_cols].iloc[-1, :].sort_values(ascending=False)
    latest_weight = latest / latest.sum()

    # test the consturcted weights df are ok
    pd.testing.assert_series_equal(latest_weight, df_weights.iloc[-1].sort_values(ascending=False))

    # adjust the market cap weight, otherwise bitcoin weight is too big
    df_weights = df_weights.apply(adjust_mkt_cap_weight, axis=1)
    # construct crypto market cap weighted price index
    df_weighted_price = {}
    for i in df_weights.columns:
        coin = i.split("_")[0]
        price_column = f"{coin}_prices"
        df_weighted_price[i] = df_weights[i] * df1[price_column]

    df_weighted_price = pd.DataFrame(df_weighted_price)
    crypto_index = df_weighted_price.sum(1)
    return 100 * crypto_index / crypto_index.iloc[0]  # start from 100


def construct_big_market_index(df, crypto_index):
    """
    Assume in the future when crypto becomes just another equity like asset that a normal investment would consider
    Construct an indext that is 50% weighted in SPY and 50% in crypto market

    """
    s_spy = 100 * df["SPY_Close"] / df["SPY_Close"].iloc[0]
    s_big_market = 0.5 * s_spy + 0.5 * crypto_index
    return s_big_market


def get_all_returns(df, s_big_market, s_crypto_market):
    """prepare data for PCA
    Prices are converted into pct changes
    Interest rates, volatility and spreads are directly used
    """
    df = df.copy()
    cols = []
    level_tickers = [
        "^VIX",  # VIX index
        "^IRX",  # 13 week T-Bill
        "^FVX",  # 5 year Treasury
        "^TNX",  # 10 year Treasury
        "^TYX",  # 30 year Treasury
    ]
    for i in df.columns:  # find price columns, but ignore tickers are not prices
        if ("_prices" in i or "_Close" in i) and i.split("_")[0] not in level_tickers:
            cols.append(i)
    df_rets = df[cols].copy()
    df_rets["spy"] = df["SPY_Close"]
    df_rets["big_market"] = s_big_market
    df_rets["crypto_market"] = s_crypto_market
    df_rets = df_rets.pct_change()
    for i in df.columns:
        if i.split("_")[0] in level_tickers:
            df_rets[i] = df[i]
    spread_cols = []
    for i in df.columns:
        if "_spreads" in i and "Time_" not in i:
            spread_cols.append(i)
    df_rets["median_spread"] = df[spread_cols].fillna(0).apply(np.median, axis=1)  # bitcoin bid-ask spread among different exchanges
    return df_rets.dropna()
