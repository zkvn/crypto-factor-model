import pandas as pd
import glob
import os
from loguru import logger
import numpy as np

"""
functions for merge data and calculate historical return and volatilities
"""


def read_pickle_prices(yahoo_or_gecko="gecko", skip_columns=None, rename_columns=None):
    """Load downloaded CoinGecko data from 'data' folder and merge them into a big data frame"""
    pickles = glob.glob("./data/*.pkl")
    df1 = None
    for p in pickles:
        file_name = os.path.basename(p)
        if file_name.startswith(f"{yahoo_or_gecko}_"):
            ticker = file_name.split("_")[1]  # file convention
            logger.info(ticker)
            df2 = pd.read_pickle(p)

            # drop some columns if list skip_columns is not empty
            if skip_columns is not None:
                df2 = df2.drop(skip_columns, axis=1)

            # rename some columns if dict skip_columns is not empty
            if rename_columns is not None:
                df2 = df2.rename(columns=rename_columns)

            # add coin name to columns, e.g. bitcoin_prices, bitcoin_market_caps
            df2 = df2.rename(columns={k: f"{ticker}_{k}" for k in df2.columns})
            if df1 is None:
                df1 = df2
            else:
                df1 = df1.join(df2)
    return df1


def change_UTC_to_EST(df_UTC):
    """
    Yahoo prices are as of EST but they are downloaded by Pandas with a time UTC 00:00
    CoinGecko prices are as of UTC 00:00 but are downloaded using local timezone,  i.e. as of 08:00 HKT

    As a price from CoinGecko as of 2024-10-31 00:00 UTC is around 2024-10-30 8:00 PM EST
    The following adustments are made to use consistent time 00:00 UTC or around 8AM HKT

    1. remove timezone from yahoo and gecko price
    2. adjust gecko timestamp to EST by deducting 1 hour and 8 hours
    3. remove timezone from gecko prices

    Then the prices are as of EST close
    """
    df_UTC.index = df_UTC.index - pd.Timedelta(1, unit="d")
    df_UTC.index = df_UTC.index - pd.Timedelta(hours=8)
    df_UTC.index = df_UTC.index.tz_localize(None)
    return df_UTC


def merge_gecko_yahoo_bitcoinity():
    # merge yahoo and gecko
    df_yahoo = read_pickle_prices("yahoo", skip_columns=["Open", "High", "Low", "Close", "Volume"], rename_columns={"Adj Close": "Close"})
    df_yahoo.index = df_yahoo.index.tz_localize(None)
    df_gecko = read_pickle_prices("gecko")
    df_gecko = change_UTC_to_EST(df_gecko)
    df = pd.merge(df_gecko, df_yahoo, how="inner", left_index=True, right_index=True)

    # merge bitcoinity bid ask spred
    df_bid_ask = pd.read_csv("./data/bitcoinity_data.csv")
    df_bid_ask.index = pd.DatetimeIndex(df_bid_ask["Time"])
    df_bid_ask.index = df_bid_ask.index - pd.Timedelta(1, unit="d")
    df_bid_ask.index = df_bid_ask.index.tz_localize(None)
    df_bid_ask.columns = [f"{i}_spreads" for i in df_bid_ask.columns]
    df = pd.merge(df, df_bid_ask, how="inner", left_index=True, right_index=True)

    return df


def calc_hist_vol(df, ticker_price_column, window):
    """annualized historical volatility using daily percentage returns and sample stdev (pandas deault to sample stdev)"""
    ret = df[ticker_price_column].pct_change().dropna()
    vol = ret[-window:].std() * np.sqrt(252)
    return ret.mean() * 252, vol


def calc_hist_vol_all(df):
    df_result = pd.DataFrame()
    df_ret = pd.DataFrame()
    for days in [7, 30, 90, 252]:
        result = {}
        returns = {}
        for column in df.columns:
            if "prices" in column or "Close" in column and "^" not in column:
                ticker = column.split("_")[0]
                returns[ticker], result[ticker] = calc_hist_vol(df, column, days)
        df_result[days] = result
        df_ret[days] = returns
    return df_ret, df_result


def plot_prices_culm_change(df):
    columns = []
    for column in df.columns:
        if "prices" in column or "Close" in column and "^" not in column:
            columns.append(column)
    df[columns].pct_change().cumsum().plot()


def filter_price_columns(df):
    columns = []
    for column in df.columns:
        if "prices" in column or "Close" in column and "^" not in column:
            columns.append(column)
    return df[columns].pct_change().dropna()


def plot_volumes(df):
    columns = []
    for column in df.columns:
        if "volume" in column:
            columns.append(column)
    df[columns].plot()


def plot_mkt_cap(df):
    columns = []
    for column in df.columns:
        if "market_caps" in column:
            columns.append(column)
    df[columns].plot()


if __name__ == "__main__":

    df1 = merge_gecko_yahoo_bitcoinity()
    print(df1)
