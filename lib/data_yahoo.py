import yfinance as yf
from loguru import logger
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime


def get_price(ticker):
    """
    Use yahoo finance python library to get historical data for a ticker
    start date and end date are set as past 1 year
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=1)
    df = yf.download(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df.columns = df.columns.to_flat_index()
    df.columns = [s[0] for s in df.columns]
    df.to_pickle(f"./data/yahoo_{ticker}_ohlc.pkl")


def run():
    for ticker in [
        "SPY",  # SPY
        "^VIX",  # VIX index
        "^XAU",  # Gold price
        "^IRX",  # 13 week T-Bill
        "^FVX",  # 5 year Treasury
        "^TNX",  # 10 year Treasury
        "^TYX",  # 30 year Treasury
        "RIOT",  # RIOT platform
        "NVDA",  # graphic cards
        "SQ",  # block
    ]:
        logger.info(f"Downloading {ticker} data from yahoo")
        get_price(ticker)


if __name__ == "__main__":
    run()
