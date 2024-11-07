import requests
import json
import time
from datetime import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import conf as settings
from loguru import logger
import pandas as pd


class Crypto_Historic_Price:
    """
    Simple class for downloading data using coingecko API
    Refer to the page below for details
    https://docs.coingecko.com/reference/introduction
    """

    def __init__(self, currency="usd"):
        self.currency = currency.lower()
        self.end_date = datetime.now()
        self.start_date = self.end_date - relativedelta(years=1)
        # self.base_url = r"https://pro-api.coingecko.com/api/v3"  # Premium Account API URL
        self.base_url = r"https://api.coingecko.com/api/v3"  # Free Account

    def populate_date(self, date1):
        """Convert Date to UNIX Timestamp"""
        if isinstance(date1, datetime):
            return int(time.mktime(date1.timetuple()))
        elif isinstance(date1, str):
            date2 = parse(date1)
            return int(time.mktime(date2.timetuple()))
        return None

    def get_historic_price(self, token: str):
        """
        Get historical chart data for past 1 year
        Refer to the page below for details
        https://docs.coingecko.com/reference/coins-id-market-chart-range
        """
        url = f"{self.base_url}/coins/{token}/market_chart/range"

        data = {}  # Populate Parameters
        data["vs_currency"] = self.currency
        data["from"] = self.populate_date(self.start_date)
        data["to"] = self.populate_date(self.end_date)

        headers = {}
        headers["Content-type"] = "application/json"
        headers["x_cg_pro_api_key"] = settings.API_KEY
        response = requests.get(url, headers=headers, params=data)

        if response.status_code != 200:
            logger.error(f"CoinGecko status error {response.status_code} ")
            return
        info = response.json()
        result = {}
        for key in info.keys():
            s = pd.Series()
            for ind in info[key]:
                date1 = datetime.fromtimestamp(ind[0] / 1000.0)
                s[date1] = ind[1]
            result[key] = s

        return pd.DataFrame(result)

    def run(self):
        for token in ["binancecoin", "tether", "ripple", "dogecoin", "usd-coin", "bitcoin", "ethereum", "solana", "chainlink"]:
            logger.info(f"Download {token} data from CoinGecko")
            df = self.get_historic_price(token)
            if df is not None:  # download failed
                df.to_pickle(f"./data/gecko_{token}_price_volume_marketcap.pkl")


def run():
    obj = Crypto_Historic_Price(currency="USD")
    obj.run()


if __name__ == "__main__":
    run()
    logger.info("Done")
