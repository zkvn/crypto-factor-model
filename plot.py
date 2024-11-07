import os
from random import randint
from datetime import datetime
from dateutil.relativedelta import relativedelta

import lib.data_gecko as data_gecko
import matplotlib.pyplot as plt


class Generate_Chart:
    def __init__(self, currency="usd"):
        self.currency = currency

    # ---------------------------------------------------------------
    #   Generate Line chart from Crypto prices from last 1 years
    # ---------------------------------------------------------------
    def generate_line_chart(self, token_list: list):

        obj = data_gecko.Crypto_Historic_Price(currency=self.currency)
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=1)

        crypto_price = {}
        for token in token_list:
            crypto_price[token] = obj.get_historic_price(token, start_date, end_date)

        for crypto, prices in crypto_price.items():
            x_axis = [ind[0] for ind in prices]  # Date
            y_axis = [ind[1] for ind in prices]  # Crypto Prices

            plt.plot(x_axis, y_axis, label=crypto)

        plt.legend()  # Show Legend
        plt.title("Crypto Line Chart", fontsize=18)  # Add Chart Title
        plt.xlabel("Date", fontsize=12)  # Add X-Axis
        plt.ylabel("Price", fontsize=12)  # Add Y-Axis
        # plt.show() # Show the chart on the screen

        rand_num = randint(100_000, 999_999)
        image_file = f"{os.getcwd()}/output/crpto_chart_{rand_num}.png"

        plt.savefig(image_file, bbox_inches="tight")  # Save the chart as an image


def generate_historic_chart():

    token_list = ["bitcoin", "ethereum", "tether", "usd-coin", "dogecoin"]

    obj = Generate_Chart()
    obj.generate_line_chart(token_list)


if __name__ == "__main__":
    generate_historic_chart()
    print("Done")
