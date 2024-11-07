import requests
import json
import conf as settings
import sys

url = "https://api.coingecko.com/api/v3/coins/list"

API_KEY = settings.API_KEY
headers = {"accept": "application/json", "x-cg-pro-api-key": "API_KEY"}

response = requests.get(url, headers=headers)
r_list = response.json()


def find_token_id(symbol="usdc"):
    for item in r_list:
        if symbol == item["symbol"]:
            print(item)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        symbol = args[0]
    else:
        symbol = "usdt"
    r = find_token_id(symbol)
    print(r)
