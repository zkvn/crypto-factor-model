# Overview
This is a demo of using factor model analyze crypto market in 2023-2024.
- GARCH model and K-means for volatility cluster
- PCA factor analysis and stress testing
- CAPM style cross market model



# Installation
1. Clone this git to a local folder and CD into it
2. Create a venv folder for all dependencies
`python -m venv "venv"`
3. Activate it 
`source venv/bin/activate`  (linux/mac)
`./venv/scripts/activate`   (Windows)
4. Install the dependencies
`python -m pip install -r requirements.txt`
5. Setup Jupyter Notebook Environment
```
source venv/bin/activate
pip install jupyter
pip install ipython && pip install ipykernel
ipython kernel install --user --name=factor-model
python -m ipykernel install --user --name=factor-model
```
6. Select the kernel factor-model in Jupyter
7. Open one of the notebooks and take a look


# Factor Model
- Market factor: a market cap weighted index of SPY and a few selected crypto
- Macro factors
    - VIX Implied volaility of SPX options
    - XAU Gold price 
    - DXY Dollar index
    - Interest Rates
- (TODO )Momentum factor: weekly returns of a winner crypto portfolio mins a loser portfolio

# Future study
- Machine learning approach cross-validation for some parameters
- Identify regime change 
- Use OHLC to estimate volatility
- Use high frequency data, like hourly 
- Trading volume impact on volatility
- Liquidity bid-ask spread as a risk factor from different exchanges for different coins
- Machine learning approach using non-time series data 


# Data Naming Convention
In the data folder, data file is named after a source, ticker and data content type, e.g. "yahoo_SPY_ohlc.pkl"
This pickle file is from yahoo, the ticker is SPY, and it has OHLC data


# Error Handling
1. In lib.data_gecko.py file, if response from CoinGecko doesn't have status_code 200 for a coin, then it will skip to next coin


# Reference
- [Get Historical MarketPrice from CoinGecko in Python](https://github.com/kirancshet/Get-Crypto-Historic-Chart-CoinGecko)
- [ARCH/GARCH Volatility Forecasting in Python](https://goldinlocks.github.io/ARCH_GARCH-Volatility-Forecasting/)
- [PCA projection and reconstruction in scikit-learn](https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn)
- [Applicability Analysis of Cryptocurrency Market Based on Capital Asset Pricing Model, Yuhan Liu](https://www.researchgate.net/publication/377732492_Applicability_Analysis_of_Cryptocurrency_Market_Based_on_Capital_Asset_Pricing_Model)
- [Quantifying bitcoin patterns with PCA](https://www.coinbase.com/en-gb/institutional/research-insights/research/monthly-outlook/quantifying-bitcoin-patterns-with-pca-may-2023)
- [Detail for Monthly Momentum Factor (Mom)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html)
- [Factor Structure in Cryptocurrency Return and Volatility](https://www.researchgate.net/publication/352550633_Factor_Structure_in_Cryptocurrency_Return_and_Volatility)
- [A Detailed Case Study on Crypto Multi-factor Risk Analysis, Numerai](https://forum.numer.ai/t/a-detailed-case-study-on-crypto-multi-factor-risk-analysis/7682)
- [A three-factor pricing model for cryptocurrencies, Shen, D., Urquhart, A](https://centaur.reading.ac.uk/85321/3/A%20Three-factor%20Pricing%20Model%20for%20Cryptocurrency_R1.pdf)
- [A Detailed Case Study on Crypto Multi-factor Risk Analysis – by Ahan M R](https://drive.google.com/file/d/1378ZJbdqqP2DBlrPS1Pg6oHQYD7hyB9j/view?usp=sharing)
- [A Factor Model for Cryptocurrency Returns, PCA, Daniele Bianchi, Mykola Babiak](http://wp.lancs.ac.uk/fofi2022/files/2022/08/FoFI-2022-056-Daniele-Bianchi.pdf)
- [Crypto Model Factoring: Data Challenge Podium](https://blog.oceanprotocol.com/crypto-model-factoring-data-challenge-podium-b56950db6f43)
- [How to build a factor model?](https://quant.stackexchange.com/questions/17125/how-to-build-a-factor-model)
- [Lecture 12: Factor Pricing, Prof. Markus K. Brunnermeier, Princeton](https://www.princeton.edu/~markus/teaching/Fin501/12Lecture.pdf)
- [Decoding Bitcoin: leveraging macro- and micro-factors in time series analysis for price prediction](https://peerj.com/articles/cs-2314.pdf)
- [Bitcoin price prediction based on fear & greed index](https://www.shs-conferences.org/articles/shsconf/pdf/2024/01/shsconf_icdeba2023_02015.pdf)
- Capital Asset Pricing Model (CAPM)
- Fama–French three-factor model
- Fama and MacBeth regressions
- Arbitrage pricing theory
- Barra Factor Model