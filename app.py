import lib.data_gecko as data_gecko
import lib.data_yahoo as data_yahoo
import lib.data_preprocessing as preprocessing
import lib.factor_creation as factor_creation
from loguru import logger


def download_data():
    logger.info("Starting to download data")
    data_gecko.run()
    data_yahoo.run()


def preprocess_data():
    logger.info("Starting to preprocess data")
    df1 = preprocessing.merge_gecko_yahoo_bitcoinity()
    df_hist_ret, df_vol = preprocessing.calc_hist_vol_all(df1)
    # preprocessing.plot_prices_culm_change(df1)
    # preprocessing.plot_volumes(df1)
    # preprocessing.plot_mkt_cap(df1)
    return df1, df_hist_ret, df_vol


def construct_market_factors(df1):
    # df1, df_ret, df_vol = preprocess_data()
    s_crypto_market = factor_creation.construct_crypto_index(df1)
    s_big_market = factor_creation.construct_big_market_index(df1, s_crypto_market)
    return s_crypto_market, s_big_market


def construct_pca_factors(df1, s_big_market, s_crypto_market, n=5, standardize_return=True):
    # df1, df_ret, df_vol = preprocess_data()
    # s_crypto_market, s_big_market = construct_market_factors(df1)
    df_rets = factor_creation.get_all_returns(df1, s_big_market, s_crypto_market)
    pc_df, df_factor_loadings = factor_creation.construct_PCA(df_rets, n=n, standardize_return=standardize_return)
    return df_rets, pc_df, df_factor_loadings


if __name__ == "__main__":

    download_data()
    # preprocess_data()

    logger.info("Done")
