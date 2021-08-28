from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class Passive:

    # def __init__(self):

    @staticmethod
    def get_first_month(self):
        df = pd.read_csv('/files/NAFTRAC_20180131.csv')
        df = df.loc[:, ['Ticker', 'Peso (%)']]
        df.dropna(subset=['Peso (%)'], inplace=True)
        df['Ticker'] = df['Ticker'].str.replace('*', '')
        df_tickers = list(df.Ticker)
        df.set_index('Ticker', inplace=True)
        for i in df_tickers:
            try:
                stock = i + ".MX"
                df.loc[i, 'Price'] = yf.download(stock,
                                                 start='2018-01-31',
                                                 progress=False
                                                 ).loc[:, 'Adj Close'][0]
            except:
                pass
        return print(df)

