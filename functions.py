from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import datetime


class Passive:

    def __init__(self):
        pass

    @staticmethod
    def read_file() -> pd.DataFrame:
        df = pd.read_csv('~\\Documents\\GitHub\\MyST\\files\\NAFTRAC_20180131.csv', skiprows=2)
        df = df.loc[:, ['Ticker', 'Peso (%)']]
        df.dropna(subset=['Peso (%)'], inplace=True)
        df['Ticker'] = df['Ticker'].str.replace('*', '', regex = False)
        df['Ticker'] = df['Ticker'].str.replace('MEXCHEM', 'ORBIA')
        df['Ticker'] = df['Ticker'].str.replace('LIVEPOLC.1', 'LIVEPOLC-1', regex = False)
        df['Ticker'] = df['Ticker'].str.replace('GFREGIOO', 'RA')
        df.set_index('Ticker', inplace=True)
        df.drop(['KOFL', 'MXN', 'BSMXB'], inplace=True)
        return df

    def get_historical(self) -> pd.DataFrame:
        df = self.read_file()
        df_tickers = df.index.to_list()
        historical = pd.DataFrame()
        for i in df_tickers:
            try:
                stock = i + ".MX"
                historical[i] = yf.download(stock, start='2018-01-31', progress=False).loc[:, 'Adj Close']
            except:
                pass

        holiday = ['2018-03-30', '2018-03-29']
        end_date = datetime.datetime.today()
        start_date = datetime.datetime(2018, 1, 1)
        periods = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        es_holiday = pd.tseries.offsets.CustomBusinessMonthEnd(holidays=holiday)
        s = pd.date_range('2018-01-01', periods=periods, freq=es_holiday)
        historical = historical.reindex(index=s)

        return historical

    def get_passive_table(self) -> pd.DataFrame:
        historic = self.get_historical()
        historic_ = pd.DataFrame(historic.iloc[0, :])
        historic_.columns = (['Price'])
        weights = self.read_file()
        historic_ = pd.concat([historic_, weights], axis=1)
        historic_['Comision_por_Accion'] = historic_['Price'] * 0.00125
        presupuesto = 1000000
        historic_['Acciones'] = (presupuesto * (historic_['Peso (%)']) / 100) / (
                    historic_['Price'] + historic_['Comision_por_Accion'])
        historic_['Acciones'] = historic_['Acciones'].apply(np.floor)
        historic_['Capital'] = np.round(historic_['Acciones'] * (historic_['Price'] + historic_['Comision_por_Accion']),
                                        2)

        # CASH
        historic_.loc['CASH'] = [100 - sum(historic_['Peso (%)']), 0, 0, 0,
                                 ((100 - sum(historic_['Peso (%)'])) / 100) * presupuesto]

        for i in historic.columns:
            historic['Capital_' + i] = np.round(
                historic_['Acciones'][i] * (historic[i] + historic_['Comision_por_Accion'][i]), 2)

        historic['Capital_CASH'] = historic_['Capital']['CASH']
        num_acc = len(historic_) - 1
        historic['Capital_Total'] = historic.iloc[:, num_acc:].sum(axis=1)

        # Rend
        historic['Rend'] = historic['Capital_Total'].pct_change()
        historic.fillna(0, inplace=True)

        return historic.iloc[:, -2:]

    def get_pre_pandemic(self) -> pd.DataFrame:
        df = self.get_passive_table()
        df = df.iloc[:25]
        df['Rend_accum'] = (df['Capital_Total'] - df['Capital_Total'][0]) / df['Capital_Total'][0]
        return df

    def get_in_pandemic(self) -> pd.DataFrame:
        df = self.get_passive_table()
        df = df.iloc[25:]
        df['Rend_accum'] = (df['Capital_Total'] - df['Capital_Total'][0]) / df['Capital_Total'][0]
        return df
