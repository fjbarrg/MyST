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


class Active:

    def __init__(self):
        pass

    @staticmethod
    def get_historical() -> pd.DataFrame:
        return Passive().get_historical()

    def get_sharpe(self) -> pd.DataFrame:
        df = self.get_historical()
        ret = df.pct_change().dropna()
        w = []
        sharpe = []
        s = []
        er = []
        r = ret.mean()
        cov = np.cov(r)
        for j in range(100000):
            a = np.random.random(len(df.columns))
            a /= np.sum(a)
            ren = a.dot(r)
            ss = np.sqrt(a.dot(cov).dot(a.T))
            sh = ren / ss
            w.append(a)
            s.append(ss)
            er.append(ren)
            sharpe.append(sh)
        portfolio = {'Returns': er, 'Volatility': s, 'Sharpe Ratio': sharpe}
        for i, j in enumerate(df):
            portfolio[j + ' Weight'] = [Weight[i] for Weight in w]
        portfolio = pd.DataFrame(portfolio)
        order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [j + ' Weight' for j in df]
        portfolio = portfolio[order]
        max_sharpe = portfolio['Sharpe Ratio'].max()
        max_sharpe_port = portfolio.loc[portfolio['Sharpe Ratio'] == max_sharpe]
        cols = df.columns
        weights = max_sharpe_port.iloc[:, 3:]
        weights = weights.T
        weights = weights.set_index(cols)
        weights.columns = ['Peso (%)']
        weights['Peso (%)'] = weights['Peso (%)'] * 100
        return weights

    def get_active_table(self) -> pd.DataFrame:
        prueba = self.get_historical()
        weights = self.get_sharpe()
        cols = prueba.columns
        base = prueba.iloc[25:, :].copy()

        for i in cols:
            base["rend_" + i] = base[i].pct_change()
            base['Weight_' + i] = weights.loc[i, 'Peso (%)']

        base.fillna(0, inplace=True)
        base.reset_index(inplace=True)

        for i in cols:
            for j in range(1, len(base['AMXL'])):
                if base.loc[j, "rend_" + i] >= 0.05:
                    base.loc[j, 'Weight_' + i] = base.loc[j - 1, 'Weight_' + i] * 1.025
                elif base.loc[j, "rend_" + i] <= -0.05:
                    base.loc[j, 'Weight_' + i] = base.loc[j - 1, 'Weight_' + i] * 0.975
                else:
                    base.loc[j, 'Weight_' + i] = base.loc[j - 1, 'Weight_' + i]

        col_weights = ["Weight_" + i for i in cols]
        base['total_weights'] = base.loc[:, col_weights].sum(axis=1)

        for j in range(len(base['AMXL'])):
            for i in cols:
                if base.loc[j, 'total_weights'] > 100:
                    excedente = (base.loc[j, 'total_weights'] - 100) / len(cols)
                    base.loc[j, 'Weight_' + i] = base.loc[j, 'Weight_' + i] - excedente

        base['total_weights'] = base.loc[:, col_weights].sum(axis=1)
        presupuesto = 1000000

        for i in cols:
            base['Acciones_' + i] = (presupuesto * (base['Weight_' + i] / 100)) / (base[i] + (base[i] * 0.00125))
            base['Acciones_' + i] = base['Acciones_' + i].apply(np.floor)
            base['Comision_' + i] = base[i] * 0.00125
            base['Capital_' + i] = np.round(base['Acciones_' + i] * (base[i] + base['Comision_' + i]), 2)

        base['CASH'] = round(((100 - base['total_weights']) / 100) * presupuesto, 2)

        col_capital = ["Capital_" + i for i in cols] + ['CASH']
        base['total_Capital'] = round(base.loc[:, col_capital].sum(axis=1), 2)

        diff_acc_cols = ["Diff_Acc_" + i for i in cols]
        acciones_cols = ["Capital_" + i for i in cols]

        # Ciclo por fecha
        for i in range(1, len(base['AMXL'])):
            # Ciclo nuevo valor portafolio
            for r in cols:
                base.loc[i, 'Capital_' + r] = np.round(
                    base.loc[i - 1, 'Acciones_' + r] * (base.loc[i, r] + base.loc[i, 'Comision_' + r]), 2)

            cash = base.loc[i - 1, 'CASH']
            base.loc[i, 'total_Capital'] = round(base.loc[i, acciones_cols].sum() + cash, 2)

            # Ciclo cambio num acciones.
            for w in cols:
                base.loc[i, 'Acciones_' + w] = (base.loc[i, 'total_Capital'] * (base.loc[i, 'Weight_' + w] / 100)) / (
                            base.loc[i, w] + (base.loc[i, w] * 0.00125))
                base.loc[i, 'Acciones_' + w] = np.floor(base.loc[i, 'Acciones_' + w])
                base.loc[i, 'Comision_' + w] = base.loc[i, w] * 0.00125
                base.loc[i, 'Capital_' + w] = np.round(
                    base.loc[i, 'Acciones_' + w] * (base.loc[i, w] + base.loc[i, 'Comision_' + w]), 2)
                base.loc[i, 'Diff_Acc_' + w] = base.loc[i, 'Acciones_' + w] - base.loc[i - 1, 'Acciones_' + w]

            df_iter = base.loc[i, :]
            df_iter = df_iter[diff_acc_cols]

            df_venta = df_iter[df_iter < 0]
            # Ciclo Venta
            for j in cols:
                try:
                    cash = cash + (abs(df_venta.loc['Diff_Acc_' + j]) * (base.loc[i, j] - base.loc[i, 'Comision_' + j]))
                except:
                    pass
            # Ciclo Compra
            df_compra = df_iter[df_iter > 0]
            for k in cols:
                try:
                    if cash <= df_compra.loc['Diff_Acc_' + k] * (base.loc[i, k] + base.loc[i, 'Comision_' + k]):
                        base.loc[i, 'Weight_' + k] = base.loc[i - 1, 'Weight_' + k]
                        base.loc[i, 'Acciones_' + k] = base.loc[i - 1, 'Acciones_' + k]
                        base.loc[i, 'Diff_Acc_' + k] = 0
                    else:
                        cash = cash - (df_compra.loc['Diff_Acc_' + k] * (base.loc[i, k] + base.loc[i, 'Comision_' + k]))
                except:
                    pass

            base.loc[i, 'CASH'] = round(cash, 0)
            base.loc[i, 'total_Capital'] = round(base.loc[i, col_capital].sum(), 2)
            base.fillna(0, inplace=True)

        base.set_index('index', inplace=True)
        return base

    def get_in_pandemic(self) -> pd.DataFrame:
        df = self.get_active_table()
        prueba = self.get_historical()
        cols = prueba.columns
        col_capital = ["Capital_" + i for i in cols] + ['CASH']
        df['total_Capital'] = round(df.loc[:, col_capital].sum(axis=1), 2)
        df['Rend'] = df['total_Capital'].pct_change()
        df.fillna(0, inplace=True)
        df['Rend_accum'] = (df['total_Capital'] - df['total_Capital'][0]) / df['total_Capital'][0]
        return df[['total_Capital', 'Rend', 'Rend_accum']]

    def get_historical_operations(self) -> pd.DataFrame:
        base = self.get_active_table()
        base.reset_index(inplace=True)
        prueba = self.get_historical()
        cols = prueba.columns
        df_operaciones = pd.DataFrame()
        df_operaciones['comisión_acumulada'] = 0
        for i in range(1, len(base['AMXL'])):
            for j in cols:
                df_operaciones.loc[i, 'titulos_totales_' + j] = base.loc[i, 'Acciones_' + j]
                df_operaciones.loc[i, 'titulos_comprados_' + j] = base.loc[i, 'Diff_Acc_' + j]
                df_operaciones.loc[i, 'precio_' + j] = base.loc[i, j]

            com = 0
            for j in cols:
                com += base.loc[i, 'Comision_' + j]
            df_operaciones.loc[i, 'comisión'] = com
            for j in cols:
                try:
                    df_operaciones.loc[i, 'comisión_acumulada'] = round(
                        df_operaciones.loc[i, 'comisión'] + df_operaciones.loc[i - 1, 'comisión_acumulada'], 0)
                except:
                    df_operaciones.loc[i, 'comisión_acumulada'] = round(df_operaciones.loc[i, 'comisión'], 0)

        df_operaciones['comision_acum'] = df_operaciones['comisión_acumulada']
        df_operaciones.drop(['comisión_acumulada'], axis=1, inplace=True)
        df_operaciones['index'] = base.loc[1:, 'index'].to_list()
        df_operaciones.set_index('index', inplace=True)
        return df_operaciones


class Metrics:

    def __init__(self):
        pass

    @staticmethod
    def get_metrics() -> pd.DataFrame:
        passive = Passive().get_in_pandemic()
        active = Active().get_in_pandemic()
        rf = 0.0429
        metric_df = pd.DataFrame(data={'medida': ['rend_m', 'rend_c', 'sharpe'],
                                       'decripcion': ['Rendimiento Promedio Mensual', 'Rendimiento mensual acumulado',
                                                      'Sharpe Ratio'],
                                       'inv_activa': [active['Rend'].mean(), active['Rend_accum'][-1],
                                                      (active['Rend_accum'][-1] - rf) / active['Rend'].std()],
                                       'inv_pasiva': [passive['Rend'].mean(), passive['Rend_accum'][-1],
                                                      (passive['Rend_accum'][-1] - rf) / passive['Rend'].std()]})
        return metric_df
