"""ONS with a correlation-distance diversification constraint: alongside the usual
Newton-step projection onto the simplex, the candidate weight vector is also
constrained to stay within a maximum average pairwise "distance"
(sqrt(2*(1-correlation))) across the portfolio, re-estimated from a trailing
correlation matrix each day. Same universe/data as ons.py for consistency.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False


class ONS():
    def __init__(self, initial_data, delta=1, beta=1e2, eta=0.0):
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.historical_data = initial_data
        self.update_correlation_matrix()
        self.max_distance = 89

    def init_weights(self, columns):
        return np.ones(columns) / columns

    def init_step(self, X):
        m = X.shape[1]
        self.A = np.eye(m)
        self.b = np.zeros(m)

    def step(self, r, p, history, max_distance):
        if history is None or len(history) == 0:
            grad = r / np.dot(p, r)
        else:
            all_grads = np.array([rt / np.dot(p, rt) for rt in history])
            grad = np.sum(all_grads, axis=0)

        reg_factor = 0.01
        self.A += np.outer(grad, grad) + reg_factor * np.eye(self.A.shape[0])
        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)
        self.b += (1 + 1.0 / self.beta) * grad
        pp = self.projection_in_norm(self.delta * np.dot(self.A_inv, self.b), self.A, max_distance)
        return pp * (1 - self.eta) + np.ones(len(r)) / float(len(r)) * self.eta

    def projection_in_norm(self, x, M, max_distance):
        """Projection onto the simplex, further constrained by average pairwise
        correlation-distance <= max_distance (diversification constraint)."""
        m = M.shape[0]
        P = matrix(2 * M)
        q = matrix(-2 * np.dot(M, x))
        G = matrix(np.vstack([-np.eye(m), self.distance_matrix.sum(axis=0).to_numpy()]))
        h = matrix(np.hstack([np.zeros(m), max_distance]))
        A = matrix(1.0, (1, m))
        b = matrix(1.0)
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol["x"]).flatten()

    def update_correlation_matrix(self):
        if not self.historical_data.empty:
            correlation_matrix = self.historical_data.corr()
            self.distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

    def update_max_distance(self):
        if hasattr(self, 'distance_matrix'):
            mean_distance = self.distance_matrix.stack().mean()
            self.max_distance = mean_distance * 89


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    tickers = [
        "MSFT", "AAPL", "NVDA", "AMZN", "META",
        "AVGO", "GOOGL", "GOOG", "TSLA", "COST",
        "AMD", "NFLX", "PEP", "ADBE", "LIN",
        "CSCO", "TMUS", "QCOM", "INTC", "INTU",
        "CMCSA", "AMAT", "TXN", "AMGN", "ISRG",
        "HON", "MU", "LRCX", "BKNG", "VRTX",
        "REGN", "SBUX", "ADP", "ADI", "MDLZ",
        "KLAC", "GILD", "PANW", "SNPS", "ASML",
        "CDNS", "PDD", "MELI", "MAR", "CRWD",
        "CSX", "ABNB", "PYPL", "CTAS", "ORLY",
        "PCAR", "NXPI", "MNST", "MRVL", "ROP",
        "CEG", "WDAY", "CPRT", "ADSK", "DXCM",
        "FTNT", "DASH", "ROST", "MCHP", "LULU",
        "ODFL", "AEP", "KHC", "FAST", "IDXX",
        "PAYX", "KDP", "GEHC", "CHTR", "MRNA",
        "CSGP", "AZN", "TTD", "DDOG", "EXC",
        "EA", "FANG", "CTSH", "CDW", "BKR",
        "VRSK", "BIIB", "ON", "TEAM",
        "ANSS", "XEL", "ZS", "GFS", "DLTR",
        "MDB", "TTWO", "ILMN", "WBD",
        "WBA", "SIRI"
    ]
    start = '2023-03-16'
    end = '2024-03-21'

    df2 = pd.read_csv("data/22-23.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    if df2.columns[0] not in tickers:
        df2 = df2.drop(df2.columns[0], axis=1)
    df2 = df2[tickers].fillna(0)

    df = pd.read_csv("data/23-24.csv")
    df = df.drop(df.columns[0], axis=1)
    df = df[tickers].fillna(0)

    result_df = pd.DataFrame()
    result_df['equal_weighted_portfolio_return'] = df.mul(1 / len(tickers)).sum(axis=1)
    result_df['cum_return'] = (1 + result_df['equal_weighted_portfolio_return']).cumprod() - 1

    ons = ONS(df2)
    p = ons.init_weights(len(tickers))
    ons.init_step(np.zeros((1, len(tickers))))
    result_df['ons_daily_return'] = np.nan
    daily_risk_free_rate = (1 + 0.05) ** (1 / len(df)) - 1

    history = []
    for i, row in df[1:].iterrows():
        if i > 1:
            ons.historical_data = pd.concat([df2, df.iloc[:i]], ignore_index=True)
            ons.update_correlation_matrix()
            ons.update_max_distance()
        r = row.to_numpy(dtype=float)
        new_p = ons.step(r, p, np.array(history), ons.max_distance)
        new_p = np.where(new_p < 1e-5, 0, new_p)
        history.append(r)
        result_df.loc[i, 'ons_daily_return'] = np.dot(new_p, r)
        p = new_p

    result_df['ons_cumulative_return'] = (1 + result_df['ons_daily_return']).cumprod() - 1
    result_df['Excess Returns'] = result_df['ons_daily_return'] - daily_risk_free_rate
    result_df = result_df.drop(columns=['equal_weighted_portfolio_return', 'Excess Returns'])

    qqq = yf.download('QQQ', start=start, end=end, progress=False)['Close'].pct_change().fillna(0).reset_index(drop=True)
    result_df['qqq_cum'] = (1 + qqq).cumprod() - 1

    last_cum = result_df['ons_cumulative_return'].dropna().iloc[-1]
    apy = (last_cum + 1) ** (252 / len(result_df)) - 1
    ann_std = result_df['ons_daily_return'].std() * np.sqrt(len(result_df))
    sharpe = last_cum / ann_std

    print(result_df.tail())
    print(f"\ncumulative return: {last_cum:.4f}")
    print(f"annualized return: {apy:.4f}")
    print(f"annualized std:    {ann_std:.4f}")
    print(f"sharpe:            {sharpe:.3f}")

    plt.figure(figsize=(14, 7))
    plt.plot(result_df.index, result_df['ons_cumulative_return'], label='ONS + Distance Constraint')
    plt.plot(result_df.index, result_df['qqq_cum'], label='QQQ Benchmark', alpha=0.75)
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Trading day')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/distance_threshold_result.png', dpi=120)
    print("\nsaved plot -> results/distance_threshold_result.png")
