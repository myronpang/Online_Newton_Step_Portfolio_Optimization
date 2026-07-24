"""Online Newton Step (Agarwal, Hazan, Kale & Schapire, 2006) portfolio algorithm,
benchmarked against an equal-weighted ("Universal") portfolio over the same universe.
"""
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

solvers.options['show_progress'] = False


class ONS():
    def __init__(self, delta=1, beta=1e2, eta=0.0):
        self.delta = delta
        self.beta = beta
        self.eta = eta

    def init_weights(self, columns):
        return np.ones(columns) / columns

    def init_step(self, X):
        m = X.shape[1]
        self.A = np.asmatrix(np.eye(m))
        self.b = np.asmatrix(np.zeros(m)).T

    def step(self, r, p, history):
        grad = np.asmatrix(r / np.dot(p, r)).T
        self.A += np.outer(grad, grad.T)
        self.A_inv = np.linalg.pinv(self.A)
        self.b += (1 + 1.0 / self.beta) * grad
        pp = self.projection_in_norm(self.delta * self.A_inv * self.b, self.A)
        return pp * (1 - self.eta) + np.ones(len(r)) / float(len(r)) * self.eta

    def projection_in_norm(self, x, M):
        """Projection of x to the simplex, in the norm induced by matrix M."""
        m = M.shape[0]
        P = matrix(2 * M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m, 1)))
        A = matrix(np.ones((1, m)))
        b = matrix(1.0)
        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol["x"])


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

    df = pd.read_csv("data/23-24.csv")
    df = df.drop(df.columns[0], axis=1)
    df = df[tickers].fillna(0)

    result_df = pd.DataFrame()
    result_df['equal_weighted_portfolio_return'] = df.mul(1 / len(tickers)).sum(axis=1)
    result_df['cum_return'] = (1 + result_df['equal_weighted_portfolio_return']).cumprod() - 1

    ons = ONS()
    ons.init_step(np.zeros((1, len(tickers))))
    p = ons.init_weights(len(tickers))
    result_df['ons_daily_return'] = np.nan

    for i, row in df[1:].iterrows():
        r = row.to_numpy(dtype=float)
        new_p = ons.step(r, p, None)
        result_df.loc[i, 'ons_daily_return'] = float(np.dot(new_p, r))
        p = new_p  # carry the updated portfolio into the next step

    result_df['ons_cumulative_return'] = (1 + result_df['ons_daily_return']).cumprod() - 1
    result_df = result_df.drop(columns=['equal_weighted_portfolio_return', 'ons_daily_return'])

    ons_cum = result_df['ons_cumulative_return'].dropna().iloc[-1]
    uni_cum = result_df['cum_return'].dropna().iloc[-1]
    print(result_df.tail())
    print(f"\nONS cumulative return:       {ons_cum:.4f}")
    print(f"Universal cumulative return:  {uni_cum:.4f}")
