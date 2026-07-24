"""ONS + volatility filter (Algorithm 2 in the poster): standard ONS Newton step and
simplex projection, then zero the weight of any stock whose trailing K-day return
std exceeds threshold V, and renormalize the remaining weights.

Reconstructed from the poster's pseudocode -- see README for how closely this
reproduces the published Table 1 / Table 2 figures (methodology match, not an
exact numerical reproduction).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


class ONS_VolatilityFilter():
    def __init__(self, n, delta=1.0, beta=0.01, K=10, V=0.05):
        self.delta = delta
        self.beta = beta
        self.K = K
        self.V = V
        self.A = np.eye(n)
        self.b = np.zeros(n)

    def step(self, r, p, ret_history):
        grad = r / np.dot(p, r)
        self.A += np.outer(grad, grad)
        self.b += (1 + 1.0 / self.beta) * grad
        A_inv = np.linalg.pinv(self.A)
        y = self.project(self.delta * np.dot(A_inv, self.b), self.A)

        if len(ret_history) >= self.K:
            recent = np.array(ret_history[-self.K:])
            vol = recent.std(axis=0)
            mask = vol > self.V
            if mask.any() and not mask.all():
                y = y.copy()
                y[mask] = 0.0
                s = y.sum()
                if s > 0:
                    y = y / s
        return y

    def project(self, x, M):
        m = M.shape[0]
        P = matrix(2 * M)
        q = matrix(-2 * np.dot(M, x))
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m, 1)))
        A = matrix(1.0, (1, m))
        b = matrix(1.0)
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol["x"]).flatten()


if __name__ == "__main__":
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

    def run(beta=0.01, K=10, V=0.05):
        n = len(tickers)
        ons = ONS_VolatilityFilter(n, delta=1.0, beta=beta, K=K, V=V)
        p = np.ones(n) / n
        daily, ret_history = [], []
        for _, row in df[1:].iterrows():
            r = row.to_numpy(dtype=float)
            new_p = ons.step(r, p, ret_history)
            ret_history.append(r)
            daily.append(float(np.dot(new_p, r)))
            p = new_p
        daily = pd.Series(daily)
        cum_series = (1 + daily).cumprod() - 1
        ann_std = daily.std() * np.sqrt(len(daily))
        return cum_series, ann_std, cum_series.iloc[-1] / ann_std

    print(f"{'K':>4} {'V':>6} | {'cum_return':>10} {'ann_std':>8} {'sharpe':>7}")
    for K, V in [(10, 0.05), (15, 0.07), (20, 0.10)]:
        cum_series, std, sharpe = run(K=K, V=V)
        print(f"{K:>4} {V:>6} | {cum_series.iloc[-1]:>10.4f} {std:>8.4f} {sharpe:>7.3f}")

    # headline chart: K=20, V=0.10 -- closest to the poster's own published figure
    headline_cum, headline_std, headline_sharpe = run(K=20, V=0.10)
    qqq = yf.download('QQQ', start='2023-03-16', end='2024-03-21', progress=False)['Close']
    qqq_cum = (qqq.pct_change().fillna(0).reset_index(drop=True) + 1).cumprod() - 1

    plt.figure(figsize=(14, 7))
    plt.plot(headline_cum.index, headline_cum.values, label=f'ONS + Volatility Filter (K=20, V=0.10, Sharpe={headline_sharpe:.2f})')
    plt.plot(qqq_cum.index, qqq_cum.values, label='QQQ Benchmark', alpha=0.75)
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Trading day')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/volatility_filter_result.png', dpi=120)
    print("\nsaved plot -> results/volatility_filter_result.png")
