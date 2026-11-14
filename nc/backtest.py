"""Walk-forward next-candle backtest for one model and one timeframe. Writes rows.csv (one row per forecast).
Each forecast only sees candles up to and including the last CLOSED candle; it is graded on the candles after it."""
import os, sys, time, csv, numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0, os.path.dirname(__file__)); import models

MODEL, TF = os.environ["NC_MODEL"], os.environ["NC_TF"]
SYMS = os.environ.get("NC_SYMS", "EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X,USDCAD=X,EURJPY=X,GBPJPY=X,BTC-USD,ETH-USD,SOL-USD,XRP-USD,GC=F,CL=F,NQ=F,ES=F").split(",")
N, CTX, H = int(os.environ.get("NC_POINTS", "800")), int(os.environ.get("NC_CTX", "256")), 3
BUDGET = float(os.environ.get("NC_BUDGET_MIN", "150")) * 60; T0 = time.time()
PERIOD = {"5m": "59d", "15m": "59d", "1h": "720d"}[TF]

D = {}
for s in SYMS:
    for a in range(3):
        try:
            df = yf.download(s, period=PERIOD, interval=TF, auto_adjust=False, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"]).iloc[:-1]
            if len(df) >= CTX + 200:
                D[s] = dict(ts=(df.index.astype("int64") // 10**9).values, o=df["Open"].values.astype(float), h=df["High"].values.astype(float),
                            l=df["Low"].values.astype(float), c=df["Close"].values.astype(float), v=df["Volume"].fillna(0).values.astype(float))
            break
        except Exception as e: print("download", s, a, e); time.sleep(5)
    print(s, len(D.get(s, {}).get("c", [])), flush=True)
fn = models.load(MODEL); print("model loaded", MODEL, round(time.time() - T0), "s", flush=True)

f = open("rows.csv", "w", newline=""); wr = csv.writer(f)
wr.writerow("sym tf ts last atr p10_1 p50_1 p90_1 p50_3 a1 a3 o1 h1 l1".split()); done = 0
BLK = 64                                    # newest block first, symbols interleaved, so a time-out still leaves balanced data
for b in range(0, N, BLK):
    for s, d in D.items():
        n = len(d["c"]); idx = [n - H - 1 - k for k in range(b, min(b + BLK, N)) if n - H - 1 - k >= CTX]
        if not idx: continue
        W = [{k: d[k][i - CTX + 1:i + 1] for k in d} for i in idx]
        try: P = fn(W, H)
        except Exception as e:
            import traceback; traceback.print_exc(); continue
        for i, p in zip(idx, P):
            tr = np.maximum(d["h"][i - 13:i + 1], d["c"][i - 14:i]) - np.minimum(d["l"][i - 13:i + 1], d["c"][i - 14:i])
            wr.writerow([s, TF, int(d["ts"][i]), d["c"][i], float(tr.mean()), p[0, 0], p[0, 1], p[0, 2], p[2, 1], d["c"][i + 1], d["c"][i + 3], d["o"][i + 1], d["h"][i + 1], d["l"][i + 1]])
        done += len(idx)
    f.flush(); print("block", b, "rows", done, "elapsed", round(time.time() - T0), "s", flush=True)
    if time.time() - T0 > BUDGET: print("time budget reached"); break
f.close()
r = pd.read_csv("rows.csv"); r = r[r.a1 != r["last"]]
print("\n%s %s: %d graded forecasts, next-candle direction hit rate %.2f%%" % (MODEL, TF, len(r), 100 * (np.sign(r.p50_1 - r["last"]) == np.sign(r.a1 - r["last"])).mean()))
