"""Forecast history for AI Backtesting (owner, 16 Nov 2026).
For each market and timeframe, walk through history and, at every closed candle t, ask one model for the next candle's
close using ONLY candles up to t. Pushes {sym: {tf: [[t, close_t, p50_next], ...]}} to chartpulse.io (r=fchpush, GitHub OIDC).
The site combines the four models into backtest indicators (votes and average expected move)."""
import os, sys, time, json, urllib.request, numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nc")); import models

MODEL = os.environ["FC_MODEL"]; SYMS = os.environ.get("FC_SYMS", "^NSEI,^NSEBANK,BTC-USD,GC=F,EURUSD=X").split(",")
TFS = os.environ.get("FC_TFS", "1h,1d").split(","); CTX = int(os.environ.get("FC_CTX", "256"))
BUDGET = float(os.environ.get("FC_BUDGET_MIN", "320")) * 60; T0 = time.time()
PERIOD = {"1h": "729d", "1d": "6y"}

def get(s, tf):
    for a in range(3):
        try:
            df = yf.download(s, period=PERIOD[tf], interval=tf, auto_adjust=False, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"]).iloc[:-1]
            idx = df.index.tz_convert("UTC").tz_localize(None) if df.index.tz is not None else df.index
            return dict(ts=idx.values.astype("datetime64[s]").astype("int64"), o=df["Open"].values.astype(float), h=df["High"].values.astype(float),
                        l=df["Low"].values.astype(float), c=df["Close"].values.astype(float), v=df["Volume"].fillna(0).values.astype(float))
        except Exception as e: print("download", s, tf, a, e); time.sleep(5)
    return None

def push(payload):
    tok = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN"); url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    t = json.loads(urllib.request.urlopen(urllib.request.Request(url + "&audience=chartpulse.io", headers={"Authorization": "bearer " + tok}), timeout=30).read())["value"]
    body = json.dumps(payload, separators=(",", ":")).encode()
    r = urllib.request.urlopen(urllib.request.Request("https://chartpulse.io/api.php?r=fchpush", data=body, method="POST",
        headers={"Content-Type": "application/json", "X-GH-OIDC": t, "User-Agent": "chartpulse-ml"}), timeout=120)
    print("push", r.status, r.read()[:200], flush=True)

fn = models.load(MODEL); print("model", MODEL, "loaded", round(time.time() - T0), "s", flush=True)
for s in SYMS:
    for tf in TFS:
        d = get(s, tf)
        if not d or len(d["c"]) < CTX + 50: print("skip", s, tf); continue
        n = len(d["c"]); idx = list(range(CTX - 1, n)); rows = []
        for b in range(0, len(idx), 64):
            ii = idx[b:b + 64]; W = [{k: d[k][i - CTX + 1:i + 1] for k in d} for i in ii]
            try: P = fn(W, 1)
            except Exception as e: print("fail", s, tf, e); continue
            for i, p in zip(ii, P):
                if np.isfinite(p[0, 1]): rows.append([int(d["ts"][i]), round(float(d["c"][i]), 6), round(float(p[0, 1]), 6)])
            if time.time() - T0 > BUDGET: break
        hit = np.mean([np.sign(rows[k][2] - rows[k][1]) == np.sign(rows[k + 1][1] - rows[k][1]) for k in range(len(rows) - 1)]) if len(rows) > 1 else 0
        print(s, tf, "forecasts", len(rows), "next-candle hit %.1f%%" % (100 * hit), "elapsed", round(time.time() - T0), "s", flush=True)
        try: push({"model": MODEL, "sym": s, "tf": tf, "rows": rows})
        except Exception as e: print("push failed", e)
        if time.time() - T0 > BUDGET: print("budget reached"); sys.exit(0)
