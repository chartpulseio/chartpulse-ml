"""Next Candle live loop (ChartPulse, 15 Nov 2026). Every 5 minutes: read the closed 5m candles, ask three forecast
models for the next candle, grade the earlier forecasts against what really happened, push everything to the site.
Rule shown as a signal (from the walk-forward backtest): forex only, all three models agree on the direction."""
import os, sys, json, time, subprocess, datetime, math, random
import numpy as np, pandas as pd, requests

URL = os.environ.get("CP_URL", "https://chartpulse.io/api.php")
MODELS = [m for m in os.environ.get("NC_MODELS", "chronos2,timesfm,moirai2,kronos").split(",") if m]
FOREX = "EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCAD=X USDCHF=X NZDUSD=X EURJPY=X GBPJPY=X EURGBP=X AUDJPY=X EURAUD=X".split()
OTHER = "BTC-USD ETH-USD SOL-USD XRP-USD PAXG-USD".split()
SYMS = FOREX + OTHER; CTX, H, STEP = 256, 12, 300
HOST = os.environ.get("NC_HOST", "github")
RUN_MIN = float(os.environ.get("NC_MINUTES", "330")); T0 = time.time(); FAKE = os.environ.get("NC_FAKE") == "1"
BT = {"run": "walk-forward backtest, 15 Nov 2026, 4 models, 5m candles, about 9 days", "forex_all_agree": [53.5, 58.4], "all_agree": [53.0, 56.6],
      "single_model": [50.1, 52.5], "n": 3089, "band_inside": 80}

def headers():
    h = {"User-Agent": "chartpulse-nc-live"}
    kf = os.environ.get("NC_KEY_FILE")
    if kf and os.path.exists(kf): h["X-NC-Key"] = open(kf).read().strip(); return h
    u, t = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL"), os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if u and t:
        r = requests.get(u + "&audience=chartpulse.io", headers={"Authorization": "bearer " + t}, timeout=30); r.raise_for_status()
        h["X-GH-OIDC"] = r.json()["value"]
    return h

class Worker:
    def __init__(self, name):
        self.name = name; py = os.environ.get("NC_PY_" + name.upper(), sys.executable)
        self.p = subprocess.Popen([py, os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"), name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        print("worker", name, json.loads(self.p.stdout.readline()), flush=True)
    def ask(self, W, h):
        self.p.stdin.write(json.dumps({"W": W, "H": h}) + "\n"); self.p.stdin.flush(); r = json.loads(self.p.stdout.readline())
        if "error" in r: raise RuntimeError(self.name + ": " + r["error"])
        return np.asarray(r["P"], dtype=float)
class Fake:
    def __init__(self, name): self.name = name
    def ask(self, W, h):
        out = []
        for c in W:
            c = np.asarray(c["c"] if isinstance(c, dict) else c); sd = np.std(np.diff(c[-50:])) or 1e-9; d = random.choice([-1, 1]) * sd * 0.1
            out.append([[c[-1] + d - 1.28 * sd * math.sqrt(k + 1), c[-1] + d, c[-1] + d + 1.28 * sd * math.sqrt(k + 1)] for k in range(h)])
        return np.asarray(out)

def candles(now, interval="5m", period="5d", step=STEP):
    """Closed candles per symbol: (ts[], close[]). A candle counts as closed once its time is over."""
    import yfinance as yf
    out = {}
    try: df = yf.download(SYMS, period=period, interval=interval, auto_adjust=False, progress=False, group_by="ticker", threads=True)
    except Exception as e: print("download failed", e); return out
    for s in SYMS:
        try:
            x = df[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]); idx = x.index
            ts = np.asarray((idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx).values.astype("datetime64[s]").astype("int64"))
            c = x["Close"].values.astype(float); ok = ts + step <= now; ts, c = ts[ok], c[ok]
            if len(c) >= CTX: out[s] = (ts, c, {k: x[K].values.astype(float)[ok] for k, K in (("o", "Open"), ("h", "High"), ("l", "Low"), ("v", "Volume"))})
        except Exception: pass
    return out

def day_block(W, S, now, D1):
    """24-hour forecast from 1h candles (owner, 15 Nov 2026): median path and 80% band for the next 24 hours, graded a day later."""
    HD = 24; still = []
    for o in S["dopen"]:
        d = D1.get(o["s"]); k = np.where(d[0] >= o["t"] + HD * 3600)[0] if d is not None else []
        if len(k) and d[0][k[0]] - o["t"] <= 72 * 3600:
            c = float(d[1][k[0]]); day = datetime.datetime.utcfromtimestamp(o["t"]).strftime("%Y-%m-%d"); mv = np.sign(c - o["last"])
            if mv != 0 and o["side"] != 0: a = S["days"].setdefault(day, {}).setdefault("d24dir", [0, 0]); a[0 if o["side"] == mv else 1] += 1
            a = S["days"].setdefault(day, {}).setdefault("d24rng", [0, 0]); a[0 if o["lo"] <= c <= o["hi"] else 1] += 1
        elif now - o["t"] < 96 * 3600: still.append(o)
    S["dopen"] = still
    syms = [s for s in SYMS if s in D1]; rows = []
    if not syms: return rows
    X = [D1[s][1][-CTX:].tolist() for s in syms]; P = {}
    for w in W:
        if w.name == "kronos": continue       # Kronos votes on the next 5m candle only
        try: P[w.name] = w.ask(X, HD)
        except Exception as e: print("day model failed", e, flush=True)
    if not P: return rows
    band = [m for m in ("chronos2", "moirai2") if m in P] or list(P)
    for i, s in enumerate(syms):
        ts, c = D1[s][:2]; last = float(c[-1]); t = int(ts[-1]); d = dec(last)
        med = np.mean([P[m][i, :, 1] for m in P], axis=0); lo = np.mean([P[m][i, :, 0] for m in band], axis=0); hi = np.mean([P[m][i, :, 2] for m in band], axis=0)
        v = {m: int(np.sign(P[m][i, HD - 1, 1] - last)) for m in P}; tot = sum(v.values()); side = int(np.sign(tot))
        rows.append({"s": s, "mkt": "forex" if s in FOREX else ("gold" if s.startswith("PAXG") else "crypto"), "t": t, "last": round(last, d), "dec": d, "open": int(now - t <= 3 * 3600),
                     "v": v, "up": sum(1 for x in v.values() if x > 0), "n": len(v), "side": side, "agree": int(abs(tot) == len(v) and len(v) >= 3),
                     "chg": round(100 * (float(med[-1]) / last - 1), 3), "past": [round(float(x), d) for x in c[-48:]],
                     "med": [round(float(x), d) for x in med], "lo": [round(float(x), d) for x in lo], "hi": [round(float(x), d) for x in hi]})
        if now - t <= 3 * 3600 and not any(o["s"] == s and t - o["t"] < 4 * 3600 for o in S["dopen"]):
            S["dopen"].append({"s": s, "t": t, "last": last, "side": side, "lo": float(lo[-1]), "hi": float(hi[-1])})
    return rows

def dec(p): return 3 if p > 50 else (5 if p > 0.5 else 6)
def main():
    def vps_alive():
        """The VPS is the main forecaster. A GitHub run only works while the VPS is silent."""
        if HOST == "vps" or FAKE: return False
        try: o = requests.get(URL, params={"r": "nc"}, timeout=20).json(); return o.get("host") == "vps" and (o.get("age") or 9e9) < 420
        except Exception: return False
    if vps_alive(): print("VPS forecaster is live; nothing to do"); open("skip", "w").write("1"); return
    S = {"hist": [], "open": [], "rng": [], "days": {}, "models": {}, "zs": [], "dopen": [], "day": [], "day_at": 0}
    try:
        old = requests.get(URL, params={"r": "nc", "full": 1}, timeout=30).json()
        if isinstance(old.get("state"), dict): S.update({k: old["state"].get(k, S[k]) for k in S})
        if isinstance(old.get("day"), list): S["day"] = old["day"]; S["day_at"] = int(old.get("day_at") or 0)
    except Exception as e: print("no earlier state:", e)
    W = [Fake(m) if FAKE else Worker(m) for m in MODELS]; last_bar = {}
    while time.time() - T0 < RUN_MIN * 60:
        now = int(time.time()); nxt = (now // STEP + 1) * STEP + 20
        if not FAKE and not os.environ.get("NC_ONCE"): time.sleep(max(0, nxt - time.time()))
        now = int(time.time()); D = candles(now) if not FAKE else fake_candles(now)
        # ---- grade earlier forecasts with the candles that have closed since
        still = []
        for o in S["open"]:
            d = D.get(o["s"]); hit = None
            if d is not None:
                k = np.where(d[0] == o["t"] + STEP)[0]
                if len(k): hit = float(d[1][k[0]])
            if hit is None:
                if now - o["t"] < 3 * 3600: still.append(o)
                continue
            mv = np.sign(hit - o["last"]); day = datetime.datetime.utcfromtimestamp(o["t"]).strftime("%Y-%m-%d")
            for m, v in o["v"].items():
                if mv != 0: a = S["models"].setdefault(m, [0, 0]); a[0 if v == mv else 1] += 1
            if mv != 0 and o["side"] != 0:
                key = "rule" if o["sig"] else ("agree" if o["agree"] else "split"); a = S["days"].setdefault(day, {}).setdefault(key, [0, 0]); a[0 if o["side"] == mv else 1] += 1
            if o["sig"] or o["agree"]:
                S["hist"].append({"t": o["t"] + STEP, "s": o["s"], "side": o["side"], "last": o["last"], "close": hit, "sig": int(o["sig"]),
                                  "res": "flat" if mv == 0 else ("win" if o["side"] == mv else "loss")})
        S["open"] = still; S["hist"] = S["hist"][-300:]
        still = []
        for r in S["rng"]:                                   # was the close one hour later inside the forecast range?
            d = D.get(r["s"]); k = np.where(d[0] == r["t"] + H * STEP)[0] if d is not None else []
            if len(k):
                a = S["days"].setdefault(datetime.datetime.utcfromtimestamp(r["t"]).strftime("%Y-%m-%d"), {}).setdefault("range", [0, 0])
                a[0 if r["lo"] <= float(d[1][k[0]]) <= r["hi"] else 1] += 1
            elif now - r["t"] < 4 * 3600: still.append(r)
        S["rng"] = still
        for day in sorted(S["days"])[:-14]: S["days"].pop(day)
        # ---- new forecasts, only for markets whose last closed candle is fresh
        live = [s for s in SYMS if s in D and now - int(D[s][0][-1]) <= 3 * STEP]
        rows = []
        if live:
            X = [D[s][1][-CTX:].tolist() for s in live]; P = {}
            XK = [dict(ts=D[s][0][-CTX:].tolist(), c=D[s][1][-CTX:].tolist(), **{k: np.nan_to_num(v[-CTX:]).tolist() for k, v in D[s][2].items()}) for s in live]
            for w in W:
                try: P[w.name] = w.ask(XK, 1) if w.name == "kronos" else w.ask(X, H)
                except Exception as e: print("model failed", e, flush=True)
            for i, s in enumerate(live):
                ts, c = D[s][:2]; last = float(c[-1]); bar = int(ts[-1]); atr = float(np.mean(np.abs(np.diff(c[-15:])))) or 1e-12
                v = {m: int(np.sign(P[m][i, 0, 1] - last)) for m in P}; tot = sum(v.values()); n = len(v)
                agree = n >= 3 and abs(tot) == n; side = int(np.sign(tot)); fx = s in FOREX; sig = bool(agree and fx)
                band = [m for m in ("chronos2", "moirai2") if m in P] or list(P)
                q = lambda k, j: float(np.mean([P[m][i, k, j] for m in band])) if band else last
                z = abs(float(np.mean([P[m][i, 0, 1] - last for m in P])) / atr) if P else 0.0
                if agree: S["zs"].append(round(z, 4)); S["zs"] = S["zs"][-600:]
                strong = bool(agree and len(S["zs"]) >= 100 and z >= float(np.quantile(S["zs"], 0.8)))
                d = dec(last)
                rows.append({"s": s, "mkt": "forex" if fx else ("gold" if s.startswith("PAXG") else "crypto"), "bar": bar, "last": round(last, d), "dec": d,
                             "v": v, "up": sum(1 for x in v.values() if x > 0), "n": n, "side": side, "agree": int(agree), "sig": int(sig), "strong": int(strong),
                             "next": [round(q(0, 0), d), round(q(0, 1), d), round(q(0, 2), d)], "h1": [round(q(H - 1, 0), d), round(q(H - 1, 2), d)]})
                if last_bar.get(s) != bar and n:
                    last_bar[s] = bar; S["open"].append({"s": s, "t": bar, "last": last, "v": v, "side": side, "agree": int(agree), "sig": int(sig)})
                    S["rng"].append({"s": s, "t": bar, "lo": q(H - 1, 0), "hi": q(H - 1, 2)})
        closed = [s for s in SYMS if s not in live]
        if now - int(S.get("day_at") or 0) >= 3300 or not S["day"]:
            try:
                D1 = fake_candles(now) if FAKE else candles(now, "1h", "60d", 3600); dr = day_block(W, S, now, D1)
                if dr: S["day"] = dr; S["day_at"] = now
            except Exception as e:
                import traceback; traceback.print_exc()
        def tot(key, days):
            w = l = 0
            for dname in sorted(S["days"])[-days:]:
                a = S["days"][dname].get(key, [0, 0]); w += a[0]; l += a[1]
            return {"w": w, "l": l, "rate": round(100 * w / (w + l), 1) if w + l >= 20 else None}
        out = {"at": now, "host": HOST, "tf": "5m", "models": MODELS, "rows": rows, "closed": closed,
               "stats": {"rule": {"d1": tot("rule", 1), "d7": tot("rule", 7)}, "agree": {"d7": tot("agree", 7)}, "split": {"d7": tot("split", 7)}, "range": {"d7": tot("range", 7)},
                         "models": {m: {"w": a[0], "l": a[1], "rate": round(100 * a[0] / (a[0] + a[1]), 1) if sum(a) >= 50 else None} for m, a in S["models"].items()}},
               "recent": [h for h in S["hist"] if h["sig"]][-40:][::-1], "bt": BT,
               "day": S["day"], "day_at": S["day_at"], "day_stats": {"dir": tot("d24dir", 14), "range": tot("d24rng", 14)}, "state": {k: v for k, v in S.items() if k != "day"}}
        if FAKE: return out
        if vps_alive(): print("VPS forecaster took over; stopping"); open("skip", "w").write("1"); return
        try:
            r = requests.post(URL, params={"r": "ncpush"}, headers={**headers(), "Content-Type": "application/json"}, data=json.dumps(out), timeout=60)
            print(datetime.datetime.utcnow().strftime("%H:%M:%S"), "live", len(live), "signals", sum(x["sig"] for x in rows), "open", len(S["open"]), "push", r.status_code, r.text[:120], flush=True)
        except Exception as e: print("push failed", e, flush=True)
        if os.environ.get("NC_ONCE"): break

def fake_candles(now):
    rng = np.random.default_rng(1); out = {}; t = (now // STEP) * STEP - STEP * np.arange(400)[::-1] - STEP
    for s in SYMS:
        c = 1.1 + np.cumsum(rng.normal(0, 0.0003, 400)); out[s] = (t, c, {"o": c, "h": c, "l": c, "v": c * 0})
    return out
if __name__ == "__main__":
    r = main()
    if FAKE: print(json.dumps({k: r[k] for k in ("stats", "closed", "day_stats")})[:400]); print(json.dumps(r["day"][0])[:700]); json.dumps(r)
