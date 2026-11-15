"""ChartPulse liquidation collector (owner, 15 Nov 2026). Runs 24/7 on the Big VPS.
Listens to the public liquidation streams of Binance, Bybit and OKX USDT perpetuals, keeps the last 24 hours in memory
(saved to disk every minute), and every 15 seconds pushes totals, per-coin numbers, hourly series and the biggest
recent liquidations to chartpulse.io. No exchange keys are used; the site push uses the key in /opt/liq/key.
Note: Binance sends at most one liquidation per symbol per second (the largest), so its totals are a floor."""
import asyncio, json, os, re, time, collections, urllib.request, ssl, sys
import websockets

URL = os.environ.get("CP_URL", "https://chartpulse.io/api.php?r=liqpush")
KEY = open(os.environ.get("LIQ_KEY_FILE", "/opt/liq/key")).read().strip() if os.path.exists(os.environ.get("LIQ_KEY_FILE", "/opt/liq/key")) else ""
STATE = os.environ.get("LIQ_STATE", "/opt/liq/state.json")
DAY = 86400
EV = collections.deque()            # (ts, ex, coin, side, px, usd)
SEEN = {"Binance": 0, "Bybit": 0, "OKX": 0}
SENT_ALERT = set(); START = time.time()

def coin_of(sym):
    s = re.sub(r"(-USDT-SWAP|USDT|-USDT)$", "", sym.upper())
    s = re.sub(r"^(1000000|10000|1000|100)(?=[A-Z])", "", s)
    return s
def add(ts, ex, sym, side, px, usd):
    if usd <= 0 or px <= 0: return
    EV.append((int(ts), ex, coin_of(sym), side, float(px), float(usd))); SEEN[ex] = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def get_json(u):
    req = urllib.request.Request(u, headers={"User-Agent": "chartpulse-liq"})
    with urllib.request.urlopen(req, timeout=20, context=ssl.create_default_context()) as r: return json.loads(r.read())

async def binance():
    u = "wss://fstream.binance.com/ws/!forceOrder@arr"
    while True:
        try:
            async with websockets.connect(u, ping_interval=60, ping_timeout=60, max_size=2**20) as ws:
                log("binance connected")
                async for m in ws:
                    o = json.loads(m).get("o") or {}
                    px = float(o.get("ap") or 0) or float(o.get("p") or 0); q = float(o.get("z") or o.get("q") or 0)
                    if str(o.get("s", "")).endswith("USDT"): add(int(o.get("T", time.time() * 1000)) / 1000, "Binance", o["s"], "long" if o.get("S") == "SELL" else "short", px, px * q)
        except Exception as e: log("binance", repr(e)[:200]); await asyncio.sleep(5)

async def bybit():
    u = "wss://stream.bybit.com/v5/public/linear"
    while True:
        try:
            t = get_json("https://api.bybit.com/v5/market/tickers?category=linear")["result"]["list"]
            syms = [x["symbol"] for x in sorted(t, key=lambda x: -float(x.get("turnover24h") or 0)) if x["symbol"].endswith("USDT")][:250]
            async with websockets.connect(u, ping_interval=None, max_size=2**20) as ws:
                for i in range(0, len(syms), 10): await ws.send(json.dumps({"op": "subscribe", "args": ["allLiquidation." + s for s in syms[i:i + 10]]}))
                log("bybit connected", len(syms))
                async def ping():
                    while True: await asyncio.sleep(20); await ws.send('{"op":"ping"}')
                pt = asyncio.create_task(ping())
                try:
                    async for m in ws:
                        j = json.loads(m)
                        for d in (j.get("data") or []) if str(j.get("topic", "")).startswith("allLiquidation") else []:
                            px = float(d.get("p") or 0); add(int(d.get("T", 0)) / 1000, "Bybit", d.get("s", ""), "long" if d.get("S") == "Buy" else "short", px, px * float(d.get("v") or 0))
                finally: pt.cancel()
        except Exception as e: log("bybit", repr(e)[:200]); await asyncio.sleep(5)

async def okx():
    u = "wss://ws.okx.com:8443/ws/v5/public"
    while True:
        try:
            ct = {x["instId"]: float(x.get("ctVal") or 0) for x in get_json("https://www.okx.com/api/v5/public/instruments?instType=SWAP")["data"]}
            async with websockets.connect(u, ping_interval=None, max_size=2**20) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}))
                log("okx connected", len(ct))
                async def ping():
                    while True: await asyncio.sleep(25); await ws.send("ping")
                pt = asyncio.create_task(ping())
                try:
                    async for m in ws:
                        if m == "pong": continue
                        j = json.loads(m)
                        for g in j.get("data") or []:
                            inst = g.get("instId", "")
                            if "-USDT-" not in inst: continue
                            for d in g.get("details") or []:
                                px = float(d.get("bkPx") or 0); usd = float(d.get("sz") or 0) * ct.get(inst, 0) * px
                                long_ = d.get("posSide") == "long" or (d.get("posSide") == "net" and d.get("side") == "sell")
                                add(int(d.get("ts", 0)) / 1000, "OKX", inst, "long" if long_ else "short", px, usd)
                finally: pt.cancel()
        except Exception as e: log("okx", repr(e)[:200]); await asyncio.sleep(5)

def summary():
    now = time.time()
    while EV and EV[0][0] < now - DAY: EV.popleft()
    tot = {k: {"long": 0.0, "short": 0.0, "n": 0} for k in ("m5", "h1", "h4", "h24")}; win = {"m5": 300, "h1": 3600, "h4": 14400, "h24": DAY}
    ex = {e: {"long": 0.0, "short": 0.0, "n": 0} for e in SEEN}
    coins = {}; hours = collections.defaultdict(lambda: [0.0, 0.0]); m5 = collections.defaultdict(lambda: [0.0, 0.0])
    for ts, e, c, sd, px, usd in EV:
        age = now - ts; j = 0 if sd == "long" else 1
        for k, w in win.items():
            if age <= w: tot[k][sd] += usd; tot[k]["n"] += 1
        ex[e][sd] += usd; ex[e]["n"] += 1
        hours[int(ts // 3600) * 3600][j] += usd
        if age <= 4 * 3600: m5[int(ts // 300) * 300][j] += usd
        r = coins.setdefault(c, {"c": c, "h1": [0.0, 0.0], "h4": [0.0, 0.0], "h24": [0.0, 0.0], "max": 0.0, "n": 0})
        r["h24"][j] += usd; r["n"] += 1; r["max"] = max(r["max"], usd)
        if age <= 3600: r["h1"][j] += usd
        if age <= 14400: r["h4"][j] += usd
    rd = lambda v: round(v)
    top = sorted(coins.values(), key=lambda r: -(r["h24"][0] + r["h24"][1]))[:120]
    for r in top: r["h1"] = list(map(rd, r["h1"])); r["h4"] = list(map(rd, r["h4"])); r["h24"] = list(map(rd, r["h24"])); r["max"] = rd(r["max"])
    big = [e for e in EV if e[5] >= 20000][-80:][::-1]
    largest = sorted(EV, key=lambda e: -e[5])[:12]
    ev = lambda e: {"t": e[0], "ex": e[1], "c": e[2], "side": e[3], "px": e[4], "v": rd(e[5])}
    alerts = []
    for e in EV:
        if e[5] >= 1_500_000 and now - e[0] < 900:
            k = (e[0], e[1], e[2], round(e[5]))
            if k not in SENT_ALERT: SENT_ALERT.add(k); alerts.append(ev(e))
    return {"at": int(now), "since": int(EV[0][0]) if EV else int(now), "up": int(now - START),
            "tot": {k: {"long": rd(v["long"]), "short": rd(v["short"]), "n": v["n"]} for k, v in tot.items()},
            "ex": {k: {"long": rd(v["long"]), "short": rd(v["short"]), "n": v["n"], "age": int(now - SEEN[k]) if SEEN[k] else None} for k, v in ex.items()},
            "hours": [[t, rd(a), rd(b)] for t, (a, b) in sorted(hours.items())][-25:], "m5": [[t, rd(a), rd(b)] for t, (a, b) in sorted(m5.items())],
            "coins": top, "big": [ev(e) for e in big], "largest": [ev(e) for e in largest], "alerts": alerts}

def push(p):
    data = json.dumps(p, separators=(",", ":")).encode()
    req = urllib.request.Request(URL, data=data, method="POST", headers={"Content-Type": "application/json", "X-NC-Key": KEY, "User-Agent": "chartpulse-liq"})
    with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r: return r.status, r.read()[:120]

async def pusher():
    last_save = 0
    while True:
        await asyncio.sleep(15)
        try:
            p = await asyncio.to_thread(summary)
            st, body = await asyncio.to_thread(push, p)
            if p["alerts"] or int(time.time()) % 300 < 15: log("push", st, body, "events", len(EV), {k: round(v["long"] + v["short"]) for k, v in p["tot"].items()})
        except Exception as e: log("push failed", repr(e)[:200])
        if time.time() - last_save > 60:
            last_save = time.time()
            try:
                tmp = STATE + ".tmp"; open(tmp, "w").write(json.dumps(list(EV))); os.replace(tmp, STATE)
            except Exception as e: log("save failed", e)

async def main():
    try:
        for e in json.load(open(STATE)):
            if e[0] > time.time() - DAY: EV.append(tuple(e))
        log("restored", len(EV), "events")
    except Exception: pass
    await asyncio.gather(binance(), bybit(), okx(), pusher())

if __name__ == "__main__":
    if not KEY and "--dry" not in sys.argv: print("no key file", file=sys.stderr)
    asyncio.run(main())
