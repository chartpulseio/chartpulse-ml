"""ChartPulse ML job (GitHub Actions, hourly). Pulls the work list from chartpulse.io, runs
TimesFM 2.5 (daily forecasts, once a day), FinBERT (new headlines) and TradingAgents (queued symbols),
then pushes the results back. Secrets: CP_KEY (same as config.php ml_push_key), GOOGLE_API_KEY (Gemini)."""
import os, sys, json, time, datetime, traceback
import numpy as np, requests

URL = os.environ["CP_URL"]
def H0():
    h = {"User-Agent": "chartpulse-ml-job"}
    if len(os.environ.get("CP_KEY", "")) >= 20: h["X-CP-Key"] = os.environ["CP_KEY"]
    u, t = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL"), os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if u and t:
        r = requests.get(u + "&audience=chartpulse.io", headers={"Authorization": "bearer " + t}, timeout=30); r.raise_for_status()
        h["X-GH-OIDC"] = r.json()["value"]
    return h
H = H0()
feed = requests.get(URL, params={"r": "mlfeed"}, headers=H, timeout=60); feed.raise_for_status(); F = feed.json()
out = {"forecasts": {}, "sentiment": {}, "agents": {}}
FULL = bool(F["need_forecast"]); FC = F["symbols"] if FULL else F.get("priority", [])
print("feed:", len(F["symbols"]), "symbols,", len(F["headlines"]), "headlines,", len(F["agents"]), "agent jobs, full forecast:", FULL, "forecast now:", len(FC))

def push():
    r = requests.post(URL, params={"r": "mlpush"}, headers={**H0(), "Content-Type": "application/json"}, data=json.dumps(out), timeout=120)
    print("push:", r.status_code, r.text[:300])

# ---- FinBERT ----
if F["headlines"]:
    try:
        from transformers import pipeline
        fin = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
        for t, r in zip(F["headlines"], fin(F["headlines"], batch_size=32)):
            d = {z["label"].lower(): z["score"] for z in r}
            out["sentiment"][t] = round(d.get("positive", 0) - d.get("negative", 0), 4)
    except Exception: traceback.print_exc()

# ---- TimesFM 2.5 (daily closes from Yahoo Finance, once a day) ----
if FC:
    try:
        import yfinance as yf, timesfm, torch
        torch.set_num_threads(os.cpu_count() or 2)
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=64, normalize_inputs=True, use_continuous_quantile_head=True,
                  force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        syms = FC; h = int(F.get("horizon", 10))
        for i in range(0, len(syms), 60):
            chunk = syms[i:i + 60]
            df = yf.download(chunk, period="5y", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
            S, names = [], []
            for s in chunk:
                try: c = (df[s]["Close"] if len(chunk) > 1 else df["Close"]).dropna().values.astype(np.float32)
                except Exception: continue
                if len(c) >= 64: S.append(c[-1024:]); names.append(s)
            if not S: continue
            _, q = m.forecast(horizon=h, inputs=S); q = np.asarray(q)
            for k, s in enumerate(names):
                out["forecasts"][s] = [{"h": j + 1, "p10": float(q[k, j, 1]), "p25": float((q[k, j, 2] + q[k, j, 3]) / 2), "p50": float(q[k, j, 5]),
                                        "p75": float((q[k, j, 7] + q[k, j, 8]) / 2), "p90": float(q[k, j, 9])} for j in range(h)]
            print("forecasts", len(out["forecasts"]))
    except Exception: traceback.print_exc()
out["full"] = FULL and len(out["forecasts"]) > 50
push(); out = {"forecasts": {}, "sentiment": {}, "agents": {}}

# ---- TradingAgents (queued symbols), pushed one by one ----
if F["agents"] and os.environ.get("GOOGLE_API_KEY"):
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({"llm_provider": "google", "deep_think_llm": "gemini-3.5-flash", "quick_think_llm": "gemini-3.5-flash-lite",
                "max_debate_rounds": 1, "max_risk_discuss_rounds": 1})
    for job in F["agents"]:
        try:
            ta = TradingAgentsGraph(debug=False, config=cfg)
            st, dec = ta.propagate(job["symbol"], job["date"])
            g = lambda k: str(st.get(k, ""))[:6000] if isinstance(st, dict) else ""
            out["agents"][job["symbol"]] = {"date": job["date"], "decision": str(dec)[:200], "market": g("market_report"), "sentiment": g("sentiment_report"),
                "news": g("news_report"), "fundamentals": g("fundamentals_report"), "plan": g("investment_plan"), "trader": g("trader_investment_plan"), "final": g("final_trade_decision")}
        except Exception:
            traceback.print_exc(); out["agents"][job["symbol"]] = {"date": job["date"], "decision": "", "error": traceback.format_exc()[-1500:]}
        push(); out["agents"] = {}
print("done")
