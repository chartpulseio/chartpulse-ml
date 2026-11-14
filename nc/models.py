"""Next-candle forecasters behind one interface (ChartPulse, 15 Nov 2026).
load(name) -> fn(windows, H) where windows = list of dict(ts=int64[s], o,h,l,c,v = float arrays, same length)
fn returns float array [n, H, 3] = (p10, p50, p90) of the close for the next H candles (p10/p90 NaN when the model has none)."""
import os, sys, numpy as np

def _chronos2():
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(os.environ.get("NC_CHRONOS", "amazon/chronos-2"), device_map="cpu")
    def fn(W, H):
        out = np.full((len(W), H, 3), np.nan, dtype=np.float64)
        for i in range(0, len(W), 64):
            chunk = [torch.tensor(np.asarray(w["c"], dtype=np.float32)) for w in W[i:i + 64]]
            q, _ = pipe.predict_quantiles(inputs=chunk, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
            for k, x in enumerate(q):
                out[i + k] = np.asarray(x, dtype=np.float64).reshape(-1, H, 3)[0]
        return out
    return fn

def _timesfm():
    import torch, timesfm
    torch.set_num_threads(os.cpu_count() or 2)
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=16, normalize_inputs=True, use_continuous_quantile_head=True,
              force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    def fn(W, H):
        out = np.full((len(W), H, 3), np.nan, dtype=np.float64)
        for i in range(0, len(W), 64):
            _, q = m.forecast(horizon=H, inputs=[np.asarray(w["c"], dtype=np.float32) for w in W[i:i + 64]]); q = np.asarray(q)
            out[i:i + q.shape[0], :, 0] = q[:, :H, 1]; out[i:i + q.shape[0], :, 1] = q[:, :H, 5]; out[i:i + q.shape[0], :, 2] = q[:, :H, 9]
        return out
    return fn

def _moirai2():
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    mod = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"); cache = {}
    def fn(W, H):
        n = len(W[0]["c"]); key = (n, H)
        if key not in cache:
            cache[key] = Moirai2Forecast(module=mod, prediction_length=H, context_length=n, target_dim=1,
                                         feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).create_predictor(batch_size=64)
        ds = ListDataset([{"start": pd.Period("2024-01-01 00:00", freq="h"), "target": np.asarray(w["c"], dtype=np.float32)} for w in W], freq="h")
        out = np.full((len(W), H, 3), np.nan, dtype=np.float64)
        for k, f in enumerate(cache[key].predict(ds)):
            for j, lv in enumerate((0.1, 0.5, 0.9)): out[k, :, j] = np.asarray(f.quantile(lv))[:H]
        return out
    return fn

def _kronos():
    import pandas as pd, torch
    sys.path.insert(0, os.environ.get("KRONOS_DIR", "Kronos"))
    from model import Kronos, KronosTokenizer, KronosPredictor
    torch.set_num_threads(os.cpu_count() or 2)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained(os.environ.get("NC_KRONOS", "NeoQuasar/Kronos-small"))
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    S = int(os.environ.get("NC_KRONOS_SAMPLES", "3"))
    def fn(W, H):
        out = np.full((len(W), H, 3), np.nan, dtype=np.float64)
        for i in range(0, len(W), 32):
            ch = W[i:i + 32]; dfs, xs, ys = [], [], []
            for w in ch:
                dfs.append(pd.DataFrame({"open": w["o"], "high": w["h"], "low": w["l"], "close": w["c"], "volume": w["v"]}).astype("float64"))
                ts = np.asarray(w["ts"], dtype="int64"); step = int(np.median(np.diff(ts[-50:])))
                xs.append(pd.Series(pd.to_datetime(ts, unit="s"))); ys.append(pd.Series(pd.to_datetime(ts[-1] + step * np.arange(1, H + 1), unit="s")))
            res = pred.predict_batch(df_list=dfs, x_timestamp_list=xs, y_timestamp_list=ys, pred_len=H, T=1.0, top_p=0.9, sample_count=S, verbose=False)
            for k, r in enumerate(res): out[i + k, :, 1] = r["close"].values[:H]
        return out
    return fn

LOADERS = {"chronos2": _chronos2, "timesfm": _timesfm, "moirai2": _moirai2, "kronos": _kronos}
def load(name): return LOADERS[name]()
