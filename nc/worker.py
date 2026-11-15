"""One model in its own Python environment. Reads JSON lines {"W": [[closes...], ...], "H": n} on stdin,
answers {"P": [[[p10, p50, p90] per step] per series]} on stdout. Started by nc/live.py."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models
out = sys.stdout; sys.stdout = sys.stderr              # libraries may print; keep the pipe clean
fn = models.load(sys.argv[1]); out.write(json.dumps({"ready": sys.argv[1]}) + "\n"); out.flush()
for line in sys.stdin:
    try:
        q = json.loads(line); W = [{k: np.asarray(v, dtype=np.int64 if k == "ts" else np.float64) for k, v in c.items()} if isinstance(c, dict) else {"c": np.asarray(c, dtype=np.float64)} for c in q["W"]]
        P = fn(W, int(q["H"])); out.write(json.dumps({"P": np.nan_to_num(P, nan=0.0).round(8).tolist()}) + "\n")
    except Exception as e:
        import traceback; traceback.print_exc(); out.write(json.dumps({"error": str(e)[:300]}) + "\n")
    out.flush()
