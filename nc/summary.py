"""Merge the per-model rows and print what the ensemble is really worth. First half of time = 'train', second half = 'test'."""
import glob, sys, numpy as np, pandas as pd
R = {}
for p in glob.glob("art/**/rows.csv", recursive=True):
    m = p.split("/")[1].split("-")[1]; R.setdefault(m, []).append(pd.read_csv(p))
R = {m: pd.concat(v) for m, v in R.items()}; M = sorted(R); print("models:", M, {m: len(R[m]) for m in M})
base = None
for m in M:
    x = R[m][["sym", "tf", "ts", "last", "atr", "a1", "a3", "p10_1", "p50_1", "p90_1", "p50_3"]].rename(columns={"p10_1": m + "_lo", "p50_1": m, "p90_1": m + "_hi", "p50_3": m + "_3"})
    base = x if base is None else base.merge(x.drop(columns=["last", "atr", "a1", "a3"]), on=["sym", "tf", "ts"], how="inner")
d = base.sort_values(["tf", "sym", "ts"]).reset_index(drop=True); d = d[d.a1 != d["last"]].copy()
d["up"] = (d.a1 > d["last"]).astype(int); d["up3"] = (d.a3 > d["last"]).astype(int)
d["half"] = d.groupby(["tf", "sym"]).ts.transform(lambda s: (s.rank(pct=True) > 0.5).astype(int))
d["prev"] = d.groupby(["tf", "sym"])["last"].diff()
for m in M: d[m + "_s"] = np.sign(d[m] - d["last"]); d[m + "_z"] = (d[m] - d["last"]) / d.atr
d["votes"] = sum(d[m + "_s"] for m in M); d["zavg"] = sum(d[m + "_z"] for m in M) / len(M)
mkt = lambda s: "crypto" if s.endswith("-USD") else ("forex" if s.endswith("=X") else "futures"); d["mkt"] = d.sym.map(mkt)
def hr(x, sig, col="up"):
    x = x[sig.loc[x.index] != 0]; s = sig.loc[x.index]
    return (len(x), 100 * ((s > 0).astype(int) == x[col]).mean() if len(x) else float("nan"))
def line(name, x, sig, col="up"):
    a, b = hr(x[x.half == 0], sig, col), hr(x[x.half == 1], sig, col)
    print("| %-34s | %6d | %5.1f%% | %6d | %5.1f%% |" % (name, a[0], a[1], b[0], b[1]))
print("\nmerged windows:", len(d), d.groupby("tf").size().to_dict())
for tf in sorted(d.tf.unique()):
    x = d[d.tf == tf]; print("\n## %s  (always-up base rate %.1f%%)\n| rule | n 1st half | hit | n 2nd half | hit |\n|---|---|---|---|---|" % (tf, 100 * x.up.mean()))
    line("baseline: momentum", x, np.sign(d.prev)); line("baseline: mean reversion", x, -np.sign(d.prev))
    for m in M: line(m, x, d[m + "_s"])
    for m in M:
        z = d[m + "_z"].abs(); thr = z[x.index][x.half == 0].quantile(0.9); line(m + " top-10% conviction", x, d[m + "_s"].where(z >= thr, 0))
    line("ensemble: majority", x, np.sign(d.votes)); line("ensemble: all %d agree" % len(M), x, np.sign(d.votes).where(d.votes.abs() == len(M), 0))
    line("ensemble: >=%d agree" % (len(M) - 1), x, np.sign(d.votes).where(d.votes.abs() >= len(M) - 2, 0))
    za = d.zavg.abs()
    for q in (0.8, 0.9, 0.95):
        thr = za[x.index][x.half == 0].quantile(q)
        line("all agree + top-%d%% size" % round(100 - 100 * q), x, np.sign(d.votes).where((d.votes.abs() == len(M)) & (za >= thr), 0))
    line("all agree, graded 3 candles later", x, np.sign(d.votes).where(d.votes.abs() == len(M), 0), "up3")
    for k in sorted(x.mkt.unique()): line("all agree · " + k, x[x.mkt == k], np.sign(d.votes).where(d.votes.abs() == len(M), 0))
    print("\nper symbol, all agree (both halves):")
    for s in sorted(x.sym.unique()):
        y = x[x.sym == s]; a = hr(y, np.sign(d.votes).where(d.votes.abs() == len(M), 0)); print("  %-10s n=%5d hit=%.1f%%" % (s, a[0], a[1]))
d.to_csv("merged.csv", index=False)

print("\n## Range forecast quality (what the bands are good for)\n| model | tf | next close inside p10-p90 (ideal 80%) | band width / ATR | next candle's low-high stays inside band |\n|---|---|---|---|---|")
raw = {m: R[m] for m in M}
for m in M:
    if m == "kronos": continue
    for tf in sorted(raw[m].tf.unique()):
        x = raw[m][raw[m].tf == tf]; inside = ((x.a1 >= x.p10_1) & (x.a1 <= x.p90_1)).mean(); w = ((x.p90_1 - x.p10_1) / x.atr).median()
        hl = ((x.l1 >= x.p10_1) & (x.h1 <= x.p90_1)).mean()
        print("| %s | %s | %.1f%% | %.2f | %.1f%% |" % (m, tf, 100 * inside, w, 100 * hl))
print("\n## Pullback entry test: a limit order 0.5 band-widths toward p10/p90 from the last close, in the majority direction, graded at the next close")
for tf in sorted(d.tf.unique()):
    x = d[d.tf == tf].merge(raw[M[0]][["sym", "tf", "ts", "h1", "l1"]], on=["sym", "tf", "ts"]); s = np.sign(x.votes)
    lo = x[[m + "_lo" for m in M if m != "kronos"]].mean(axis=1); hi = x[[m + "_hi" for m in M if m != "kronos"]].mean(axis=1)
    lim = np.where(s > 0, x["last"] - 0.5 * (x["last"] - lo), x["last"] + 0.5 * (hi - x["last"]))
    fill = np.where(s > 0, x.l1 <= lim, x.h1 >= lim) & (s != 0); pnl = np.where(s > 0, x.a1 - lim, lim - x.a1) / x.atr
    mk = (s != 0); pm = np.where(s > 0, x.a1 - x["last"], x["last"] - x.a1) / x.atr
    print("  %s: market-order avg %.4f ATR (n=%d) | limit fills %.1f%%, avg when filled %.4f ATR, win %.1f%%" % (tf, pm[mk].mean(), mk.sum(), 100 * fill.sum() / max(mk.sum(), 1), pnl[fill].mean(), 100 * (pnl[fill] > 0).mean()))
