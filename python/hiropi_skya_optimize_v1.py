#!/usr/bin/env python3
"""
Hiropi_Skya 再現版  Python 検証・探索 v1
対応: hiropi_skya_indicator_v1.pine / hiropi_skya_strategy_v1.pine（ロジック・約定ルール同一）

使い方
  # 1) 直近2日でパラメーター探索（5通貨ペア）
  python3 hiropi_skya_optimize_v1.py search --data DIR --days 2 --out result_search
  # 2) 方法そのものの検証: 毎日「直前2日のベスト」を翌日に使ったら勝てるか
  python3 hiropi_skya_optimize_v1.py walkforward --data DIR --from 2026-07-01 --to 2026-09-29 --out result_wf

データ
  DIR に <ペア小文字><年>.csv（<TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>、UTC+0）
  または TradingView 書き出し CSV（time,open,high,low,close。time は UNIX秒 or ISO日時）を <ペア小文字>*.csv で置く。
  1分足。重複時刻は除去。

約定ルール（Pine と同じ）
  ・確定足でシグナル → 次の足の始値でエントリー。保有中は新規シグナル無視。
  ・TP/SL は足内経路「始値→近い方の高安→もう一方→終値」。始値が既に超えていれば始値で約定。
  ・時間切れ/停止帯決済は確定足で判定 → 次の足の始値。
  ・損益(pips) = 値幅/pip × 方向 − スプレッド。
  ・その日 N 連勝（M分以内）で当日の新規停止。日の切替は タイムゾーン基準 dayReset 時。
"""
import argparse, glob, itertools, os, sys
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # numba が無くても動く（遅い）
    HAVE_NUMBA = False
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

PAIRS = ["usdjpy", "eurjpy", "gbpjpy", "audjpy", "chfjpy"]

# ───────── 設定（Pine の input と同じ名前・既定値）─────────
@dataclass
class Cfg:
    tz: str = "Asia/Tokyo"
    stops: tuple = (("0500", "0830"),)   # 有効な停止帯（既定: タイ 03:00-06:30）
    close_in_stop: bool = False
    rci_len: int = 20
    rci_src: str = "close"               # close / open
    rci_thr: float = 82
    mode: int = 0                        # 0=逆張り 1=順張り
    di_len: int = 14
    adx_smooth: int = 14
    adx_upper: float = 45
    adx_lower: float = 33
    adx_state: bool = True
    tp: float = 7
    sl: float = 6
    time_exit: int = 0
    stop_on_win: bool = True
    win_n: int = 3
    win_within: int = 60
    day_reset: int = 6
    spread: float = 0.5
    reverse: bool = False


def pip_of(pair):
    return 0.1 if pair.startswith("xau") else (0.01 if pair.endswith("jpy") else 0.0001)

# ───────── データ ─────────
def load_pair(data_dir, pair):
    files = sorted(glob.glob(os.path.join(data_dir, f"{pair}*.csv")) + glob.glob(os.path.join(data_dir, f"{pair.upper()}*.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        cols = [c.strip("<>").lower() for c in d.columns]
        d.columns = cols
        if "dtyyyymmdd" in cols:
            ts = pd.to_datetime(d["dtyyyymmdd"].astype(str) + d["time"].astype(int).astype(str).str.zfill(4), format="%Y%m%d%H%M", utc=True)
        else:
            tcol = "time" if "time" in cols else cols[0]
            v = d[tcol]
            ts = pd.to_datetime(v, unit="s", utc=True) if np.issubdtype(v.dtype, np.number) else pd.to_datetime(v, utc=True)
        dfs.append(pd.DataFrame({"t": ts, "o": d["open"].astype(float), "h": d["high"].astype(float),
                                 "l": d["low"].astype(float), "c": d["close"].astype(float)}))
    df = pd.concat(dfs).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return df

# ───────── 指標（Pine と同じ計算）─────────
@njit(cache=True)
def rci_arr(src, n):
    out = np.full(src.shape[0], np.nan)
    for k in range(n - 1, src.shape[0]):
        s = 0.0
        for i in range(n):
            vi = src[k - i]
            gt = 0.0; eq = 0.0
            for j in range(n):
                vj = src[k - j]
                if vj > vi:
                    gt += 1
                elif vj == vi:
                    eq += 1
            pr = 1 + gt + (eq - 1) / 2
            d = (i + 1) - pr
            s += d * d
        out[k] = (1 - 6 * s / (n * (n * n - 1))) * 100
    return out


def rma(x, n):
    out = np.full(len(x), np.nan)
    a = 1.0 / n
    prev = np.nan
    buf = []
    for i, v in enumerate(x):
        if np.isnan(prev):
            buf.append(v)
            if len(buf) > n:
                buf.pop(0)
            if len(buf) == n and not np.any(np.isnan(buf)):
                prev = float(np.mean(buf))
        else:
            prev = a * v + (1 - a) * prev
        out[i] = prev
    return out


def adx_arr(h, l, c, di_len, smooth):
    up = np.r_[np.nan, h[1:] - h[:-1]]
    dn = np.r_[np.nan, l[:-1] - l[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = np.nan
    trur = rma(tr, di_len)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 100 * rma(pdm, di_len) / trur
        m = 100 * rma(mdm, di_len) / trur
        s = p + m
        x = np.abs(p - m) / np.where(s == 0, 1, s)
    return 100 * rma(x, smooth)


def regime(adx, lower, upper, state):
    n = len(adx)
    ok = np.zeros(n, bool)
    armed = False
    prev = np.nan
    for i in range(n):
        a = adx[i]
        if not np.isnan(prev) and not np.isnan(a) and a > lower and prev <= lower:
            armed = True
        if np.isnan(a) or a < lower or a > upper:
            armed = False
        inr = (not np.isnan(a)) and lower <= a <= upper
        ok[i] = (armed and inr) if state else inr
        prev = a
    return ok


def signals(rci, thr, mode, reg):
    r1 = np.r_[np.nan, rci[:-1]]
    with np.errstate(invalid="ignore"):
        up_lo = (rci > -thr) & (r1 <= -thr)
        up_hi = (rci > thr) & (r1 <= thr)
        dn_hi = (rci < thr) & (r1 >= thr)
        dn_lo = (rci < -thr) & (r1 >= -thr)
    buy = up_lo if mode == 0 else up_hi
    sell = dn_hi if mode == 0 else dn_lo
    sig = np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)
    sig[~reg] = 0
    return sig

# ───────── 時刻系 ─────────
def time_arrays(df, tz, stops, day_reset, start_utc=None):
    lt = df["t"].dt.tz_convert(tz)
    mod = (lt.dt.hour * 60 + lt.dt.minute).to_numpy()
    in_stop = np.zeros(len(df), bool)
    for a, b in stops:
        s = int(a[:2]) * 60 + int(a[2:]); e = int(b[:2]) * 60 + int(b[2:])
        in_stop |= ((mod >= s) & (mod < e)) if s < e else ((mod >= s) | (mod < e))
    shifted = lt - pd.Timedelta(hours=day_reset)
    day = (shifted.dt.year * 10000 + shifted.dt.month * 100 + shifted.dt.day).to_numpy()
    tmin = (df["t"].astype("int64") // 60_000_000_000).to_numpy()  # UTC 分
    allow = ~in_stop
    if start_utc is not None:
        allow &= df["t"].to_numpy() >= np.datetime64(start_utc.tz_convert("UTC").tz_localize(None))
    return in_stop, allow, day, tmin

# ───────── 売買シミュレーション（Pine の仮想売買と同一）─────────
@njit(cache=True)
def simulate(o, h, l, sig, allow, in_stop, day, tmin, i0, i1, tp, sl, pip, spread,
             time_exit, close_in_stop, stop_on_win, win_n, win_within, rev):
    n = i1
    res_pips = np.zeros(4096); res_exit_i = np.zeros(4096, np.int64); res_entry_i = np.zeros(4096, np.int64)
    nt = 0
    pos = 0; entry = 0.0; entry_t = 0; entry_i = 0
    pend_dir = 0; pend_close = False
    cur_day = -1; halted = False
    wins = np.zeros(64, np.int64); nw = 0
    i = i0
    while i < n:
        if day[i] != cur_day:
            cur_day = day[i]; halted = False; nw = 0
        closed = False; p = 0.0
        if pend_close and pos != 0:
            p = (o[i] - entry) / pip * pos - spread; closed = True; pos = 0
        pend_close = False
        if pend_dir != 0 and pos == 0:
            pos = pend_dir; entry = o[i]; entry_t = tmin[i]; entry_i = i
        pend_dir = 0
        if pos != 0 and not closed:
            tpp = entry + pos * tp * pip; slp = entry - pos * sl * pip
            hf = (h[i] - o[i]) <= (o[i] - l[i])
            ex = np.nan
            if pos == 1:
                if o[i] >= tpp or o[i] <= slp: ex = o[i]
                elif hf:
                    if h[i] >= tpp: ex = tpp
                    elif l[i] <= slp: ex = slp
                else:
                    if l[i] <= slp: ex = slp
                    elif h[i] >= tpp: ex = tpp
            else:
                if o[i] <= tpp or o[i] >= slp: ex = o[i]
                elif hf:
                    if h[i] >= slp: ex = slp
                    elif l[i] <= tpp: ex = tpp
                else:
                    if l[i] <= tpp: ex = tpp
                    elif h[i] >= slp: ex = slp
            if not np.isnan(ex):
                if ex == tpp: p = tp - spread
                elif ex == slp: p = -sl - spread
                else: p = (ex - entry) / pip * pos - spread
                closed = True; pos = 0
        if closed:
            if nt < 4096:
                res_pips[nt] = p; res_exit_i[nt] = i; res_entry_i[nt] = entry_i
            nt += 1
            if p > 0:
                if nw < 64:
                    wins[nw] = tmin[i]; nw += 1
            else:
                nw = 0
            if stop_on_win and nw >= win_n:
                if win_within == 0 or (tmin[i] - wins[nw - win_n]) <= win_within:
                    halted = True
        if pos != 0:
            if time_exit > 0 and (tmin[i] + 1 - entry_t) >= time_exit: pend_close = True
            if close_in_stop and in_stop[i]: pend_close = True
        d = sig[i] * (-1 if rev else 1)
        if pos == 0 and not pend_close and d != 0 and allow[i] and not halted and i + 1 < n:
            pend_dir = d
        # 高速化: 手ぶらで保留も無ければ次のシグナルまで飛ばす
        if pos == 0 and pend_dir == 0 and not pend_close:
            j = i + 1
            while j < n and sig[j] == 0:
                j += 1
            i = j
        else:
            i += 1
    k = min(nt, 4096)
    return res_pips[:k], res_entry_i[:k], res_exit_i[:k]


def stats(p):
    n = len(p)
    if n == 0:
        return dict(trades=0, winrate=0.0, pf=0.0, net_pips=0.0, avg=0.0, maxdd=0.0)
    w = p[p > 0].sum(); ls = -p[p <= 0].sum()
    eq = np.cumsum(p); dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    return dict(trades=n, winrate=round(float(100 * (p > 0).mean()), 1), pf=round(float(w / ls), 2) if ls > 0 else 99.0,
                net_pips=round(float(p.sum()), 1), avg=round(float(p.mean()), 2), maxdd=round(float(dd), 1))

# ───────── 探索グリッド ─────────
GRID_FULL = dict(src=["close", "open"], rci_len=list(range(9, 31)), rci_thr=[70, 74, 77, 80, 82, 85, 88],
                 mode=[0, 1], adx=[(lo, up) for up in (40, 45, 50, 60, 100) for lo in (20, 25, 30, 33)],
                 adx_state=[True, False], tp=[5, 6, 7, 8, 10], sl=[5, 6, 7, 8, 10], rev=[False, True])
GRID_SMALL = dict(src=["close"], rci_len=[9, 11, 12, 14, 18, 20, 24], rci_thr=[77, 80, 82, 85],
                  mode=[0, 1], adx=[(lo, up) for up in (45, 50, 100) for lo in (25, 33)],
                  adx_state=[True, False], tp=[6, 7, 8], sl=[6, 7, 8], rev=[False, True])


class PairData:
    def __init__(self, df, pair, base: Cfg):
        self.df = df; self.pair = pair; self.pip = pip_of(pair); self.base = base
        self.o = df["o"].to_numpy(); self.h = df["h"].to_numpy(); self.l = df["l"].to_numpy(); self.c = df["c"].to_numpy()
        self.in_stop, self.allow0, self.day, self.tmin = time_arrays(df, base.tz, base.stops, base.day_reset)
        self._rci = {}; self._adx = None; self._reg = {}

    def rci(self, src, n):
        k = (src, n)
        if k not in self._rci:
            self._rci[k] = rci_arr(self.c if src == "close" else self.o, n)
        return self._rci[k]

    def reg(self, lo, up, st):
        if self._adx is None:
            self._adx = adx_arr(self.h, self.l, self.c, self.base.di_len, self.base.adx_smooth)
        k = (lo, up, st)
        if k not in self._reg:
            self._reg[k] = regime(self._adx, lo, up, st)
        return self._reg[k]


def run_grid(pd_: PairData, i0, i1, grid, min_trades, base: Cfg, keep=30):
    rows = []
    sig_keys = itertools.product(grid["src"], grid["rci_len"], grid["rci_thr"], grid["mode"], grid["adx"], grid["adx_state"])
    for src, n, thr, mode, (lo, up), st in sig_keys:
        sig = signals(pd_.rci(src, n), thr, mode, pd_.reg(lo, up, st))
        if not sig[i0:i1].any():
            continue
        for tp, sl, rev in itertools.product(grid["tp"], grid["sl"], grid["rev"]):
            p, _, _ = simulate(pd_.o, pd_.h, pd_.l, sig, pd_.allow0, pd_.in_stop, pd_.day, pd_.tmin, i0, i1,
                               float(tp), float(sl), pd_.pip, base.spread, base.time_exit, base.close_in_stop,
                               base.stop_on_win, base.win_n, base.win_within, rev)
            if len(p) < min_trades:
                continue
            s = stats(p)
            rows.append((s["net_pips"], dict(src=src, rci_len=n, rci_thr=thr, mode="逆張り" if mode == 0 else "順張り",
                                             adx_lower=lo, adx_upper=up, adx_state=st, tp=tp, sl=sl,
                                             reverse=rev, **s)))
    rows.sort(key=lambda r: -r[0])
    return [r[1] for r in rows[:keep]]


def window_index(df, start, end):
    t = df["t"]
    return int(t.searchsorted(start)), int(t.searchsorted(end))


def last_n_days(df, tz, n_days, day_reset):
    lt = df["t"].dt.tz_convert(tz) - pd.Timedelta(hours=day_reset)
    cnt = lt.dt.date.value_counts()
    days = sorted(d for d, k in cnt.items() if k >= 360)   # 6時間未満しか無い日（週末の端など）は数えない
    sel = days[-n_days:]
    start = pd.Timestamp(sel[0]).tz_localize(tz) + pd.Timedelta(hours=day_reset)
    return start, df["t"].iloc[-1] + pd.Timedelta(minutes=1)

# ───────── コマンド ─────────
def cmd_search(a, base):
    grid = GRID_FULL if a.grid == "full" else GRID_SMALL
    os.makedirs(a.out, exist_ok=True)
    allrows = []
    for pair in a.pairs:
        df = load_pair(a.data, pair)
        if df is None:
            print(f"[{pair}] データなし"); continue
        start, end = last_n_days(df, base.tz, a.days, base.day_reset)
        pdx = PairData(df, pair, base)
        i0, i1 = window_index(df, start, end)
        top = run_grid(pdx, i0, i1, grid, a.min_trades, base, keep=a.keep)
        for r in top:
            r.update(pair=pair, window=f"{start:%Y-%m-%d %H:%M}〜{end.tz_convert(base.tz):%Y-%m-%d %H:%M} ({base.tz})",
                     graph_value=round(r["net_pips"] / 100, 2), target_ok=r["net_pips"] >= a.target)
        allrows += top
        best = top[0] if top else None
        print(f"[{pair}] {start:%m/%d %H:%M}〜 ベスト: " + (", ".join(f"{k}={v}" for k, v in best.items() if k != "window") if best else "該当なし"))
    out = pd.DataFrame(allrows)
    out.to_csv(os.path.join(a.out, "search_top.csv"), index=False, encoding="utf-8-sig")
    print(f"保存: {a.out}/search_top.csv  (target={a.target}pips 以上を target_ok=True)")


def cmd_walkforward(a, base):
    """毎日: 直前 lookback 日でベストを選ぶ → その日1日だけ使う。方法の実力を測る。"""
    grid = GRID_FULL if a.grid == "full" else GRID_SMALL
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for pair in a.pairs:
        df = load_pair(a.data, pair)
        if df is None:
            print(f"[{pair}] データなし"); continue
        pdx = PairData(df, pair, base)
        lt = df["t"].dt.tz_convert(base.tz) - pd.Timedelta(hours=base.day_reset)
        days = sorted(d for d in lt.dt.date.unique() if pd.Timestamp(a.dfrom).date() <= d <= pd.Timestamp(a.dto).date())
        alldays = sorted(lt.dt.date.unique())
        for d in days:
            k = alldays.index(d)
            if k < a.days:
                continue
            ts = lambda x: pd.Timestamp(x).tz_localize(base.tz) + pd.Timedelta(hours=base.day_reset)
            i0, i1 = window_index(df, ts(alldays[k - a.days]), ts(d))
            j0, j1 = window_index(df, ts(d), ts(d) + pd.Timedelta(days=1))
            top = run_grid(pdx, i0, i1, grid, a.min_trades, base, keep=1)
            if not top or top[0]["net_pips"] < a.target:
                rows.append(dict(pair=pair, day=str(d), traded=False, is_net=top[0]["net_pips"] if top else 0, oos_net=0.0, oos_trades=0))
                continue
            b = top[0]
            sig = signals(pdx.rci(b["src"], b["rci_len"]), b["rci_thr"], 0 if b["mode"] == "逆張り" else 1,
                          pdx.reg(b["adx_lower"], b["adx_upper"], b["adx_state"]))
            p, _, _ = simulate(pdx.o, pdx.h, pdx.l, sig, pdx.allow0, pdx.in_stop, pdx.day, pdx.tmin, j0, j1,
                               float(b["tp"]), float(b["sl"]), pdx.pip, base.spread, base.time_exit, base.close_in_stop,
                               base.stop_on_win, base.win_n, base.win_within, b["reverse"])
            s = stats(p)
            rows.append(dict(pair=pair, day=str(d), traded=True, is_net=b["net_pips"], oos_net=s["net_pips"],
                             oos_trades=s["trades"], params={k2: b[k2] for k2 in ("src", "rci_len", "rci_thr", "mode", "adx_lower", "adx_upper", "adx_state", "tp", "sl", "reverse")}))
            print(f"[{pair}] {d} 学習{b['net_pips']:+.1f} → 翌日{s['net_pips']:+.1f} ({s['trades']}回)")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "walkforward_days.csv"), index=False, encoding="utf-8-sig")
    if len(out):
        t = out[out.traded]
        summ = t.groupby("pair").agg(days=("day", "count"), oos_net=("oos_net", "sum"), oos_trades=("oos_trades", "sum"),
                                     win_days=("oos_net", lambda x: int((x > 0).sum())))
        summ.loc["合計"] = summ.sum()
        summ.to_csv(os.path.join(a.out, "walkforward_summary.csv"), encoding="utf-8-sig")
        print(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "walkforward"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--days", type=int, default=2, help="学習に使う直近日数")
    ap.add_argument("--target", type=float, default=150, help="採用ライン(pips)。グラフ1.50=150pips")
    ap.add_argument("--min-trades", type=int, default=6)
    ap.add_argument("--grid", choices=["full", "small"], default="full")
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--tz", default="Asia/Tokyo")
    ap.add_argument("--from", dest="dfrom", default="2026-07-01")
    ap.add_argument("--to", dest="dto", default="2026-09-29")
    ap.add_argument("--out", default="result")
    a = ap.parse_args()
    base = Cfg(tz=a.tz, spread=a.spread)
    print(f"numba: {'あり' if HAVE_NUMBA else 'なし（遅い。pip install numba 推奨）'}")
    (cmd_search if a.cmd == "search" else cmd_walkforward)(a, base)


if __name__ == "__main__":
    main()
