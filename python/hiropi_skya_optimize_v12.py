#!/usr/bin/env python3
"""
Hiropi_Skya 再現版  Python 検証・探索 v12
対応: hiropi_skya_indicator_v4.pine / hiropi_skya_strategy_v4.pine / hiropi_skya_auto_strategy_v15.pine（ロジック・約定ルール同一）

使い方
  # 1) 直近2日でパラメーター探索（5通貨ペア）
  python3 hiropi_skya_optimize_v12.py search --data DIR --days 2 --out result_search
  # 3) 自動最適化（Pine hiropi_skya_auto_strategy_v15 と同じ）: 30分ごとに直近48時間で90候補を比べて切り替え
  python3 hiropi_skya_optimize_v12.py adaptive --data DIR --from 2026-09-01 --to 2026-09-29 --out result_adaptive
  # 2) 方法そのものの検証: 毎日「直前2日のベスト」を翌日に使ったら勝てるか
  python3 hiropi_skya_optimize_v12.py walkforward --data DIR --from 2026-07-01 --to 2026-09-29 --out result_wf

データ
  DIR に <ペア小文字><年>.csv（<TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>、UTC+0）
  または TradingView 書き出し CSV（time,open,high,low,close。time は UNIX秒 or ISO日時）を <ペア小文字>*.csv で置く。
  1分足。重複時刻は除去。

約定ルール（Pine と同じ）
  ・確定足でシグナル → 次の足の始値でエントリー。保有中は新規シグナル無視。
  ・TP/SL は足内経路「始値→近い方の高安→もう一方→終値」。始値が既に超えていれば始値で約定。
  ・時間切れ/停止帯決済は確定足で判定 → 次の足の始値。
  ・損益(pips) = 値幅/pip × 方向 − スプレッド。
  ・ドテン: 保有中に逆シグナル → 次の足の始値で決済して反対に入る（--no-doten で無効）。
  ・既定は Pine v4/v5 の既定と同じ（停止帯なし・連勝停止なし）。運用条件は --thai-stop --win-stop で。
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
    stops: tuple = ()                    # 有効な停止帯（--thai-stop で タイ 03:00-06:30 = 東京 05:00-08:30）
    close_in_stop: bool = False
    rci_len: int = 9
    rci_src: str = "close"               # close / open
    rci_thr: float = 82
    mode: int = 2                        # 0=逆張り 1=順張り 2=押し目（v4既定。オリジナルに一番近い）
    di_len: int = 14
    adx_smooth: int = 14
    adx_upper: float = 45
    adx_lower: float = 33
    adx_state: bool = True
    tp: float = 7
    sl: float = 6
    time_exit: int = 0
    stop_on_win: bool = False
    win_n: int = 3
    win_within: int = 60
    day_reset: int = 6
    spread: float = 0.5
    reverse: bool = False
    doten: bool = True


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


def rci_fast(src, n, chunk=40000):
    """rci_arr と同じ値（同値は平均順位）を numpy で一括計算。numba が無いとき用。"""
    from numpy.lib.stride_tricks import sliding_window_view
    src = np.asarray(src, float)
    out = np.full(src.shape[0], np.nan)
    if src.shape[0] < n:
        return out
    W = sliding_window_view(src, n)[:, ::-1]          # 列0 = 最新
    t = np.arange(1, n + 1, dtype=float)
    for s0 in range(0, W.shape[0], chunk):
        w = W[s0:s0 + chunk]
        gt = (w[:, None, :] > w[:, :, None]).sum(2)
        eq = (w[:, None, :] == w[:, :, None]).sum(2)
        pr = 1 + gt + (eq - 1) / 2
        d2 = ((t[None, :] - pr) ** 2).sum(1)
        out[n - 1 + s0:n - 1 + s0 + w.shape[0]] = (1 - 6 * d2 / (n * (n * n - 1))) * 100
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


def signals(rci, thr, mode, reg, c=None):
    r1 = np.r_[np.nan, rci[:-1]]
    with np.errstate(invalid="ignore"):
        up_lo = (rci > -thr) & (r1 <= -thr)
        up_hi = (rci > thr) & (r1 <= thr)
        dn_hi = (rci < thr) & (r1 >= thr)
        dn_lo = (rci < -thr) & (r1 >= -thr)
    if mode == 2:
        dc = np.r_[0.0, c[1:] - c[:-1]]
        with np.errstate(invalid="ignore"):
            buy = (rci > thr) & (dc < 0)
            sell = (rci < -thr) & (dc > 0)
    else:
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
    # UTC の分（v4 までは datetime の内部単位が ns 以外だと値がずれるバグがあった）
    tmin = ((df["t"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    allow = ~in_stop
    if start_utc is not None:
        allow &= df["t"].to_numpy() >= np.datetime64(start_utc.tz_convert("UTC").tz_localize(None))
    return in_stop, allow, day, tmin

# ───────── 売買シミュレーション（Pine の仮想売買と同一）─────────
def simulate(o, h, l, sig, allow, in_stop, day, tmin, i0, i1, tp, sl, pip, spread,
             time_exit, close_in_stop, stop_on_win, win_n, win_within, rev, doten):
    """tp / sl は数値（全取引共通）でも、足ごとの配列（シグナル足の値をそのエントリーに使う）でもよい"""
    n = len(o)
    tp_a = np.full(n, float(tp)) if np.isscalar(tp) else np.asarray(tp, float)
    sl_a = np.full(n, float(sl)) if np.isscalar(sl) else np.asarray(sl, float)
    return _simulate(o, h, l, sig, allow, in_stop, day, tmin, i0, i1, tp_a, sl_a, pip, spread,
                     time_exit, close_in_stop, stop_on_win, win_n, win_within, rev, doten)


@njit(cache=True)
def _simulate(o, h, l, sig, allow, in_stop, day, tmin, i0, i1, tp, sl, pip, spread,
             time_exit, close_in_stop, stop_on_win, win_n, win_within, rev, doten):
    n = i1
    res_pips = np.zeros(200000); res_exit_i = np.zeros(200000, np.int64); res_entry_i = np.zeros(200000, np.int64)
    nt = 0
    pos = 0; entry = 0.0; entry_t = 0; entry_i = 0
    pend_dir = 0; pend_close = False
    pend_tp = 0.0; pend_sl = 0.0; cur_tp = 0.0; cur_sl = 0.0
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
            cur_tp = pend_tp; cur_sl = pend_sl
        pend_dir = 0
        if pos != 0 and not closed:
            tpp = entry + pos * cur_tp * pip; slp = entry - pos * cur_sl * pip
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
                if ex == tpp: p = cur_tp - spread
                elif ex == slp: p = -cur_sl - spread
                else: p = (ex - entry) / pip * pos - spread
                closed = True; pos = 0
        if closed:
            if nt < 200000:
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
        if d != 0 and allow[i] and not halted and i + 1 < n:
            if pos == 0 and not pend_close:
                pend_dir = d; pend_tp = tp[i]; pend_sl = sl[i]
            elif pos != 0 and doten and d == -pos:
                pend_close = True
                pend_dir = d; pend_tp = tp[i]; pend_sl = sl[i]
        # 高速化: 手ぶらで保留も無ければ次のシグナルまで飛ばす
        if pos == 0 and pend_dir == 0 and not pend_close:
            j = i + 1
            while j < n and sig[j] == 0:
                j += 1
            i = j
        else:
            i += 1
    k = min(nt, 200000)
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
                 mode=[2, 1, 0], adx=[(lo, up) for up in (40, 45, 50, 60, 100) for lo in (20, 25, 30, 33)],
                 adx_state=[True, False], tp=[5, 6, 7, 8, 10], sl=[5, 6, 7, 8, 10], rev=[False, True])
GRID_SMALL = dict(src=["close"], rci_len=[9, 11, 12, 14, 18, 20, 24], rci_thr=[77, 80, 82, 85],
                  mode=[2, 1, 0], adx=[(lo, up) for up in (45, 50, 100) for lo in (25, 33)],
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
            f = rci_arr if HAVE_NUMBA else rci_fast
            self._rci[k] = f(self.c if src == "close" else self.o, n)
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
        sig = signals(pd_.rci(src, n), thr, mode, pd_.reg(lo, up, st), pd_.c)
        if not sig[i0:i1].any():
            continue
        for tp, sl, rev in itertools.product(grid["tp"], grid["sl"], grid["rev"]):
            p, _, _ = simulate(pd_.o, pd_.h, pd_.l, sig, pd_.allow0, pd_.in_stop, pd_.day, pd_.tmin, i0, i1,
                               float(tp), float(sl), pd_.pip, base.spread, base.time_exit, base.close_in_stop,
                               base.stop_on_win, base.win_n, base.win_within, rev, base.doten)
            if len(p) < min_trades:
                continue
            s = stats(p)
            rows.append((s["net_pips"], dict(src=src, rci_len=n, rci_thr=thr, mode=["逆張り", "順張り", "押し目"][mode],
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
            sig = signals(pdx.rci(b["src"], b["rci_len"]), b["rci_thr"], ["逆張り", "順張り", "押し目"].index(b["mode"]),
                          pdx.reg(b["adx_lower"], b["adx_upper"], b["adx_state"]), pdx.c)
            p, _, _ = simulate(pdx.o, pdx.h, pdx.l, sig, pdx.allow0, pdx.in_stop, pdx.day, pdx.tmin, j0, j1,
                               float(b["tp"]), float(b["sl"]), pdx.pip, base.spread, base.time_exit, base.close_in_stop,
                               base.stop_on_win, base.win_n, base.win_within, b["reverse"], base.doten)
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


def parse_list(sv, f=float):
    return [f(x) for x in sv.split(",") if x.strip()]


def cmd_adaptive(a, base):
    """Pine hiropi_skya_auto_strategy_v15 と同じ手順:
    候補ごとに全期間を仮想売買 → 見直し間隔の区切りをまたいだ最初の足で、直近 lookback の損益で候補を選ぶ
    → 実売買はその時点の選択候補のシグナルだけで行う。"""
    os.makedirs(a.out, exist_ok=True)
    lens = parse_list(a.cand_lens, int); thrs = parse_list(a.cand_thrs)
    adxs = [tuple(float(v) for v in x.split("-")) for x in a.cand_adx.split(",")]
    revs = {"both": [False, True], "off": [False], "on": [True]}[a.cand_rev]
    tpsls = [tuple(float(v) for v in x.split(":")) for x in a.cand_tpsl.split(",")]
    cands = [(L, T, lo, up, rv, tp, sl) for L in lens for T in thrs for (lo, up) in adxs for rv in revs for (tp, sl) in tpsls]
    wins = [float(x) for x in a.wins.split(",")] if a.select == "ratio" else [a.lookback_h]
    w_slots = [max(1, int(round(w * 60 / a.reopt))) for w in wins]
    n_slots = max(w_slots)
    mode = ["逆張り", "順張り", "押し目"].index(a.mode)
    summary = []
    for pair in a.pairs:
        df = load_pair(a.data, pair)
        if df is None:
            print(f"[{pair}] データなし"); continue
        # 評価期間の分だけ前から読み込む（助走）
        t0 = pd.Timestamp(a.dfrom, tz=base.tz) - pd.Timedelta(hours=max(wins) + 24)
        t1 = pd.Timestamp(a.dto, tz=base.tz) + pd.Timedelta(days=1)
        df = df[(df.t >= t0) & (df.t < t1)].reset_index(drop=True)
        P = PairData(df, pair, base)
        n = len(df)
        sigs = np.zeros((len(cands), n), np.int8)
        # 見直しの足（Pine と同じ: 足の開始時刻が新しい区切りに入った最初の足）
        slot = (P.tmin // a.reopt)
        chk_idx = np.where(np.r_[True, slot[1:] != slot[:-1]])[0]
        cum = np.zeros((len(cands), len(chk_idx))); ntr = np.zeros((len(cands), len(chk_idx)))  # 見直し時点の累積値だけ持つ（メモリ節約）
        cand_tp = np.array([c[5] for c in cands]); cand_sl = np.array([c[6] for c in cands])
        for k, (L, T, lo, up, rv, tp, sl) in enumerate(cands):
            sg = signals(P.rci("close", L), T, mode, P.reg(lo, up, base.adx_state), P.c)
            if rv:
                sg = -sg
            sigs[k] = sg
            p, _, ex = simulate(P.o, P.h, P.l, sg, P.allow0, P.in_stop, P.day, P.tmin, 0, n,
                                float(tp), float(sl), P.pip, base.spread, base.time_exit, base.close_in_stop,
                                False, base.win_n, base.win_within, False, base.doten)
            cb = np.zeros(n); tb = np.zeros(n)
            np.add.at(cb, ex, p); np.add.at(tb, ex, 1)
            cum[k] = np.cumsum(cb)[chk_idx]; ntr[k] = np.cumsum(tb)[chk_idx]
        # 見直し（Pine と同じ: 足の開始時刻が新しい区切りに入った最初の足で、その足までの結果を使う）
        slot = (P.tmin // a.reopt)

        sel = np.full(n, -1)
        hist_c, hist_t = [], []
        cur = -1; log = []
        for j, i in enumerate(chk_idx):
            hist_c.append(cum[:, j]); hist_t.append(ntr[:, j])
            cnt = len(hist_c) - 1
            if a.select == "ratio":
                # v10 / Pine v13: 評価期間の候補（8h・6h・4h）ごとに 後半/前半 の伸び率を出し、一番高いものを選ぶ
                best_s = -np.inf; best = (-1, np.nan, 0)
                n_ok = 0
                for wi, ws in enumerate(w_slots):
                    back = min(ws, cnt); old = cnt - back; mid = cnt - back // 2
                    net = cum[:, j] - hist_c[old]; rec = cum[:, j] - hist_c[mid]; trs = ntr[:, j] - hist_t[old]
                    first = net - rec
                    okw = (first >= a.min_first) & (first > 0) & (rec > first) & (trs >= a.min_trades)   # v12: 前半 < 後半 のみ
                    n_ok += int(okw.sum())
                    if okw.any():
                        ratio = np.where(okw, rec / np.where(first > 0, first, 1), -np.inf)
                        k = int(np.argmax(ratio))   # 同率なら番号の小さい候補（Pine と同じ）
                        if ratio[k] > best_s:
                            best_s = ratio[k]; best = (k, float(net[k]), wi)
                cur = best[0]
                sel[i:] = cur
                if len(hist_c) > n_slots + 1:
                    hist_c[len(hist_c) - n_slots - 2] = None; hist_t[len(hist_t) - n_slots - 2] = None
                log.append((df.t.iloc[i], cur, best[1], n_ok))
                continue
            back = min(n_slots, cnt)
            old = cnt - back; mid = cnt - back // 2
            net = cum[:, j] - hist_c[old]; rec = cum[:, j] - hist_c[mid]; trs = ntr[:, j] - hist_t[old]
            if a.recent == "accel":      # 後半が前半以上（後半 >= 全体の半分）
                rec_ok = (rec > 0) & (rec >= net - rec)
            elif a.recent == "plus":     # 後半もプラス
                rec_ok = rec > 0
            else:
                rec_ok = np.ones_like(rec, bool)
            ok = (net >= a.target) & (trs >= a.min_trades) & rec_ok
            cur = int(np.argmax(np.where(ok, net, -np.inf))) if ok.any() else -1
            sel[i:] = cur
            if len(hist_c) > n_slots + 1:
                hist_c[len(hist_c) - n_slots - 2] = None; hist_t[len(hist_t) - n_slots - 2] = None
            log.append((df.t.iloc[i], cur, float(net[cur]) if cur >= 0 else np.nan, int(ok.sum())))
        live = np.where(sel >= 0, sigs[np.clip(sel, 0, None), np.arange(n)], 0).astype(np.int8)
        live_tp = cand_tp[np.clip(sel, 0, None)]; live_sl = cand_sl[np.clip(sel, 0, None)]
        start_i = int(df.t.searchsorted(pd.Timestamp(a.dfrom, tz=base.tz)))
        allow = P.allow0.copy(); allow[:start_i] = False
        p, ei, xi = simulate(P.o, P.h, P.l, live, allow, P.in_stop, P.day, P.tmin, 0, n,
                             live_tp, live_sl, P.pip, base.spread, base.time_exit, base.close_in_stop,
                             base.stop_on_win, base.win_n, base.win_within, False, base.doten)
        st = stats(p)
        print(f"[{pair}] 実売買 {st}")
        tr = pd.DataFrame({"entry": df.t.iloc[ei].dt.tz_convert(base.tz).values, "exit": df.t.iloc[xi].dt.tz_convert(base.tz).values, "pips": p})
        tr.to_csv(os.path.join(a.out, f"{pair}_trades.csv"), index=False, encoding="utf-8-sig")
        lg = pd.DataFrame(log, columns=["time", "sel", "net", "n_ok"])
        lg["params"] = [str(cands[k]) if k >= 0 else "なし" for k in lg.sel]
        lg.to_csv(os.path.join(a.out, f"{pair}_selection.csv"), index=False, encoding="utf-8-sig")
        tr["day"] = pd.to_datetime(tr.exit).dt.date
        daily = tr.groupby("day").pips.agg(["count", "sum"])
        daily.to_csv(os.path.join(a.out, f"{pair}_daily.csv"), encoding="utf-8-sig")
        summary.append(dict(pair=pair, **st, days=len(daily), win_days=int((daily["sum"] > 0).sum())))
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "adaptive_summary.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "walkforward", "adaptive"])
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
    ap.add_argument("--thai-stop", action="store_true", help="タイ時間 03:00-06:30 はエントリーなし")
    ap.add_argument("--win-stop", action="store_true", help="1時間以内に3連勝でその日終了")
    ap.add_argument("--no-doten", action="store_true", help="ドテンしない")
    # adaptive（Pine v5 と同じ既定値）
    ap.add_argument("--reopt", type=int, default=30, help="見直し間隔（分）")
    ap.add_argument("--preset", choices=["8h", "48h", "custom"], default="8h", help="8h=8時間・30pips（Pine v8 既定）/ 48h=48時間・100pips / custom=--lookback-h と --target")
    ap.add_argument("--lookback-h", type=float, default=8, help="評価期間（時間・custom 時）")
    ap.add_argument("--select", choices=["ratio", "net"], default="ratio", help="ratio=後半/前半の伸び率が最大（Pine v13 既定）/ net=これまでの採用ライン方式")
    ap.add_argument("--wins", default="8,6,4", help="ratio 方式の評価期間の候補（時間・前半後半は半分ずつ）")
    ap.add_argument("--min-first", type=float, default=20, help="ratio 方式: 前半の最低 pips（小さすぎる前半で伸び率が極端になるのを防ぐ）")
    ap.add_argument("--recent", choices=["plus", "accel", "none"], default="plus", help="後半の条件: plus=後半もプラス / accel=後半が前半以上 / none=条件なし")
    ap.add_argument("--cand-lens", default="7,9,11,14,18")
    ap.add_argument("--cand-thrs", default="77,82,87")
    ap.add_argument("--cand-adx", default="25-45,33-45,33-60")
    ap.add_argument("--cand-rev", choices=["both", "off", "on"], default="both")
    ap.add_argument("--cand-tpsl", default="7:6", help="利確:損切り の組（pips）")
    ap.add_argument("--mode", choices=["押し目", "順張り", "逆張り"], default="押し目")
    a = ap.parse_args()
    base = Cfg(tz=a.tz, spread=a.spread, doten=not a.no_doten, stop_on_win=a.win_stop)
    if a.thai_stop:
        base.stops = (("0500", "0830"),) if a.tz == "Asia/Tokyo" else (("0300", "0630"),)
    print(f"numba: {'あり' if HAVE_NUMBA else 'なし（遅い。pip install numba 推奨）'}")
    if a.cmd == "adaptive":
        if a.preset == "8h":
            a.lookback_h, a.target = 8, 30
        elif a.preset == "48h":
            a.lookback_h, a.target = 48, 100
        if a.min_trades == 6 and "--min-trades" not in sys.argv:
            a.min_trades = 5
    {"search": cmd_search, "walkforward": cmd_walkforward, "adaptive": cmd_adaptive}[a.cmd](a, base)


if __name__ == "__main__":
    main()
