#!/usr/bin/env python3
"""
4時間高安の逆張り（NY時間） Python 検証 v1
対応: fade4h_ny_strategy_v1.pine / fade4h_ny_indicator_v1.pine（同じロジック・同じ既定値）

ロジック
  ・1分足で、直前 N 本（既定240）の高値を終値が初めて上抜け → 売り、安値を初めて下抜け → 買い（逆張り）
  ・ADX(14,14) >= 下限（既定25、0で条件なし）のときだけ
  ・エントリーは NY時間（タイ時間 19:00〜04:00 = UTC 12〜21時）のシグナル足だけ
  ・次の足の始値で成行、利確20・損切り15pips、保有中のシグナルは無視（ドテンなし）
  ・足内の約定順は Pine と同じ仮定（始値→近い方の高安→もう一方→終値）
  ・損益(pips) = 値幅 − コスト（既定2.2 = CHFJPY のスプレッド約1.2 + 手数料約0.95）

使い方
  python3 fade4h_ny_backtest_v1.py --data <年別CSVのフォルダ> --pairs chfjpy --from 2019-01-06 --to 2026-09-30 --out result_fade
  （<ペア小文字><年>.csv、<TICKER>,<DTYYYYMMDD>,<TIME>,... UTC。hiropi_skya_optimize_v15.py と同じフォルダに置いて実行）
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiropi_skya_optimize_v15 as m


def fade_signals(P, lookN, adx_min, follow=False):
    h, l, c = P.h, P.l, P.c
    hh = pd.Series(h).rolling(lookN).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(lookN).min().shift(1).to_numpy()
    c1 = np.r_[np.nan, c[:-1]]; hh1 = np.r_[np.nan, hh[:-1]]; ll1 = np.r_[np.nan, ll[:-1]]
    adx = m.adx_arr(h, l, c, 14, 14)
    with np.errstate(invalid="ignore"):
        up = (c > hh) & ~(c1 > hh1)
        dn = (c < ll) & ~(c1 < ll1)
        if adx_min > 0:
            ok = adx >= adx_min
            up &= ok; dn &= ok
    s = np.where(dn, 1, np.where(up, -1, 0)).astype(np.int8)
    return -s if follow else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", nargs="+", default=["chfjpy"])
    ap.add_argument("--from", dest="dfrom", default="2019-01-06")
    ap.add_argument("--to", dest="dto", default="2026-09-30")
    ap.add_argument("--look", type=int, default=240)
    ap.add_argument("--adx-min", type=float, default=25)
    ap.add_argument("--tp", type=float, default=20)
    ap.add_argument("--sl", type=float, default=15)
    ap.add_argument("--cost", type=float, default=2.2)
    ap.add_argument("--utc-from", type=int, default=12, help="エントリーできる時間帯（UTC時）の開始")
    ap.add_argument("--utc-to", type=int, default=21, help="同 終了（この時は含まない）")
    ap.add_argument("--follow", action="store_true", help="逆神モード（更新方向に乗る）")
    ap.add_argument("--out", default="result_fade")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allrows = []
    for pair in a.pairs:
        df = m.load_pair(a.data, pair)
        if df is None:
            print(f"[{pair}] データなし"); continue
        P = m.PairData(df, pair, m.Cfg())
        uh = df.t.dt.hour.to_numpy()
        sess = (uh >= a.utc_from) & (uh < a.utc_to) if a.utc_from < a.utc_to else ((uh >= a.utc_from) | (uh < a.utc_to))
        sg = fade_signals(P, a.look, a.adx_min, a.follow)
        i0 = int(df.t.searchsorted(pd.Timestamp(a.dfrom, tz="UTC"))); i1 = int(df.t.searchsorted(pd.Timestamp(a.dto, tz="UTC")))
        p, ei, xi = m.simulate(P.o, P.h, P.l, sg, P.allow0 & sess, P.in_stop, P.day, P.tmin, i0, i1,
                               a.tp, a.sl, P.pip, a.cost, 0, False, False, 3, 60, False, False)
        tr = pd.DataFrame({"entry_utc": df.t.iloc[ei].values, "exit_utc": df.t.iloc[xi].values,
                           "dir": [int(np.sign(sg[i - 1])) if i > 0 else 0 for i in ei], "pips": p})
        tr.to_csv(os.path.join(a.out, f"{pair}_trades.csv"), index=False, encoding="utf-8-sig")
        tr["year"] = pd.to_datetime(tr.entry_utc).dt.year; tr["month"] = pd.to_datetime(tr.entry_utc).dt.strftime("%Y-%m")
        y = tr.groupby("year").pips.agg(["count", "sum", "mean"]).round(2)
        y["win%"] = tr.groupby("year").pips.apply(lambda s: round(100 * (s > 0).mean(), 1))
        mon = tr.groupby("month").pips.sum()
        print(f"\n[{pair}] 損切り{a.sl}:利確{a.tp} / {a.look}本 / ADX≥{a.adx_min} / コスト{a.cost}")
        print(y.to_string())
        print(f"合計 {tr.pips.sum():.1f}pips・{len(tr)}回・プラスの月 {int((mon > 0).sum())}/{len(mon)}")
        y.assign(pair=pair).to_csv(os.path.join(a.out, f"{pair}_yearly.csv"), encoding="utf-8-sig")
        allrows.append(dict(pair=pair, trades=len(tr), net_pips=round(tr.pips.sum(), 1), avg=round(tr.pips.mean(), 3) if len(tr) else 0,
                            plus_months=int((mon > 0).sum()), months=len(mon)))
    pd.DataFrame(allrows).to_csv(os.path.join(a.out, "summary.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
