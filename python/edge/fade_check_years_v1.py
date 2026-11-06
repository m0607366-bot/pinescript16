# 「240本ブレイクアウトの逆（高値・安値の更新を逆張り）」を別の年でも確かめる v1
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude'); import hiropi_skya_optimize_v15 as m
pair, folder, start, end = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
df = m.load_pair(folder, pair); b = m.Cfg(); P = m.PairData(df, pair, b)
t = df.t; i0 = int(t.searchsorted(pd.Timestamp(start, tz='UTC'))); i1 = int(t.searchsorted(pd.Timestamp(end, tz='UTC')))
uh = t.dt.hour.to_numpy()
sess = {'NY(UTC12-21)': (uh >= 12) & (uh < 21), '全時間': np.ones(len(df), bool), '東京(UTC0-8)': (uh >= 0) & (uh < 8)}
h, l, c = P.h, P.l, P.c; adx = m.adx_arr(h, l, c, 14, 14)
def bo(N, am):
    hh = pd.Series(h).rolling(N).max().shift(1).to_numpy(); ll = pd.Series(l).rolling(N).min().shift(1).to_numpy()
    c1 = np.r_[np.nan, c[:-1]]; hh1 = np.r_[np.nan, hh[:-1]]; ll1 = np.r_[np.nan, ll[:-1]]
    with np.errstate(invalid='ignore'):
        buy = (c > hh) & ~(c1 > hh1); sell = (c < ll) & ~(c1 < ll1)
        if am: buy &= adx >= am; sell &= adx >= am
    return np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)
rows = []
for N, am, (sn, sm), (tp, sl) in itertools.product((120, 240, 480), (0, 25), [('NY(UTC12-21)', sess['NY(UTC12-21)'])], ((20, 15), (25, 20))):
    sg = -bo(N, am)   # 逆（フェード）
    p, ei, xi = m.simulate(P.o, P.h, P.l, sg, P.allow0 & sm, P.in_stop, P.day, P.tmin, i0, i1, float(tp), float(sl), P.pip, 0.0, 0, False, False, 3, 60, False, False)
    mon = pd.Series(p, index=t.iloc[ei].dt.strftime('%Y-%m').values).groupby(level=0).sum()
    rows.append(dict(pair=pair, N=N, adx=am, session=sn, tp=tp, sl=sl, n=len(p), avg=round(float(p.mean()), 3) if len(p) else np.nan,
                     net15=round(float((p - 1.5).sum()), 1), months=len(mon), plus_months_net=int(((mon - 1.5 * pd.Series(1, index=t.iloc[ei].dt.strftime('%Y-%m').values).groupby(level=0).sum()) > 0).sum())))
pd.DataFrame(rows).to_csv(f'/home/claude/edge/fade_{pair}_{start[:4]}.csv', index=False)
print(pair, start, 'done')
