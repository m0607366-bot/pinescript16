# シグナル × 時間帯 × 利確損切りの「コスト前」成績を、前半(2026-01〜06)/後半(07〜09)で比べる v1
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude'); import hiropi_skya_optimize_v15 as m
pair = sys.argv[1]
df = m.load_pair('/home/claude/data1m', pair); b = m.Cfg(); P = m.PairData(df, pair, b)
t = df.t; i0 = int(t.searchsorted(pd.Timestamp('2026-01-06', tz='UTC'))); im = int(t.searchsorted(pd.Timestamp('2026-07-01', tz='UTC')))
uh = t.dt.hour.to_numpy()
sessions = {'全時間': np.ones(len(df), bool),
            '東京(UTC0-8)': (uh >= 0) & (uh < 8),
            'ロンドン(UTC7-16)': (uh >= 7) & (uh < 16),
            'NY(UTC12-21)': (uh >= 12) & (uh < 21),
            'ロンドンNY重複(UTC12-16)': (uh >= 12) & (uh < 16)}
h, l, c = P.h, P.l, P.c
adx = m.adx_arr(h, l, c, 14, 14)
def breakout(N, adx_min):
    hh = pd.Series(h).rolling(N).max().shift(1).to_numpy(); ll = pd.Series(l).rolling(N).min().shift(1).to_numpy()
    c1 = np.r_[np.nan, c[:-1]]; hh1 = np.r_[np.nan, hh[:-1]]; ll1 = np.r_[np.nan, ll[:-1]]
    with np.errstate(invalid='ignore'):
        buy = (c > hh) & ~(c1 > hh1); sell = (c < ll) & ~(c1 < ll1)
        if adx_min > 0:
            buy &= adx >= adx_min; sell &= adx >= adx_min
    return np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)
sigdefs = {}
for L in (9, 14, 23):
    for mode, nm in ((2, '押し目'), (1, '順張り'), (0, '逆張り')):
        sigdefs[f'RCI{L} {nm} ±82 ADX25-45'] = m.signals(P.rci('close', L), 82, mode, P.reg(25, 45, True), P.c)
for N in (20, 60, 240):
    for am in (0, 25):
        sigdefs[f'ブレイクアウト{N}本' + (f' ADX≥{am}' if am else '')] = breakout(N, am)
rows = []
for (sn, sg), (ssn, smask), (tp, sl) in itertools.product(sigdefs.items(), sessions.items(), ((7, 6), (12, 10), (20, 15))):
    for rev in (False, True):
        s2 = -sg if rev else sg
        p, ei, _ = m.simulate(P.o, P.h, P.l, s2, P.allow0 & smask, P.in_stop, P.day, P.tmin, i0, len(df), float(tp), float(sl), P.pip, 0.0, 0, False, False, 3, 60, False, False)
        r = dict(pair=pair, signal=sn, rev=rev, session=ssn, tp=tp, sl=sl)
        for nm, msk in (('前半', ei < im), ('後半', ei >= im)):
            q = p[msk]; r[nm + '_n'] = len(q); r[nm + '_avg'] = round(float(q.mean()), 3) if len(q) else np.nan; r[nm + '_sum'] = round(float(q.sum()), 1)
        rows.append(r)
pd.DataFrame(rows).to_csv(f'/home/claude/edge/{pair}.csv', index=False)
print(pair, 'done', len(rows))
