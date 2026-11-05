# 固定パラメーターの前半(2026-01〜06)/後半(07〜09)検証 v1。コスト(pips)は引数。hiropi_skya_optimize_v13 を使用
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude'); import hiropi_skya_optimize_v13 as m
pair=sys.argv[1]; cost=float(sys.argv[2])
df=m.load_pair('/home/claude/data1m',pair); b=m.Cfg(); P=m.PairData(df,pair,b)
t=df.t; tz='Asia/Tokyo'
i0=int(t.searchsorted(pd.Timestamp('2026-01-06',tz=tz))); im=int(t.searchsorted(pd.Timestamp('2026-07-01',tz=tz))); i1=len(df)
lens=[6,10,14,18,23]; thrs=[77,82,87]; adxs=[(25,35),(30,40),(35,45),(25,45),(33,45),(33,60)]
tpsl=[(7,6),(9,7),(10,8),(12,10),(14,10),(15,12),(18,14),(20,15)]
rows=[]
for L,T,(lo,up),rv in itertools.product(lens,thrs,adxs,[False,True]):
    sg=m.signals(P.rci('close',L),T,2,P.reg(lo,up,True),P.c)
    if rv: sg=-sg
    for tp,sl in tpsl:
        dt=False
        r=dict(pair=pair,rci=L,thr=T,adx=f'{lo}-{up}',rev=rv,tp=tp,sl=sl,doten=dt)
        p,ei,_=m.simulate(P.o,P.h,P.l,sg,P.allow0,P.in_stop,P.day,P.tmin,i0,i1,float(tp),float(sl),P.pip,cost,0,False,False,3,60,False,dt)
        for nm,msk in (('IS',ei<im),('OOS',ei>=im)):
            s=m.stats(p[msk]); r[nm+'_n']=s['trades']; r[nm+'_pips']=s['net_pips']; r[nm+'_pf']=s['pf']
        rows.append(r)
pd.DataFrame(rows).to_csv(f'/home/claude/opt/{pair}_c{cost}.csv',index=False)
print(pair,'done',len(rows))
