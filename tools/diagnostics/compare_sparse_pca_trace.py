from pathlib import Path
import sys
import numpy as np

trace=Path(sys.argv[1]); X=np.loadtxt(sys.argv[2],delimiter=',',skiprows=1); init=np.loadtxt(sys.argv[3],delimiter=',',skiprows=1)
s=init.reshape(-1); Xd=X.copy(); scores=[]; loadings=[]

def norm(a):
    return a.copy() if np.all(a==0) else a/np.sqrt(a@a)
def sparse(x, lam=.2):
    if np.all(x==0): lam=0
    elif lam>=np.max(np.abs(x)):
        d=np.diff(np.sort(np.unique(np.abs(x)))); d=d[d>1e-8]
        lam=np.max(np.abs(x))-(d[-1]/2 if d.size else 1e-6)
    return np.sign(x)*np.maximum(np.abs(x)-lam,0)
def check(name,a):
    r=np.loadtxt(trace/name,delimiter=',',ndmin=1)
    d=np.max(np.abs(a.reshape(-1)-r.reshape(-1)))
    return d
for c in range(1,7):
    for i in range(1,41):
        raw=Xd.T@s; nw=norm(raw); sw=sparse(nw); w=norm(sw); s=Xd@w
        vals={'raw_w':raw,'norm_w':nw,'sparse_w':sw,'w':w,'s':s}
        diffs={k:check(f'c{c:02d}_i{i:02d}_{k}.csv',v) for k,v in vals.items()}
        if max(diffs.values())>1e-12:
            print('FIRST DIFFERENCE',c,i,diffs); raise SystemExit(1)
    scores.append(s.copy()); loadings.append(w.copy())
    if c<6:
        S=np.column_stack(scores); cross=S.T@S; e=0
        while True:
            try: B=np.linalg.solve(cross+1e-18*10**e*np.eye(cross.shape[0]),S.T@X); break
            except np.linalg.LinAlgError: e+=1
        Xd=X-S@B
        d=check(f'c{c:02d}_residual_after_deflation.csv',Xd)
        print('component',c,'post-deflation max diff',format(d,'.17g'))
print('All iteration states agree within 1e-12')
