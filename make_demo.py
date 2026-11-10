"""Create a synthetic fixture from a PRNG only; this is not the licensed paper sample."""
from pathlib import Path
import numpy as np
import pandas as pd

def make(path):
    rng=np.random.default_rng(20261109)
    rows=[]
    for firm in range(1,31):
        size=rng.lognormal(12,1)
        for year in range(2010,2025):
            v=size*rng.lognormal(0,.5,25)
            v[17]=size/10
            v[18]=rng.lognormal(2,.4)
            v[23]=rng.normal(0,.15)*size
            v[24]=v[19]-v[20]
            label=int(rng.random()<(.07+.2*(v[23]<0)))
            row=dict(firm_id=f'{990000+firm:06d}',year=year,label=label)
            row.update({f'N{i+1}':float(x) for i,x in enumerate(v)})
            if (firm+year)%17==0:row['N3']=np.nan
            rows.append(row)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')

if __name__=='__main__':
    make(Path(__file__).resolve().parent/'data/synthetic_demo.csv')
