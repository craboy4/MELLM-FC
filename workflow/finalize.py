"""Freeze final pool on development validation, then fit <=2021 and score 2022-24 once."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluator import metrics
from common import *

def select(ev,pool,cfg):
    status(phase='final_selection_validation_only')
    full=list(pool.exprs)
    remaining=sorted(range(len(full)),key=lambda i:-abs(pool.weights[i]))
    selected=[];score,_=ev.score([],record=False);trace=[]
    for _ in range(min(cfg['selection']['max_factors'],len(full))):
        def trial(i):
            s,_=ev.score([full[j] for j in selected+[i]],record=False)
            return i,s
        with ThreadPoolExecutor(max_workers=cfg['numeric_workers']) as ex: results=list(ex.map(trial,remaining))
        i,proposal=max(results,key=lambda p:p[1]['validation']['AUC'])
        force=len(selected)<min(cfg['selection']['min_factors'],len(full))
        if proposal['validation']['AUC']<=score['validation']['AUC']+1e-5 and not force:break
        selected.append(i);remaining.remove(i);score=proposal
        trace.append(dict(added=str(full[i]),validation_auc=score['validation']['AUC'],minimum_count_forced=force))
        if not remaining:break
    subset=[full[i] for i in selected]
    fullscore,_=ev.score(full,record=False);basescore,_=ev.score([],record=False)
    options=[('selected',subset,score),('full',full,fullscore),('Base',[],basescore)]
    name,expressions,best=max(options,key=lambda p:p[2]['validation']['AUC'])
    receipt=dict(selection=name,expressions=[str(x) for x in expressions],score=best,trace=trace,
        alternatives={n:dict(expressions=[str(x) for x in expr],score=s) for n,expr,s in options},test_opened=False,utc=now())
    save(RUN/'final_selection.json',receipt)
    return expressions,best

def finalize(ev,pool,cfg):
    expressions,_=select(ev,pool,cfg)
    save(RUN/'generation_complete.json',dict(status='complete',rounds=cfg['rounds'],
         expressions=[str(e) for e in expressions],test_opened=False,
         note='Generation-stage output; use reproduce.py for the archived selected main-table recipe.'))
    status(state='complete',phase='generation_complete_validation_only',completed_rounds=cfg['rounds'])
    event('complete',rounds=cfg['rounds'],test_opened=False)
