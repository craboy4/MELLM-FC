"""CPU processes for unchanged expression evaluation; no model/API operations."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from fraud_alpha.expression import Feature

_FRAME=None
_TRAIN=None
_MASK=None
_LIMITER=None
_ACTIVE_BACKENDS=[]

def _initialize(frame,train,mask,features):
    global _FRAME,_TRAIN,_MASK,_LIMITER
    _FRAME=frame;_TRAIN=train;_MASK=mask
    Feature.set_available_features(features)
    _LIMITER=threadpool_limits(limits=1,user_api='blas')

def _evaluate(expr):
    tick=time.perf_counter()
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        values=np.asarray(expr.evaluate(_FRAME),float)
        before=np.asarray(expr.evaluate(_TRAIN),float)
    if values.shape!=(len(_FRAME),):raise ValueError('Expression shape mismatch')
    if not np.allclose(values[_MASK],before,equal_nan=True):raise ValueError('Future dependency or row alignment failure')
    values=np.where(np.isfinite(values),values,np.nan)
    return dict(values=values,worker_pid=os.getpid(),seconds=time.perf_counter()-tick)

class FeatureValueBackend:
    def __init__(self,frame,train,mask,features,workers=6):
        pythonw=Path(sys.executable).with_name('pythonw.exe')
        if os.name=='nt' and pythonw.exists():mp.set_executable(str(pythonw))
        self.executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=_initialize,initargs=(frame,train,mask,features))
        self.closed=False
        _ACTIVE_BACKENDS.append(self)
    def submit(self,expr):return self.executor.submit(_evaluate,expr)
    def evaluate(self,expr):return self.submit(expr).result()['values']
    def close(self):
        if not self.closed:
            self.closed=True;self.executor.shutdown(wait=True,cancel_futures=True)
            if self in _ACTIVE_BACKENDS:_ACTIVE_BACKENDS.remove(self)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def close_active_backends():
    for backend in list(_ACTIVE_BACKENDS):backend.close()
