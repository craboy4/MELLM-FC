"""Local-only experiment I/O, immutable API receipts, and progress records."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone
import urllib.request
import urllib.error
from email.utils import parsedate_to_datetime
import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = Path(os.environ.get('MELLM_WORKDIR', str(ROOT / 'outputs' / 'generation'))).resolve()
RUN = DATA / 'run'
BASE = [f'N{i}' for i in range(1, 26)]
LOCK = threading.RLock()

def now():
    return datetime.now(timezone.utc).isoformat()

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple, set)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def dumps(x):
    return json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False)

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with LOCK:
        tmp=path.with_name(path.name+'.tmp')
        tmp.write_text(dumps(value)+'\n',encoding='utf-8')
        for attempt in range(6):
            try:os.replace(tmp,path);break
            except PermissionError:
                if attempt==5:raise
                time.sleep(0.03*(attempt+1))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(x): return hashlib.sha256(dumps(x).encode()).hexdigest()
def symbol(value): return str(int(float(value))).zfill(6)

def event(kind, **fields):
    RUN.mkdir(parents=True, exist_ok=True)
    row=clean(dict(utc=now(),kind=kind,**fields))
    with LOCK:
        with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if kind in ('error','api_retry','complete'):
            with (RUN/'alerts.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def status(**fields):
    with LOCK:
        p=RUN/'status.json'; old=read(p) if p.exists() else {}
        old.update(clean(fields),updated_utc=now())
        save(p,old)

def load_local_key():
    key=os.environ.get('MELLM_API_KEY')
    if not key: raise RuntimeError('Set MELLM_API_KEY in your environment; no credential stores are searched.')
    return key

class ContextWindowExceeded(RuntimeError):
    pass

def is_context_rejection(message):
    text=str(message).lower()
    return ('exceeds the context window' in text or 'context_length_exceeded' in text
            or 'maximum context length' in text)

class LocalLuna:
    def __init__(self, config):
        self.config=config
        self.key=load_local_key()
        self.semaphore=threading.BoundedSemaphore(3)
        self.active=0

    def _http(self, route, payload=None):
        body=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
        req=urllib.request.Request(self.config['llm']['base_url']+route,data=body,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs): return None
        with urllib.request.build_opener(NoRedirect).open(req,timeout=900) as r:
            return json.load(r)

    def verify(self):
        response=self._http('/models')
        assert self.config['llm']['model'] in [x['id'] for x in response['data']]
        return {'endpoint':self.config['llm']['base_url'],'model':self.config['llm']['model'],'authentication':'user-configured endpoint; credential not exported'}

    def call(self, folder, system, prompt, purpose):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        payload=dict(model='gpt-6-luna',messages=[{'role':'system','content':system},{'role':'user','content':prompt}],
                     reasoning_effort='medium',service_tier='priority',stream=False)
        request_hash=identity(payload)
        with self.semaphore:
            if (folder/'request.json').exists():
                assert identity(read(folder/'request.json'))==request_hash, 'Resume input differs from saved request'
            else: save(folder/'request.json',payload)
            (folder/'prompt.txt').write_text(prompt,encoding='utf-8')
            (folder/'system.txt').write_text(system,encoding='utf-8')
            if (folder/'response.json').exists():
                response=read(folder/'response.json')
            else:
                for failed in folder.glob('attempt_*.failed.json'):
                    message=read(failed).get('message','')
                    if is_context_rejection(message): raise ContextWindowExceeded(message)
                ceiling=5
                for attempt in range(1,ceiling+1):
                    prefix=folder/f'attempt_{attempt:02d}'
                    if prefix.with_suffix('.started.json').exists():
                        if prefix.with_suffix('.failed.json').exists(): continue
                        raise RuntimeError(f'Uncertain API outcome in {folder}; do not silently redispatch')
                    save(prefix.with_suffix('.started.json'),dict(utc=now(),purpose=purpose,request_hash=request_hash))
                    with LOCK:
                        self.active+=1; status(api_active=self.active)
                    started=time.monotonic()
                    try:
                        response=self._http('/chat/completions',payload)
                        save(folder/'response.json',response)
                    except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                        code=getattr(exc,'code',None)
                        message=str(exc).replace(self.key,'[redacted]')
                        if isinstance(exc,urllib.error.HTTPError): message=exc.read().decode('utf-8',errors='replace').replace(self.key,'[redacted]')
                        if code is None:
                            save(prefix.with_suffix('.uncertain.json'),dict(utc=now(),message=message,outcome='unknown; no automatic redispatch'))
                            event('error',purpose=purpose,attempt=attempt,detail='Uncertain transport outcome; request retained, no redispatch',transport_detail=message)
                            raise RuntimeError(f'Uncertain transport outcome in {folder}; do not redispatch') from None
                        save(prefix.with_suffix('.failed.json'),dict(utc=now(),status=code,message=message,upstream_cost_unknown=True))
                        if is_context_rejection(message):
                            event('context_rejection',purpose=purpose,attempt=attempt,http_status=code,detail=message)
                            raise ContextWindowExceeded(message) from None
                        retryable=code in (429,500,502,503,504) and attempt<ceiling
                        delay=[2,5,10,20,60][(attempt-1)%5]
                        if code in (429,503):delay=max(delay,60)
                        header=(getattr(exc,'headers',None) or {}).get('Retry-After')
                        if header:
                            try:requested=float(header)
                            except (ValueError,TypeError):
                                try:requested=parsedate_to_datetime(header).timestamp()-time.time()
                                except (ValueError,TypeError,OverflowError):requested=0
                            if np.isfinite(requested):delay=max(delay,requested)
                        event('api_retry' if retryable else 'error',purpose=purpose,attempt=attempt,http_status=code,detail=message,retry_seconds=delay if retryable else None)
                        if not retryable:raise RuntimeError(message) from None
                        time.sleep(delay)
                    else:
                        event('api_complete',purpose=purpose,seconds=time.monotonic()-started,usage=response.get('usage'),request_characters=len(prompt)+len(system))
                        break
                    finally:
                        with LOCK:
                            self.active-=1; status(api_active=self.active)
                else: raise RuntimeError('Transport retry budget exhausted')
            assert response.get('model','').lower()=='gpt-6-luna', 'Different model returned; stop'
            choice=response['choices'][0]
            if choice.get('finish_reason') not in ('stop',None):
                raise RuntimeError(f'Incomplete model response: {choice.get("finish_reason")}; response retained, no truncation/retry for quality')
            text=choice['message'].get('content')
            if not isinstance(text,str) or not text.strip(): raise RuntimeError('Empty model content; receipt saved')
            (folder/'response.txt').write_text(text,encoding='utf-8')
            return text
