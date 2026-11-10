"""Explicit opt-in launcher for the optional, costly, Windows research-generation workflow."""
import argparse
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workdir',type=Path,default=Path(__file__).resolve().parent/'outputs/generation')
    ap.add_argument('--allow-api',action='store_true')
    ap.add_argument('--allow-development-context-upload',action='store_true')
    ap.add_argument('--stop-after',type=int)
    args=ap.parse_args()
    if not args.allow_api or not args.allow_development_context_upload:
        ap.error('Explicit --allow-api and --allow-development-context-upload are required. Numeric reproduction uses reproduce.py and requires neither.')
    if os.name!='nt':
        ap.error('The archived generation launcher uses Windows process locking. Numeric reproduction is platform-independent; this launcher is supported on Windows only.')
    endpoint=os.environ.get('MELLM_BASE_URL','').rstrip('/')
    u=urlparse(endpoint)
    if not u.hostname or u.username or u.password or u.query or u.fragment:
        ap.error('Set MELLM_BASE_URL to your API endpoint without credentials or query parameters.')
    if u.scheme!='https' and not (u.scheme=='http' and u.hostname in ('localhost','127.0.0.1','::1')):
        ap.error('Use HTTPS, or HTTP on a loopback endpoint only.')
    if not os.environ.get('MELLM_API_KEY'):
        ap.error('Set MELLM_API_KEY in your environment. Never put it in source files.')
    os.environ['MELLM_WORKDIR']=str(args.workdir.resolve())
    sys.path.insert(0,str(Path(__file__).resolve().parent/'workflow'))
    import common
    original_init=common.LocalLuna.__init__
    def init(self,config):
        original_init(self,config)
        self.config={**config,'llm':{**config['llm'],'base_url':endpoint}}
    common.LocalLuna.__init__=init
    import run_experiment
    sys.argv=[sys.argv[0]]+(['--stop-after',str(args.stop_after)] if args.stop_after is not None else [])
    run_experiment.main()

if __name__=='__main__':main()
