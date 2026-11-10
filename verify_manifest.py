"""Verify published file hashes without reading any private input or making API requests."""
from pathlib import Path
import hashlib,json

def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.json').read_text('utf-8'))
    bad=[]
    for name,expected in manifest['files'].items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            bad.append(name)
    if bad:raise SystemExit('Manifest mismatch: '+', '.join(bad))
    print(f"PASS: {len(manifest['files'])} published file hashes verified.")

if __name__=='__main__':main()
