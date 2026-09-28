"""Download the Rotterdam breast-cancer data used in the application.

We obtain `cancer.rda` from a fixed version of the R package `survival`. After downloading the file,
we check its checksum so that the analysis is run on the same data file each time. Run this script once
before running the Rotterdam analysis.
"""
from pathlib import Path
import hashlib, urllib.request

DATA = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/cran/survival/3.8-6/data/cancer.rda'
MD5 = '77bee419c5f0f8133f4a566f0defc048'

if __name__ == '__main__':
    dest = DATA/'cancer.rda'
    urllib.request.urlretrieve(URL, dest)
    md5 = hashlib.md5(dest.read_bytes()).hexdigest()
    if md5 != MD5:
        raise SystemExit(f'checksum mismatch for {dest}: {md5} (expected {MD5})')
    print(f'cancer.rda: {dest.stat().st_size:,} bytes, checksum OK')
