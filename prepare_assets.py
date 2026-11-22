"""Prepare pinned public datasets and pretrained weights; verify all file hashes."""
import argparse
import csv
import hashlib
import shutil
import time
import urllib.request
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REV='52fe75dec70c1e8046493b4e3256f8f752a4d39d'
AM=f'AMdataset-{REV}'
MANIFEST=ROOT/'assets_manifest.csv'
WEIGHTS={
    'efficientnetv2-m_notop.h5': ('efficientnet_v2','1b8dc055df72dde80d614482840fe342'),
    'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5': ('inception_resnet_v2','d19885ff4a710c122648d3b5c3b684e4'),
}
KAGGLE='https://www.kaggle.com/api/v1/datasets/download/justin900429/3d-printer-defected-dataset?datasetVersionNumber=1'
KAGGLE_HASH='a24229ab7f3f4802fb7e0316b373f2b915b1faf30b00c45143479eacfaab9bcd'


def digest(path, algorithm='sha256'):
    with path.open('rb') as f:
        h=hashlib.new(algorithm)
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        return h.hexdigest()


def valid(path, target, algorithm='sha256'):
    return path.is_file() and digest(path,algorithm)==target


def download(url, dest, expected=None, algorithm='sha256'):
    if expected and valid(dest,expected,algorithm): return
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.part')
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'IDAT7215-reproducibility'})
            with urllib.request.urlopen(request,timeout=120) as response,tmp.open('wb') as out:
                shutil.copyfileobj(response,out,length=1024*1024)
            if expected and not valid(tmp,expected,algorithm): raise ValueError(f'Hash mismatch: {dest.name}')
            tmp.replace(dest)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt==3: raise
            time.sleep(2**attempt)


def entries():
    with MANIFEST.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def record_manifest():
    rows=[]
    for study,dirname,count in [('one',f'{AM}/dataset',271),('two',f'{AM}/Augmented_data',3017),('three','study3',1557)]:
        root=ROOT/'data'/dirname
        paths=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'} and not p.name.startswith('.'))
        assert len(paths)==count
        rows.extend(dict(study=study,path=p.relative_to(ROOT/'data').as_posix(),sha256=digest(p)) for p in paths)
    with MANIFEST.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['study','path','sha256']);w.writeheader();w.writerows(rows)


def check():
    rows=entries()
    missing=[row for row in rows if not valid(ROOT/'data'/row['path'],row['sha256'])]
    bad_weights=[name for name,(_,h) in WEIGHTS.items() if not valid(ROOT/'.keras'/'models'/name,h,'md5')]
    return missing,bad_weights


def extract_exact(archive, rows, prefix):
    with zipfile.ZipFile(archive) as z:
        available=set(z.namelist())
        for row in rows:
            rel=row['path']
            member=rel if prefix==AM else rel.removeprefix('study3/')
            if member not in available: continue
            dest=ROOT/'data'/rel
            if valid(dest,row['sha256']): continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix(dest.suffix+'.part')
            with z.open(member) as src,tmp.open('wb') as out: shutil.copyfileobj(src,out)
            if valid(tmp,row['sha256']): tmp.replace(dest)
            else: tmp.unlink(missing_ok=True)


def prepare():
    missing,bad_weights=check()
    am=[r for r in missing if r['study'] in ('one','two')]
    if am:
        # Pinned archive is preferred; individual raw files provide a resumable fallback.
        archive=ROOT/'downloads'/'AMdataset.zip'
        try:
            if not archive.exists(): download(f'https://codeload.github.com/mahsan2/AMdataset/zip/{REV}',archive)
            extract_exact(archive,am,AM)
        except Exception as e: print(f'Archive unavailable; retrying individual files: {e}',flush=True)
        am=[r for r in am if not valid(ROOT/'data'/r['path'],r['sha256'])]
        def one(row):
            raw=row['path'].removeprefix(AM+'/')
            download(f'https://raw.githubusercontent.com/mahsan2/AMdataset/{REV}/{urllib.parse.quote(raw)}',ROOT/'data'/row['path'],row['sha256'])
        with ThreadPoolExecutor(max_workers=5) as pool:
            for i,_ in enumerate(pool.map(one,am),1):
                if i%100==0 or i==len(am):print(f'AMdataset remaining files {i}/{len(am)}',flush=True)
    three=[r for r in missing if r['study']=='three']
    if three:
        archive=ROOT/'downloads'/'study3.zip'
        download(KAGGLE,archive,KAGGLE_HASH)
        extract_exact(archive,three,'study3')
    for name in bad_weights:
        folder,expected=WEIGHTS[name]
        download(f'https://storage.googleapis.com/tensorflow/keras-applications/{folder}/{name}',ROOT/'.keras'/'models'/name,expected,'md5')
    return check()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Only validate local assets, no downloads')
    parser.add_argument('--record-manifest',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.record_manifest: record_manifest()
    else:
        missing,weights=check() if args.check else prepare()
        if missing or weights:
            print(f'Missing or changed images: {len(missing)}; weights: {weights}')
            raise SystemExit(1)
        print('All 4,845 images and both pretrained weight files passed hash verification.')
