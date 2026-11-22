"""Project entry point: audit predictions and report exact accuracies."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def call(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('results','prepare','check-assets'),nargs='?',default='results')
    args=p.parse_args()
    if args.action=='results':
        call('verify.py','--model','InceptionResNetV2')
        call('verify.py','--model','EfficientNetV2M')
        call('summarize_augmentation.py')
        call('paper_tables.py')
    elif args.action=='prepare': call('prepare_assets.py')
    else: call('prepare_assets.py','--check')
