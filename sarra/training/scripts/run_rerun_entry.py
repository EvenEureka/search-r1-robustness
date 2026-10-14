"""Import an immutable adapter snapshot before launching distributed training (entry point used by train_3b.sh)."""
import argparse
import os
import runpy
import sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
a,remaining=p.parse_known_args()
assert (a.source/'training_adapter/rerun_main.py').is_file()
sys.path.insert(0,str(a.source.resolve()))
os.environ['PYTHONPATH']=str(a.source.resolve())+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['SKYRL_PYTHONPATH_EXPORT']='1'
sys.argv=[sys.argv[0],*remaining]
runpy.run_module('training_adapter.rerun_main',run_name='__main__')
