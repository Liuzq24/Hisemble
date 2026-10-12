#!/usr/bin/env python3
"""Launch all 24 exact-core convergence jobs; preserve per-dataset logs/status."""
import argparse, concurrent.futures, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent

def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path, value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--iterations',type=int,default=100)
    a=p.parse_args()
    if a.iterations < 50: p.error("Batch original-metric export requires at least 50 iterations")
    datasets=json.loads((HERE/'datasets.json').read_text())
    assert len(datasets)==24 and len(set(datasets))==24
    missing=[]
    for d in datasets:
        for rel in [f'ADATA/{d}_scBasset.pkl',f'ADATA/{d}_snap.pkl',f'lsi_results/ADATA/{d}_lsi.pkl']:
            if not (a.data_root/rel).is_file(): missing.append(rel)
    if missing: raise SystemExit('Missing inputs: '+json.dumps(missing))
    a.run_root.mkdir(parents=True,exist_ok=True)
    (a.run_root/'logs').mkdir(exist_ok=True)
    (a.run_root/'datasets').mkdir(exist_ok=True)
    status_path=a.run_root/'batch_status.json'
    if status_path.exists(): raise SystemExit('Existing batch_status.json: refusing duplicate launch')
    status={'started_utc':now(),'controller_pid':os.getpid(),'python':sys.executable,'workers':a.workers,'threads_per_job':a.threads,'target_iterations':a.iterations,'expected_datasets':24,'expected_view_trajectories':72,'status':'running','datasets':{d:{'status':'queued'} for d in datasets}}
    atomic(status_path,status)
    environment=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        environment[key]=str(a.threads)
    environment['PYTHONUNBUFFERED']='1'
    # A fresh directory per job avoids replacing any existing source/results.
    def run(d):
        output=a.run_root/'datasets'/d
        command=[sys.executable,str(HERE/'run_core_convergence.py'),'--data-root',str(a.data_root),'--output-dir',str(output),'--dataset',d,'--iterations',str(a.iterations),'--chunk-size','2048','--seed','42']
        with (a.run_root/'logs'/f'{d}.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
            return d,proc,now()
    active={}
    pending=list(datasets)
    while pending or active:
        while pending and len(active)<a.workers:
            d=pending.pop(0)
            try:
                d,proc,start=run(d)
                active[d]=proc
                status['datasets'][d]={'status':'running','pid':proc.pid,'started_utc':start}
            except Exception as error:
                status['datasets'][d]={'status':'failed_to_launch','error':str(error),'finished_utc':now()}
            atomic(status_path,status)
        time.sleep(5)
        for d,proc in list(active.items()):
            code=proc.poll()
            if code is None: continue
            complete=(a.run_root/'datasets'/d/'COMPLETE.json').is_file()
            status['datasets'][d].update(status='complete' if code==0 and complete else 'failed',exit_code=code,finished_utc=now())
            del active[d]
            atomic(status_path,status)
    failures=[d for d,s in status['datasets'].items() if s['status']!='complete']
    status.update(status='failed' if failures else 'complete',finished_utc=now(),failed_datasets=failures)
    atomic(status_path,status)
    if failures: raise SystemExit('Dataset failures: '+json.dumps(failures))
    subprocess.run([sys.executable,str(HERE/'export_original_metrics.py'),'--input-dir',str(a.run_root/'datasets'),'--output-dir',str(a.run_root/'original_metrics'),'--iterations','50','--require-all'],check=True,env=environment)
    status['original_metrics_status']='exported'
    atomic(status_path,status)

if __name__=='__main__': main()
