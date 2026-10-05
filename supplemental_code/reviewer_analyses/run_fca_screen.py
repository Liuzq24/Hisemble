"""Numerical FCA screen: retain every completed outcome; no plotting."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
DATASETS=['FCA_fetal_33184180_'+x for x in ['cerebellum','pancreas','stomach','muscle','intestine']]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--timeout-seconds',type=int,default=14400);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 statuses=[]
 if not a.summarize_only:
  for d in DATASETS:
   for v in ['full','no_diffusion']:
    cmd=[sys.executable,str(Path(__file__).with_name('component_ablation.py')),'--data-root',str(a.data_root),'--dataset',d,'--variant',v,'--output-dir',str(a.output_dir),'--k','15','--iterations','20','--seed','42']
    with (a.output_dir/f'{d}__{v}.log').open('w') as f:
     try:
      result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout_seconds);status='completed' if result.returncode==0 else 'failed'
     except subprocess.TimeoutExpired:status='timeout'
    statuses.append({'dataset':d,'variant':v,'status':status})
    (a.output_dir/'run_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
 rows=[]
 for d in DATASETS:
  files=[a.output_dir/f'{d}__{v}.json' for v in ['full','no_diffusion']]
  if not all(f.exists() for f in files):continue
  full,no=[json.loads(f.read_text()) for f in files]
  assert full['n_cells']==no['n_cells'] and not full['subsampled'] and not no['subsampled']
  rows.append({'dataset':d,'n_cells':full['n_cells'],'full_ARI':full['ARI'],'no_diffusion_ARI':no['ARI'],'delta_ARI':full['ARI']-no['ARI'],'full_NMI':full['NMI'],'no_diffusion_NMI':no['NMI'],'delta_NMI':full['NMI']-no['NMI'],'positive_both':full['ARI']>no['ARI'] and full['NMI']>no['NMI']})
 if not rows:raise RuntimeError('No completed paired comparisons; inspect run_status.json')
 with (a.output_dir/'screen_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 positive=[r for r in rows if r['positive_both'] and not r['dataset'].endswith('cerebellum')]
 selected=max(positive,key=lambda r:r['delta_ARI'])['dataset'] if positive else None
 (a.output_dir/'screen_summary.json').write_text(json.dumps({'exploratory':True,'all_completed_comparisons':rows,'selected':selected,'rule':'Both ARI and NMI positive; largest ARI improvement among positive new candidates'},indent=2)+'\n')
if __name__=='__main__':main()
