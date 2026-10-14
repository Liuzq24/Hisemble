import csv,json,pickle,subprocess,sys,tempfile
from pathlib import Path
import anndata as ad,numpy as np,scanpy as sc
from scipy.sparse import diags
from sklearn.datasets import make_blobs
root=Path(__file__).resolve().parents[1];analyses=root/'analyses'
def run(*args):subprocess.run([sys.executable,*map(str,args)],check=True)
run('-m','unittest','discover','-s',root/'tests','-p','test_*.py')
run(analyses/'convergence/test_runner_equivalence.py')
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);run(root/'examples/quickstart_synthetic.py','--output-dir',tmp)
 run(root/'examples/run_hisemble_from_h5ad.py','--input',tmp/'synthetic_hisemble.h5ad','--output',tmp/'reloaded.h5ad','--keys','view_a,view_b','--metrics','gaussian,gaussian','--resolution','0.5')
 # Verify the documented numeric result and exact H5AD reload reproducibility.
 original=ad.read_h5ad(tmp/'synthetic_hisemble.h5ad');reloaded=ad.read_h5ad(tmp/'reloaded.h5ad')
 assert original.n_obs==180 and original.obs['hisemble_clusters'].nunique()==4
 assert original.obsp['connectivities'].shape==(180,180) and original.obsp['connectivities'].nnz==7372
 assert original.obs_names.equals(reloaded.obs_names)
 assert np.array_equal(original.obs['hisemble_clusters'].astype(str),reloaded.obs['hisemble_clusters'].astype(str))
 assert (original.obsp['connectivities']!=reloaded.obsp['connectivities']).nnz==0
 # Exact scheduling-equivalence check, same graph and same seeds.
 x,y=make_blobs(n_samples=90,centers=3,n_features=8,random_state=42)
 d='Chen-2019-ATAC';data=tmp/'inputs';(data/'ADATA').mkdir(parents=True);(data/'lsi_results/ADATA').mkdir(parents=True)
 for suffix,key,folder in [('scBasset','projection','ADATA'),('snap','X_spectral','ADATA'),('lsi','X_lsi','lsi_results/ADATA')]:
  obj=ad.AnnData(np.zeros((90,1)),obs={'cell_type':y.astype(str)});obj.obsm[key]=x.astype(np.float32)
  with (data/folder/f'{d}_{suffix}.pkl').open('wb') as f:pickle.dump(obj,f)
 for script,out in [('seed_stability.py','serial'),('seed_stability_parallel.py','parallel')]:
  args=[analyses/script,'--data-root',data,'--output-dir',tmp/out,'--datasets',d,'--seeds','0','42','101']
  if out=='parallel':args+=['--seed-workers','2']
  run(*args)
 def rows(out):
  with (tmp/out/'seed_stability_per_seed.csv').open() as f:return list(csv.DictReader(f))
 assert rows('serial')==rows('parallel'),'Serial and parallel results differ'
 # Scanpy/DPT installed and interoperable with graph slots.
 n=100;a=ad.AnnData(np.zeros((n,1)));g=diags([np.ones(n-1),np.ones(n-1)],[-1,1],shape=(n,n),format='csr')
 a.obsp['connectivities']=g;a.uns['neighbors']={'connectivities_key':'connectivities','distances_key':'distances','params':{'method':'Hisemble consensus graph','n_neighbors':15}};a.uns['iroot']=0
 sc.tl.diffmap(a,n_comps=15);sc.tl.dpt(a,n_dcs=10,n_branchings=0);assert np.isfinite(a.obs['dpt_pseudotime']).all()
 for name in ['component_ablation.py','seed_stability.py','seed_stability_parallel.py','degraded_view_analysis.py','lsi_neighbor_analysis.py','lineage_trajectory_analysis.py']:run(analyses/name,'--help')
print('PASS: all tests (core/API and convergence), synthetic and AnnData examples, serial/parallel equality, finite graph-based DPT, six CLI entry points')
