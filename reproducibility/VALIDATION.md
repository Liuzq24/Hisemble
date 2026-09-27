# Reproducibility environment validation

Validation date: 2026-09-27

The Docker image was built from the repository root with:

```bash
docker build -f reproducibility/Dockerfile -t hisemble-core:revision .
```

The clean build completed successfully on a Linux/arm64 container runtime.
The resulting local image identifier was:

```text
sha256:b03912ba9cb1a532fb0ac91d489db0765b60e7c3d0845183eb31dcb490d8d6f8
```

The following checks passed inside the container:

1. Import and version checks for NumPy 1.24.3, SciPy 1.10.1,
   scikit-learn 1.3.2, igraph 0.11.8, AnnData 0.9.2,
   pandas 2.0.3, Matplotlib 3.7.5, and psutil 5.9.8.
2. All six tests in `test_hisemble_io.py`.
3. Two core tests confirming that both iterative diffusion and direct averaging
   are invariant to permutations of the input-view order.
4. Two package API tests: an AnnData/Seurat interoperability test and an
   end-to-end synthetic `run_hisemble` test covering graph construction, two
   diffusion iterations, Leiden clustering, and AnnData output storage.

The image tag and digest above document the revision-stage local validation.
They are not a permanent public identifier. Replace the provisional tag in the
manuscript and response letter with the final GitHub release or archival DOI
after the Supplemental Code package is frozen.
