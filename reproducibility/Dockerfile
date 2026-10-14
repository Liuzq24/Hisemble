FROM python:3.10.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/opt/hisemble:/opt/hisemble/analyses

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY reproducibility/requirements-core.txt /tmp/requirements-core.txt
COPY reproducibility/requirements-trajectory.txt /tmp/requirements-trajectory.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements-core.txt -r /tmp/requirements-trajectory.txt

COPY hisemble/ /opt/hisemble/hisemble/
COPY analyses/ /opt/hisemble/analyses/
COPY tests/ /opt/hisemble/tests/
COPY examples/ /opt/hisemble/examples/
COPY workflows/ /opt/hisemble/workflows/
COPY results/ /opt/hisemble/results/
COPY reproducibility/ /opt/hisemble/reproducibility/
COPY setup.py prepare_workflows.py config.example.json /opt/hisemble/
COPY README.md Reproduction_Tutorial.md ANALYSIS_MAP.md INPUTS_AND_ENVIRONMENTS.md DEPENDENCY_IMPORTS.csv LICENSE /opt/hisemble/
WORKDIR /work

RUN python -c "import hisemble; print('Hisemble core environment OK')"

CMD ["bash"]
