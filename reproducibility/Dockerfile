FROM python:3.10.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/opt/hisemble:/opt/hisemble/supplemental_code/reviewer_analyses

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY reproducibility/requirements-core.txt /tmp/requirements-core.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements-core.txt

COPY . /opt/hisemble
WORKDIR /work

RUN python -c "import hisemble; print('Hisemble core environment OK')"

CMD ["bash"]
