# Hisemble 🧬

> **A scalable representation fusion framework for robust ensemble clustering of single-cell data.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚠️ Prerequisites

Hisemble is designed as a lightweight, downstream representation fusion aggregator. **It operates on pre-computed embeddings.**

Before running Hisemble, you should have already extracted low-dimensional embeddings using your preferred upstream tools (e.g., scBasset, SnapATAC2, LSI, scVI) and stored them in the `.obsm` attribute of your Scanpy `AnnData` object. We do not restrict or include these heavy upstream models in our dependencies to maintain maximum flexibility and speed.

---

## 🛠️ Installation

**Step 1: Clone the repository**
```bash
git clone [https://github.com/Liuzq24/Hisemble.git](https://github.com/Liuzq24/Hisemble.git)
cd Hisemble
```

**Step 2: Create and activate a Conda environment**
```bash
conda create -n Hisemble python=3.8.18 -y
conda activate Hisemble
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install Hisemble (Editable mode)**
```bash
pip install -e .
```