# AdaPD experiment notebooks

This directory contains the executed, standalone notebooks used to generate
the experimental results underlying the optimal-transport and Max-Cut figures
in the paper. Saved outputs are retained so that the reported runs can be
inspected without repeating the longer experiments.

## Contents

```text
.
|-- maxcut_experiments.ipynb
|-- ot_experiments.ipynb
|-- download_gset_g1.py
|-- generate_ot_cost.py
|-- requirements.txt
|-- requirements-data.txt
`-- datasets/
    |-- GSet/
    |   `-- G1_ground_truth.pkl
    `-- OTDD/
        `-- cifar10_mnist_2000x2000_resnet18_ground_truth.npz
```

The GSet G1 graph is downloaded separately because its official distribution
page does not state an explicit redistribution license. The OT cost matrix is
generated locally through the pinned OTDD pipeline. The two compact
ground-truth cache files are distributed with the repository.

## Create the notebook environment

Use Python 3.10 or 3.11. From this directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, scipy, matplotlib, sdpap; print('Notebook dependencies OK')"
```

The Max-Cut notebook uses SDPA for the SDP ground truth when the supplied cache
is unavailable. The OT notebook uses SciPy HiGHS for its LP ground truth. No
CVXPY, CVXOPT, MOSEK, or SCS installation is required.

## Prepare the datasets

### GSet G1

Download G1 from its official Stanford distribution and check its SHA-256
checksum with:

```bash
python download_gset_g1.py
```

The script compares the downloaded file against the paper-instance checksum
below. A mismatch produces a warning but does not stop execution:

```text
73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665
```

### CIFAR-10/MNIST OT cost matrix

Generate the matrix used by the released notebook at:

```text
datasets/OTDD/cifar10_mnist_2000x2000_resnet18.npy
```

The expected SHA-256 checksum is:

```text
9d56668033e35a27bbc7d77a6151b58ee803b70603bba09bf87aa9f34e64740f
```

The matrix contains squared Euclidean distances between frozen ResNet-18
ImageNet-1K V1 features for 2,000 CIFAR-10 samples and 2,000 MNIST samples. To
generate it from the source datasets, install the additional dependencies and
run:

```bash
python -m pip install -r requirements-data.txt
python generate_ot_cost.py
```

The generator downloads CIFAR-10, MNIST, and the pretrained ResNet-18 weights
through OTDD and Torchvision. The pinned package versions and explicit model
weight identifier define the released instance. The script reports the output
checksum and warns if it differs from the reference above.

## Run the experiments

```bash
source .venv/bin/activate
python -m jupyter lab
```

Open either notebook and run the cells in order. Jupyter should use the Python
kernel from the active `.venv`; select that environment if prompted. The
committed configuration uses 1,000 iterations for each parameter-tuning run
and 10,000 iterations for each selected long run. For a quick smoke test,
reduce `TUNING_MAXITER` and `LONG_MAXITER` before running.

Both notebooks write CSV files under `results/`. The individual long-run CSV
files are the numerical inputs used to construct the paper figures.

## Licensing and provenance

The notebook and utility code is released under the MIT License in `LICENSE`.
The generated data artifacts are covered by `DATA_LICENSE.md`. Third-party
software and dataset provenance are documented in `THIRD_PARTY_NOTICES.md`.
