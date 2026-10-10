# Third-party notices and data provenance

## GSet G1

The Max-Cut experiment uses the GSet G1 benchmark instance distributed from
Yinyu Ye's Stanford page:

https://web.stanford.edu/~yyye/yyye/Gset/

The G1 file is not redistributed in this repository because the distribution
page does not state an explicit license. `download_gset_g1.py` downloads the
file from that page and compares its SHA-256 digest with the instance used in
the paper, issuing a warning if they differ.

## OT cost matrix

The OT matrix is generated locally from CIFAR-10 and MNIST using frozen
ResNet-18 ImageNet-1K V1 features. The construction is implemented in
`generate_ot_cost.py` and uses the data-loading setup from the Optimal
Transport Dataset Distance (OTDD) project:

https://github.com/microsoft/otdd

OTDD is Copyright (c) Microsoft Corporation and is distributed under the MIT
License. A copy of that license is included in `licenses/OTDD-MIT.txt`.

The source datasets and pretrained weights are not redistributed here. They
are obtained through OTDD and torchvision when the generation script is run:

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- MNIST: https://yann.lecun.com/exdb/mnist/
- Torchvision ResNet-18 weights:
  https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

The generated matrix contains pairwise feature-space costs, not source images
or labels. Its exact provenance and checksum are recorded in `README.md`.
