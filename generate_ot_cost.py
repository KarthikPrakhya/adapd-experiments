#!/usr/bin/env python3
"""Regenerate the CIFAR-10/MNIST cost matrix used by the OT notebook."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from otdd.pytorch.datasets import load_torchvision_data


DEFAULT_OUTPUT = Path(
    "datasets/OTDD/cifar10_mnist_2000x2000_resnet18.npy"
)
REFERENCE_SHA256 = (
    "9d56668033e35a27bbc7d77a6151b58ee803b70603bba09bf87aa9f34e64740f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_features(loader, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    features = []
    with torch.no_grad():
        for images, _ in loader:
            features.append(model(images.to(device)).detach().cpu())
    return torch.cat(features, dim=0).numpy().astype(np.float64)


def generate_cost_matrix(n: int, m: int, seed: int, device_name: str) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_name)
    source_loaders = load_torchvision_data(
        "CIFAR10", resize=28, maxsize=n
    )[0]
    target_loaders = load_torchvision_data(
        "MNIST", resize=28, to3channels=True, maxsize=m
    )[0]

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    source = extract_features(source_loaders["train"], model, device)[:n]
    target = extract_features(target_loaders["train"], model, device)[:m]

    source_norms = np.sum(source**2, axis=1)
    target_norms = np.sum(target**2, axis=1)
    cost = source_norms[:, None] + target_norms[None, :] - 2.0 * source @ target.T
    return np.maximum(cost, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--m", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite")

    output.parent.mkdir(parents=True, exist_ok=True)
    cost = generate_cost_matrix(args.n, args.m, args.seed, args.device)
    np.save(output, cost)

    digest = sha256(output)
    print(f"Wrote {cost.shape} {cost.dtype} cost matrix to {output}")
    print(f"SHA-256: {digest}")
    if (args.n, args.m, args.seed) == (2000, 2000, 0):
        if digest == REFERENCE_SHA256:
            print("The generated file matches the reference release matrix.")
        else:
            print(
                "The generated matrix does not match the reference release matrix. "
                "Upstream dataset, package, preprocessing, or model-weight versions "
                f"may differ. Expected SHA-256: {REFERENCE_SHA256}."
            )


if __name__ == "__main__":
    main()
