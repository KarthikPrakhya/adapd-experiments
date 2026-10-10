#!/usr/bin/env python3
"""Download the GSet G1 graph and check its canonical SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import warnings
from urllib.request import Request, urlopen


G1_URL = "https://web.stanford.edu/~yyye/yyye/Gset/G1"
G1_SHA256 = "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665"
DEFAULT_OUTPUT = Path("datasets/GSet/G1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path, force: bool = False) -> None:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        actual = sha256(output)
        if actual == G1_SHA256:
            print(f"GSet G1 already present and verified: {output}")
            return
        warnings.warn(
            f"Existing GSet G1 checksum differs from the paper instance: "
            f"expected {G1_SHA256}, found {actual}. Continuing with {output}; "
            "use --force to download it again.",
            RuntimeWarning,
        )
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(G1_URL, headers={"User-Agent": "AdaPD-reproducibility/1.0"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="G1.", suffix=".download", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request) as response:
                shutil.copyfileobj(response, temporary)

        actual = sha256(temporary_path)
        if actual != G1_SHA256:
            warnings.warn(
                "Downloaded GSet G1 checksum differs from the paper instance: "
                f"expected {G1_SHA256}, found {actual}. Continuing with the "
                "downloaded file.",
                RuntimeWarning,
            )
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if actual == G1_SHA256:
        print(f"Downloaded and verified GSet G1: {output}")
    else:
        print(f"Downloaded GSet G1 with a checksum warning: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, force=args.force)


if __name__ == "__main__":
    main()
