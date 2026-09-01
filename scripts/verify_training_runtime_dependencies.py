#!/usr/bin/env python3
"""Verify the deliberate dependency boundary in the trainer process."""

from importlib.metadata import PackageNotFoundError, version

EXPECTED = {
    "cuda-bindings": "12.9.7",
    "cuda-python": "12.9.7",
    "grpcio": "1.83.0",
    "grpcio-health-checking": "1.83.0",
    "grpcio-reflection": "1.83.0",
    "numpy": "1.26.4",
    "openai": "2.54.0",
    "pillow": "11.3.0",
    "protobuf": "7.35.1",
    "setuptools": "80.9.0",
    "tifffile": "2025.5.10",
}
SERVER_ONLY = (
    "langgraph-api",
    "langgraph-cli",
    "langgraph-runtime-inmem",
    "grpcio-tools",
    "flash-qla",
)
COMPILER_ONLY = (
    "nvidia-cuda-nvcc",
    "nvidia-cuda-tileiras",
    "nvidia-nvvm",
)


def main() -> None:
    for package, expected in EXPECTED.items():
        actual = version(package)
        if actual != expected:
            raise SystemExit(
                f"training runtime mismatch for {package}: expected {expected}, got {actual}"
            )
    for package in SERVER_ONLY:
        try:
            actual = version(package)
        except PackageNotFoundError:
            continue
        raise SystemExit(
            f"training prefix contains server-only package {package} {actual}"
        )
    for package in COMPILER_ONLY:
        actual = version(package)
        if not actual.startswith("13.2."):
            raise SystemExit(
                f"unexpected FlashInfer compiler tool {package} version {actual}"
            )

    import grpc
    import langchain_openai
    import langgraph
    import openai
    import sglang
    import torch

    if torch.version.cuda != "12.9":
        raise SystemExit(f"training CUDA runtime drifted to {torch.version.cuda}")

    print(
        "training runtime dependency verification passed",
        grpc.__version__,
        openai.__version__,
        langchain_openai.__file__,
        langgraph.__path__,
        sglang.__file__,
    )


if __name__ == "__main__":
    main()
