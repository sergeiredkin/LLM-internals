"""Verify that the local environment is ready for LLM training."""

import platform
import sys

import torch


def gib(nbytes: int) -> float:
    return nbytes / 2**30


def main() -> None:
    print(f"Python:       {platform.python_version()} ({sys.executable})")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA build:   {torch.version.cuda}")
    print(f"CUDA ready:   {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: PyTorch cannot access a CUDA GPU.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    print(f"GPU:           {props.name}")
    print(f"Compute cap.:  {props.major}.{props.minor}")
    print(f"VRAM:          {gib(total):.2f} GiB total, {gib(free):.2f} GiB free")
    print(f"BF16:          {torch.cuda.is_bf16_supported()}")
    print(
        "SDPA:          "
        f"{hasattr(torch.nn.functional, 'scaled_dot_product_attention')}"
    )

    # Exercise allocation, matrix multiplication, autograd, and synchronization.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x = torch.randn(1024, 1024, device=device, dtype=dtype, requires_grad=True)
    loss = (x @ x.T).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"CUDA smoke test passed ({dtype}, loss={loss.item():.4f}).")

    if gib(free) < 8:
        print("WARNING: Less than 8 GiB was free before the smoke test.")
        print("Stop Ollama or other GPU workloads before model training.")


if __name__ == "__main__":
    main()
