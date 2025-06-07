import torch
import math
import numpy as np

# -----------------------------------------------------------------------------
# Simple table-based Asymmetric Numeral Systems (tANS) implementation in Python
# -----------------------------------------------------------------------------
# The implementation follows a range variant of tANS.  It is not optimised for
# performance but mimics the interface of ``arithmetic.py`` used in the CUDA
# version.  All operations run on the CPU and operate on PyTorch tensors.


def gaussian_cdf(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Gaussian cumulative distribution function."""
    return 0.5 * torch.erfc(-(x - mean) / (scale * math.sqrt(2.0)))


def calculate_cdf(mean: torch.Tensor,
                  scale: torch.Tensor,
                  Q: torch.Tensor,
                  min_value: torch.Tensor,
                  max_value: torch.Tensor) -> torch.Tensor:
    """Replicates ``calculate_cdf`` from the CUDA implementation.

    Parameters are identical to the original version.  ``mean`` and ``scale`` are
    one dimensional tensors describing Gaussian parameters.  ``Q`` may either be
    a scalar tensor or a tensor matching the size of ``mean``.
    """
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    Q = Q.to(mean.dtype)
    scale = torch.clamp(scale, min=1e-9)

    values = torch.arange(min_value.item() - 0.5,
                          max_value.item() + 1.5,
                          device=mean.device,
                          dtype=mean.dtype)
    values = values.view(1, -1) * Q.unsqueeze(-1)

    cdf = gaussian_cdf(values, mean.unsqueeze(-1), scale.unsqueeze(-1))
    return cdf.to(torch.float32)


# ---------- tANS helper functions -------------------------------------------

_PRECISION = 16
_STATE_BITS = 16
_MASK = (1 << _PRECISION) - 1


def _prepare_freq(row_cdf: torch.Tensor, precision: int = _PRECISION):
    pmf = (row_cdf[1:] - row_cdf[:-1]).cpu().numpy()
    freq = np.maximum(1, np.round(pmf * (1 << precision)).astype(np.int64))
    diff = (1 << precision) - int(freq.sum())
    freq[-1] += diff
    cum = np.zeros_like(freq)
    cum[1:] = np.cumsum(freq[:-1])
    return freq, cum


def arithmetic_encode(sym: torch.Tensor,
                      cdf: torch.Tensor,
                      chunk_size: int,
                      N: int,
                      Lp: int,
                      precision: int = _PRECISION):
    """Encode ``sym`` using a simple tANS implementation.

    The returned byte stream and counts mirror the arithmetic coder interface.
    """
    sym_np = sym.to(torch.int64).cpu().numpy()
    cdf_cpu = cdf.cpu()
    state = 1 << precision
    out_words = []

    for i in range(N - 1, -1, -1):
        row_cdf = cdf_cpu[i]
        freq, cum = _prepare_freq(row_cdf, precision)
        s = int(sym_np[i])
        f = int(freq[s])
        c = int(cum[s])
        while state >= (f << _STATE_BITS):
            out_words.append(state & 0xFFFF)
            state >>= _STATE_BITS
        state = ((state // f) << precision) + (state % f) + c

    out_words.append(state)
    encoded = np.array(out_words[::-1], dtype=np.uint16).tobytes()
    byte_stream = torch.frombuffer(encoded, dtype=torch.uint8).clone()
    cnt = torch.tensor([len(encoded)], dtype=torch.int32)
    return byte_stream, cnt


def arithmetic_decode(cdf: torch.Tensor,
                      byte_stream: torch.Tensor,
                      cnt: torch.Tensor,
                      chunk_size: int,
                      N: int,
                      Lp: int,
                      precision: int = _PRECISION):
    """Decode a tANS encoded byte stream."""
    device = cdf.device
    data = byte_stream.cpu().numpy().tobytes()
    words = np.frombuffer(data, dtype=np.uint16)
    idx = len(words) - 1
    state = int(words[idx])
    idx -= 1

    out = np.zeros(N, dtype=np.int16)
    cdf_cpu = cdf.cpu()
    mask = (1 << precision) - 1

    for i in range(N):
        row_cdf = cdf_cpu[i]
        freq, cum = _prepare_freq(row_cdf, precision)
        x = state & mask
        s = np.searchsorted(cum + freq, x, side="right") - 1
        out[i] = s
        f = int(freq[s])
        c = int(cum[s])
        state = f * (state >> precision) + x - c
        while state < (1 << _STATE_BITS) and idx >= 0:
            state = (state << _STATE_BITS) | int(words[idx])
            idx -= 1

    return torch.tensor(out, dtype=torch.int16, device=device)
