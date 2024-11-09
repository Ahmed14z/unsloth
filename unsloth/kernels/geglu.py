import triton
import triton.language as tl
import torch
from .utils import calculate_settings, triton_tanh

@triton.jit
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    # Kernel implementation remains the same
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)

    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask = mask)

def geglu_exact_forward_kernel(gate, up):
    """
    Multi-GPU compatible GEGLU forward pass
    
    Args:
        gate: Input tensor (batch, seq_len, hd)
        up: Input tensor (batch, seq_len, hd) on same device as gate
    Returns:
        Output tensor on same device as inputs
    """
    assert gate.device == up.device, "Gate and up tensors must be on same device"
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device=gate.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _exact_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024,)
    return out

@triton.jit
def _exact_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    # Kernel implementation remains the same
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)

    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row
    
    f_row = f_row.to(DW_row.dtype)
    h_row  =  f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row

    t = 0.3989422804014327  # 1/sqrt(2*pi)
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    tl.store(DW + offsets, h_row,  mask = mask)
    tl.store(e  + offsets, df_row, mask = mask)
    tl.store(g  + offsets, de_row, mask = mask)

def geglu_exact_backward_kernel(DW, e, g):
    """
    Multi-GPU compatible GEGLU backward pass
    
    Args:
        DW, e, g: Input tensors, must all be on same device
    Returns:
        Gradient tensors on same device as inputs
    """
    assert DW.device == e.device == g.device, "All input tensors must be on same device"
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _exact_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE=1024,)
    return DW, e, g

@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    # Kernel implementation remains the same
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = 0.7978845608028654
    
    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)

    f_row = 0.5 * e_row * (
        triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) \
        + 1.0
    )
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask = mask)

def geglu_approx_forward_kernel(gate, up):
    """
    Multi-GPU compatible approximate GEGLU forward pass
    
    Args:
        gate: Input tensor (batch, seq_len, hd)
        up: Input tensor (batch, seq_len, hd) on same device as gate
    Returns:
        Output tensor on same device as inputs
    """
    assert gate.device == up.device, "Gate and up tensors must be on same device"
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device=gate.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _approx_forward_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024,)
    return out

@triton.jit
def _approx_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr,):
    # Kernel implementation remains the same
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)

    s = 0.7978845608028654
    a = s * e_row
    b = a * 0.044715 * e_row * e_row
    T = 1.0 + triton_tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b) 
    df_de = T2 + Q2

    f_row = T2 * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row  =  f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    tl.store(DW + offsets, h_row,  mask = mask)
    tl.store(e  + offsets, df_row, mask = mask)
    tl.store(g  + offsets, de_row, mask = mask)

def geglu_approx_backward_kernel(DW, e, g):
    """
    Multi-GPU compatible approximate GEGLU backward pass
    
    Args:
        DW, e, g: Input tensors, must all be on same device
    Returns:
        Gradient tensors on same device as inputs
    """
    assert DW.device == e.device == g.device, "All input tensors must be on same device"
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _approx_backward_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE=1024,)
    return DW, e, g