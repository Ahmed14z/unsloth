# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import torch
import triton
from packaging.version import Version
import bitsandbytes as bnb
import ctypes

# Set device and initialize CUDA stream for multi-GPU
device = torch.device(f"cuda:{torch.distributed.get_rank()}" if torch.distributed.is_initialized() else "cuda")
CUDA_STREAM = torch.cuda.current_stream(device=device)

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2

# AMP compatibility for torch version >= 2.4
if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")

# Triton tanh compatibility based on version
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
else:
    import triton.language as tl
    triton_tanh = tl.math.tanh

# bitsandbytes CUDA functions initialization
HAS_CUDA_STREAM = Version(bnb.__version__) > Version("0.43.3")
CUDA_STREAM = torch.cuda.current_stream(device=device) if HAS_CUDA_STREAM else None
get_ptr = bnb.functional.get_ptr

cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16

def calculate_settings(n: int) -> (int, int,):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps: int = 4
    if BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >= 8192: num_warps = 16
    elif BLOCK_SIZE >= 2048: num_warps = 8
    return BLOCK_SIZE, num_warps

def QUANT_STATE(W):
    return getattr(W, "quant_state", None)

def get_lora_parameters(proj):
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight
    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None
    active_adapter = proj.active_adapters[0] if hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s

def get_lora_parameters_bias(proj):
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    W = base_layer.weight
    bias = base_layer.bias
    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None, bias
    active_adapter = proj.active_adapters[0] if hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s, bias

def quant_state_unpack(quant_state):
    if isinstance(quant_state, list):
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, *_ = state2
    else:
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    return absmax, shape, dtype, blocksize, offset, state2

# Dequantization function
def fast_dequantize(W, quant_state=None, out=None):
    if quant_state is None: return W
    absmax, shape, dtype, blocksize, offset, state2 = quant_state_unpack(quant_state)
    code2 = state2.code
    absmax2 = state2.absmax
    blocksize2 = state2.blocksize

    if out is None:
        out = torch.empty(shape, dtype=dtype, device=device)
    else:
        assert out.shape == shape
        assert out.dtype == dtype

    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(n_elements_absmax, dtype=torch.float32, device=device)
    ptr_out_absmax = get_ptr(out_absmax)
    cdequantize_blockwise_fp32(get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax, ctypes.c_int(blocksize2), ctypes.c_int(n_elements_absmax), CUDA_STREAM)
    out_absmax += offset

    fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else cdequantize_blockwise_bf16_nf4
    fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out), ctypes.c_int(blocksize), ctypes.c_int(out.numel()), CUDA_STREAM)
    return out.t() if W.shape[0] == 1 else out

# Matrix-vector multiplication with quantization
def fast_gemv(X, W, quant_state, out=None):
    if quant_state is None: return torch.matmul(X, W, out=out)
    absmax, shape, dtype, blocksize, offset, state2 = quant_state_unpack(quant_state)
    code2 = state2.code
    absmax2 = state2.absmax
    blocksize2 = state2.blocksize
    bout = shape[0]

    if out is None:
        out = torch.empty((1, 1, bout), dtype=dtype, device=device)

    df = torch.empty(absmax.shape, dtype=torch.float32, device=device)
    cdequantize_blockwise_fp32(get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), get_ptr(df), ctypes.c_int(blocksize2), ctypes.c_int(df.numel()), CUDA_STREAM)
    df += offset
    absmax = df

    fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else cgemm_4bit_inference_naive_bf16
    fx(ctypes.c_int32(shape[0]), ctypes.c_int32(1), ctypes.c_int32(shape[1]), get_ptr(X), get_ptr(W), get_ptr(absmax), get_ptr(code2), get_ptr(out), ctypes.c_int32(shape[0]), ctypes.c_int32((X.shape[2] + 1) // 2), ctypes.c_int32(shape[0]), ctypes.c_int32(blocksize), CUDA_STREAM)
    return out

# Linear layer with LoRA weights
def fast_linear_forward(proj, X, temp_lora=None, out=None):
    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1:
        return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)
    if W_quant is None:
        out = torch.matmul(X, W.t().to(device), out=out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W.to(device), W_quant, out=out)
    else:
        W = fast_dequantize(W.t().to(device), W_quant)
        out = torch.matmul(X, W, out=out)
    if lora_A is not None:
        temp_lora = handle_lora(out, X, lora_A, lora_B, lora_S, bsz, in_dim, temp_lora)
    if bias is not None:
        out += bias
    return out

# Helper functions for LoRA operations
def handle_lora(out, X, lora_A, lora_B, lora_S, bsz, in_dim, temp_lora):
    out_dim = out.shape[2]
    dtype = X.dtype
    lora_A, lora_B = lora_A.to(dtype).to(device), lora_B.to(dtype).to(device)
    if bsz == 1:
        temp_lora = torch.mv(lora_A, X.ravel().to(device), out=temp_lora)
        out.addmv_(lora_B, temp_lora, alpha=lora_S)
    else:
        temp_lora = torch.mm(X.view(bsz, in_dim).to(device), lora_A.t(), out=temp_lora)
        out.addmm_(temp_lora, lora_B.t(), alpha=lora_S)
    return out.view(bsz, 1, out_dim)
def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype
    # Dequantize W if required
    W = fast_dequantize(W.t().to(device), W_quant)
    
    # Adjust reshape conditionally based on input dimensions
    reshape = X.dim() == 3
    if reshape:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])

    # Initial matrix multiplication of X and W
    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W  # Free memory if no longer needed

    # Add LoRA components if A and B are defined
    if A is not None and B is not None:
        # Ensure that A and B have compatible shapes for multiplication
        A, B = A.t(), B.t()  # Adjust transpose based on shape requirements
        try:
            out += (X @ A.to(dtype).to(device)) @ (s * B.to(dtype).to(device))
        except RuntimeError as e:
            print(f"Error in matmul_lora: {e}")
            print("Shape of X:", X.shape)
            print("Shape of A:", A.shape)
            print("Shape of B:", B.shape)
            print("Shape of s:", s.shape)
            raise e

    return out.view(batch, seq_len, -1) if reshape else out
