import triton
import triton.language as tl
import torch
from .utils import calculate_settings
import torch.distributed as dist


def setup_distributed(local_rank):
    """Initialize distributed training"""
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank


@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)

    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask = mask)


def swiglu_fg_kernel(e, g, device=None):
    """Multi-GPU version of SwiGLU forward pass"""
    if device is None:
        device = e.device
    
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    
    # Create output tensor on the same device as input
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fg_kernel[grid](e, g, h, n_elements, BLOCK_SIZE=1024)
    
    return h


@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE : tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask = mask, other = 0)
    e_row  = tl.load(e  + offsets, mask = mask, other = 0).to(tl.float32)
    g_row  = tl.load(g  + offsets, mask = mask, other = 0)

    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row  =  f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    tl.store(DW + offsets, h_row,  mask = mask)
    tl.store(e  + offsets, df_row, mask = mask)
    tl.store(g  + offsets, de_row, mask = mask)


def swiglu_DWf_DW_dfg_kernel(DW, e, g, device=None):
    """Multi-GPU version of SwiGLU backward pass"""
    if device is None:
        device = DW.device
        
    batch_seq_len, hd = e.shape
    n_elements = e.numel()
    
    # Ensure all tensors are on the same device
    DW = DW.to(device)
    e = e.to(device)
    g = g.to(device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _DWf_DW_dfg_kernel[grid](DW, e, g, n_elements, BLOCK_SIZE=1024)
    
    return DW, e, g


class DistributedSwiGLU(torch.nn.Module):
    """Wrapper class for distributed SwiGLU operations"""
    def __init__(self):
        super().__init__()
        self.device = torch.cuda.current_device()

    def forward(self, e, g):
        return swiglu_fg_kernel(e, g, self.device)

    def backward(self, DW, e, g):
        return swiglu_DWf_DW_dfg_kernel(DW, e, g, self.device)


def shard_tensor(tensor, world_size, rank):
    """Split tensor across multiple GPUs"""
    batch_size = tensor.size(0)
    split_size = batch_size // world_size
    return tensor[rank * split_size:(rank + 1) * split_size]


def gather_tensor(tensor, world_size, rank):
    """Gather tensor from multiple GPUs"""
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)

