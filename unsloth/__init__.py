# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings, importlib, sys
from packaging.version import Version
import os, re, subprocess, inspect
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training if multi-GPU is available
def init_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        return 0  # Default to single GPU if LOCAL_RANK is not set

local_rank = init_distributed()  # Initialize distributed environment

# Adjust CUDA_VISIBLE_DEVICES logic to support multi-GPU
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = os.environ["CUDA_VISIBLE_DEVICES"]
    if not devices.isdigit():
        first_id = devices.split(",")[0]
        warnings.warn(
            f"Unsloth: 'CUDA_VISIBLE_DEVICES' is currently {devices} \n"
            "Detected multiple CUDA devices. Unsloth now supports multi-GPU with DistributedDataParallel.\n"
            f"Setting primary CUDA device to {first_id} for the main process."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = devices  # Keep all devices for multi-GPU
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pass

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
pass

# Log Unsloth is being used
os.environ["UNSLOTH_IS_PRESENT"] = "1"

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"
        "We have some installation instructions on our Github page."
    )
except Exception as exception:
    raise exception
pass

# Ensure Pytorch version 2.x is being used
torch_version = torch.__version__.split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch < 2):
    raise ImportError("Unsloth only supports Pytorch 2 for now. Please update your Pytorch to 2.1.\n"
                      "We have some installation instructions on our Github page.")
elif (major_torch == 2) and (minor_torch < 2):
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
pass

# Torch 2.4 has including_emulation
major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = (major_version >= 8)

old_is_bf16_supported = torch.cuda.is_bf16_supported
if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):
    def is_bf16_supported(including_emulation=False):
        return old_is_bf16_supported(including_emulation)
    torch.cuda.is_bf16_supported = is_bf16_supported
else:
    def is_bf16_supported(): return SUPPORTS_BFLOAT16
    torch.cuda.is_bf16_supported = is_bf16_supported
pass

# Try loading bitsandbytes and triton
import bitsandbytes as bnb

if "SPACE_AUTHOR_NAME" not in os.environ and "SPACE_REPO_NAME" not in os.environ:
    import triton
    libcuda_dirs = lambda: None
    if Version(triton.__version__) >= Version("3.0.0"):
        try:
            from triton.backends.nvidia.driver import libcuda_dirs
        except:
            pass
    else:
        from triton.common.build import libcuda_dirs

    try:
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        libcuda_dirs()
    except:
        warnings.warn(
            "Unsloth: Running `ldconfig /usr/lib64-nvidia` to link CUDA."
        )

        if os.path.exists("/usr/lib64-nvidia"):
            os.system("ldconfig /usr/lib64-nvidia")
        elif os.path.exists("/usr/local"):
            possible_cudas = subprocess.check_output(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
            find_cuda = re.compile(r"[\s](cuda\-[\d\.]{2,})$")
            possible_cudas = [find_cuda.search(x) for x in possible_cudas]
            possible_cudas = [x.group(1) for x in possible_cudas if x is not None]

            if len(possible_cudas) == 0:
                os.system("ldconfig /usr/local/")
            else:
                find_number = re.compile(r"([\d\.]{2,})")
                latest_cuda = np.argsort([float(find_number.search(x).group(1)) for x in possible_cudas])[::-1][0]
                latest_cuda = possible_cudas[latest_cuda]
                os.system(f"ldconfig /usr/local/{latest_cuda}")
        pass

        importlib.reload(bnb)
        importlib.reload(triton)
        try:
            libcuda_dirs = lambda: None
            if Version(triton.__version__) >= Version("3.0.0"):
                try:
                    from triton.backends.nvidia.driver import libcuda_dirs
                except:
                    pass
            else:
                from triton.common.build import libcuda_dirs
            cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
            libcuda_dirs()
        except:
            warnings.warn(
                "Unsloth: CUDA is not linked properly.\n"
                "Try running `python -m bitsandbytes` then `python -m xformers.info`\n"
                "We tried running `ldconfig /usr/lib64-nvidia` ourselves, but it didn't work.\n"
                "You need to run in your terminal `sudo ldconfig /usr/lib64-nvidia` yourself, then import Unsloth.\n"
                "Also try `sudo ldconfig /usr/local/cuda-xx.x` - find the latest cuda version.\n"
                "Unsloth will still run for now, but maybe it might crash - let's hope it works!"
            )
    pass
pass

# Check for unsloth_zoo
try:
    import unsloth_zoo
except:
    raise ImportError("Unsloth: Please install unsloth_zoo via `pip install unsloth-zoo`")
pass

from .models import *
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *
