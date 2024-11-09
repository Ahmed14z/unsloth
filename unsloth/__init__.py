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

import warnings, importlib, sys, os, re, subprocess, inspect
from packaging.version import Version
import numpy as np

# Enable multi-GPU compatibility
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Set flag for Unsloth usage
os.environ["UNSLOTH_IS_PRESENT"] = "1"

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Unsloth: PyTorch is not installed. Visit https://pytorch.org/.\n"
        "Installation instructions are available on our GitHub page."
    )

# Ensure compatibility with PyTorch 2+
torch_version = Version(torch.__version__)
if torch_version < Version("2.0.0"):
    raise ImportError("Unsloth requires PyTorch 2.0 or later. Please update your PyTorch.")
elif torch_version < Version("2.2.0"):
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

# Verify CUDA device capability
major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = major_version >= 8

# Override torch's bfloat16 support check if necessary
if hasattr(torch.cuda, 'is_bf16_supported') and "including_emulation" in str(inspect.signature(torch.cuda.is_bf16_supported)):
    def is_bf16_supported(including_emulation=False):
        return torch.cuda.is_bf16_supported(including_emulation)
    torch.cuda.is_bf16_supported = is_bf16_supported
else:
    def is_bf16_supported():
        return SUPPORTS_BFLOAT16
    torch.cuda.is_bf16_supported = is_bf16_supported

# Load bitsandbytes and triton, handling possible CUDA linking issues
import bitsandbytes as bnb
import triton

def attempt_cuda_linking():
    if os.path.exists("/usr/lib64-nvidia"):
        os.system("ldconfig /usr/lib64-nvidia")
    elif os.path.exists("/usr/local"):
        possible_cudas = subprocess.check_output(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
        cuda_versions = [re.search(r"(cuda\-[\d\.]+)$", line) for line in possible_cudas]
        cuda_versions = [match.group(1) for match in cuda_versions if match]
        if cuda_versions:
            latest_cuda = max(cuda_versions, key=lambda ver: float(re.search(r"([\d\.]+)", ver).group(1)))
            os.system(f"ldconfig /usr/local/{latest_cuda}")
        else:
            os.system("ldconfig /usr/local")

if "SPACE_AUTHOR_NAME" not in os.environ and "SPACE_REPO_NAME" not in os.environ:
    try:
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        if Version(triton.__version__) >= Version("3.0.0"):
            from triton.backends.nvidia.driver import libcuda_dirs
        else:
            from triton.common.build import libcuda_dirs
        libcuda_dirs()
    except:
        warnings.warn("Unsloth: Running CUDA linking command to resolve dependencies.")
        attempt_cuda_linking()
        importlib.reload(bnb)
        importlib.reload(triton)
        try:
            cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
            if Version(triton.__version__) >= Version("3.0.0"):
                from triton.backends.nvidia.driver import libcuda_dirs
            else:
                from triton.common.build import libcuda_dirs
            libcuda_dirs()
        except:
            warnings.warn(
                "Unsloth: CUDA linking unsuccessful. Please run `ldconfig /usr/lib64-nvidia` or `ldconfig /usr/local/cuda-xx.x`.\n"
                "Unsloth may continue with limited functionality."
            )

# Ensure `unsloth_zoo` is installed
try:
    import unsloth_zoo
except ImportError:
    raise ImportError("Unsloth: Please install `unsloth_zoo` via `pip install unsloth-zoo`")

# Import main Unsloth modules
from .models import *
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *
