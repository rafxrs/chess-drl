import os
import torch
import numpy

# Get the directory containing PyTorch
torch_dir = os.path.dirname(torch.__file__)
# Find and rename the OpenMP DLL
torch_omp_path = os.path.join(torch_dir, "lib", "libiomp5md.dll")
if os.path.exists(torch_omp_path):
    os.rename(torch_omp_path, torch_omp_path + ".bak")
    print(f"Renamed {torch_omp_path} to prevent conflicts")