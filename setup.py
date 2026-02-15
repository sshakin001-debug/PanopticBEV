from os import path, listdir
import platform

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the base directory early so it can be used in make_extension
here = path.abspath(path.dirname(__file__))


def get_cuda_architectures():
    """
    Get CUDA architectures including sm_120 for RTX 5060 Ti.
    Compiles for multiple architectures for compatibility.
    
    Supports:
    - sm_70: V100, GTX 1080 Ti
    - sm_75: RTX 2080, T4
    - sm_80: A100, RTX 3090
    - sm_86: RTX 3080, 3090, 4090 (Ampere/Ada)
    - sm_89: RTX 4060, 4070, 4080 (Ada)
    - sm_90: H100 (Hopper)
    - sm_120: RTX 5060 Ti, 5070, 5080, 5090 (Blackwell)
    """
    import torch
    
    # Base architectures for broad compatibility
    arch_list = [
        "70",   # V100, GTX 1080 Ti
        "75",   # RTX 2080, T4
        "80",   # A100, RTX 3090
        "86",   # RTX 3080, 3090, 4090 (Ampere/Ada)
        "89",   # RTX 4060, 4070, 4080 (Ada)
        "90",   # H100 (Hopper)
        "120",  # RTX 5060 Ti, 5070, 5080, 5090 (Blackwell)
    ]
    
    # Check if we can detect the current GPU's architecture
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        current_arch = f"{capability[0]}{capability[1]}"
        if current_arch not in arch_list:
            arch_list.append(current_arch)
            print(f"[setup.py] Detected GPU arch sm_{current_arch}, adding to build")
    
    # Generate NVCC flags
    nvcc_flags = []
    for arch in arch_list:
        nvcc_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    
    return nvcc_flags


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


def make_extension(name, package):
    nvcc_flags = [
        "--expt-extended-lambda",
        "-O3",
        "--use_fast_math",
    ] + get_cuda_architectures()
    
    # Windows-specific flags
    cxx_flags = ["-O3"]
    if platform.system() == 'Windows':
        # MSVC-compatible flags
        cxx_flags = ["/O2"]
        nvcc_flags.append("-Xcompiler=/wd4819")  # Disable Unicode warnings
        # Allow VS 2025 (unsupported but works)
        nvcc_flags.append("-allow-unsupported-compiler")
        # Fix for PyTorch header ambiguity with VS 2025
        nvcc_flags.append("-DTORCH_DYNAMO_DISABLE_COMPILED_AUTOGRAD")
        # Force 64-bit compilation to fix pointer size mismatch
        nvcc_flags.append("-m64")
        nvcc_flags.append("-Xcompiler=-m64")
        nvcc_flags.append("--machine=64")
    
    return CUDAExtension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join("src", name)),
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        include_dirs=[
            path.join(here, "include"),  # Absolute path to include directory
            path.join(here, "src", name)  # Absolute path to source directory
        ],
    )


setuptools.setup(
    # Meta-data
    name="PanopticBEV",
    author="Nikhil Gosala",
    author_email="gosalan@cs.uni-freiburg.de",
    description="PanopticBEV Model Code",
    version="1.0.0",
    url="http://panoptic-bev.cs.uni-freiburg.de/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],

    python_requires=">=3.7, <4",

    # Package description
    packages=[
        "panoptic_bev",
        "panoptic_bev.algos",
        "panoptic_bev.config",
        "panoptic_bev.data",
        "panoptic_bev.models",
        "panoptic_bev.modules",
        "panoptic_bev.modules.heads",
        "panoptic_bev.utils",
        "panoptic_bev.utils.bbx",
        "panoptic_bev.utils.nms",
        "panoptic_bev.utils.parallel",
        "panoptic_bev.utils.roi_sampling",
    ],
    ext_modules=[
        make_extension("nms", "panoptic_bev.utils"),
        make_extension("bbx", "panoptic_bev.utils"),
        make_extension("roi_sampling", "panoptic_bev.utils")
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
