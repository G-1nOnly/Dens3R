# Installation

## Create a conda environment
```bash
conda create --name auto_recon -y python=3.9
conda activate auto_recon
python -m pip install --upgrade pip setuptools
```

## Clone the repo
```shell
git clone --recurse-submodules git@github.com:zju3dv/AutoRecon.git
```

## Install AutoDecomp
```bash
cd third_party/AutoDecomp
```
Then, please install AutoDecomp based on its [installation guide](https://github.com/zju3dv/AutoDecomp/blob/main/docs/INSTALL.md).
We expect that it is installed in `third_party/AutoDecomp`.

## Install other dependencies
Install pytorch with CUDA (this repo has been tested with CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install faiss following the [official guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
```bash
# for example
conda install -c conda-forge faiss-gpu
```

## Install AutoRecon
```bash
cd path/to/AutoRecon
pip install -e .
# install tab completion
ns-install-cli
```
