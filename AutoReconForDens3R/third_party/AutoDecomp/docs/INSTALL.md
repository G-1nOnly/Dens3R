# Installation
## Clone the repo
```shell
git clone --recurse-submodules git@github.com:zju3dv/AutoDecomp.git
```

## Install pytorch
The code is teseted with torch==1.13.1.
```shell
# e.g., install with conda
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# e.g., install with pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Install pytorch3d following the [official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```shell
# for example
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Install [hloc](https://github.com/cvg/Hierarchical-Localization)
```shell
cd third_party/Hierarchical-Localization
pip install -e .
```

## Install LoFTR
```shell
cd third_party/LoFTR
pip install -e .
```
Please also download the [LoFTR checkpoints](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) and put them under `third_party/LoFTR/weights/`.

## Install AutoDecomp
```shell
cd path/to/AutoDecomp
pip install -e .
```