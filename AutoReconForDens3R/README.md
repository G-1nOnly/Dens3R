# AutoRecon: Automated 3D Object Discovery and Reconstruction

### [Project Page](https://zju3dv.github.io/autorecon) | [Paper](https://zju3dv.github.io/autorecon/files/autorecon.pdf)

![teaser](assets/teaser_video.gif)

> [AutoRecon: Automated 3D Object Discovery and Reconstruction](https://zju3dv.github.io/autorecon/files/autorecon.pdf)  
> Yuang Wang, Xingyi He, Sida Peng, Haotong Lin, Hujun Bao, Xiaowei Zhou  
> CVPR 2023

# About
This is a refactored version of the [AutoRecon](https://zju3dv.github.io/autorecon) project based on the NeRFStudio and the SDFStudio codebase. We separate the project into two parts. The coarse decomposition part is implemented in the [AutoDecomp](https://github.com/zju3dv/AutoDecomp) repo, which can be used as a general tool for 3D object discovery and preprocessing casual captures for object reconstruction. The neural surface reconstruction part is implemented here.


# Installation
Please refer to the [installation guide](docs/INSTALL.md) for detailed instructions.


# Run the pipeline
## Run the pipeline with your own data
Here we take a demo data as an example. In this example, we assume only a sequential stream of images is available. You can easily adapt your data to use AutoRecon.
1. Download the demo data from [Google Drive](https://drive.google.com/drive/folders/1IFbK9b7gzqwh9QkZe6zoLmcSFr_zW3rJ?usp=drive_link) and put it under `data` (-> `data/custom_data_example/...`)
2. Run the pipeline with `exps/code-release/run_pipeline_demo_low-res.sh`

> NOTE: In the demo script, we assume the images come from a sequential video stream and use sequential matching for SfM. If you have unordered images, the default setting might lead to inferior results and it is recommended to use exhaustive matching or vocab-tree instead.


## Run the pipeline with annotated data in the IDR format
1. Download the BlendedMVS data from [GoogleDrive](https://drive.google.com/drive/folders/1ZLQ0hap6o_Tjr7S6H_EAn17pz4_qFluW?usp=sharing) and put it under `data` (-> `data/BlendedMVS/...`)
2. Run one of the scripts in `exps/code-release/bmvs`

## Run the pipeline with CO3D data
1. Download the CO3D data from [GoogleDrive](https://drive.google.com/drive/folders/1u-ugNhwFVtV6TKZ2J29iwcdQY2iwVnoi?usp=sharing) and put it under `data` (-> `data/CO3D_DEMO/...`)
2. Run one of the scripts in `exps/code-release/co3d_demo`


# Extract Mesh
You can take the following script as a reference to extract mesh:
```bash
# Extract mesh with MC
LOG_DIR="path_to_log_dir"

MC_RES=512
MESH_FN="extracted_mesh_res-${MC_RES}.ply"
MESH_PATH="${LOG_DIR}/${MESH_FN}"

ns-extract-mesh \
	--load-config $LOG_DIR/config.yml \
    --load-dir $LOG_DIR/sdfstudio_models \
	--output-path $MESH_PATH \
    --chunk_size 25000 --store_float16 True \
    --resolution $MC_RES \
    --use_train_scene_box True \
    --seg_aware_sdf False \
    --remove_internal_geometry None \
    --remove_non_maximum_connected_components True \
    --close_holes False --simplify_mesh_final False
```

> NOTE: We postprocess the extracted mesh before evaluation by removing possible internal geometries with ambient occlusion. The postprocessing code depends on pymeshlab which only works on some machines. You can remove possible internal geometries manually with ambient occlusion using MeshLab.


# Citation
If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{wang2023autorecon,
  title={AutoRecon: Automated 3D Object Discovery and Reconstruction},
  author={Wang, Yuang and He, Xingyi and Peng, Sida and Lin, Haotong and Bao, Hujun and Zhou, Xiaowei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21382--21391},
  year={2023}
}
```


# Acknowledgement
This code is built upon the awesome projects including [nerfstudio](https://github.com/nerfstudio-project/nerfstudio/), [sdfstudio](https://github.com/autonomousvision/sdfstudio/blob/master/README.md), [nerfacc](https://github.com/KAIR-BAIR/nerfacc), [tyro](https://github.com/brentyi/tyro) and more. Thanks for these great projects!
