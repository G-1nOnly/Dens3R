# AutoDecomp: 3D object discovery from casual object captures
This is the coarse decomposition part of the method proposed in [AutoRecon: Automated 3D Object Discovery and Reconstruction](https://zju3dv.github.io/autorecon/files/autorecon.pdf). It can be used to preprocess a casual capture (object-centric multi-view images or a video) which estimate the camera poses with SfM and localize the salient foreground object for further reconstruction.

## Install
Please install AutoDecomp following [INSTALL.md](docs/INSTALL.md).

## Inference
### Inference with demo data
Here we takes `assets/custom_data_example/co3d_chair` as an example.
You can run automatic foreground scene decomposition with: `scripts/run_pipeline_demo_low-res.sh`.
You should get a similar visualization as in `assets/custom_data_example/co3d_chair/vis_decomposition.html`.

You can take the data structure and the script as a reference to run the pipeline on your own data.

### Inference with CO3D data
1. Download the demo data from [Google Drive](https://drive.google.com/drive/folders/1wgtV2WycT2zXVPCMQYm05q-0SIH2ZpER?usp=drive_link) and put them under `data/`.
2. Run one of the script in `scripts/test_pipeline_co3d_manual-poses/cvpr` (use low-res images for feature matching and DINO features) or `scripts/test_pipeline_co3d_manual-poses` (use high-res images for feature matching and DINO features) to run the inference pipeline.
3. We save camera poses, decomposition results and visualization to `path_to_the_instance/auto-deocomp_sfm-transformer`.

### Inference with annotated data in the IDR format
We also support import camera poses saved in the IDR format and localize the foreground object. You can run one of the script in `scripts/test_pipeline_bmvs/cvpr` or `scripts/test_pipeline_bmvs` for reference.

## Citation
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