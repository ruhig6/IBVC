# IBVC: Interpolation-driven B-frame Video Compression

The official PyTorch implementation of [IBVC](https://arxiv.org/abs/2309.13835).

Authors: Chenming Xu, Meiqin Liu, Chao Yao, Weisi Lin, Yao Zhao

## Baseline

We offer a solution for training DCVC that can replicate the performance described in the article, and we have made the code openly available.

## Introduction

We propose a simple yet effective structure called Interpolation-driven B-frame Video Compression (IBVC). Our approach only involves two major operations: video frame interpolation and artifact reduction compression. IBVC introduces a bit-rate free MEMC based on interpolation, which avoids optical-flow quantization and additional compression distortions. Later, to reduce duplicate bit-rate consumption and focus on unaligned artifacts, a residual guided masking encoder is deployed to adaptively select the meaningful contexts with interpolated multi-scale dependencies. In addition, a conditional spatio-temporal decoder is proposed to eliminate location errors and artifacts instead of using MEMC coding in other methods. The experimental results on B-frame coding demonstrate that IBVC has significant improvements compared to the relevant state-of-the-art methods.

## Implemention

The code manuscript has been open-sourced to the [IBVC_net.py](https://github.com/ruhig6/IBVC/blob/main/DCVC/subnet/src/models/IBVC_net.py), and the specific training, testing, and pre-trained model is coming soon.

## Acknowledgement

The implementation is based on [DCVC](https://github.com/microsoft/DCVC/tree/main/DCVC) and [IFRNet](https://github.com/ltkong218/IFRNet).

## Citation

If you find this work useful for your research, please cite:
```
@article{li2021deep,
  title={Deep Contextual Video Compression},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
@inproceedings{kong2022ifrnet,
  title={{IFRNet}: Intermediate feature refine network for efficient frame interpolation},
  author={Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1969--1978},
  year={2022}
}
@article{xu2024ibvc,
  title={{IBVC}: Interpolation-driven B-frame Video Compression},
  author={Xu, Chenming and Liu, Meiqin and Yao, Chao and Lin, Weisi and Zhao, Yao},
  journal={arXiv preprint arXiv:2309.13835},
  year={2024}
}
```
