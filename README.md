# Workshop in Machine Learning for Computer Graphics 
Exploring Generative Image Inpainting with Contextual Attention

Forked from JiahuiYu's repo, the following repo was used for testing the [model](https://arxiv.org/abs/1801.07892) by Jiahui Yu et. al as part of a Workshop on the subject of GANs and computer vision. Our goal was to study the model both theoretically and through ablation studies, and to attempt in fine tuning it on a specific domain- images of faces. 

In the following [report](https://docs.google.com/document/d/1Ha68clQ0eCSrFMmmjQS6u1lJ4hhUVU-cjZA0XITdIYU/edit?usp=sharing) we present:
* Ablation studies of important stages in the model
    * The Attention Path
    * The Hallucination Path
    *  The Spatial Mask
* The Usefulness of adding perceptual loss in the case of Faces.

<img src="https://preview.ibb.co/hpBMSp/Capture.jpg" width="75%"/> 

For running and training the model see original instructions below.


# Generative Image Inpainting with Contextual Attention

[CVPR 2018 Paper](https://arxiv.org/abs/1801.07892) | [ArXiv](https://arxiv.org/abs/1801.07892) | [Project](http://jiahuiyu.com/deepfill) | [Demo](http://jiahuiyu.com/deepfill) | [YouTube](https://youtu.be/xz1ZvcdhgQ0) | [BibTex](#citing)

## Run

0. Requirements:
    * Install python3.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
    * Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (run `pip install git+https://github.com/JiahuiYu/neuralgym`).
1. Training:
    * Prepare training images filelist and shuffle it ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)). 
    * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
3. Testing:
    * Run `python test.py --image examples/input.png --mask examples/mask.png --output examples/output.png --checkpoint model_logs/your_model_dir`.
4. Still have questions?
    * If you still have questions (e.g.: How filelist looks like? How to use multi-gpus? How to do batch testing?), please first search over closed issues. If the problem is not solved, please open a new issue.

## Pretrained models

[CelebA-HQ](https://drive.google.com/open?id=1lpluFXyWDxTY6wcjixQGWX8jxUUMlyBW) | [Places2](https://drive.google.com/open?id=1M3AFy7x9DqXaI-fINSynW7FJSXYROfv-) | [CelebA](https://drive.google.com/open?id=1sP8ViF3mxUMN--xpKqonEeW9d8S8pJEo) | [ImageNet](https://drive.google.com/open?id=136APWSdPRAF7-XoS8sMBTLV-X3f-ogE0) | 

Download the model dirs and put it under `model_logs/` (rename `checkpoint.txt` to `checkpoint` because google drive automatically add ext after download). Run testing or resume training as described above. All models are trained with images of resolution 256x256 and largest hole size 128x128, above which the results may be deteriorated. We provide several example test cases. Please run:

```bash
# Places2 512x680 input
python test.py --image examples/places2/wooden_input.png --mask examples/places2/wooden_mask.png --output examples/output.png --checkpoint_dir model_logs/release_places2_256
# CelebA 256x256 input
python test.py --image examples/celeba/celebahr_patches_164036_input.png --mask examples/center_mask_256.png --output examples/output.png --checkpoint_dir model_logs/release_celeba_256/
# CelebA-HQ 256x256 input
# Please visit CelebA-HQ demo at: jhyu.me/demo
# ImageNet 256x256 input
python test.py --image examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_input.png --mask examples/center_mask_256.png --output examples/output.png --checkpoint_dir model_logs/release_imagenet_256
```
Citing
```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```
