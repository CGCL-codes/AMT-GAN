# AMT-GAN

The official implementation of our CVPR 2022 paper "**Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**".[[Paper](https://arxiv.org/abs/2203.03121)] 

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.11.0](https://img.shields.io/badge/pytorch-1.11.0-red.svg?style=plastic)

## Abstract
While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality.\
In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a  new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

<img src="pipeline.png"/>

## Latest Update
**2022/5/10**   We have renewed the demo test dataset (100 pics to 1000) for a broader evaluation.\
**2022/4/15**   We have released the official implementation code.

## Setup
- **Get code**
```shell 
git clone https://github.com/CGCL-codes/AMT-GAN.git
```

- **Build environment**
```shell
cd AMT-GAN
# use anaconda to build environment 
conda create -n AMT-GAN python=3.8
conda activate AMT-GAN
# install packages
pip install -r requirements.txt
```

- **Download assets**
  - Pretrained face recognition models and datasets are needed to train and test AMT-GAN, please download these assets at:
    [[Google](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view?usp=sharing)] [[Baidu](https://pan.baidu.com/s/1hiIV1GVZTwV1o2Q4DfC2Cg)] pw:1bpv
  - Unzip the assets.zip file in ```AMT-GAN/assets```, this file contains the pre-trained FR models, the training data for AMT-GAN, and a subset of CelebA-HQ for evaluation.\
*Please note that we do not own the datasets, for more information about them, check out [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch).*

- **Download checkpoints**
  - The pretrained checkpoints is available at:
    [[Google](https://drive.google.com/file/d/1QSsH_s8fKAwhFgSBv5014yGtewlmNJkI/view?usp=sharing)] [[Baidu](https://pan.baidu.com/s/1cxxe7TkeQ4zGAk4xLi3e_w)] pw:qxfd
  - Unzip the assets.zip file in ```AMT-GAN/checkpoints```
  
- **The final project should be like this:**
    ```shell
    AMT-GAN
    └- assets
       └- datasets
       └- models
    └- backbone
    └- checkpoints
       └- G.pth
       └- ...
    └- ...
    ```

## Quick Start
- **Train AMT-GAN**
```shell 
python train.py  # results saved in /log
```
- **Simple evaluation on local models and Face++**
```shell 
python test.py  # generated images saved in /assets/datasets/save
```

## Customize
- **Target identity**
  - Put target face image in ```/assets/datasets/target```
  - Modify ```TARGET_PATH``` in ```train.train_net```
  
- **Local(training) models**
  - Modify ```MODELS``` in ```configs.yaml```, such as ```['facenet', 'ir152', 'irse50'] → ['facenet', 'ir152', 'mobile_face']```
  - To load your own pretrained models, modify ```backbone.solver.build_model``` and ```configs.yaml``` accordingly.

- **Local(testing) models**
  - Modify ```args.model_names``` in ```test.attack_local_models```

- **Test image**
  - Put test face image in ```/assets/datasets/test```, assert that the test image is from the same identy of target face image.
  - Modify ```args.target_path``` in ```test.attack_local_models, test.attack_faceplusplus``` accordingly.

- **Reference images**
  - Put reference images (for makeup transfer) in ```/assets/datasets/reference```

*Please note that if you want to train a new AMT-GAN, G.pth, H.pth, D_A.pth and D_B.pth in /checkpoints should be deleted or renamed.*

## Acknowledge

Some of the codes are built upon [PSGAN](https://github.com/wtjiang98/PSGAN), pretrained face recognition models are from [Adv-Makeup](https://github.com/TencentYoutuResearch/Adv-Makeup).

## BibTeX 
```bibtex
@InProceedings{Hu_2022_CVPR,
    author    = {Hu, Shengshan and Liu, Xiaogeng and Zhang, Yechao and Li, Minghui and Zhang, Leo Yu and Jin, Hai and Wu, Libing},
    title     = {Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-Robust Makeup Transfer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {15014-15023}
}
```
