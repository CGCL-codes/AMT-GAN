# AMT-GAN

The official implementation of our CVPR 2022 paper "**Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**".[[Paper](https://arxiv.org/abs/2203.03121)] 

## Abstract

While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality.

In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a  new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

## Update

**2022/3/22**   We are sorting out our code. The PyTorch version will be released very soon. Stay Tuned!


## BibTeX

```bibtex
@article{hu2022protecting,
  title =   {Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer},
  author =  {Hu, Shengshan and Liu, Xiaogeng and Zhang, Yechao and Li, Minghui and Zhang, Leo Yu and Jin, Hai and Wu, Libing},
  journal = {arXiv preprint arXiv:2203.03121},
  year =    {2022}
}
```
