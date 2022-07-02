# Co-domain Symmetry for Complex-Valued Deep Learning [CVPR 2022]

Official code implementation for CDS. 

Requires Pytorch 1.9.0 and CUDA 11.0

### Instructions:

To train the CDS-E model on various encodings of CIFAR10/100/SVHN datasets, please run

`bash scripts/rgb_e.sh`

To train the CDS-I model on various encodings of CIFAR10/100/SVHN datasets, please run

`bash scripts/rgb_i.sh`

To train the CDS-MSTAR model on MSTAR dataset, please run

`bash scripts/mstar.sh`

To train the CDS-Large model, please run

`python train_big.py`

CDS-Large pre-trained model [weights](https://drive.google.com/file/d/16Eka8UKDQdqzutdKxiYw1ss1gGtejwk9/view?usp=sharing)



Please cite our work as:

```
@InProceedings{Singhal_2022_CVPR,
    author    = {Singhal, Utkarsh and Xing, Yifei and Yu, Stella X.},
    title     = {Co-Domain Symmetry for Complex-Valued Deep Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {681-690}
}
```
