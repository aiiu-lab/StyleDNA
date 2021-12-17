# StyleDNA: A High-Fidelity Age and Gender Aware Kinship Face Synthesizer

## Description

This is the official repository of "StyleDNA: A High-Fidelity Age and Gender Aware Kinship Face Synthesizer" paper for inferencing. 

## Get Started

#### Prerequisites

- Linux or macOS
- Python3
- PyTorch == 1.9.0+cu111
- dlib ==19.22.1

#### Pretrained Model Weights

- [Face Predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

  Shape predictor from dlib.

- [StyleGAN2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)

  StyleGANw model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.

- Other pertained model weights



#### Inference Notebook

We provide a Jupyter notebook running on Google Colab () for fast inferecing.



## Inferencing

#### Pretrained Weights

Download all the pretrained models and put in *./pretrained_model/*

#### Inferencing

Having trained your model, you can use `./inference.py` to apply the model on a set of images.
For example,

```
python3 inference.py --mom_path ./test/mom.png --dad_path ./test/dad.png
```



## Credits

We borrow code from [stylegan2](https://github.com/rosinality/stylegan2-pytorch), [pSp](https://github.com/eladrich/pixel2style2pixel), and [dlib](http://dlib.net/face_landmark_detection.py.html) example code.


