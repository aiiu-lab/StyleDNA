# StyleDNA: A High-Fidelity Age and Gender Aware Kinship Face Synthesizer

## Description

This is the official repository of "[StyleDNA: A High-Fidelity Age and Gender Aware Kinship Face Synthesizer]()" paper.

## Get Started

#### Prerequisites

- Linux or macOS
- Python3
- PyTorch == 1.9.0+cu111
- dlib == 19.22.1

#### Pretrained Model Weights

- [Face Predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

  Shape predictor from dlib.

- [StyleGAN2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)

  StyleGAN2 model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.

- Other pertained model weights



#### Inference Notebook

We provide a Jupyter notebook version running on [Google Colab](https://colab.research.google.com/drive/1FHf5ftbYtAfvODEqj5lp-S1cir44UniT?usp=sharing) for fast inferecing.



## Inferencing

#### Pretrained Weights

Download all the pretrained model weights and put in *./pretrained_model/*

#### Inferencing

Having trained your model, you can use `./inference.py` to test the model on a set of images.
For example,

```
python3 inference.py --mom_path ./test/mom.png --dad_path ./test/dad.png
```



## Credits

We borrow code from
 - stylegan2: https://github.com/rosinality/stylegan2-pytorch  
  Copyright (c) 2019 Kim Seonghyeon  
  License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

 - pSp: https://github.com/eladrich/pixel2style2pixel  
  Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
  License (MIT) https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE  

 - InsightFace_Pytorch: https://github.com/TreB1eN/InsightFace_Pytorch  
  Copyright (c) 2018 TreB1eN  
  License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

 - dlib: http://dlib.net/face_landmark_detection.py.html  
  License (BSL-1.0) https://github.com/davisking/dlib/blob/master/LICENSE.txt  

 - Face alignment: https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5



## Citation

If you use this code for your research, please cite our paper.  
[StyleDNA: A High-Fidelity Age and Gender Aware Kinship Face Synthesizer]()

'''

'''

